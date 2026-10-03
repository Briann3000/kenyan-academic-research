"""Dense semantic search engine using BAAI/bge-small-en-v1.5 ONNX runtime, L2-normalized embeddings, and NumPy exact dot product."""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from src.search.models import SearchResult, SearchResponse, RetrievalDocument
from src.search.corpus import RetrievalCorpus
from src.ingestion.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL_REPO = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_SEQ_LENGTH = 512

EMBEDDINGS_CACHE_FILE = PROCESSED_DATA_DIR / "paper_embeddings.npy"
EMBEDDINGS_METADATA_FILE = PROCESSED_DATA_DIR / "embeddings_metadata.json"


class BGEOnnxEmbedder:
    """Lightweight BGE embedder using Tokenizers and ONNX Runtime CPU inference."""

    def __init__(self, model_repo: str = DEFAULT_MODEL_REPO, max_seq_length: int = MAX_SEQ_LENGTH):
        self.model_repo = model_repo
        self.max_seq_length = max_seq_length
        self.tokenizer: Optional[Tokenizer] = None
        self.session: Optional[ort.InferenceSession] = None
        self._load_model()

    def _load_model(self):
        """Downloads and initializes the ONNX model and tokenizer from HuggingFace."""
        logger.info(f"Loading ONNX weights and tokenizer for '{self.model_repo}'...")
        
        # Download tokenizer.json
        tok_path = hf_hub_download(repo_id=self.model_repo, filename="tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=self.max_seq_length)
        self.tokenizer.enable_padding(length=self.max_seq_length)

        # Download ONNX model weights (bge-small has official onnx/model.onnx)
        try:
            model_path = hf_hub_download(repo_id=self.model_repo, filename="onnx/model.onnx")
        except Exception:
            # Fallback to root model.onnx if in root
            model_path = hf_hub_download(repo_id=self.model_repo, filename="model.onnx")

        # Configure ONNX CPU Session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        logger.info(f"Initialized ONNX InferenceSession for '{self.model_repo}' successfully.")

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """Encodes a list of text strings into [N, 384] dense embeddings."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch_texts)

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            # Prepare feed dict matching model inputs
            feed = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }

            # Filter feed inputs to only those expected by ONNX graph
            input_names = [inp.name for inp in self.session.get_inputs()]
            feed = {k: v for k, v in feed.items() if k in input_names}

            outputs = self.session.run(None, feed)
            # BGE uses CLS token pooling (first token: outputs[0][:, 0])
            last_hidden_state = outputs[0]
            cls_embeddings = last_hidden_state[:, 0]

            if normalize:
                # Unit L2 normalization
                norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                cls_embeddings = cls_embeddings / norms

            all_embeddings.append(cls_embeddings.astype(np.float32))

        return np.vstack(all_embeddings)


class SemanticSearchEngine:
    """Dense semantic retrieval engine backed by BGE embeddings and exact NumPy dot product."""

    def __init__(
        self,
        corpus: RetrievalCorpus,
        model_name: str = DEFAULT_MODEL_REPO,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        embeddings_cache_file: Optional[Path] = EMBEDDINGS_CACHE_FILE,
        metadata_cache_file: Optional[Path] = EMBEDDINGS_METADATA_FILE
    ):
        self.corpus = corpus
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.embeddings_cache_file = Path(embeddings_cache_file) if embeddings_cache_file else None
        self.metadata_cache_file = Path(metadata_cache_file) if metadata_cache_file else None

        self.embedder: Optional[BGEOnnxEmbedder] = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.paper_id_list: List[str] = [doc.paper_id for doc in self.corpus.documents]
        self.paper_id_to_idx: Dict[str, int] = {p_id: i for i, p_id in enumerate(self.paper_id_list)}

        self._initialize_embedder()
        self._load_or_generate_embeddings()

    def _initialize_embedder(self):
        """Initializes the BGE ONNX embedder."""
        self.embedder = BGEOnnxEmbedder(model_repo=self.model_name, max_seq_length=MAX_SEQ_LENGTH)

    def _load_or_generate_embeddings(self):
        """Loads cached embeddings if valid, otherwise computes and caches them."""
        if (
            self.embeddings_cache_file
            and self.embeddings_cache_file.exists()
            and self.metadata_cache_file
            and self.metadata_cache_file.exists()
        ):
            try:
                with open(self.metadata_cache_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if (
                    meta.get("model_name") == self.model_name
                    and meta.get("corpus_size") == len(self.corpus)
                    and meta.get("paper_ids") == self.paper_id_list
                ):
                    logger.info(f"Loading cached document embeddings from {self.embeddings_cache_file}...")
                    self.doc_embeddings = np.load(self.embeddings_cache_file)
                    if self.doc_embeddings.shape == (len(self.corpus), 384):
                        return
            except Exception as exc:
                logger.warning(f"Failed to load cached embeddings: {exc}. Recomputing...")

        self.generate_and_cache_embeddings()

    def generate_and_cache_embeddings(self) -> Tuple[np.ndarray, float]:
        """Encodes all corpus documents with L2 normalization and saves to cache."""
        texts = [doc.search_text for doc in self.corpus.documents]
        logger.info(f"Encoding {len(texts)} documents using '{self.model_name}'...")

        start_time = time.perf_counter()
        embeddings = self.embedder.encode(texts, batch_size=32, normalize=True)
        duration_s = time.perf_counter() - start_time
        logger.info(f"Generated {embeddings.shape} embeddings in {duration_s:.2f}s")

        self.doc_embeddings = embeddings.astype(np.float32)

        if self.embeddings_cache_file and self.metadata_cache_file:
            self.embeddings_cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.embeddings_cache_file, self.doc_embeddings)
            meta = {
                "model_name": self.model_name,
                "dimension": int(self.doc_embeddings.shape[1]),
                "corpus_size": len(self.corpus),
                "generation_duration_seconds": round(duration_s, 2),
                "paper_ids": self.paper_id_list,
                "normalization": "L2",
                "similarity": "cosine_inner_product",
                "max_seq_length": MAX_SEQ_LENGTH
            }
            with open(self.metadata_cache_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Cached embeddings and metadata to {self.embeddings_cache_file}")

        return self.doc_embeddings, duration_s

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Executes a dense semantic retrieval query using BGE asymmetric encoding and cosine similarity."""
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query)}")

        cleaned_query = query.strip()
        if not cleaned_query or top_k <= 0:
            return SearchResponse(
                query=query,
                retrieval_method="semantic",
                total_hits=0,
                latency_ms=0.0,
                results=[]
            )

        start_time = time.perf_counter()

        # Apply official BGE query instruction prefix
        formatted_query = f"{self.query_instruction}{cleaned_query}"
        query_vec = self.embedder.encode([formatted_query], normalize=True)[0]

        # Dot product of L2 normalized vectors = exact Cosine Similarity
        scores = np.dot(self.doc_embeddings, query_vec)

        # Build scored list and apply deterministic tie-breaking: score DESC, paper_id ASC
        scored_pairs = [
            (self.corpus.documents[idx], float(scores[idx]))
            for idx in range(len(scores))
        ]
        scored_pairs.sort(key=lambda x: (-x[1], x[0].paper_id))

        results: List[SearchResult] = []
        effective_k = min(top_k, len(scored_pairs))

        for rank, (doc, score) in enumerate(scored_pairs[:effective_k], 1):
            snippet = doc.abstract[:200] + "..." if doc.abstract else doc.title
            results.append(SearchResult(
                paper_id=doc.paper_id,
                title=doc.title,
                score=round(score, 4),
                rank=rank,
                publication_year=doc.publication_year,
                doi=doc.doi,
                retrieval_method="semantic",
                snippet=snippet,
                metadata={
                    "model_name": self.model_name,
                    "dimension": int(self.doc_embeddings.shape[1]),
                    "query_instruction": self.query_instruction
                }
            ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResponse(
            query=query,
            retrieval_method="semantic",
            total_hits=len(results),
            latency_ms=round(elapsed_ms, 3),
            results=results
        )
