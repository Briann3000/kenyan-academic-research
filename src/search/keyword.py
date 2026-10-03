"""Keyword retrieval engine implementing standard BM25 (Okapi BM25)."""

import re
import time
import logging
from typing import List, Optional
from rank_bm25 import BM25Okapi

from src.search.models import SearchResult, SearchResponse, RetrievalDocument
from src.search.corpus import RetrievalCorpus

logger = logging.getLogger(__name__)

# Standard English stopwords (minimal clean set for deterministic normalization)
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenizes text using alphanumeric word boundaries and lowercasing."""
    if not text or not isinstance(text, str):
        return []
    tokens = re.findall(r"\b\w+\b", text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STANDARD_STOPWORDS and len(t) > 1]
    return tokens


class BM25SearchEngine:
    """Standard Okapi BM25 Keyword Search Engine (rank-bm25 0.2.2)."""

    def __init__(
        self,
        corpus: RetrievalCorpus,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True
    ):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.build_index()

    def build_index(self):
        """Builds the BM25 inverted index from all document search_text entries."""
        start_time = time.perf_counter()
        self.tokenized_corpus = [
            tokenize_text(doc.search_text, remove_stopwords=self.remove_stopwords)
            for doc in self.corpus.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"Built BM25 index for {len(self.corpus.documents)} papers in {elapsed_ms:.2f}ms")

    def search(self, query: str, top_k: int = 10) -> SearchResponse:
        """Executes a BM25 keyword search against the indexed corpus."""
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string, got {type(query)}")
        
        cleaned_query = query.strip()
        if not cleaned_query:
            return SearchResponse(
                query=query,
                retrieval_method="bm25",
                total_hits=0,
                latency_ms=0.0,
                results=[]
            )

        if top_k <= 0:
            return SearchResponse(
                query=query,
                retrieval_method="bm25",
                total_hits=0,
                latency_ms=0.0,
                results=[]
            )

        start_time = time.perf_counter()
        query_tokens = tokenize_text(cleaned_query, remove_stopwords=self.remove_stopwords)

        if not query_tokens:
            # Query was made entirely of stopwords or non-alphanumeric chars
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                query=query,
                retrieval_method="bm25",
                total_hits=0,
                latency_ms=round(elapsed_ms, 3),
                results=[]
            )

        scores = self.bm25.get_scores(query_tokens)
        
        # Pair documents with scores, filtering out zero/negative scores if desired,
        # or sorting all positive matching documents
        scored_pairs = []
        for idx, score in enumerate(scores):
            if score > 0.0:
                scored_pairs.append((self.corpus.documents[idx], float(score)))

        # Sort descending by score, tie-breaking deterministically by paper_id
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
                retrieval_method="bm25",
                snippet=snippet,
                metadata={
                    "k1": self.k1,
                    "b": self.b,
                    "matched_query_tokens": query_tokens
                }
            ))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResponse(
            query=query,
            retrieval_method="bm25",
            total_hits=len(results),
            latency_ms=round(elapsed_ms, 3),
            results=results
        )
