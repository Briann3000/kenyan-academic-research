"""Normalization utilities to transform raw OpenAlex works into clean tabular/JSON structures for profiling."""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def reconstruct_abstract(abstract_inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstructs text from OpenAlex's abstract_inverted_index.
    
    OpenAlex format: {"Word": [pos1, pos2], "Other": [pos3]}
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return None

    try:
        # Find maximum position index
        max_pos = -1
        for word, positions in abstract_inverted_index.items():
            if positions:
                max_pos = max(max_pos, max(positions))

        if max_pos < 0:
            return None

        # Build position list
        words = [""] * (max_pos + 1)
        for word, positions in abstract_inverted_index.items():
            for pos in positions:
                if 0 <= pos <= max_pos:
                    words[pos] = word

        # Join tokens into continuous text
        abstract_text = " ".join(words).strip()
        return abstract_text if abstract_text else None
    except Exception as exc:
        logger.warning(f"Error reconstructing abstract inverted index: {exc}")
        return None


def normalize_work(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes a raw OpenAlex work dictionary into a clean, flat profiling record."""
    work_id = raw.get("id", "")
    openalex_id = work_id.replace("https://openalex.org/", "") if work_id else None
    doi = raw.get("doi")
    title = raw.get("title")
    publication_year = raw.get("publication_year")
    publication_date = raw.get("publication_date")
    work_type = raw.get("type")
    cited_by_count = raw.get("cited_by_count", 0)

    # Primary location / journal source
    primary_loc = raw.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    source_name = source.get("display_name")
    source_issn = source.get("issn_l")

    # Open Access
    oa_info = raw.get("open_access") or {}
    is_oa = oa_info.get("is_oa", False)
    oa_status = oa_info.get("oa_status")
    oa_url = oa_info.get("oa_url")

    # Abstract
    abstract_inverted = raw.get("abstract_inverted_index")
    abstract_text = reconstruct_abstract(abstract_inverted)
    abstract_available = abstract_text is not None and len(abstract_text.strip()) > 0
    abstract_word_count = len(abstract_text.split()) if abstract_available else 0

    # Authors & Affiliations
    authorships = raw.get("authorships") or []
    author_list: List[Dict[str, Any]] = []
    institution_list: List[Dict[str, Any]] = []
    has_ke_institution = False
    has_international_institution = False

    for auth in authorships:
        author_data = auth.get("author") or {}
        author_id = author_data.get("id", "").replace("https://openalex.org/", "") if author_data.get("id") else None
        author_name = author_data.get("display_name")

        if author_name:
            author_list.append({
                "id": author_id,
                "name": author_name,
                "raw_affiliation_strings": auth.get("raw_affiliation_strings", [])
            })

        for inst in auth.get("institutions") or []:
            inst_id = inst.get("id", "").replace("https://openalex.org/", "") if inst.get("id") else None
            inst_name = inst.get("display_name")
            inst_country = inst.get("country_code")
            inst_ror = inst.get("ror")

            if inst_name:
                institution_list.append({
                    "id": inst_id,
                    "name": inst_name,
                    "country_code": inst_country,
                    "ror": inst_ror
                })

            if inst_country == "KE":
                has_ke_institution = True
            elif inst_country is not None:
                has_international_institution = True

    # Topics & Concepts
    primary_topic = raw.get("primary_topic") or {}
    primary_topic_name = primary_topic.get("display_name")
    primary_topic_subfield = (primary_topic.get("subfield") or {}).get("display_name")
    primary_topic_field = (primary_topic.get("field") or {}).get("display_name")
    primary_topic_domain = (primary_topic.get("domain") or {}).get("display_name")

    topics = [
        {
            "id": t.get("id", "").replace("https://openalex.org/", ""),
            "name": t.get("display_name"),
            "score": t.get("score")
        }
        for t in (raw.get("topics") or [])
        if t.get("display_name")
    ]

    concepts = [
        {
            "id": c.get("id", "").replace("https://openalex.org/", ""),
            "name": c.get("display_name"),
            "level": c.get("level"),
            "score": c.get("score")
        }
        for c in (raw.get("concepts") or [])
        if c.get("display_name")
    ]

    # Topic labels for fallback representation
    topic_labels = [t["name"] for t in topics]

    # Fallback search text formulation (Title + Topics + Abstract if available)
    fallback_parts = []
    if title:
        fallback_parts.append(title)
    if topic_labels:
        fallback_parts.append(", ".join(topic_labels))
    if abstract_text:
        fallback_parts.append(abstract_text)
    search_text = ". ".join(fallback_parts)

    return {
        "id": openalex_id,
        "raw_id": work_id,
        "doi": doi,
        "title": title,
        "publication_year": publication_year,
        "publication_date": publication_date,
        "type": work_type,
        "source_name": source_name,
        "source_issn": source_issn,
        "is_oa": is_oa,
        "oa_status": oa_status,
        "oa_url": oa_url,
        "abstract_available": abstract_available,
        "abstract_text": abstract_text,
        "abstract_word_count": abstract_word_count,
        "search_text": search_text,
        "authors_count": len(author_list),
        "authors": author_list,
        "institutions_count": len(institution_list),
        "institutions": institution_list,
        "has_ke_institution": has_ke_institution,
        "has_international_institution": has_international_institution,
        "primary_topic": primary_topic_name,
        "primary_topic_subfield": primary_topic_subfield,
        "primary_topic_field": primary_topic_field,
        "primary_topic_domain": primary_topic_domain,
        "topics_count": len(topics),
        "topics": topics,
        "concepts_count": len(concepts),
        "concepts": concepts,
        "cited_by_count": cited_by_count
    }
