"""Metadata quality analysis and metric profiling engine."""

from collections import Counter
from typing import Dict, Any, List, Optional
import math
import logging

logger = logging.getLogger(__name__)


def compute_field_completeness(records: List[Dict[str, Any]], field_extractors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Calculates presence, missing count, and percentages for specified fields."""
    total = len(records)
    completeness = []

    for field_name, extractor in field_extractors.items():
        present = sum(1 for r in records if extractor(r))
        missing = total - present
        pct_present = (present / total * 100.0) if total > 0 else 0.0
        pct_missing = (missing / total * 100.0) if total > 0 else 0.0

        completeness.append({
            "field": field_name,
            "total": total,
            "present": present,
            "missing": missing,
            "present_pct": round(pct_present, 2),
            "missing_pct": round(pct_missing, 2)
        })

    return completeness


def calculate_summary_stats(values: List[float]) -> Dict[str, Any]:
    """Calculates min, max, mean, median for a numeric list."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n
    median_val = sorted_vals[n // 2] if n % 2 != 0 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    return {
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": round(mean_val, 2),
        "median": round(median_val, 2)
    }


def profile_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runs a full empirical profiling audit on a normalized records list."""
    total = len(records)
    if total == 0:
        return {"total_records": 0, "status": "empty"}

    # Field extractors
    field_extractors = {
        "OpenAlex ID": lambda r: bool(r.get("id")),
        "DOI": lambda r: bool(r.get("doi")),
        "Title": lambda r: bool(r.get("title")),
        "Publication Year": lambda r: r.get("publication_year") is not None,
        "Publication Date": lambda r: bool(r.get("publication_date")),
        "Work Type": lambda r: bool(r.get("type")),
        "Source / Journal": lambda r: bool(r.get("source_name")),
        "Abstract (Reconstructed)": lambda r: bool(r.get("abstract_available")),
        "Authors": lambda r: len(r.get("authors", [])) > 0,
        "Author Identifiers": lambda r: any(a.get("id") for a in r.get("authors", [])),
        "Institutions / Affiliations": lambda r: len(r.get("institutions", [])) > 0,
        "Kenyan Institution Affiliation": lambda r: r.get("has_ke_institution", False),
        "Institution ROR ID": lambda r: any(i.get("ror") for i in r.get("institutions", [])),
        "Primary Topic": lambda r: bool(r.get("primary_topic")),
        "Topics List": lambda r: len(r.get("topics", [])) > 0,
        "Concepts List": lambda r: len(r.get("concepts", [])) > 0,
        "Open Access Status": lambda r: bool(r.get("oa_status")),
        "Citations Count": lambda r: r.get("cited_by_count") is not None
    }

    completeness = compute_field_completeness(records, field_extractors)

    # Distributions
    authors_counts = [r.get("authors_count", 0) for r in records]
    institutions_counts = [r.get("institutions_count", 0) for r in records]
    topics_counts = [r.get("topics_count", 0) for r in records]
    concepts_counts = [r.get("concepts_count", 0) for r in records]
    citations_counts = [r.get("cited_by_count", 0) for r in records]
    abstract_words = [r.get("abstract_word_count", 0) for r in records if r.get("abstract_available")]

    years_dist = Counter(r.get("publication_year") for r in records if r.get("publication_year"))
    types_dist = Counter(r.get("type") for r in records if r.get("type"))
    oa_dist = Counter(r.get("oa_status") for r in records if r.get("oa_status"))
    domains_dist = Counter(r.get("primary_topic_domain") for r in records if r.get("primary_topic_domain"))
    fields_dist = Counter(r.get("primary_topic_field") for r in records if r.get("primary_topic_field"))

    # Institution breakdown
    ke_institutions = Counter()
    intl_institutions = Counter()
    intl_countries = Counter()

    for r in records:
        for inst in r.get("institutions", []):
            name = inst.get("name")
            country = inst.get("country_code")
            if not name:
                continue
            if country == "KE":
                ke_institutions[name] += 1
            elif country:
                intl_institutions[f"{name} ({country})"] += 1
                intl_countries[country] += 1

    # Edge cases & Anomalies
    doi_list = [r.get("doi").lower() for r in records if r.get("doi")]
    title_list = [r.get("title").lower().strip() for r in records if r.get("title")]
    id_list = [r.get("id") for r in records if r.get("id")]

    duplicate_dois = {doi: count for doi, count in Counter(doi_list).items() if count > 1}
    duplicate_titles = {title: count for title, count in Counter(title_list).items() if count > 1}
    duplicate_ids = {i: count for i, count in Counter(id_list).items() if count > 1}

    anomalous_years = [r.get("publication_year") for r in records if r.get("publication_year") and (r.get("publication_year") < 1960 or r.get("publication_year") > 2027)]
    ke_flag_no_ke_inst = sum(1 for r in records if not r.get("has_ke_institution"))
    intl_collaborations = sum(1 for r in records if r.get("has_ke_institution") and r.get("has_international_institution"))

    return {
        "total_records": total,
        "completeness": completeness,
        "distributions": {
            "authors_per_work": calculate_summary_stats(authors_counts),
            "institutions_per_work": calculate_summary_stats(institutions_counts),
            "topics_per_work": calculate_summary_stats(topics_counts),
            "concepts_per_work": calculate_summary_stats(concepts_counts),
            "citations_per_work": calculate_summary_stats(citations_counts),
            "abstract_word_length": calculate_summary_stats(abstract_words),
            "publication_years": dict(sorted(years_dist.items(), key=lambda x: str(x[0]))),
            "work_types": dict(types_dist.most_common(15)),
            "oa_statuses": dict(oa_dist.most_common(10)),
            "topic_domains": dict(domains_dist.most_common(10)),
            "topic_fields": dict(fields_dist.most_common(10))
        },
        "affiliations": {
            "top_domestic_institutions": dict(ke_institutions.most_common(20)),
            "top_international_institutions": dict(intl_institutions.most_common(20)),
            "top_collaborating_countries": dict(intl_countries.most_common(15)),
            "international_collaboration_count": intl_collaborations,
            "international_collaboration_pct": round(intl_collaborations / total * 100.0, 2) if total > 0 else 0
        },
        "edge_cases": {
            "duplicate_ids_count": len(duplicate_ids),
            "duplicate_dois_count": len(duplicate_dois),
            "duplicate_titles_count": len(duplicate_titles),
            "sample_duplicate_dois": list(duplicate_dois.keys())[:5],
            "sample_duplicate_titles": list(duplicate_titles.keys())[:5],
            "records_without_identified_ke_institution": ke_flag_no_ke_inst,
            "anomalous_years_count": len(anomalous_years),
            "anomalous_years": anomalous_years[:5]
        }
    }
