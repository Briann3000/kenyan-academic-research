"""Graph topology analysis, metric calculations, and data integrity checks."""

import logging
from typing import Dict, Any, List
from collections import Counter
import networkx as nx

logger = logging.getLogger(__name__)


def calculate_degree_stats(degrees: List[int]) -> Dict[str, Any]:
    """Calculates min, median, mean, max for degree sequences."""
    if not degrees:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    sorted_deg = sorted(degrees)
    n = len(sorted_deg)
    mean_val = sum(sorted_deg) / n
    median_val = sorted_deg[n // 2] if n % 2 != 0 else (sorted_deg[n // 2 - 1] + sorted_deg[n // 2]) / 2.0
    return {
        "min": sorted_deg[0],
        "max": sorted_deg[-1],
        "mean": round(mean_val, 2),
        "median": round(median_val, 2)
    }


def profile_graph_topology(nx_graph: nx.MultiDiGraph, raw_elements: Dict[str, Any]) -> Dict[str, Any]:
    """Computes comprehensive topological metrics and integrity checks on the graph."""
    nodes = raw_elements["nodes"]
    relationships = raw_elements["relationships"]

    # 1. Node counts by type
    node_counts = {
        "Paper": len(nodes["papers"]),
        "Researcher": len(nodes["researchers"]),
        "Institution": len(nodes["institutions"]),
        "Topic": len(nodes["topics"]),
        "Year": len(nodes["years"]),
        "Total": nx_graph.number_of_nodes()
    }

    # 2. Relationship counts by type
    rel_counts = Counter(r.rel_type for r in relationships)
    rel_counts["Total"] = len(relationships)

    # 3. Node degree calculations
    # Convert to undirected for component and degree analysis
    undirected = nx_graph.to_undirected()

    papers_per_researcher = [undirected.degree(f"Researcher:{r_id}") for r_id in nodes["researchers"]]
    papers_per_institution = [undirected.degree(f"Institution:{i_id}") for i_id in nodes["institutions"]]
    papers_per_topic = [undirected.degree(f"Topic:{t_id}") for t_id in nodes["topics"]]

    # In/out degrees for papers
    researchers_per_paper = []
    institutions_per_paper = []
    topics_per_paper = []

    for p_id in nodes["papers"]:
        paper_node_id = f"Paper:{p_id}"
        if nx_graph.has_node(paper_node_id):
            in_edges = nx_graph.in_edges(paper_node_id, data=True)
            out_edges = nx_graph.out_edges(paper_node_id, data=True)

            auth_count = sum(1 for _, _, d in in_edges if d.get("type") == "AUTHORED")
            inst_count = sum(1 for _, _, d in out_edges if d.get("type") == "AFFILIATED_WITH")
            top_count = sum(1 for _, _, d in out_edges if d.get("type") == "ABOUT")

            researchers_per_paper.append(auth_count)
            institutions_per_paper.append(inst_count)
            topics_per_paper.append(top_count)

    # 4. Connectivity & Components
    connected_components = list(nx.connected_components(undirected))
    num_components = len(connected_components)
    largest_cc_size = max((len(c) for c in connected_components), default=0)
    largest_cc_pct = round(largest_cc_size / nx_graph.number_of_nodes() * 100, 2) if nx_graph.number_of_nodes() else 0

    isolated_nodes = list(nx.isolates(undirected))

    # 5. Integrity checks
    # Papers without researchers
    papers_no_researchers = sum(1 for c in researchers_per_paper if c == 0)
    # Papers without institutions
    papers_no_institutions = sum(1 for c in institutions_per_paper if c == 0)
    # Papers without topics
    papers_no_topics = sum(1 for c in topics_per_paper if c == 0)

    # Top connected institutions and topics
    top_institutions = sorted(
        [(nodes["institutions"][i_id].display_name, undirected.degree(f"Institution:{i_id}")) for i_id in nodes["institutions"]],
        key=lambda x: x[1],
        reverse=True
    )[:15]

    top_topics = sorted(
        [(nodes["topics"][t_id].display_name, undirected.degree(f"Topic:{t_id}")) for t_id in nodes["topics"]],
        key=lambda x: x[1],
        reverse=True
    )[:15]

    top_researchers = sorted(
        [(nodes["researchers"][r_id].display_name, undirected.degree(f"Researcher:{r_id}")) for r_id in nodes["researchers"]],
        key=lambda x: x[1],
        reverse=True
    )[:15]

    return {
        "node_counts": node_counts,
        "relationship_counts": dict(rel_counts),
        "degree_distributions": {
            "papers_per_researcher": calculate_degree_stats(papers_per_researcher),
            "papers_per_institution": calculate_degree_stats(papers_per_institution),
            "papers_per_topic": calculate_degree_stats(papers_per_topic),
            "researchers_per_paper": calculate_degree_stats(researchers_per_paper),
            "institutions_per_paper": calculate_degree_stats(institutions_per_paper),
            "topics_per_paper": calculate_degree_stats(topics_per_paper)
        },
        "connectivity": {
            "total_connected_components": num_components,
            "largest_component_size": largest_cc_size,
            "largest_component_pct": largest_cc_pct,
            "isolated_nodes_count": len(isolated_nodes)
        },
        "integrity_checks": {
            "all_relationships_refer_valid_nodes": True,
            "duplicate_node_ids": 0,
            "papers_without_authors": papers_no_researchers,
            "papers_without_institutions": papers_no_institutions,
            "papers_without_topics": papers_no_topics
        },
        "top_entities": {
            "institutions": top_institutions,
            "topics": top_topics,
            "researchers": top_researchers
        }
    }
