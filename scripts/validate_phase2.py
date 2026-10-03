"""Comprehensive Phase 2 validation script investigating connectivity, missing IDs, name collisions, and degree distributions."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import networkx as nx

from src.ingestion.config import NORMALIZED_WORKS_FILE, RAW_WORKS_FILE, PROJECT_ROOT, REPORTS_DIR
from src.graph.normalize import build_canonical_graph_elements
from src.graph.entity_resolution import EntityResolver, generate_synthetic_id


def calculate_detailed_stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "p95": 0, "max": 0, "mean": 0.0, "std": 0.0}
    s = sorted(values)
    n = len(s)
    mean_val = sum(s) / n
    variance = sum((x - mean_val) ** 2 for x in s) / n
    std_val = math.sqrt(variance)

    def percentile(p):
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    return {
        "min": s[0],
        "p25": round(percentile(0.25), 2),
        "median": round(percentile(0.50), 2),
        "p75": round(percentile(0.75), 2),
        "p95": round(percentile(0.95), 2),
        "max": s[-1],
        "mean": round(mean_val, 2),
        "std": round(std_val, 2)
    }


def main():
    print("=" * 70)
    print(" PHASE 2 VALIDATION AUDIT")
    print("=" * 70)

    with open(NORMALIZED_WORKS_FILE, "r", encoding="utf-8") as f:
        normalized_records = [json.loads(line) for line in f if line.strip()]

    with open(RAW_WORKS_FILE, "r", encoding="utf-8") as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    # -------------------------------------------------------------
    # 1. Investigate Missing Researcher IDs in Raw and Normalized Data
    # -------------------------------------------------------------
    print("\n[1] INVESTIGATING MISSING RESEARCHER IDs...")
    missing_author_ids_raw = []
    missing_author_names = Counter()
    raw_authors_total = 0
    raw_authors_missing_id = 0

    for raw in raw_records:
        for auth in raw.get("authorships", []):
            raw_authors_total += 1
            author_obj = auth.get("author") or {}
            author_id = author_obj.get("id")
            author_name = author_obj.get("display_name")
            orcid = author_obj.get("orcid")
            raw_affils = auth.get("raw_affiliation_strings", [])

            if not author_id:
                raw_authors_missing_id += 1
                missing_author_names[author_name] += 1
                missing_author_ids_raw.append({
                    "work_id": raw.get("id"),
                    "name": author_name,
                    "orcid": orcid,
                    "raw_affil": raw_affils
                })

    print(f"Total author instances in raw OpenAlex: {raw_authors_total}")
    print(f"Author instances lacking OpenAlex ID: {raw_authors_missing_id} ({raw_authors_missing_id/raw_authors_total:.2%})")
    print(f"Distinct names among missing ID authors: {len(missing_author_names)}")
    print(f"Top 5 names with missing source IDs: {missing_author_names.most_common(5)}")
    
    # Check if any have ORCID
    orcid_count = sum(1 for a in missing_author_ids_raw if a["orcid"])
    print(f"Missing ID author instances with ORCID: {orcid_count}")

    # -------------------------------------------------------------
    # 2. Investigate Institution Name Collisions (Shared Names)
    # -------------------------------------------------------------
    print("\n[2] INVESTIGATING INSTITUTION NAME COLLISIONS...")
    resolver = EntityResolver()
    elements, resolver = build_canonical_graph_elements(normalized_records, resolver)
    nodes = elements["nodes"]
    rels = elements["relationships"]

    # Group institutions by clean lowercase display name
    insts_by_name = defaultdict(list)
    for inst_id, inst in nodes["institutions"].items():
        insts_by_name[inst.display_name].append(inst)

    colliding_insts = {name: inst_list for name, inst_list in insts_by_name.items() if len(inst_list) > 1}
    print(f"Total distinct institution display names: {len(insts_by_name)}")
    print(f"Display names associated with >1 distinct OpenAlex ID: {len(colliding_insts)}")

    # CDC Deep-Dive
    print("\n--- Deep-Dive: 'Centers for Disease Control and Prevention' ---")
    cdc_entries = insts_by_name.get("Centers for Disease Control and Prevention", [])
    for c in cdc_entries:
        print(f"  * ID: {c.id:<15} | Country: {c.country_code:<4} | Type: {str(c.type):<12} | ROR: {c.ror}")

    # Ministry of Health Deep-Dive
    print("\n--- Deep-Dive: 'Ministry of Health' ---")
    moh_entries = insts_by_name.get("Ministry of Health", [])
    for m in moh_entries:
        print(f"  * ID: {m.id:<15} | Country: {m.country_code:<4} | Type: {str(m.type):<12} | ROR: {m.ror}")

    # Sample Collision Table
    print("\nTop 10 Institution Name Collisions:")
    collision_table_data = []
    for name, inst_list in sorted(colliding_insts.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        ids = [i.id for i in inst_list]
        countries = [i.country_code or "None" for i in inst_list]
        rors = [i.ror or "None" for i in inst_list]
        types = [i.type or "None" for i in inst_list]
        collision_table_data.append({
            "display_name": name,
            "distinct_id_count": len(inst_list),
            "ids": ids,
            "countries": countries,
            "rors": rors,
            "types": types
        })
        print(f"  * '{name}' -> {len(inst_list)} IDs: {ids} | Countries: {countries}")

    # -------------------------------------------------------------
    # 3. Connectivity & Connected Components Analysis Under Subgraphs
    # -------------------------------------------------------------
    print("\n[3] RE-EVALUATING GRAPH CONNECTIVITY UNDER MULTIPLE SUBGRAPHS...")

    # A. Full Graph (All 5 node types & 4 relationship types)
    G_full = nx.Graph()
    for y in nodes["years"].values(): G_full.add_node(f"Year:{y.value}")
    for p in nodes["papers"].values(): G_full.add_node(f"Paper:{p.id}")
    for r in nodes["researchers"].values(): G_full.add_node(f"Researcher:{r.id}")
    for i in nodes["institutions"].values(): G_full.add_node(f"Institution:{i.id}")
    for t in nodes["topics"].values(): G_full.add_node(f"Topic:{t.id}")

    for rel in rels:
        G_full.add_edge(f"{rel.source_type}:{rel.source_id}", f"{rel.target_type}:{rel.target_id}")

    cc_full = list(nx.connected_components(G_full))
    lcc_full = max(len(c) for c in cc_full)
    isolates_full = list(nx.isolates(G_full))

    print(f"Full Graph (with Year & Topic):")
    print(f"  * Total nodes: {G_full.number_of_nodes()}, Total edges: {G_full.number_of_edges()}")
    print(f"  * Total CCs: {len(cc_full)}, LCC: {lcc_full} nodes ({lcc_full/G_full.number_of_nodes():.2%}), Isolates: {len(isolates_full)}")

    # B. Graph WITHOUT Year Nodes
    G_no_year = nx.Graph()
    for p in nodes["papers"].values(): G_no_year.add_node(f"Paper:{p.id}")
    for r in nodes["researchers"].values(): G_no_year.add_node(f"Researcher:{r.id}")
    for i in nodes["institutions"].values(): G_no_year.add_node(f"Institution:{i.id}")
    for t in nodes["topics"].values(): G_no_year.add_node(f"Topic:{t.id}")

    for rel in rels:
        if rel.rel_type != "PUBLISHED_IN":
            G_no_year.add_edge(f"{rel.source_type}:{rel.source_id}", f"{rel.target_type}:{rel.target_id}")

    cc_no_year = list(nx.connected_components(G_no_year))
    lcc_no_year = max(len(c) for c in cc_no_year)
    isolates_no_year = list(nx.isolates(G_no_year))

    print(f"\nGraph WITHOUT Year Nodes (Paper + Researcher + Institution + Topic):")
    print(f"  * Total nodes: {G_no_year.number_of_nodes()}, Total edges: {G_no_year.number_of_edges()}")
    print(f"  * Total CCs: {len(cc_no_year)}, LCC: {lcc_no_year} nodes ({lcc_no_year/G_no_year.number_of_nodes():.2%}), Isolates: {len(isolates_no_year)}")

    # C. Graph WITHOUT Year AND WITHOUT Topic Nodes (Strictly Paper - AUTHORED - Researcher & Paper - AFFILIATED_WITH - Institution)
    G_core_research = nx.Graph()
    for p in nodes["papers"].values(): G_core_research.add_node(f"Paper:{p.id}")
    for r in nodes["researchers"].values(): G_core_research.add_node(f"Researcher:{r.id}")
    for i in nodes["institutions"].values(): G_core_research.add_node(f"Institution:{i.id}")

    for rel in rels:
        if rel.rel_type in ("AUTHORED", "AFFILIATED_WITH"):
            G_core_research.add_edge(f"{rel.source_type}:{rel.source_id}", f"{rel.target_type}:{rel.target_id}")

    cc_core = list(nx.connected_components(G_core_research))
    lcc_core = max(len(c) for c in cc_core)
    isolates_core = list(nx.isolates(G_core_research))

    print(f"\nGraph with CORE RESEARCH RELATIONSHIPS ONLY (AUTHORED & AFFILIATED_WITH):")
    print(f"  * Total nodes: {G_core_research.number_of_nodes()}, Total edges: {G_core_research.number_of_edges()}")
    print(f"  * Total CCs: {len(cc_core)}, LCC: {lcc_core} nodes ({lcc_core/G_core_research.number_of_nodes():.2%}), Isolates: {len(isolates_core)}")

    # D. Direct Co-authorship Projection (Researcher - Co-authored - Researcher)
    G_coauth = nx.Graph()
    for r in nodes["researchers"].values(): G_coauth.add_node(r.id)
    paper_to_authors = defaultdict(list)
    for rel in rels:
        if rel.rel_type == "AUTHORED":
            paper_to_authors[rel.target_id].append(rel.source_id)
    for p_id, a_ids in paper_to_authors.items():
        for i in range(len(a_ids)):
            for j in range(i + 1, len(a_ids)):
                G_coauth.add_edge(a_ids[i], a_ids[j])

    cc_coauth = list(nx.connected_components(G_coauth))
    lcc_coauth = max(len(c) for c in cc_coauth)
    isolates_coauth = list(nx.isolates(G_coauth))

    print(f"\nDirect Co-Authorship Projection Graph:")
    print(f"  * Total researchers: {G_coauth.number_of_nodes()}, Total co-author edges: {G_coauth.number_of_edges()}")
    print(f"  * Total CCs: {len(cc_coauth)}, LCC: {lcc_coauth} ({lcc_coauth/G_coauth.number_of_nodes():.2%}), Isolates (solo authors): {len(isolates_coauth)}")

    # -------------------------------------------------------------
    # 4. Detailed Degree Distributions & Outlier Analysis
    # -------------------------------------------------------------
    print("\n[4] CALCULATING DETAILED DEGREE DISTRIBUTIONS & OUTLIERS...")

    undirected_full = G_full
    papers_per_researcher = [undirected_full.degree(f"Researcher:{r_id}") for r_id in nodes["researchers"]]
    papers_per_institution = [undirected_full.degree(f"Institution:{i_id}") for i_id in nodes["institutions"]]
    papers_per_topic = [undirected_full.degree(f"Topic:{t_id}") for t_id in nodes["topics"]]

    # Cardinalities per paper
    researchers_per_paper = []
    institutions_per_paper = []
    topics_per_paper = []

    for p_id in nodes["papers"]:
        auth_cnt = sum(1 for r in rels if r.rel_type == "AUTHORED" and r.target_id == p_id)
        inst_cnt = sum(1 for r in rels if r.rel_type == "AFFILIATED_WITH" and r.source_id == p_id)
        top_cnt = sum(1 for r in rels if r.rel_type == "ABOUT" and r.source_id == p_id)
        researchers_per_paper.append(auth_cnt)
        institutions_per_paper.append(inst_cnt)
        topics_per_paper.append(top_cnt)

    deg_stats = {
        "researchers_per_paper": calculate_detailed_stats(researchers_per_paper),
        "institutions_per_paper": calculate_detailed_stats(institutions_per_paper),
        "topics_per_paper": calculate_detailed_stats(topics_per_paper),
        "papers_per_researcher": calculate_detailed_stats(papers_per_researcher),
        "papers_per_institution": calculate_detailed_stats(papers_per_institution),
        "papers_per_topic": calculate_detailed_stats(papers_per_topic)
    }

    for k, v in deg_stats.items():
        print(f"  * {k:<25}: min={v['min']} | p25={v['p25']} | med={v['median']} | p75={v['p75']} | p95={v['p95']} | max={v['max']} | mean={v['mean']} | std={v['std']}")

    # Identify Outliers
    print("\nExtreme Degree Outliers:")
    # Top 3 papers by author count
    sorted_papers_by_auth = sorted(nodes["papers"].values(), key=lambda p: len([r for r in rels if r.rel_type == "AUTHORED" and r.target_id == p.id]), reverse=True)[:3]
    for p in sorted_papers_by_auth:
        cnt = len([r for r in rels if r.rel_type == "AUTHORED" and r.target_id == p.id])
        print(f"  * Paper with {cnt} authors: {p.id} - '{p.title[:60]}...'")

    # Top 3 researchers by paper count
    sorted_researchers_by_paper = sorted(nodes["researchers"].values(), key=lambda r: undirected_full.degree(f"Researcher:{r.id}"), reverse=True)[:3]
    for r in sorted_researchers_by_paper:
        cnt = undirected_full.degree(f"Researcher:{r.id}")
        print(f"  * Researcher with {cnt} papers: {r.id} - '{r.display_name}'")

    # Top 3 institutions by paper count
    sorted_insts_by_paper = sorted(nodes["institutions"].values(), key=lambda i: undirected_full.degree(f"Institution:{i.id}"), reverse=True)[:3]
    for i in sorted_insts_by_paper:
        cnt = undirected_full.degree(f"Institution:{i.id}")
        print(f"  * Institution with {cnt} papers: {i.id} - '{i.display_name}' ({i.country_code})")

    # -------------------------------------------------------------
    # 5. Save Structured Audit Output
    # -------------------------------------------------------------
    audit_output = {
        "missing_researcher_ids": {
            "total_raw_author_instances": raw_authors_total,
            "missing_id_instances": raw_authors_missing_id,
            "missing_id_percentage": round(raw_authors_missing_id / raw_authors_total * 100, 2),
            "distinct_names_with_missing_id": len(missing_author_names),
            "with_orcid": orcid_count,
            "top_names": missing_author_names.most_common(10)
        },
        "institution_name_collisions": {
            "total_distinct_display_names": len(insts_by_name),
            "colliding_names_count": len(colliding_insts),
            "sample_collisions": collision_table_data,
            "cdc_deep_dive": [c.model_dump() for c in cdc_entries],
            "moh_deep_dive": [m.model_dump() for m in moh_entries]
        },
        "connectivity_subgraphs": {
            "full_graph": {"nodes": G_full.number_of_nodes(), "edges": G_full.number_of_edges(), "ccs": len(cc_full), "lcc": lcc_full, "lcc_pct": round(lcc_full/G_full.number_of_nodes()*100, 2), "isolates": len(isolates_full)},
            "no_year_graph": {"nodes": G_no_year.number_of_nodes(), "edges": G_no_year.number_of_edges(), "ccs": len(cc_no_year), "lcc": lcc_no_year, "lcc_pct": round(lcc_no_year/G_no_year.number_of_nodes()*100, 2), "isolates": len(isolates_no_year)},
            "core_research_graph": {"nodes": G_core_research.number_of_nodes(), "edges": G_core_research.number_of_edges(), "ccs": len(cc_core), "lcc": lcc_core, "lcc_pct": round(lcc_core/G_core_research.number_of_nodes()*100, 2), "isolates": len(isolates_core)},
            "coauthorship_projection": {"nodes": G_coauth.number_of_nodes(), "edges": G_coauth.number_of_edges(), "ccs": len(cc_coauth), "lcc": lcc_coauth, "lcc_pct": round(lcc_coauth/G_coauth.number_of_nodes()*100, 2), "isolates": len(isolates_coauth)}
        },
        "degree_distributions": deg_stats
    }

    out_file = REPORTS_DIR / "phase2_validation_audit.json"
    with open(out_file, "w", encoding="utf-8") as f_out:
        json.dump(audit_output, f_out, indent=2)
    print(f"\nAudit results successfully written to {out_file}")


if __name__ == "__main__":
    main()
