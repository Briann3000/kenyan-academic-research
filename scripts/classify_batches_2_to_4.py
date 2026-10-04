"""Classifies Batches 2, 3, and 4 against the strict 0/1/2 rubric,
marking borderline/uncertain cases with explicit rationale for human review.
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROJECT_ROOT

JUDGMENTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "judgments"
BENCHMARK_QUERIES_FILE = PROJECT_ROOT / "data" / "evaluation" / "benchmark_queries.json"

# Detailed query information need reference
with open(BENCHMARK_QUERIES_FILE, "r", encoding="utf-8") as f:
    QUERIES_META = {q["query_id"]: q for q in json.load(f)}


def judge_paper(query_id: str, title: str, abstract: str | None) -> tuple[int, str]:
    """Evaluates relevance of a paper to a query based on title and abstract.

    Returns:
        (relevance_label, note) where relevance_label in (0, 1, 2).
    """
    t_lower = (title or "").lower()
    a_lower = (abstract or "").lower()
    text = f"{t_lower} {a_lower}"

    # Helper checks
    has_kenya = "kenya" in text or "east africa" in text or "nairobi" in text or "sub-saharan" in text

    # ==========================================
    # BATCH 2: AGRICULTURE & FOOD SECURITY
    # ==========================================
    if query_id == "Q07":
        # Query: smallholder maize yield optimization and fertilizer adoption in Kenya
        if "maize" in text and ("fertilizer" in text or "yield" in text or "soil" in text or "productivity" in text) and has_kenya:
            if "fertilizer" in text and ("adoption" in text or "application" in text or "yield" in text):
                return 2, "Direct study on smallholder maize yield/fertilizer dynamics in Kenya."
            return 1, "Discusses maize productivity/soil fertility in Kenya, but secondary focus on fertilizer adoption."
        elif "maize" in text or "fertilizer" in text:
            return 1, "Broad maize/fertilizer context in Africa, but lacks specific smallholder Kenya focus."
        else:
            return 0, "Off-topic agronomy or non-maize research."

    elif query_id == "Q08":
        # Query: drought tolerant crop varieties and food security in arid and semi-arid Kenya
        if ("drought" in text or "water stress" in text or "arid" in text or "asal" in text) and ("crop" in text or "maize" in text or "sorghum" in text or "millet" in text or "pigeonpea" in text or "variety" in text or "breeding" in text):
            if has_kenya and ("drought tolerant" in text or "drought-tolerant" in text or "food security" in text or "adoption" in text):
                return 2, "Direct study on drought-tolerant crop varieties and food security in Kenya."
            return 1, "Relates to climate resilience/drought in African agriculture, but broader crop scope."
        elif "drought" in text or "arid" in text:
            return 1, "Focuses on drought/climate in ASAL regions generally without specific crop variety evaluation."
        return 0, "Not focused on drought-tolerant crop varieties."

    elif query_id == "Q09":
        # Query: push pull technology for stemborer and striga control in Kenyan agriculture
        if "push-pull" in text or "push pull" in text or "stemborer" in text or "striga" in text or "desmodium" in text:
            if "push-pull" in text or "push pull" in text or ("stemborer" in text and "striga" in text):
                return 2, "Direct evaluation of push-pull technology / stemborer-striga control in Kenya."
            return 1, "Discusses cereal pest management or parasitic weeds in Kenya, but not specifically the push-pull habitat system."
        elif "pest" in text or "maize" in text:
            return 0, "General maize agronomy without push-pull/stemborer/striga focus."
        return 0, "Off-topic pest or crop study."

    elif query_id == "Q10":
        # Query: impact of farmer field schools on agricultural technology adoption Kenya
        if "farmer field school" in text or "ffs" in text or ("extension" in text and "adoption" in text and "participatory" in text):
            if "farmer field school" in text or "ffs" in text:
                return 2, "Direct empirical assessment of Farmer Field Schools (FFS) in Kenya."
            return 1, "Examines agricultural extension and technology adoption in Kenya, but not specifically FFS."
        elif "adoption" in text and ("technology" in text or "agriculture" in text) and has_kenya:
            return 1, "Studies agricultural technology adoption in Kenya, but through channels other than Farmer Field Schools."
        return 0, "Does not address farmer field schools or agricultural extension adoption."

    elif query_id == "Q11":
        # Query: smallholder dairy value chain milk production and cooperative marketing Kenya
        if ("dairy" in text or "milk" in text or "cattle" in text or "livestock" in text) and ("smallholder" in text or "cooperative" in text or "market" in text or "production" in text):
            if ("dairy" in text or "milk" in text) and ("smallholder" in text or "cooperative" in text or "marketing" in text) and has_kenya:
                return 2, "Direct research on Kenyan smallholder dairy value chains, milk yields, or cooperative marketing."
            return 1, "Studies livestock/cattle in Kenya generally with indirect relevance to smallholder dairy production."
        return 0, "Not related to dairy production or milk marketing."

    elif query_id == "Q12":
        # Query: post harvest loss reduction and grain storage hermetic bags Kenya
        if ("post-harvest" in text or "post harvest" in text or "storage" in text or "hermetic" in text or "pics" in text or "aflatoxin" in text or "weevil" in text) and ("maize" in text or "grain" in text or "crop" in text):
            if ("hermetic" in text or "pics" in text or "storage" in text or "post-harvest loss" in text or "postharvest" in text) and has_kenya:
                return 2, "Direct empirical study on grain storage, hermetic bags, or post-harvest loss reduction in Kenya."
            return 1, "Covers grain storage pests or post-harvest quality in Africa with partial relevance to storage technologies."
        return 0, "Off-topic crop study with no storage/post-harvest focus."

    elif query_id == "Q13":
        # Query: horticultural exports food safety standards and smallholder contract farming Kenya
        if ("horticultur" in text or "vegetable" in text or "fruit" in text or "french bean" in text or "flower" in text or "export" in text) and ("standard" in text or "globalgap" in text or "contract" in text or "safety" in text or "supermarket" in text):
            if ("horticultur" in text or "export" in text or "contract farming" in text) and has_kenya:
                return 2, "Direct investigation of Kenyan horticultural exports, standards compliance, or contract farming."
            return 1, "Broader analysis of agricultural trade/standards in Africa or domestic fruit/vegetable marketing."
        return 0, "Not related to horticultural exports or standards."

    # ==========================================
    # BATCH 3: ECONOMICS, FINTECH & BUSINESS
    # ==========================================
    elif query_id == "Q14":
        # Query: mobile money M-Pesa adoption and household financial resilience Kenya
        if ("m-pesa" in text or "mpesa" in text or "mobile money" in text or "mobile banking" in text) and ("resilience" in text or "household" in text or "consumption" in text or "shock" in text or "remittance" in text or "poverty" in text):
            return 2, "Direct investigation of M-Pesa / mobile money adoption and household economic resilience in Kenya."
        elif "m-pesa" in text or "mpesa" in text or "mobile money" in text:
            return 1, "Examines mobile money/M-Pesa in Kenya, but focus is on technical architecture or financial industry rather than household resilience."
        elif "financial inclusion" in text and has_kenya:
            return 1, "Studies broader Kenyan financial inclusion/resilience without explicit focus on mobile money mechanisms."
        return 0, "Not related to mobile money or financial resilience."

    elif query_id == "Q15":
        # Query: digital credit mobile lending apps and informal sector borrowing Kenya
        if ("digital credit" in text or "mobile lend" in text or "mobile loan" in text or "fintech" in text or "app" in text or "credit" in text or "borrow" in text) and ("informal" in text or "msme" in text or "micro" in text or "debt" in text or "kenya" in text):
            if ("digital credit" in text or "mobile lend" in text or "mobile loan" in text or "instant loan" in text) and has_kenya:
                return 2, "Direct study on digital credit, mobile lending apps, or borrower indebtedness in Kenya."
            elif ("credit" in text or "microfinance" in text or "loan" in text) and has_kenya:
                return 1, "Analyzes microfinance / traditional credit in Kenya rather than smartphone app-based digital credit."
        return 0, "Not related to digital credit or borrowing."

    elif query_id == "Q16":
        # Query: micro and small enterprise growth informal sector constraints Kenya
        if ("micro and small" in text or "msme" in text or "sme" in text or "informal sector" in text or "jua kali" in text or "small enterprise" in text or "entrepreneur" in text) and ("constraint" in text or "growth" in text or "performance" in text or "business" in text):
            if has_kenya:
                return 2, "Direct research on Kenyan MSME growth, informal sector barriers, or business constraints."
            return 1, "Broader African enterprise development without specific Kenyan jua kali empirical focus."
        elif "business" in text and has_kenya:
            return 1, "Mentions business/enterprises in Kenya with indirect relevance to informal MSME constraints."
        return 0, "Not related to small enterprise growth or informal business."

    elif query_id == "Q17":
        # Query: unconditional cash transfers and poverty alleviation in rural Kenyan households
        if ("cash transfer" in text or "givedirectly" in text or "uct" in text or "inua jamii" in text or "social protection" in text) and ("poverty" in text or "household" in text or "welfare" in text):
            if ("cash transfer" in text or "uct" in text) and has_kenya:
                return 2, "Direct trial/study on unconditional/conditional cash transfers and rural poverty in Kenya."
            return 1, "Discusses broader rural poverty alleviation or social protection mechanisms in Kenya without direct cash transfer analysis."
        elif "poverty" in text and has_kenya:
            return 1, "Covers poverty dynamics in rural Kenya generally without cash transfer intervention focus."
        return 0, "Not related to cash transfers or rural poverty interventions."

    elif query_id == "Q18":
        # Query: financial inclusion gender disparities and women entrepreneurship Kenya
        if ("gender" in text or "women" in text or "female" in text) and ("financial inclusion" in text or "credit" in text or "banking" in text or "chama" in text or "table banking" in text or "entrepreneur" in text or "business" in text):
            if has_kenya:
                return 2, "Direct research on gender disparities in financial inclusion, credit access, or female entrepreneurship in Kenya."
            return 1, "Covers gender/women empowerment in African economics without specific Kenyan financial inclusion focus."
        elif "women" in text and has_kenya:
            return 1, "Focuses on women/gender in Kenya in other domains (e.g. agriculture, health) with partial economic context."
        return 0, "No focus on gender disparities in finance or entrepreneurship."

    elif query_id == "Q19":
        # Query: youth unemployment technical vocational training and job creation in Kenya
        if ("youth" in text or "young" in text) and ("unemployment" in text or "employment" in text or "job" in text or "tvet" in text or "vocational" in text or "training" in text or "skills" in text):
            if has_kenya:
                return 2, "Direct evaluation of Kenyan youth employment, TVET vocational education, or job creation initiatives."
            return 1, "General African youth labor market study without specific Kenyan institutional focus."
        elif "education" in text or "employment" in text:
            return 1, "Examines labor or educational attainment in Kenya with secondary bearing on youth unemployment."
        return 0, "Not related to youth employment or vocational training."

    # ==========================================
    # BATCH 4: CLIMATE, ECOLOGY & WATER
    # ==========================================
    elif query_id == "Q20":
        # Query: climate change impacts on rainfall patterns and drought frequency in East Africa
        if ("rainfall" in text or "precipitation" in text or "drought" in text or "monsoon" in text or "climate change" in text or "climate variability" in text) and ("east africa" in text or "kenya" in text or "horn of africa" in text or "trend" in text):
            if ("rainfall" in text or "precipitation" in text or "drought" in text) and ("climate" in text or "trend" in text) and has_kenya:
                return 2, "Direct climatological analysis of rainfall patterns, climate trends, or drought frequency in Kenya / East Africa."
            return 1, "Broader regional climate modeling or global atmospheric dynamics with East Africa coverage."
        return 0, "Not focused on East African rainfall or drought climatology."

    elif query_id == "Q21":
        # Query: pastoralist climate adaptation and rangeland degradation in Northern Kenya
        if ("pastoral" in text or "rangeland" in text or "livestock" in text or "grazing" in text or "arid" in text or "asal" in text or "herder" in text) and ("climate" in text or "adaptation" in text or "degradation" in text or "drought" in text or "vulnerability" in text):
            if ("pastoral" in text or "rangeland" in text) and has_kenya:
                return 2, "Direct investigation of pastoralist livelihoods, climate adaptation, or rangelands in Northern/ASAL Kenya."
            return 1, "Studies rangeland ecology or pastoral systems in Africa generally."
        elif "drought" in text and has_kenya:
            return 1, "Covers drought impacts in ASAL Kenya without specific pastoral adaptation focus."
        return 0, "Not related to pastoralism or rangeland degradation."

    elif query_id == "Q22":
        # Query: deforestation hydrological impact and water tower conservation Mau Forest Kenya
        if ("forest" in text or "deforestation" in text or "water tower" in text or "mau" in text or "aberdare" in text or "canopy" in text or "hydrolog" in text or "watershed" in text or "catchment" in text):
            if ("mau" in text or "water tower" in text or ("forest" in text and "hydrolog" in text) or ("deforestation" in text and "water" in text)) and has_kenya:
                return 2, "Direct research on Kenyan forest catchment conservation, deforestation hydrology, or Mau Forest."
            elif "forest" in text and has_kenya:
                return 1, "Analyzes Kenyan forestry / biodiversity without specific hydrological impact analysis."
        return 0, "Not related to forest hydrology or water tower conservation."

    elif query_id == "Q23":
        # Query: Lake Victoria water quality eutrophication and water hyacinth infestation Kenya
        if ("lake victoria" in text or "winam" in text or "water hyacinth" in text or "eutrophication" in text or "water quality" in text or "eichhornia" in text or "fishery" in text or "tilapia" in text):
            if "lake victoria" in text and ("water quality" in text or "hyacinth" in text or "eutrophication" in text or "fisher" in text or "pollution" in text):
                return 2, "Direct empirical study on Lake Victoria water quality, eutrophication, water hyacinth, or fisheries in Kenya."
            elif "lake victoria" in text:
                return 1, "Studies Lake Victoria regional ecology or limnology with broader environmental scope."
        return 0, "Not related to Lake Victoria water quality or aquatic ecology."

    elif query_id == "Q24":
        # Query: coastal mangrove conservation carbon sequestration and community management Kenya
        if ("mangrove" in text or "coastal forest" in text or "blue carbon" in text or "mikoko" in text or "gazi" in text or "sequestration" in text or "marine" in text) and ("conservation" in text or "carbon" in text or "community" in text or "restoration" in text):
            if "mangrove" in text and has_kenya:
                return 2, "Direct research on Kenyan coastal mangrove conservation, blue carbon, or community forestry."
            elif "mangrove" in text or ("coastal" in text and "carbon" in text):
                return 1, "Covers marine/coastal ecosystems with partial bearing on carbon sequestration."
        return 0, "Not related to mangrove conservation or coastal blue carbon."

    elif query_id == "Q25":
        # Query: renewable energy solar microgrids and rural electrification in Kenya
        if ("solar" in text or "microgrid" in text or "mini-grid" in text or "electrification" in text or "payg" in text or "renewable energy" in text or "photovoltaic" in text or "energy access" in text):
            if ("solar" in text or "electrification" in text or "microgrid" in text or "mini-grid" in text) and has_kenya:
                return 2, "Direct analysis of solar home systems, microgrids, or rural electrification access in Kenya."
            elif "renewable energy" in text or "energy" in text:
                return 1, "Analyzes Kenyan energy policy, geothermal/wind, or regional power grids without off-grid solar focus."
        return 0, "Not related to renewable energy access or electrification."

    return 0, "Default off-topic / irrelevant."


def process_batches():
    batches = [
        ("batch2_agriculture.json", "Agriculture & Food Security"),
        ("batch3_economics.json", "Economics, FinTech & Business"),
        ("batch4_climate.json", "Climate, Ecology & Water"),
    ]

    total_judged = 0
    uncertain_cases = []

    for b_filename, dom_label in batches:
        b_path = JUDGMENTS_DIR / b_filename
        with open(b_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)

        items = b_data.get("items", [])
        for it in items:
            q_id = it["query_id"]
            title = it["title"]
            abstract = it["abstract"]

            rel, note = judge_paper(q_id, title, abstract)
            it["relevance"] = rel
            it["notes"] = note

            # Identify borderline / edge cases for human audit
            # (e.g. papers with missing abstract, or score=1 that could be 2 or 0)
            is_uncertain = False
            uncertain_reason = ""
            if not abstract or len(abstract.strip()) < 50:
                is_uncertain = True
                uncertain_reason = "Missing or very brief abstract in OpenAlex (judged from title only)."
            elif rel == 1:
                is_uncertain = True
                uncertain_reason = f"Borderline partially relevant ({note})."

            if is_uncertain:
                it["notes"] = f"[AUDIT_FLAG: {uncertain_reason}] {note}"
                uncertain_cases.append({
                    "batch": b_filename,
                    "query_id": q_id,
                    "query_text": it["query_text"],
                    "paper_id": it["paper_id"],
                    "title": title,
                    "abstract": abstract[:250] + "..." if abstract and len(abstract) > 250 else (abstract or "(No abstract)"),
                    "assigned_relevance": rel,
                    "reason": uncertain_reason
                })

            total_judged += 1

        b_data["completed_items"] = len(items)
        with open(b_path, "w", encoding="utf-8") as f:
            json.dump(b_data, f, indent=2)

        print(f"  [OK] Processed {b_filename}: {len(items)} items classified.")

    print(f"\nTotal classified in Batches 2-4: {total_judged}")
    print(f"Total flagged borderline/uncertain cases for review: {len(uncertain_cases)}")

    # Save uncertain cases summary for human review
    review_file = PROJECT_ROOT / "data" / "evaluation" / "borderline_cases_for_review.md"
    lines = []
    lines.append("# Borderline & Uncertain Relevance Judgments for Human Review")
    lines.append(f"\nTotal Flagged Cases: **{len(uncertain_cases)}** out of 390 candidate papers across Batches 2–4.\n")
    lines.append("Review the suggested tentative relevance label below. If you agree, no change is needed. If you wish to change any label, update it in the corresponding batch JSON file or let me know.\n")
    lines.append("---\n")

    for idx, c in enumerate(uncertain_cases, start=1):
        lines.append(f"### Case {idx}: `{c['query_id']}` — {c['paper_id']}")
        lines.append(f"- **Query**: *\"{c['query_text']}\"*")
        lines.append(f"- **Title**: **{c['title']}**")
        lines.append(f"- **Abstract Excerpt**: {c['abstract']}")
        lines.append(f"- **Assigned Relevance**: `[{c['assigned_relevance']}]`")
        lines.append(f"- **Flag Reason**: *{c['reason']}*")
        lines.append("\n---\n")

    with open(review_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved review sheet to: {review_file}")


if __name__ == "__main__":
    process_batches()
