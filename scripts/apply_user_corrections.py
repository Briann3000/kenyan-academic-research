"""Applies expert human corrections to Batches 2–4 in data/evaluation/judgments/."""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.config import PROJECT_ROOT

JUDGMENTS_DIR = PROJECT_ROOT / "data" / "evaluation" / "judgments"

CORRECTIONS = [
    # (Query ID, Paper ID, Corrected Relevance, Rationale)
    ("Q07", "W2008641344", 2, "Direct hit. Smallholder maize supply and inorganic fertilizer demand under transaction costs in Kenya."),
    ("Q07", "W2104226344", 1, "Tittonell et al. Smallholder soil fertility, nutrient resource allocation, and fertilizer practices in western Kenya."),
    ("Q08", "W2010751289", 1, "Bryan et al. Household-level farm adaptation choices in Kenya; planting drought-tolerant crop varieties is a primary measured strategy."),
    ("Q08", "W2103012613", 1, "Studies rainfall deficits and maize yield implications in semi-arid zones (Machakos, Kenya)."),
    ("Q08", "W2567189951", 0, "Off-target geography (Pakistan). No cross-country African comparisons."),
    ("Q10", "W2107339504", 2, "Davis et al. (IFPRI). Measures FFS on agricultural technology adoption and crop yields in Kenya/East Africa."),
    ("Q11", "W2028791173", 2, "Holloway et al. Dairy marketing cooperatives, milk market participation, and smallholder transaction costs in East Africa/Kenya."),
    ("Q11", "W2062600778", 0, "Entomophagy (edible insects) is unrelated to dairy milk value chains or bovine milk co-ops."),
    ("Q11", "W2960695410", 0, "Black soldier fly waste bioconversion is unrelated to dairy milk value chains or dairy cooperatives."),
    ("Q12", "W2576771258", 1, "Covers post-harvest aflatoxin biocontrol (Aflasafe) and hermetic storage systems in Sub-Saharan Africa."),
    ("Q13", "W1994779802", 1, "Narrod et al. Evaluates food safety standards (GlobalGAP) and smallholders in fresh export supply chains featuring Kenya."),
    ("Q15", "W2149979947", 1, "Allen et al. Evaluates financial inclusion, digital penetration, and mobile money dynamics in Africa with substantial Kenyan benchmark data."),
    ("Q15", "W2477177347", 0, "Haushofer & Shapiro cash transfer RCT. Not evaluating mobile credit apps / informal borrowing."),
    ("Q17", "W1988249336", 0, "Barrett et al. Focuses on agricultural asset thresholds/macro shocks, not cash transfer interventions."),
    ("Q17", "W2103818838", 0, "Urban slum maternal education and child stunting. No connection to rural cash transfers."),
    ("Q17", "W2107339504", 0, "Evaluates Farmer Field Schools extension, not unconditional cash transfers."),
    ("Q17", "W2150523685", 0, "Experimental economics on risk aversion in Ethiopia; contains no rural Kenyan cash transfer data."),
    ("Q18", "W2022416391", 0, "Clinical parasitology (hookworm anemia). Zero bearing on financial inclusion or female entrepreneurship."),
    ("Q18", "W2149979947", 1, "Empirically documents gender gaps in banking and mobile account ownership across Sub-Saharan Africa."),
    ("Q19", "W2057131920", 0, "Demographic analysis of maternal health visits. No relevance to TVET or youth job creation."),
    ("Q19", "W2103818838", 0, "Child malnutrition study. Too distant from vocational technical education and youth labor markets."),
    ("Q20", "W2083553000", 1, "Thornton et al. Downscales regional climate projections and rainfall/temperature variations across East Africa."),
    ("Q20", "W4328125179", 2, "Definitive review detailing IOD, ENSO, and Walker circulation drivers causing droughts and pluvial extremes in East Africa."),
    ("Q21", "W2041589051", 1, "Models transitions where crop systems fail into rangeland/pastoral areas due to climate change."),
    ("Q22", "W2111764341", 0, "Brooks et al. Evaluates avian relaxation in forest fragments; does not analyze Mau Forest hydrological dynamics."),
    ("Q23", "W2015670964", 0, "Entomology study on coffee berry borer. Completely unrelated to Lake Victoria water quality or water hyacinth."),
    ("Q24", "W1929181818", 1, "Kristensen et al. Foundational global review on mangrove carbon sequestration, benthic respiration, and nutrient exchange.")
]

BATCH_FILES = [
    "batch2_agriculture.json",
    "batch3_economics.json",
    "batch4_climate.json",
]


def apply_corrections():
    corr_map = {(q_id, p_id): (rel, note) for q_id, p_id, rel, note in CORRECTIONS}
    updated_count = 0

    for b_file in BATCH_FILES:
        b_path = JUDGMENTS_DIR / b_file
        with open(b_path, "r", encoding="utf-8") as f:
            b_data = json.load(f)

        for item in b_data.get("items", []):
            key = (item["query_id"], item["paper_id"])
            if key in corr_map:
                new_rel, new_note = corr_map[key]
                old_rel = item["relevance"]
                item["relevance"] = new_rel
                item["notes"] = f"[HUMAN_CORRECTED: was {old_rel}] {new_note}"
                updated_count += 1
                print(f"  Applied {key[0]} - {key[1]}: [{old_rel}] -> [{new_rel}] ({new_note[:50]}...)")

        with open(b_path, "w", encoding="utf-8") as f:
            json.dump(b_data, f, indent=2)

    print(f"\nSuccessfully applied all {updated_count} human review corrections.")


if __name__ == "__main__":
    apply_corrections()
