"""CLI script to discover, download, and persist Kenyan institutions from OpenAlex."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.institution_registry import InstitutionRegistry
from src.ingestion.config import INSTITUTIONS_FILE


def main():
    print("=" * 60)
    print(" DISCOVERING KENYAN INSTITUTIONS FROM OPENALEX")
    print("=" * 60)
    registry = InstitutionRegistry(INSTITUTIONS_FILE)
    institutions = registry.discover_from_openalex()
    registry.save()

    print(f"\nDiscovered {len(institutions)} institutions. Top 15 by works count:")
    print("-" * 60)
    print(f"{'Institution Name':<40} | {'Works':<8} | {'ROR'}")
    print("-" * 60)
    for inst in institutions[:15]:
        name = (inst.get("display_name") or "Unknown")[:38]
        works = inst.get("works_count", 0)
        ror = inst.get("ror") or "N/A"
        print(f"{name:<40} | {works:<8} | {ror}")
    print("=" * 60)


if __name__ == "__main__":
    main()
