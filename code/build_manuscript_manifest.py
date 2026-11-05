#!/usr/bin/env python3
"""Record hashes for the initial manuscript package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = [
    "main.tex",
    "cover_letter_measurement.tex",
    "cover_letter_measurement.pdf",
    "cover_letter_isa_transactions.tex",
    "cover_letter_isa_transactions.pdf",
    "validation_unit_table.tex",
    "ccmg_clean_noise_table.tex",
    "qualification_demo_table.tex",
    "qualification_class_audit_table.tex",
    "raw3_capacity_table.tex",
    "conventional_vs_qualification_table.tex",
    "hust_dual_validation_table.tex",
    "hust_noise_realization_table.tex",
    "hust_augmented_split_table.tex",
    "hust_noise_mismatch_table.tex",
    "hust_independent_k_table.tex",
    "hust_signal_control_table.tex",
    "qualification_operating_table.tex",
    "references.bib",
    "main.pdf",
    "DRAFT_CONFIG.md",
    "DRAFT_AUDIT.md",
    "figures/vmd_conditioned_analysis.pdf",
    "figures/ccmg_policy_comparison.pdf",
    "figures/hust_external_validation.pdf",
    "figures/additional_external_bearing_audit.pdf",
    "figures/industrial_external_audit.pdf",
    "figures/benefit_predictor_audit.pdf",
]


def record(relative: str) -> dict:
    path = ROOT / relative
    content = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main() -> None:
    manifest = {
        "algorithm": "sha256",
        "document": "initial manuscript draft",
        "files": [record(name) for name in FILES],
    }
    (ROOT / "MANUSCRIPT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Recorded {len(manifest['files'])} manuscript files")


if __name__ == "__main__":
    main()
