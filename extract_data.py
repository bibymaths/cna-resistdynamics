# SPDX-FileCopyrightText: 2025 Abhinav Mishra
# SPDX-License-Identifier: MIT
# DEPRECATED: This script has been migrated to tumorfits.dataio.
# Use `tumorfits extract-data` instead.
import sys
import warnings

warnings.warn(
    "extract_data.py is deprecated and will be removed in a future release. "
    "Use `tumorfits extract-data --data-root data/ --out-dir data/patient_data` instead.",
    DeprecationWarning,
    stacklevel=1,
)

from tumorfits.dataio import export_all_patient_data  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="[DEPRECATED] Extract patient CSVs from .RData files. "
        "Use `tumorfits extract-data` instead.",
    )
    parser.add_argument("--data-root", default="data", help="Root directory (default: data/)")
    parser.add_argument(
        "--out-dir", default="data/patient_data", help="Output directory (default: data/patient_data/)"
    )
    args = parser.parse_args()

    written = export_all_patient_data(args.data_root, args.out_dir)
    total = sum(len(v) for v in written.values())
    print(f"Extracted {total} CSV files for {len(written)} patients → {args.out_dir}")
    sys.exit(0)
