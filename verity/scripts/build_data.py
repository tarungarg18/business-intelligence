"""Build the synthetic world and load it into the DuckDB warehouse.

    python -m verity.scripts.build_data

Deterministic: the same seed always produces the same warehouse, so evaluation
runs are comparable across machines and across days.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from verity.datagen import DOCUMENTS, generate
from verity.datagen.entities import RANDOM_SEED
from verity.governance.audit import AuditLog
from verity.semantic import load_contract, load_policies
from verity.store import DEFAULT_DB_PATH, Warehouse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="warehouse path")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="generation seed")
    args = parser.parse_args(argv)

    print("Verity — building synthetic world\n" + "=" * 52)

    contract = load_contract()
    policies = load_policies()
    print(f"contract   {len(contract)} KPIs: {', '.join(sorted(contract.kpis))}")
    print(f"policies   {len(policies)} policies: {', '.join(sorted(p.id for p in policies))}")

    print(f"\ngenerating (seed={args.seed}) ...")
    data = generate(seed=args.seed)
    print(data.summary())
    print(f"documents      {len(DOCUMENTS):>7,} rows")

    db_path = Path(args.db)
    print(f"\nloading into {db_path} ...")
    with Warehouse(contract, path=db_path, audit=AuditLog()) as warehouse:
        warehouse.build(data, DOCUMENTS)
        counts = warehouse.table_counts()
        for table, count in sorted(counts.items()):
            print(f"  {table:<16} {count:>8,}")

    print("\nGround truth")
    print("-" * 52)
    gt = data.ground_truth
    for sid in gt["scenario_id"].unique():
        rows = gt[gt["scenario_id"] == sid]
        head = rows.iloc[0]
        print(
            f"{sid}  {head['label']}\n"
            f"     {head['kpi']} / {head['region']} / "
            f"{head['window_start']}..{head['window_end']} "
            f"-> expect: {head['expected_behaviour']}"
        )
        if head["driver"] is None:
            print("     (control: no planted drivers)")
            continue
        print(f"     movement {head['total_movement_pct']:+.2f}%")
        for _, row in rows.iterrows():
            print(
                f"       {row['rank']}. {row['driver']:<22}"
                f"{row['true_contribution_pp']:+7.2f} pp"
            )
        print(f"          unexplained interaction {head['interaction_residual_pp']:+7.2f} pp")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
