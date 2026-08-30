"""Command-line entry points: build the warehouse and run the demo path.

    python -m verity.scripts build     # generate the synthetic world into DuckDB
    python -m verity.scripts demo      # run the end-to-end demo and print it

Both are deterministic: the same seed always produces the same warehouse, so
evaluation runs are comparable across machines and across days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verity.analytics import assess_window, attribute_window
from verity.datagen import DOCUMENTS, SCENARIO_BY_ID, generate, scenario_series
from verity.datagen.entities import RANDOM_SEED
from verity.governance import DEMO_PRINCIPALS, route_for_assessment, summarize_cost
from verity.governance.audit import AuditLog
from verity.investigation import render_narrative
from verity.rag import build_evidence_pack
from verity.semantic import load_contract, load_policies
from verity.store import DEFAULT_DB_PATH, Warehouse
from verity.war_room import convene_war_room


def run_demo(scenario_id: str = "S1") -> dict:
    """Run the main path end to end and return every stage's output."""
    contract = load_contract()
    principal = DEMO_PRINCIPALS["west_manager"]
    scenario = SCENARIO_BY_ID[scenario_id]
    with Warehouse(contract, path=None) as warehouse:
        warehouse.build(generate(), DOCUMENTS)
        frame = scenario_series(warehouse, scenario, principal)
        assessment = assess_window(
            frame, contract[scenario.kpi], scenario.window_start, scenario.window_end
        )
        attribution = attribute_window(
            warehouse,
            kpi=scenario.kpi,
            region=scenario.region,
            start=scenario.window_start,
            end=scenario.window_end,
            scenario_id=scenario.id,
        )
        pack = build_evidence_pack(
            warehouse,
            principal,
            assessment=assessment,
            attribution=attribution,
            query="inventory promotion competitor policy approval",
        )
        narrative = render_narrative(pack, persona="ops")
        route = route_for_assessment(assessment)
        decision = convene_war_room(pack, principal) if route.war_room_allowed else None
        return {
            "assessment": assessment,
            "attribution": attribution,
            "evidence_pack": pack,
            "narrative": narrative,
            "route": route,
            "decision": decision,
            "cost": summarize_cost(route),
        }


def _demo(args: argparse.Namespace) -> int:
    result = run_demo(args.scenario)
    print(result["assessment"].explain())
    print()
    print(result["narrative"].summary)
    print()
    decision = result["decision"]
    if decision:
        print(decision.memo)
        print(decision.action_payload)
    return 0


def _build(args: argparse.Namespace) -> int:
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
        for table, count in sorted(warehouse.table_counts().items()):
            print(f"  {table:<16} {count:>8,}")

    print("\nGround truth\n" + "-" * 52)
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
            print(f"       {row['rank']}. {row['driver']:<22}{row['true_contribution_pp']:+7.2f} pp")
        print(f"          unexplained interaction {head['interaction_residual_pp']:+7.2f} pp")

    print("\nDone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="generate the synthetic world into DuckDB")
    build.add_argument("--db", default=str(DEFAULT_DB_PATH), help="warehouse path")
    build.add_argument("--seed", type=int, default=RANDOM_SEED, help="generation seed")
    build.set_defaults(func=_build)

    demo = sub.add_parser("demo", help="run the end-to-end demo path")
    demo.add_argument("--scenario", default="S1", help="scenario id, e.g. S1")
    demo.set_defaults(func=_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
