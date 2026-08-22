"""DuckDB warehouse and the governed query path.

Every read of a KPI goes through :meth:`Warehouse.kpi_series`, which:

  1. resolves entitlements from the semantic contract,
  2. applies the row filter belonging to the caller's own role,
  3. records the decision in the audit log — allowed or denied,
  4. and only then executes SQL.

There is no unguarded read path. That is what lets the rest of the system
treat "the AI cannot see what the human cannot see" as a structural property.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import duckdb
import pandas as pd

from verity.datagen.documents import DOCUMENTS, Document
from verity.datagen.generator import GeneratedData
from verity.governance.audit import AuditLog
from verity.governance.rbac import Principal, authorize_kpi
from verity.semantic.contract import KPIContract, SemanticContract

DEFAULT_DB_PATH = Path("data/warehouse.duckdb")


class AccessDenied(PermissionError):
    """Raised when a principal is refused a KPI read."""

    def __init__(self, reason: str, resource: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.resource = resource


@dataclass(frozen=True)
class KPIQuery:
    """SQL realisation of a KPI's contract formula.

    The contract carries the human-readable definition; this carries the
    executable one. ``validate_registry`` asserts the two sets stay in step, so
    adding a KPI to the contract without implementing it fails loudly at build
    time rather than silently returning nothing.
    """

    table: str
    value_sql: str
    date_column: str
    region_column: str = "region"
    extra_columns: tuple[str, ...] = ()


KPI_QUERIES: Mapping[str, KPIQuery] = {
    "net_revenue": KPIQuery(
        table="erp_sales",
        value_sql="SUM(net_revenue)",
        date_column="date",
        extra_columns=("product", "category", "channel"),
    ),
    "gross_margin_pct": KPIQuery(
        table="erp_sales",
        value_sql=(
            "CASE WHEN SUM(net_revenue) = 0 THEN NULL "
            "ELSE 100.0 * (SUM(net_revenue) - SUM(cogs)) / SUM(net_revenue) END"
        ),
        date_column="date",
        extra_columns=("product", "category"),
    ),
    "units_sold": KPIQuery(
        table="erp_sales",
        value_sql="SUM(units)",
        date_column="date",
        extra_columns=("product", "category", "channel"),
    ),
    "customer_churn": KPIQuery(
        table="crm_accounts",
        value_sql=(
            "CASE WHEN SUM(active_accounts_start) = 0 THEN NULL "
            "ELSE 100.0 * SUM(churned_accounts) / SUM(active_accounts_start) END"
        ),
        date_column="week_start",
        extra_columns=("segment",),
    ),
}


def validate_registry(contract: SemanticContract) -> None:
    """Fail loudly if the contract and the SQL registry have drifted apart."""
    declared = set(contract.kpis)
    implemented = set(KPI_QUERIES)
    missing = declared - implemented
    orphaned = implemented - declared
    problems = []
    if missing:
        problems.append(f"declared in contract but not implemented: {sorted(missing)}")
    if orphaned:
        problems.append(f"implemented but absent from contract: {sorted(orphaned)}")
    if problems:
        raise ValueError("KPI registry is out of step with the semantic contract; " + "; ".join(problems))


class Warehouse:
    """Thin governed wrapper over DuckDB."""

    def __init__(
        self,
        contract: SemanticContract,
        path: str | Path | None = DEFAULT_DB_PATH,
        audit: AuditLog | None = None,
    ) -> None:
        validate_registry(contract)
        self.contract = contract
        # Explicit None check, not `audit or AuditLog()`: an empty AuditLog is
        # falsy by length, so `or` would silently discard the caller's log and
        # write decisions somewhere they can never read them.
        self.audit = audit if audit is not None else AuditLog()
        if path is None:
            self._con = duckdb.connect(":memory:")
            self.path = None
        else:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._con = duckdb.connect(str(self.path))

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "Warehouse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._con

    # -- build -------------------------------------------------------------

    def build(self, data: GeneratedData, documents: tuple[Document, ...] = DOCUMENTS) -> None:
        """(Re)create every table from generated data. Idempotent."""
        frames = {
            "erp_sales": data.erp_sales,
            "promotion_api": data.promotion_api,
            "crm_accounts": data.crm_accounts,
            "ground_truth": data.ground_truth,
            "documents": _documents_frame(documents),
        }
        for name, frame in frames.items():
            self._con.register(f"_incoming_{name}", frame)
            self._con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _incoming_{name}")
            self._con.unregister(f"_incoming_{name}")

        self._con.execute(
            "CREATE OR REPLACE TABLE audit_log ("
            "  timestamp TIMESTAMP, user_id VARCHAR, role VARCHAR, region VARCHAR,"
            "  resource VARCHAR, action VARCHAR, result VARCHAR, reason VARCHAR,"
            "  row_filter VARCHAR, detail VARCHAR)"
        )

    def table_counts(self) -> dict[str, int]:
        tables = [r[0] for r in self._con.execute("SHOW TABLES").fetchall()]
        return {
            t: self._con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables
        }

    def flush_audit(self) -> int:
        """Persist in-memory audit entries into the warehouse."""
        rows = self.audit.to_rows()
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        self._con.register("_incoming_audit", frame)
        self._con.execute("INSERT INTO audit_log SELECT * FROM _incoming_audit")
        self._con.unregister("_incoming_audit")
        return len(rows)

    # -- governed reads ----------------------------------------------------

    def kpi_series(
        self,
        kpi_name: str,
        principal: Principal,
        *,
        start: date | None = None,
        end: date | None = None,
        region: str | None = None,
        group_by: tuple[str, ...] = (),
        freq: str = "day",
    ) -> pd.DataFrame:
        """Return a governed KPI time series.

        Raises :class:`AccessDenied` if the principal may not read the KPI, or
        requested a region outside their scope. Both outcomes are audited.
        """
        kpi = self.contract[kpi_name]
        decision = authorize_kpi(self.contract, kpi_name, principal, requested_region=region)
        self.audit.record_decision(
            decision, action="READ", start=start, end=end, group_by=list(group_by)
        )
        if not decision.allowed:
            raise AccessDenied(decision.reason, decision.resource)

        spec = KPI_QUERIES[kpi_name]
        where: list[str] = []
        params: list[Any] = []

        if start is not None:
            where.append(f"{spec.date_column} >= ?")
            params.append(start)
        if end is not None:
            where.append(f"{spec.date_column} <= ?")
            params.append(end)

        # A caller-supplied region narrows the query; the role's own row filter
        # constrains it. The filter is not caller-supplied, so it cannot be
        # widened by the request.
        effective_region = region or (principal.region if decision.row_filter else None)
        if effective_region:
            where.append(f"{spec.region_column} = ?")
            params.append(effective_region)

        invalid = [c for c in group_by if c not in spec.extra_columns + (spec.region_column,)]
        if invalid:
            raise ValueError(
                f"cannot group {kpi_name!r} by {invalid}; "
                f"available: {sorted(spec.extra_columns + (spec.region_column,))}"
            )

        bucket = _date_bucket(spec.date_column, freq)
        dims = ", ".join(group_by)
        select_dims = f", {dims}" if dims else ""
        group_dims = f", {dims}" if dims else ""

        sql = (
            f"SELECT {bucket} AS period{select_dims}, {spec.value_sql} AS value "
            f"FROM {spec.table} "
            + (f"WHERE {' AND '.join(where)} " if where else "")
            + f"GROUP BY period{group_dims} ORDER BY period{group_dims}"
        )
        frame = self._con.execute(sql, params).df()
        frame["kpi"] = kpi_name
        frame["unit"] = kpi.unit
        return frame

    def documents(self, principal: Principal) -> pd.DataFrame:
        """Documents this principal may retrieve. Filtered before ranking."""
        sql = (
            "SELECT * FROM documents "
            "WHERE list_contains(string_split(access_roles, ','), ?) "
            "ORDER BY timestamp"
        )
        return self._con.execute(sql, [principal.role]).df()

    def ground_truth(self, scenario_id: str | None = None) -> pd.DataFrame:
        if scenario_id:
            return self._con.execute(
                "SELECT * FROM ground_truth WHERE scenario_id = ? ORDER BY rank",
                [scenario_id],
            ).df()
        return self._con.execute(
            "SELECT * FROM ground_truth ORDER BY scenario_id, rank"
        ).df()

    def sql(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Escape hatch for analysis code. Not a governed read — do not expose."""
        return self._con.execute(query, params or []).df()


def _date_bucket(column: str, freq: str) -> str:
    freq = freq.lower()
    if freq in {"day", "d", "daily"}:
        return column
    if freq in {"week", "w", "weekly"}:
        return f"date_trunc('week', {column})"
    if freq in {"month", "m", "monthly"}:
        return f"date_trunc('month', {column})"
    raise ValueError(f"unsupported frequency {freq!r}; use day, week or month")


def _documents_frame(documents: tuple[Document, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": d.id,
                "source": d.source,
                "document_type": d.document_type,
                "title": d.title,
                "text": " ".join(d.text.split()),
                "timestamp": d.timestamp,
                "region": d.region or "",
                "product": d.product or "",
                "kpi": d.kpi or "",
                "source_reliability": d.source_reliability,
                "access_roles": ",".join(d.access_roles),
            }
            for d in documents
        ]
    )
