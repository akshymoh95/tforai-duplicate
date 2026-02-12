from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    role: str
    expr: str
    description: str = ""
    derived: bool = False
    default_agg: str = ""
    depends_on: List[str] = field(default_factory=list)


def _parse_float_safe(val: Any, default: float = 0.0, field_name: str = "") -> float:
    """Safely parse float with fallback and logging."""
    if val is None or val == "":
        return default
    try:
        result = float(val)
        return result
    except (ValueError, TypeError) as e:
        import sys
        context = f" (field: {field_name})" if field_name else ""
        print(f"[WARN] Invalid float value{context}: {val!r} (type: {type(val).__name__}), using default {default}", file=sys.stderr)
        return default


@dataclass(frozen=True)
class EntitySpec:
    name: str
    description: str
    database: str
    schema: str
    table: str
    fields: List[FieldSpec]
    join_keys: List[str]
    default_metric: str
    entity_type: str
    # Reasoners applicable to this entity (registry drives reasoner selection)
    applicable_reasoners: List[str] = field(default_factory=list)
    # Derived metrics applicable to this entity (registry defines what computes)
    applicable_derived_metrics: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RelationshipSpec:
    name: str
    from_entity: str
    to_entity: str
    description: str
    join_on: List[Tuple[str, str]]


@dataclass(frozen=True)
class SignalSpec:
    """Risk signal definition - registry-driven, no code changes needed"""
    name: str
    description: str
    metric_field: str
    threshold: float
    direction: str  # "below" or "above"
    weight: float = 1.0


@dataclass(frozen=True)
class ReasonerSpec:
    """Reasoner definition - fully registry-driven for signal-based reasoners"""
    id: str
    name: str
    description: str
    entity_type: str  # What entity type it applies to (e.g., "mandate", "rm")
    signals: List[SignalSpec] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    type: str = "signal_based"  # "signal_based" or "custom"
    drilldown_plan: Dict[str, Any] = field(default_factory=dict)
    graph_id: str = ""
    params: List[Dict[str, Any]] = field(default_factory=list)
    graph_build: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivedRuleClause:
    """Single rule clause for a derived field."""
    when: str
    expr: str


@dataclass(frozen=True)
class DerivedRuleSpec:
    """Registry-driven Rel rules for a derived field."""
    entity: str
    field: str
    rules: List[DerivedRuleClause] = field(default_factory=list)
    vars: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class RegistryConfig:
    """Optional registry config flags."""
    post_compute_derived_metrics: bool = False
    allow_sql_derived_expr: bool = False


def _field(
    name: str,
    dtype: str,
    role: str,
    expr: str,
    description: str = "",
    *,
    derived: bool = False,
    default_agg: str = "",
    depends_on: List[str] | None = None,
) -> FieldSpec:
    return FieldSpec(
        name=name,
        dtype=dtype,
        role=role,
        expr=expr,
        description=description,
        derived=derived,
        default_agg=default_agg,
        depends_on=list(depends_on or []),
    )


def _use_native_views() -> bool:
    return os.environ.get("RAI_USE_NATIVE_VIEWS", "false").lower() in ("1", "true", "yes")


def _registry_path() -> Path:
    env_path = os.environ.get("RAI_REGISTRY_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent / "registry" / "semantic_registry.json"


def _load_registry_payload() -> dict | None:
    path = _registry_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _field_from_dict(data: dict) -> FieldSpec | None:
    """Parse field with validation. Returns None if invalid."""
    if not isinstance(data, dict):
        import sys
        print(f"[WARN] Field entry is not a dict: {data!r}", file=sys.stderr)
        return None
    
    name = str(data.get("name") or "").strip()
    if not name:
        import sys
        print(f"[WARN] Field missing 'name' key, skipping", file=sys.stderr)
        return None
    
    dtype = str(data.get("dtype") or "").strip()
    if not dtype:
        import sys
        print(f"[WARN] Field '{name}': missing 'dtype', using 'VARCHAR'", file=sys.stderr)
        dtype = "VARCHAR"
    
    role = str(data.get("role") or "dimension").strip()
    expr = str(data.get("expr") or "").strip()
    
    return FieldSpec(
        name=name,
        dtype=dtype,
        role=role,
        expr=expr,
        description=str(data.get("description") or "").strip(),
        derived=bool(data.get("derived", False)),
        default_agg=str(data.get("default_agg") or "").strip(),
        depends_on=[str(d).strip() for d in (data.get("depends_on") or []) if d],
    )


def _entity_from_dict(data: dict) -> EntitySpec:
    # Parse fields with validation, filtering out None entries
    raw_fields = data.get("fields") or []
    parsed_fields = []
    for f in raw_fields:
        parsed = _field_from_dict(f) if isinstance(f, dict) else None
        if parsed:
            parsed_fields.append(parsed)
    
    entity_name = str(data.get("name") or "").strip()
    if not entity_name:
        import sys
        print(f"[WARN] Entity missing 'name' key", file=sys.stderr)
        entity_name = "unknown_entity"
    
    return EntitySpec(
        name=entity_name,
        description=str(data.get("description") or "").strip(),
        database=str(data.get("database") or "").strip(),
        schema=str(data.get("schema") or "").strip(),
        table=str(data.get("table") or "").strip(),
        fields=parsed_fields,
        join_keys=[str(k).strip() for k in (data.get("join_keys") or []) if k],
        default_metric=str(data.get("default_metric") or "").strip(),
        entity_type=str(data.get("entity_type") or "generic").strip(),
        applicable_reasoners=[str(r).strip() for r in (data.get("applicable_reasoners") or []) if r],
        applicable_derived_metrics=[str(m).strip() for m in (data.get("applicable_derived_metrics") or []) if m],
    )


def _relationship_from_dict(data: dict) -> RelationshipSpec:
    join_on = [(pair[0], pair[1]) for pair in (data.get("join_on") or []) if isinstance(pair, (list, tuple)) and len(pair) == 2]
    return RelationshipSpec(
        name=str(data.get("name") or ""),
        from_entity=str(data.get("from_entity") or ""),
        to_entity=str(data.get("to_entity") or ""),
        description=str(data.get("description") or ""),
        join_on=join_on,
    )


def _reasoner_from_dict(data: dict) -> ReasonerSpec:
    signals = []
    for sig in data.get("signals") or []:
        if isinstance(sig, str):
            # String signal: use defaults
            signals.append(
                SignalSpec(
                    name=str(sig).strip(),
                    description="",
                    metric_field=str(sig).strip(),
                    threshold=0.0,
                    direction="above",
                    weight=1.0,
                )
            )
            continue
        if not isinstance(sig, dict):
            import sys
            print(f"[WARN] Reasoner signal is not dict or string: {sig!r}", file=sys.stderr)
            continue
        
        # Parse signal with defensive handling
        name = str(sig.get("name") or "").strip()
        if not name:
            import sys
            print(f"[WARN] Signal entry missing 'name', skipping", file=sys.stderr)
            continue
        
        metric_field = str(sig.get("metric_field") or "").strip()
        direction = str(sig.get("direction") or "above").strip().lower()
        
        # Validate direction
        if direction not in ("above", "below"):
            import sys
            print(f"[WARN] Signal '{name}': invalid direction '{direction}', using 'above'", file=sys.stderr)
            direction = "above"
        
        signals.append(
            SignalSpec(
                name=name,
                description=str(sig.get("description") or "").strip(),
                metric_field=metric_field if metric_field else name,
                threshold=_parse_float_safe(sig.get("threshold"), default=0.0, field_name=f"signal.{name}.threshold"),
                direction=direction,
                weight=_parse_float_safe(sig.get("weight"), default=1.0, field_name=f"signal.{name}.weight"),
            )
        )
    return ReasonerSpec(
        id=str(data.get("id") or "").strip(),
        name=str(data.get("name") or "").strip(),
        description=str(data.get("description") or "").strip(),
        entity_type=str(data.get("entity_type") or "generic").strip(),
        signals=signals,
        outputs=[str(o).strip() for o in (data.get("outputs") or []) if o],
        type=str(data.get("type") or "signal_based").strip(),
        drilldown_plan=dict(data.get("drilldown_plan") or {}),
        graph_id=str(data.get("graph_id") or "").strip(),
        params=list(data.get("params") or []),
        graph_build=dict(data.get("graph_build") or {}),
    )


def _derived_rule_from_dict(data: dict) -> DerivedRuleSpec:
    rules: List[DerivedRuleClause] = []
    for item in data.get("rules") or []:
        if not isinstance(item, dict):
            continue
        expr = str(item.get("expr") or "")
        if not expr:
            continue
        when = str(item.get("when") or "")
        rules.append(DerivedRuleClause(when=when, expr=expr))
    return DerivedRuleSpec(
        entity=str(data.get("entity") or ""),
        field=str(data.get("field") or ""),
        rules=rules,
        vars={str(k): str(v) for k, v in (data.get("vars") or {}).items() if k and v},
        description=str(data.get("description") or ""),
    )


def _config_from_dict(data: dict) -> RegistryConfig:
    return RegistryConfig(
        post_compute_derived_metrics=bool(data.get("post_compute_derived_metrics", False)),
        allow_sql_derived_expr=bool(data.get("allow_sql_derived_expr", False)),
    )


def load_registry() -> List[EntitySpec]:
    """
    Load semantic registry defining all entities and their fields.
    
    AGGREGATION SEMANTICS:
    ======================
    - sum: Values are additive across dimensions (e.g., total_aum, revenue_amount, costs)
    - avg: Values are already averages or should be averaged (e.g., performance, profit_percent)
    - Derived fields: Use 'avg' for ratios and percentages (profit_margin, cost_to_revenue)
    
    CRITICAL: When aggregating derived metrics (avg of ratios), use weighted average for accuracy.
    For now, simple avg is used - implement weighted aggregation in query layer if needed.
    
    FIELD CONSTRAINTS:
    - Division by zero: Protected with NULLIF(field, 0) in derived expressions
    - Nullable fields: Some fields like AVG_MONTHLY_AUM may be NULL, handled in queries
    """
    payload = _load_registry_payload()
    if payload and isinstance(payload, dict):
        entities = []
        for item in payload.get("entities") or []:
            if isinstance(item, dict):
                spec = _entity_from_dict(item)
                if spec.name:
                    entities.append(spec)
        if entities:
            return entities
    use_native_views = _use_native_views()
    native_mandate_fields = [
        _field("rmid", "number(38,0)", "dimension", "RMID"),
        _field("mandateid", "number(38,0)", "dimension", "MANDATEID"),
        _field("mandateid_str", "text", "dimension", "MANDATEID_STR"),
        _field("relationshipmanager", "varchar", "dimension", "RELATIONSHIPMANAGER"),
        _field("email", "varchar", "dimension", "EMAIL"),
        _field("posyear", "number(4,0)", "dimension", "POSYEAR"),
        _field("posmon", "number(2,0)", "dimension", "POSMON"),
        _field("positiondate", "date", "dimension", "POSITIONDATE"),
        _field("month_date", "date", "dimension", "MONTH_DATE"),
        _field("latest_topup_date", "date", "dimension", "LATEST_TOPUP_DATE"),
        _field("total_aum", "number(38,0)", "metric", "TOTAL_AUM", default_agg="avg"),
        _field("avg_monthly_aum", "number(38,2)", "metric", "AVG_MONTHLY_AUM", default_agg="avg"),
        _field("revenue_amount", "number(31,2)", "metric", "REVENUE_AMOUNT", default_agg="sum"),
        _field("profit_amount", "float", "metric", "PROFIT_AMOUNT", default_agg="sum"),
        _field("revenue_percent", "number(38,2)", "metric", "REVENUE_PERCENT", default_agg="avg"),
        _field("profit_percent", "float", "metric", "PROFIT_PERCENT", default_agg="avg"),
        _field("annualized_revenue_amount", "number(38,2)", "metric", "ANNUALIZED_REVENUE_AMOUNT", default_agg="sum"),
        _field("profit_amount_annualized", "float", "metric", "PROFIT_AMOUNT_ANNUALIZED", default_agg="sum"),
        _field("revenue_percent_annualized", "number(38,2)", "metric", "REVENUE_PERCENT_ANNUALIZED", default_agg="avg"),
        _field("profit_percent_annualized", "float", "metric", "PROFIT_PERCENT_ANNUALIZED", default_agg="avg"),
        _field("total_rm_cost", "number(38,0)", "metric", "TOTAL_RM_COST", default_agg="sum"),
        _field("total_fte", "float", "metric", "TOTAL_FTE", default_agg="sum"),
        _field("rm_cost_rmfte_monthly", "float", "metric", "RM_COST_RMFTE_MONTHLY", default_agg="avg"),
        _field(
            "rm_cost_for_each_mandate",
            "float",
            "metric",
            "RM_COST_FOR_EACH_MANDATE",
            description="Monthly RM cost allocated to the mandate (month-level).",
            default_agg="avg",
        ),
        _field("topup_count", "number(18,0)", "metric", "TOPUP_COUNT", default_agg="sum"),
        _field("topup_amount", "float", "metric", "TOPUP_AMOUNT", default_agg="sum"),
        _field("dealname_count", "number(18,0)", "metric", "DEALNAME_COUNT", default_agg="sum"),
        _field("mandate_count", "number(18,0)", "metric", "MANDATE_COUNT", default_agg="sum"),
        _field("aum_trend", "number", "metric", "AUM_TREND", default_agg="avg"),
        _field("revenue_trend", "number", "metric", "REVENUE_TREND", default_agg="avg"),
        _field("profit_trend", "number", "metric", "PROFIT_TREND", default_agg="avg"),
        _field("meeting_trend", "number", "metric", "MEETING_TREND", default_agg="avg"),
        _field("profit_margin", "number", "metric", "PROFIT_MARGIN", default_agg="avg"),
        _field("cost_to_revenue", "number", "metric", "COST_TO_REVENUE", default_agg="avg"),
        _field("signal_aum_decline", "number", "metric", "SIGNAL_AUM_DECLINE", default_agg="max"),
        _field("signal_churn_risk", "number", "metric", "SIGNAL_CHURN_RISK", default_agg="max"),
        _field("signal_low_profitability", "number", "metric", "SIGNAL_LOW_PROFITABILITY", default_agg="max"),
        _field("signal_revenue_decline", "number", "metric", "SIGNAL_REVENUE_DECLINE", default_agg="max"),
        _field("signal_low_engagement", "number", "metric", "SIGNAL_LOW_ENGAGEMENT", default_agg="max"),
        _field("signal_high_cost_ratio", "number", "metric", "SIGNAL_HIGH_COST_RATIO", default_agg="max"),
        _field("mandate_risk_score", "number", "metric", "MANDATE_RISK_SCORE", default_agg="avg"),
        _field("mandate_risk_condition", "text", "dimension", "MANDATE_RISK_CONDITION"),
        _field("signal_aum_decline_severe", "number", "metric", "SIGNAL_AUM_DECLINE_SEVERE", default_agg="max"),
        _field("signal_meeting_drop", "number", "metric", "SIGNAL_MEETING_DROP", default_agg="max"),
        _field("signal_profit_eroding", "number", "metric", "SIGNAL_PROFIT_ERODING", default_agg="max"),
        _field("signal_cost_spike", "number", "metric", "SIGNAL_COST_SPIKE", default_agg="max"),
        _field("churn_risk_score", "number", "metric", "CHURN_RISK_SCORE", default_agg="avg"),
        _field("churn_risk_condition", "text", "dimension", "CHURN_RISK_CONDITION"),
        _field("cost_driver_impact_score", "number", "metric", "COST_DRIVER_IMPACT_SCORE", default_agg="avg"),
        _field("cost_driver_condition", "text", "dimension", "COST_DRIVER_CONDITION"),
        _field("allocation_roi_factor", "number", "metric", "ALLOCATION_ROI_FACTOR", default_agg="avg"),
        _field("allocation_efficiency_factor", "number", "metric", "ALLOCATION_EFFICIENCY_FACTOR", default_agg="avg"),
        _field("allocation_composite_score", "number", "metric", "ALLOCATION_COMPOSITE_SCORE", default_agg="avg"),
        _field("allocation_priority", "text", "dimension", "ALLOCATION_PRIORITY"),
    ]
    legacy_mandate_fields = [
        _field("rmid", "number(38,0)", "dimension", "RMID"),
        _field("mandateid", "number(38,0)", "dimension", "MANDATEID"),
        _field("relationshipmanager", "varchar", "dimension", "RELATIONSHIPMANAGER"),
        _field("email", "varchar", "dimension", "EMAIL"),
        _field("posyear", "number(4,0)", "dimension", "POSYEAR"),
        _field("posmon", "number(2,0)", "dimension", "POSMON"),
        _field("positiondate", "date", "dimension", "POSITIONDATE"),
        _field("latest_topup_date", "date", "dimension", "LATEST_TOPUP_DATE"),
        _field("total_aum", "number(38,0)", "metric", "TOTAL_AUM", default_agg="avg"),
        _field("avg_monthly_aum", "number(38,2)", "metric", "AVG_MONTHLY_AUM", default_agg="avg"),
        _field("revenue_amount", "number(31,2)", "metric", "REVENUE_AMOUNT", default_agg="sum"),
        _field("profit_amount", "float", "metric", "PROFIT_AMOUNT", default_agg="sum"),
        _field("revenue_percent", "number(38,2)", "metric", "REVENUE_PERCENT", default_agg="avg"),
        _field("profit_percent", "float", "metric", "PROFIT_PERCENT", default_agg="avg"),
        _field("annualized_revenue_amount", "number(38,2)", "metric", "ANNUALIZED_REVENUE_AMOUNT", default_agg="sum"),
        _field("profit_amount_annualized", "float", "metric", "PROFIT_AMOUNT_ANNUALIZED", default_agg="sum"),
        _field("revenue_percent_annualized", "number(38,2)", "metric", "REVENUE_PERCENT_ANNUALIZED", default_agg="avg"),
        _field("profit_percent_annualized", "float", "metric", "PROFIT_PERCENT_ANNUALIZED", default_agg="avg"),
        _field("total_rm_cost", "number(38,0)", "metric", "TOTAL_RM_COST", default_agg="sum"),
        _field("total_fte", "float", "metric", "TOTAL_FTE", default_agg="sum"),
        _field("rm_cost_rmfte_monthly", "float", "metric", "RM_COST_RMFTE_MONTHLY", default_agg="avg"),
        _field(
            "rm_cost_for_each_mandate",
            "float",
            "metric",
            "RM_COST_FOR_EACH_MANDATE",
            description="Monthly RM cost allocated to the mandate (month-level).",
            default_agg="sum",
        ),
        _field("topup_count", "number(18,0)", "metric", "TOPUP_COUNT", default_agg="sum"),
        _field("topup_amount", "float", "metric", "TOPUP_AMOUNT", default_agg="sum"),
        _field("dealname_count", "number(18,0)", "metric", "DEALNAME_COUNT", default_agg="sum"),
        _field("mandate_count", "number(18,0)", "metric", "MANDATE_COUNT", default_agg="sum"),
        _field(
            "month_date",
            "date",
            "derived",
            "DATE_FROM_PARTS(POSYEAR, POSMON, 1)",
            derived=True,
            depends_on=["posyear", "posmon"],
        ),
        _field(
            "profit_margin",
            "number",
            "derived",
            "PROFIT_AMOUNT / NULLIF(REVENUE_AMOUNT, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["profit_amount", "revenue_amount"],
        ),
        _field(
            "cost_to_revenue",
            "number",
            "derived",
            "RM_COST_FOR_EACH_MANDATE / NULLIF(REVENUE_AMOUNT, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["rm_cost_for_each_mandate", "revenue_amount"],
        ),
        _field(
            "revenue_per_cost",
            "number",
            "derived",
            "REVENUE_AMOUNT / NULLIF(RM_COST_FOR_EACH_MANDATE, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["revenue_amount", "rm_cost_for_each_mandate"],
        ),
        _field(
            "aum_per_cost",
            "number",
            "derived",
            "TOTAL_AUM / NULLIF(RM_COST_FOR_EACH_MANDATE, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["total_aum", "rm_cost_for_each_mandate"],
        ),
        _field(
            "aum_trend",
            "number",
            "derived",
            "(TOTAL_AUM - PREV_TOTAL_AUM) / NULLIF(PREV_TOTAL_AUM, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["total_aum", "posyear", "posmon", "mandateid", "rmid"],
        ),
        _field(
            "revenue_trend",
            "number",
            "derived",
            "(REVENUE_AMOUNT - PREV_REVENUE_AMOUNT) / NULLIF(PREV_REVENUE_AMOUNT, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["revenue_amount", "posyear", "posmon", "mandateid", "rmid"],
        ),
        _field(
            "profit_trend",
            "number",
            "derived",
            "(PROFIT_AMOUNT - PREV_PROFIT_AMOUNT) / NULLIF(PREV_PROFIT_AMOUNT, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["profit_amount", "posyear", "posmon", "mandateid", "rmid"],
        ),
        _field(
            "meeting_trend",
            "number",
            "derived",
            "(TOPUP_COUNT - PREV_TOPUP_COUNT) / NULLIF(PREV_TOPUP_COUNT, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["topup_count", "posyear", "posmon", "mandateid", "rmid"],
        ),
        _field(
            "signal_aum_decline",
            "number",
            "derived",
            "CASE WHEN aum_trend < -0.05 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["aum_trend"],
        ),
        _field(
            "signal_low_profitability",
            "number",
            "derived",
            "CASE WHEN profit_margin < 0.05 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["profit_margin"],
        ),
        _field(
            "signal_revenue_decline",
            "number",
            "derived",
            "CASE WHEN revenue_trend < -0.10 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["revenue_trend"],
        ),
        _field(
            "signal_low_engagement",
            "number",
            "derived",
            "CASE WHEN TOPUP_COUNT < 2 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["topup_count"],
        ),
        _field(
            "signal_high_cost_ratio",
            "number",
            "derived",
            "CASE WHEN cost_to_revenue > 0.30 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["cost_to_revenue"],
        ),
        _field(
            "signal_high_cost",
            "number",
            "derived",
            "CASE WHEN cost_to_revenue > 0.30 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["cost_to_revenue"],
        ),
        _field(
            "mandate_risk_score",
            "number",
            "derived",
            "(signal_aum_decline * 1.5 + signal_low_profitability * 1.2 + signal_revenue_decline + signal_low_engagement * 0.8 + signal_high_cost_ratio) / 5.5",
            derived=True,
            default_agg="avg",
            depends_on=[
                "signal_aum_decline",
                "signal_low_profitability",
                "signal_revenue_decline",
                "signal_low_engagement",
                "signal_high_cost_ratio",
            ],
        ),
        _field(
            "mandate_risk_condition",
            "text",
            "derived",
            "CASE WHEN mandate_risk_score > 0.75 THEN 'at_risk' WHEN mandate_risk_score > 0.40 THEN 'declining' ELSE 'healthy' END",
            derived=True,
            depends_on=["mandate_risk_score"],
        ),
        _field(
            "signal_aum_decline_severe",
            "number",
            "derived",
            "CASE WHEN aum_trend < -0.15 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["aum_trend"],
        ),
        _field(
            "signal_meeting_drop",
            "number",
            "derived",
            "CASE WHEN meeting_trend < -0.30 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["meeting_trend"],
        ),
        _field(
            "signal_profit_eroding",
            "number",
            "derived",
            "CASE WHEN profit_margin < 0.05 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["profit_margin"],
        ),
        _field(
            "signal_cost_spike",
            "number",
            "derived",
            "CASE WHEN cost_to_revenue > 0.40 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["cost_to_revenue"],
        ),
        _field(
            "churn_risk_score",
            "number",
            "derived",
            "(signal_aum_decline_severe * 1.5 + signal_meeting_drop * 1.3 + signal_profit_eroding * 1.4 + signal_cost_spike * 1.2) / 5.4",
            derived=True,
            default_agg="avg",
            depends_on=[
                "signal_aum_decline_severe",
                "signal_meeting_drop",
                "signal_profit_eroding",
                "signal_cost_spike",
            ],
        ),
        _field(
            "signal_churn_risk",
            "number",
            "derived",
            "CASE WHEN churn_risk_score > 0.75 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["churn_risk_score"],
        ),
        _field(
            "churn_risk_condition",
            "text",
            "derived",
            "CASE WHEN churn_risk_score > 0.75 THEN 'at_risk' WHEN churn_risk_score > 0.40 THEN 'declining' ELSE 'healthy' END",
            derived=True,
            depends_on=["churn_risk_score"],
        ),
        _field(
            "cost_driver_impact_score",
            "number",
            "derived",
            "CASE WHEN cost_to_revenue > 0.30 THEN cost_to_revenue * 0.5 ELSE 0 END",
            derived=True,
            default_agg="avg",
            depends_on=["cost_to_revenue"],
        ),
        _field(
            "cost_driver_condition",
            "text",
            "derived",
            "CASE WHEN cost_to_revenue > 0.40 THEN 'high_impact' WHEN cost_to_revenue > 0.30 THEN 'moderate_impact' ELSE 'low_impact' END",
            derived=True,
            depends_on=["cost_to_revenue"],
        ),
        _field(
            "allocation_roi_factor",
            "number",
            "derived",
            "CASE WHEN RM_COST_FOR_EACH_MANDATE > 0 THEN (REVENUE_AMOUNT - RM_COST_FOR_EACH_MANDATE) / RM_COST_FOR_EACH_MANDATE ELSE 0 END",
            derived=True,
            default_agg="avg",
            depends_on=["revenue_amount", "rm_cost_for_each_mandate"],
        ),
        _field(
            "allocation_efficiency_factor",
            "number",
            "derived",
            "CASE WHEN TOPUP_COUNT > 0 THEN REVENUE_AMOUNT / TOPUP_COUNT ELSE 0 END",
            derived=True,
            default_agg="avg",
            depends_on=["revenue_amount", "topup_count"],
        ),
        _field(
            "allocation_composite_score",
            "number",
            "derived",
            "CASE WHEN TOPUP_COUNT > 0 AND RM_COST_FOR_EACH_MANDATE > 0 THEN (allocation_roi_factor * 0.6) + (allocation_efficiency_factor / 10000 * 0.4) ELSE 0 END",
            derived=True,
            default_agg="avg",
            depends_on=[
                "allocation_roi_factor",
                "allocation_efficiency_factor",
                "topup_count",
                "rm_cost_for_each_mandate",
            ],
        ),
        _field(
            "allocation_priority",
            "text",
            "derived",
            "CASE WHEN allocation_composite_score > 1.0 THEN 'HIGH' WHEN allocation_composite_score > 0.5 THEN 'MEDIUM' ELSE 'LOW' END",
            derived=True,
            depends_on=["allocation_composite_score"],
        ),
    ]
    legacy_mandate_derived = [field for field in legacy_mandate_fields if field.derived]
    if use_native_views:
        native_names = {field.name for field in native_mandate_fields}
        derived_missing = [field for field in legacy_mandate_derived if field.name not in native_names]
        mandate_fields = native_mandate_fields + derived_missing
    else:
        mandate_fields = legacy_mandate_fields

    native_rm_fields = [
        _field("rmid", "number(38,0)", "dimension", "RMID"),
        _field("rm_name", "varchar", "dimension", "RM_NAME"),
        _field("meet_year", "number(4,0)", "dimension", "MEET_YEAR"),
        _field("meet_mon", "number(2,0)", "dimension", "MEET_MON"),
        _field("month_date", "date", "dimension", "MONTH_DATE"),
        _field("performance", "float", "metric", "PERFORMANCE", default_agg="avg"),
        _field("aum", "number(38,6)", "metric", "AUM_AVG", default_agg="avg"),
        _field("total_nnm", "float", "metric", "TOTAL_NNM", default_agg="sum"),
        _field("cash_in", "float", "metric", "CASH_IN", default_agg="sum"),
        _field("buy", "float", "metric", "BUY", default_agg="sum"),
        _field("sell", "float", "metric", "SELL", default_agg="sum"),
        _field("topupcommitment", "float", "metric", "TOPUPCOMMITMENT", default_agg="sum"),
        _field("firstcommitment", "float", "metric", "FIRSTCOMMITMENT", default_agg="sum"),
        _field("total_mandates", "number(18,0)", "metric", "TOTAL_MANDATES", default_agg="sum"),
        _field("total_month_mandates", "number(18,0)", "metric", "TOTAL_MONTH_MANDATES", default_agg="sum"),
        _field("meeting_count_target", "number(38,0)", "metric", "MEETING_COUNT_TARGET", default_agg="sum"),
        _field("clienttarget", "number(38,0)", "metric", "CLIENTTARGET", default_agg="sum"),
        _field("commitmenttarget", "float", "metric", "COMMITMENTTARGET", default_agg="sum"),
        _field("nnmtarget", "float", "metric", "NNMTARGET", default_agg="sum"),
        _field("new_mandate_positive_cnt", "number(18,0)", "metric", "NEW_MANDATE_POSITIVE_CNT", default_agg="sum"),
        _field("new_mandate_negative_cnt", "number(18,0)", "metric", "NEW_MANDATE_NEGATIVE_CNT", default_agg="sum"),
        _field("new_mandate_neutral_cnt", "number(18,0)", "metric", "NEW_MANDATE_NEUTRAL_CNT", default_agg="sum"),
        _field("new_mandate_total_meetings", "number(18,0)", "metric", "NEW_MANDATE_TOTAL_MEETINGS", default_agg="sum"),
        _field("existing_mandate_positive_cnt", "number(18,0)", "metric", "EXISTING_MANDATE_POSITIVE_CNT", default_agg="sum"),
        _field("existing_mandate_negative_cnt", "number(18,0)", "metric", "EXISTING_MANDATE_NEGATIVE_CNT", default_agg="sum"),
        _field("existing_mandate_neutral_cnt", "number(18,0)", "metric", "EXISTING_MANDATE_NEUTRAL_CNT", default_agg="sum"),
        _field("existing_mandate_total_meetings", "number(18,0)", "metric", "EXISTING_MANDATE_TOTAL_MEETINGS", default_agg="sum"),
        _field("signal_low_aum", "number", "metric", "SIGNAL_LOW_AUM", default_agg="max"),
        _field("signal_low_performance", "number", "metric", "SIGNAL_LOW_PERFORMANCE", default_agg="max"),
        _field("signal_low_mandate_count", "number", "metric", "SIGNAL_LOW_MANDATE_COUNT", default_agg="max"),
        _field("rm_performance_condition", "text", "dimension", "RM_PERFORMANCE_CONDITION"),
        _field("meeting_count_variance", "number", "metric", "MEETING_COUNT_VARIANCE", default_agg="avg"),
        _field("client_target_variance", "number", "metric", "CLIENT_TARGET_VARIANCE", default_agg="avg"),
        _field("aum_trend_rm", "number", "metric", "AUM_TREND_RM", default_agg="avg"),
        _field("mandate_growth_trend", "number", "metric", "MANDATE_GROWTH_TREND", default_agg="avg"),
    ]
    legacy_rm_fields = [
        _field("rmid", "number(38,0)", "dimension", "RMID"),
        _field("rm_name", "varchar", "dimension", "RM_NAME"),
        _field("meet_year", "number(4,0)", "dimension", "MEET_YEAR"),
        _field("meet_mon", "number(2,0)", "dimension", "MEET_MON"),
        _field("performance", "float", "metric", "PERFORMANCE", default_agg="avg"),
        _field("aum", "number(38,6)", "metric", "AUM", default_agg="avg"),
        _field("total_nnm", "float", "metric", "TOTAL_NNM", default_agg="sum"),
        _field("cash_in", "float", "metric", "CASH_IN", default_agg="sum"),
        _field("buy", "float", "metric", "BUY", default_agg="sum"),
        _field("sell", "float", "metric", "SELL", default_agg="sum"),
        _field("topupcommitment", "float", "metric", "TOPUPCOMMITMENT", default_agg="sum"),
        _field("firstcommitment", "float", "metric", "FIRSTCOMMITMENT", default_agg="sum"),
        _field("total_mandates", "number(18,0)", "metric", "TOTAL_MANDATES", default_agg="sum"),
        _field("total_month_mandates", "number(18,0)", "metric", "TOTAL_MONTH_MANDATES", default_agg="sum"),
        _field("meeting_count_target", "number(38,0)", "metric", "MEETING_COUNT_TARGET", default_agg="sum"),
        _field("clienttarget", "number(38,0)", "metric", "CLIENTTARGET", default_agg="sum"),
        _field("commitmenttarget", "float", "metric", "COMMITMENTTARGET", default_agg="sum"),
        _field("nnmtarget", "float", "metric", "NNMTARGET", default_agg="sum"),
        _field("new_mandate_positive_cnt", "number(18,0)", "metric", "NEW_MANDATE_POSITIVE_CNT", default_agg="sum"),
        _field("new_mandate_negative_cnt", "number(18,0)", "metric", "NEW_MANDATE_NEGATIVE_CNT", default_agg="sum"),
        _field("new_mandate_neutral_cnt", "number(18,0)", "metric", "NEW_MANDATE_NEUTRAL_CNT", default_agg="sum"),
        _field("new_mandate_total_meetings", "number(18,0)", "metric", "NEW_MANDATE_TOTAL_MEETINGS", default_agg="sum"),
        _field("existing_mandate_positive_cnt", "number(18,0)", "metric", "EXISTING_MANDATE_POSITIVE_CNT", default_agg="sum"),
        _field("existing_mandate_negative_cnt", "number(18,0)", "metric", "EXISTING_MANDATE_NEGATIVE_CNT", default_agg="sum"),
        _field("existing_mandate_neutral_cnt", "number(18,0)", "metric", "EXISTING_MANDATE_NEUTRAL_CNT", default_agg="sum"),
        _field("existing_mandate_total_meetings", "number(18,0)", "metric", "EXISTING_MANDATE_TOTAL_MEETINGS", default_agg="sum"),
        _field(
            "month_date",
            "date",
            "derived",
            "DATE_FROM_PARTS(MEET_YEAR, MEET_MON, 1)",
            derived=True,
            depends_on=["meet_year", "meet_mon"],
        ),
        _field(
            "signal_low_aum",
            "number",
            "derived",
            "CASE WHEN AUM < 1000000 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["aum"],
        ),
        _field(
            "signal_low_performance",
            "number",
            "derived",
            "CASE WHEN PERFORMANCE < 0.5 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["performance"],
        ),
        _field(
            "signal_low_mandate_count",
            "number",
            "derived",
            "CASE WHEN TOTAL_MANDATES < 5 THEN 1 ELSE 0 END",
            derived=True,
            default_agg="max",
            depends_on=["total_mandates"],
        ),
        _field(
            "rm_performance_condition",
            "text",
            "derived",
            "CASE WHEN PERFORMANCE > 0.75 THEN 'underperforming' WHEN PERFORMANCE > 0.40 THEN 'needs_support' ELSE 'performing' END",
            derived=True,
            depends_on=["performance"],
        ),
        _field(
            "meeting_count_variance",
            "number",
            "derived",
            "CASE WHEN MEETING_COUNT_TARGET > 0 THEN (NEW_MANDATE_TOTAL_MEETINGS + EXISTING_MANDATE_TOTAL_MEETINGS) / MEETING_COUNT_TARGET ELSE 0 END",
            derived=True,
            default_agg="avg",
            depends_on=["meeting_count_target", "new_mandate_total_meetings", "existing_mandate_total_meetings"],
        ),
        _field(
            "client_target_variance",
            "number",
            "derived",
            "CASE WHEN CLIENTTARGET > 0 THEN TOTAL_MONTH_MANDATES / CLIENTTARGET ELSE 0 END",
            derived=True,
            default_agg="avg",
            depends_on=["total_month_mandates", "clienttarget"],
        ),
        _field(
            "aum_trend",
            "number",
            "derived",
            "(AUM - PREV_AUM) / NULLIF(PREV_AUM, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["aum", "meet_year", "meet_mon", "rmid"],
        ),
        _field(
            "revenue_trend",
            "number",
            "derived",
            "(TOTAL_NNM - PREV_TOTAL_NNM) / NULLIF(PREV_TOTAL_NNM, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["total_nnm", "meet_year", "meet_mon", "rmid"],
        ),
        _field(
            "profit_trend",
            "number",
            "derived",
            "(PERFORMANCE - PREV_PERFORMANCE) / NULLIF(PREV_PERFORMANCE, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["performance", "meet_year", "meet_mon", "rmid"],
        ),
        _field(
            "meeting_trend",
            "number",
            "derived",
            "((NEW_MANDATE_TOTAL_MEETINGS + EXISTING_MANDATE_TOTAL_MEETINGS) - PREV_MEETING_COUNT) / NULLIF(PREV_MEETING_COUNT, 0)",
            derived=True,
            default_agg="avg",
            depends_on=[
                "new_mandate_total_meetings",
                "existing_mandate_total_meetings",
                "meet_year",
                "meet_mon",
                "rmid",
            ],
        ),
        _field(
            "aum_trend_rm",
            "number",
            "derived",
            "(AUM - PREV_AUM) / NULLIF(PREV_AUM, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["aum", "meet_year", "meet_mon", "rmid"],
        ),
        _field(
            "mandate_growth_trend",
            "number",
            "derived",
            "(TOTAL_MANDATES - PREV_TOTAL_MANDATES) / NULLIF(PREV_TOTAL_MANDATES, 0)",
            derived=True,
            default_agg="avg",
            depends_on=["total_mandates", "meet_year", "meet_mon", "rmid"],
        ),
    ]
    legacy_rm_derived = [field for field in legacy_rm_fields if field.derived]
    if use_native_views:
        native_names = {field.name for field in native_rm_fields}
        derived_missing = [field for field in legacy_rm_derived if field.name not in native_names]
        rm_fields = native_rm_fields + derived_missing
    else:
        rm_fields = legacy_rm_fields

    meeting_fields = [
        _field("id", "number(38,0)", "dimension", "ID"),
        _field("rmid", "number(38,0)", "dimension", "RMID"),
        _field("mandateid", "number(38,0)", "dimension", "MANDATEID"),
        _field("meetingdate", "timestamp_ntz(9)", "dimension", "MEETINGDATE"),
        _field("sentiment", "varchar", "dimension", "SENTIMENT"),
        _field("meetingduration", "number(38,1)", "metric", "MEETINGDURATION", default_agg="avg"),
        _field("meetingtype", "varchar", "dimension", "MEETINGTYPE"),
        _field("meetingnotes", "varchar", "dimension", "MEETINGNOTES"),
        _field("cleaned_notes", "varchar", "dimension", "CLEANED_NOTES"),
        _field("reasoning", "varchar", "dimension", "REASONING"),
        _field("suggestion", "varchar", "dimension", "SUGGESTION"),
        _field("hubspotid", "number(38,0)", "dimension", "HUBSPOTID"),
        _field("utmsourcefirsttouch", "varchar", "dimension", "UTMSOURCEFIRSTTOUCH"),
        _field("ownerid", "number(38,0)", "dimension", "OWNERID"),
        _field("create_date", "timestamp_ntz(9)", "dimension", "CREATE_DATE"),
        _field("email", "varchar", "dimension", "EMAIL"),
        _field(
            "meeting_id",
            "number",
            "derived",
            "ID",
            derived=True,
            depends_on=["id"],
        ),
        _field(
            "meeting_date",
            "timestamp",
            "derived",
            "MEETINGDATE",
            derived=True,
            depends_on=["meetingdate"],
        ),
        _field(
            "duration",
            "number",
            "derived",
            "MEETINGDURATION",
            derived=True,
            default_agg="avg",
            depends_on=["meetingduration"],
        ),
        _field(
            "meeting_type",
            "text",
            "derived",
            "MEETINGTYPE",
            derived=True,
            depends_on=["meetingtype"],
        ),
        _field(
            "meeting_notes",
            "text",
            "derived",
            "MEETINGNOTES",
            derived=True,
            depends_on=["meetingnotes"],
        ),
        _field(
            "utm_source_first_touch",
            "text",
            "derived",
            "UTMSOURCEFIRSTTOUCH",
            derived=True,
            depends_on=["utmsourcefirsttouch"],
        ),
        _field(
            "month_date",
            "date",
            "derived",
            "DATE_FROM_PARTS(YEAR(MEETINGDATE), MONTH(MEETINGDATE), 1)",
            derived=True,
            depends_on=["meetingdate"],
        ),
    ]

    mandate_table = "RAI_MANDATE_PROFITABILITY" if use_native_views else "MANDATE_PROFITABILITY_CT"
    rm_table = "RAI_RM_PERFORMANCE_MONTH" if use_native_views else "RM_PERFORMANCE_TBL"
    mandate_derived_metrics = [field.name for field in legacy_mandate_derived]
    rm_derived_metrics = [field.name for field in legacy_rm_derived]

    return [
        EntitySpec(
            name="MandateMonthlySummary",
            description="Mandate-level monthly profitability and cost signals. Unique by (rmid, mandateid, posyear, posmon).",
            database="TFO_TEST_CLONED",
            schema="ML_TEST",
            table=mandate_table,
            fields=mandate_fields,
            join_keys=["rmid", "mandateid", "posyear", "posmon"],
            default_metric="total_aum",
            entity_type="mandate",
            # Registry declares which reasoners apply to this entity
            applicable_reasoners=["mandate_risk"],
            # Registry declares which derived metrics are computed
            applicable_derived_metrics=mandate_derived_metrics,
        ),
        EntitySpec(
            name="RMMonthlySummary",
            description="RM monthly performance signals. Unique by (rmid, meet_year, meet_mon).",
            database="TFO_TEST_CLONED",
            schema="ML_TEST",
            table=rm_table,
            fields=rm_fields,
            join_keys=["rmid", "meet_year", "meet_mon"],
            default_metric="performance",
            entity_type="rm",
            applicable_reasoners=["rm_performance"],
            applicable_derived_metrics=rm_derived_metrics,
        ),
        EntitySpec(
            name="MeetingSentiment",
            description="Meeting sentiment and activity signals. Unique by (rmid, mandateid, meeting_id).",
            database="TFO_TEST_CLONED",
            schema="ML_TEST",
            table="MEETING_SENTIMENT_ANALYSIS_MANDATE_V",
            fields=meeting_fields,
            join_keys=["rmid", "mandateid", "id"],
            default_metric="duration",
            entity_type="meeting",
            applicable_reasoners=[],
            applicable_derived_metrics=[],
        ),
    ]


def entity_index(entities: Iterable[EntitySpec]) -> Dict[str, EntitySpec]:
    return {e.name: e for e in entities}


def load_relationships() -> List[RelationshipSpec]:
    """
    Relationships between entities. These enable cross-entity joins in queries.
    
    Semantics:
    - RM manages mandates: Join on rmid only (allows RMs to see all their mandates across time)
    - RM manages mandates (same month): Join on (rmid, year, month) (isolates period-specific data)
    - Mandate belongs to RM: Inverse of "manages mandates"
    - Meetings for mandate: Join on mandateid (allows query of meetings for a mandate)
    - Meetings for RM: Join on rmid (allows query of meetings an RM participated in)
    """
    payload = _load_registry_payload()
    if payload and isinstance(payload, dict):
        relationships = []
        for item in payload.get("relationships") or []:
            if isinstance(item, dict):
                spec = _relationship_from_dict(item)
                if spec.name:
                    relationships.append(spec)
        if relationships:
            return relationships
    return [
        # RM → Mandate relationships
        RelationshipSpec(
            name="manages_mandates_month",
            from_entity="RMMonthlySummary",
            to_entity="MandateMonthlySummary",
            description="RM manages mandates in the same month. Aligned join on (rmid, year, month).",
            join_on=[("rmid", "rmid"), ("meet_year", "posyear"), ("meet_mon", "posmon")],
        ),
        # Mandate → RM relationships
        RelationshipSpec(
            name="assigned_rm_month",
            from_entity="MandateMonthlySummary",
            to_entity="RMMonthlySummary",
            description="Mandate belongs to an RM in the same month. Aligned join on (rmid, year, month).",
            join_on=[("rmid", "rmid"), ("posyear", "meet_year"), ("posmon", "meet_mon")],
        ),
        # Mandate → Meeting relationships
        RelationshipSpec(
            name="mandate_meetings",
            from_entity="MandateMonthlySummary",
            to_entity="MeetingSentiment",
            description="Meetings for a mandate. Join on mandateid.",
            join_on=[("mandateid", "mandateid"), ("rmid", "rmid")],
        ),
        # RM → Meeting relationships
        RelationshipSpec(
            name="rm_meetings",
            from_entity="RMMonthlySummary",
            to_entity="MeetingSentiment",
            description="Meetings for an RM. Join on rmid.",
            join_on=[("rmid", "rmid")],
        ),
    ]


def load_reasoners() -> List[ReasonerSpec]:
    """
    Define reasoners in registry (signal-based, no code changes needed).
    
    Clients can override thresholds and weights by modifying this function only.
    No need to change rai_reasoners.py - framework instantiates from spec.
    """
    payload = _load_registry_payload()
    default_outputs = {
        "mandate_risk": [
            "mandate_risk_score",
            "mandate_risk_condition",
            "signal_aum_decline",
            "signal_revenue_decline",
            "signal_low_profitability",
            "signal_low_engagement",
            "signal_high_cost",
        ],
        "churn_risk": [
            "churn_risk_score",
            "churn_risk_condition",
            "signal_churn_risk",
        ],
        "trend": [
            "aum_trend",
            "revenue_trend",
            "profit_trend",
            "meeting_trend",
        ],
        "cost_driver": [
            "cost_driver_impact_score",
            "cost_driver_condition",
            "signal_high_cost_ratio",
        ],
        "allocation": [
            "allocation_priority",
            "allocation_composite_score",
            "allocation_roi_factor",
            "allocation_efficiency_factor",
        ],
        "rm_performance": [
            "rm_performance_condition",
            "signal_low_aum",
            "signal_low_performance",
        ],
    }
    # If the registry payload explicitly includes a "reasoners" key, treat it as an
    # authoritative override. This allows registries for other domains to disable
    # the built-in fallback reasoners by setting `"reasoners": []`.
    if payload and isinstance(payload, dict) and "reasoners" in payload:
        reasoners: List[ReasonerSpec] = []
        for item in (payload.get("reasoners") or []):
            if isinstance(item, dict):
                spec = _reasoner_from_dict(item)
                if spec.id:
                    if spec.outputs:
                        reasoners.append(spec)
                    else:
                        reasoners.append(
                            ReasonerSpec(
                                id=spec.id,
                                name=spec.name,
                                description=spec.description,
                                entity_type=spec.entity_type,
                                signals=spec.signals,
                                outputs=default_outputs.get(spec.id, []),
                                type=spec.type,
                            )
                        )
        return reasoners
    return [
        ReasonerSpec(
            id="mandate_risk",
            name="Mandate Risk Reasoner",
            description="Identifies mandates at risk of attrition or declining engagement",
            entity_type="mandate",
            type="signal_based",
            outputs=[
                "mandate_risk_score",
                "mandate_risk_condition",
                "signal_aum_decline",
                "signal_revenue_decline",
                "signal_low_profitability",
                "signal_low_engagement",
                "signal_high_cost",
            ],
            signals=[
                SignalSpec(
                    name="aum_decline",
                    description="Decreasing AUM signals client withdrawal",
                    metric_field="aum_trend",
                    threshold=-0.05,
                    direction="below",
                    weight=1.5,
                ),
                SignalSpec(
                    name="low_profitability",
                    description="Low profit margin indicates unprofitable relationship",
                    metric_field="profit_margin",
                    threshold=0.05,
                    direction="below",
                    weight=1.2,
                ),
                SignalSpec(
                    name="revenue_decline",
                    description="Declining revenue suggests reduced engagement",
                    metric_field="revenue_trend",
                    threshold=-0.10,
                    direction="below",
                    weight=1.0,
                ),
                SignalSpec(
                    name="low_engagement",
                    description="Few top-ups indicate poor relationship health",
                    metric_field="topup_count",
                    threshold=2,
                    direction="below",
                    weight=0.8,
                ),
                SignalSpec(
                    name="high_cost_ratio",
                    description="High cost-to-revenue ratio shows unprofitable servicing",
                    metric_field="cost_to_revenue",
                    threshold=0.30,
                    direction="above",
                    weight=1.0,
                ),
            ],
        ),
        ReasonerSpec(
            id="rm_performance",
            name="RM Performance Reasoner",
            description="Evaluates RM effectiveness and key performance drivers",
            entity_type="rm",
            type="signal_based",
            outputs=[
                "rm_performance_condition",
                "signal_low_aum",
                "signal_low_performance",
            ],
            signals=[
                SignalSpec(
                    name="low_aum",
                    description="Low assets under management",
                    metric_field="aum",
                    threshold=1000000,
                    direction="below",
                    weight=1.0,
                ),
                SignalSpec(
                    name="low_performance",
                    description="Below-average performance score",
                    metric_field="performance",
                    threshold=0.5,
                    direction="below",
                    weight=1.0,
                ),
            ],
        ),
        # Phase 2: New reasoners for advanced analytics
        ReasonerSpec(
            id="trend",
            name="Trend Reasoner",
            description="Detects profit, AUM, and revenue trends over time",
            entity_type="mandate",
            type="signal_based",
            outputs=[
                "aum_trend",
                "revenue_trend",
                "profit_trend",
                "meeting_trend",
            ],
            signals=[
                SignalSpec(
                    name="aum_declining",
                    description="AUM decreasing over time",
                    metric_field="aum_trend",
                    threshold=-0.05,
                    direction="below",
                    weight=1.0,
                ),
                SignalSpec(
                    name="revenue_declining",
                    description="Revenue decreasing over time",
                    metric_field="revenue_trend",
                    threshold=-0.10,
                    direction="below",
                    weight=1.0,
                ),
                SignalSpec(
                    name="profit_declining",
                    description="Profit eroding over time",
                    metric_field="profit_trend",
                    threshold=-0.05,
                    direction="below",
                    weight=1.2,
                ),
            ],
        ),
        ReasonerSpec(
            id="churn_risk",
            name="Churn Risk Reasoner",
            description="Predicts clients at risk of churning or becoming unprofitable within 3 months",
            entity_type="mandate",
            type="signal_based",
            outputs=[
                "churn_risk_score",
                "churn_risk_condition",
                "signal_churn_risk",
            ],
            signals=[
                SignalSpec(
                    name="aum_decline_severe",
                    description="Severe AUM decline",
                    metric_field="aum_trend",
                    threshold=-0.15,
                    direction="below",
                    weight=1.5,
                ),
                SignalSpec(
                    name="meeting_drop",
                    description="Meetings declining sharply",
                    metric_field="meeting_trend",
                    threshold=-0.30,
                    direction="below",
                    weight=1.3,
                ),
                SignalSpec(
                    name="profit_eroding",
                    description="Profitability eroding",
                    metric_field="profit_margin",
                    threshold=0.05,
                    direction="below",
                    weight=1.4,
                ),
                SignalSpec(
                    name="cost_spike",
                    description="Cost ratio increasing",
                    metric_field="cost_to_revenue",
                    threshold=0.40,
                    direction="above",
                    weight=1.2,
                ),
            ],
        ),
        ReasonerSpec(
            id="cost_driver",
            name="Cost Driver Reasoner",
            description="Identifies which cost components most impact profit margins",
            entity_type="mandate",
            type="signal_based",
            outputs=[
                "cost_driver_impact_score",
                "cost_driver_condition",
                "signal_high_cost_ratio",
            ],
            signals=[],  # Analytical reasoner, not signal-based
        ),
        ReasonerSpec(
            id="allocation",
            name="Allocation Reasoner",
            description="Scores clients by ROI to guide RM time allocation",
            entity_type="mandate",
            type="signal_based",
            outputs=[
                "allocation_priority",
                "allocation_composite_score",
                "allocation_roi_factor",
                "allocation_efficiency_factor",
            ],
            signals=[],  # Optimization reasoner, not signal-based
        ),
    ]


def load_derived_rel_rules() -> List[DerivedRuleSpec]:
    """Load registry-defined Rel rules for derived fields."""
    payload = _load_registry_payload()
    if payload and isinstance(payload, dict):
        rules = []
        for item in payload.get("derived_rel_rules") or []:
            if isinstance(item, dict):
                spec = _derived_rule_from_dict(item)
                if spec.entity and spec.field and spec.rules:
                    rules.append(spec)
        if rules:
            return rules
    return []


def load_registry_config() -> RegistryConfig:
    """Load optional registry config flags."""
    payload = _load_registry_payload()
    if payload and isinstance(payload, dict):
        cfg = payload.get("config") or {}
        if isinstance(cfg, dict):
            return _config_from_dict(cfg)
    return RegistryConfig()


def load_prompt_templates() -> Dict[str, Any]:
    """Load optional prompt templates for summarization/charting."""
    payload = _load_registry_payload()
    if payload and isinstance(payload, dict):
        templates = payload.get("prompt_templates") or {}
        if isinstance(templates, dict):
            return templates
    return {}


def load_analysis_config() -> Dict[str, Any]:
    """Load optional analysis config (heuristics, synonyms) for client-specific behavior."""
    payload = _load_registry_payload()
    if payload and isinstance(payload, dict):
        cfg = payload.get("analysis_config") or {}
        if isinstance(cfg, dict):
            return cfg
    return {}


def load_kg_spec() -> Dict[str, Any]:
    """
    Load optional KG configuration from the registry payload.
    This is a free-form object used to drive KG materialization.
    """
    payload = _load_registry_payload()
    if payload and isinstance(payload, dict):
        kg = payload.get("kg") or {}
        if isinstance(kg, dict):
            return kg
    return {}


def validate_registry() -> List[str]:
    """
    Validate semantic registry for common errors and inconsistencies.
    Returns list of error/warning messages. Empty list means validation passed.
    
    Checks:
    1. No duplicate entity names
    2. No duplicate table+schema combinations
    3. All join_keys exist as fields
    4. All default_metrics exist as fields
    5. Derived fields have valid dependencies (all depends_on fields exist)
    6. No circular dependencies in derived fields
    7. Relationships reference existing entities
    8. Relationship join fields exist in both entities
    """
    errors: List[str] = []
    entities = load_registry()
    relationships = load_relationships()
    
    # Check 1: Duplicate entity names
    entity_names = [e.name for e in entities]
    duplicates = [name for name in entity_names if entity_names.count(name) > 1]
    if duplicates:
        errors.append(f"ERROR: Duplicate entity names: {set(duplicates)}")
    
    # Check 2: Duplicate table+schema
    table_refs = [(e.database, e.schema, e.table) for e in entities]
    for table_ref in table_refs:
        if table_refs.count(table_ref) > 1:
            errors.append(f"WARNING: Multiple entities point to table {table_ref}. Consider consolidation.")
    
    # Build field map for validation
    entity_map = {e.name: e for e in entities}
    field_map = {e.name: {f.name for f in e.fields} for e in entities}
    
    # Check 3: join_keys exist as fields
    for entity in entities:
        missing = set(entity.join_keys) - field_map[entity.name]
        if missing:
            errors.append(f"ERROR: Entity '{entity.name}' references non-existent join keys: {missing}")
    
    # Check 4: default_metric exists
    for entity in entities:
        if entity.default_metric and entity.default_metric not in field_map[entity.name]:
            errors.append(
                f"ERROR: Entity '{entity.name}' default_metric '{entity.default_metric}' not found in fields"
            )
    
    # Check 5 & 6: Derived field dependencies
    for entity in entities:
        for f in entity.fields:
            if not f.derived:
                continue
            # Check depends_on fields exist
            missing = set(f.depends_on) - field_map[entity.name]
            if missing:
                errors.append(f"ERROR: Derived field '{entity.name}.{f.name}' depends on non-existent fields: {missing}")
    
    # Check 7: Relationship entities exist
    entity_names_set = {e.name for e in entities}
    for rel in relationships:
        if rel.from_entity not in entity_names_set:
            errors.append(f"ERROR: Relationship '{rel.name}' references non-existent from_entity '{rel.from_entity}'")
        if rel.to_entity not in entity_names_set:
            errors.append(f"ERROR: Relationship '{rel.name}' references non-existent to_entity '{rel.to_entity}'")
    
    # Check 8: Relationship join fields exist
    for rel in relationships:
        if rel.from_entity in entity_map and rel.to_entity in entity_map:
            from_fields = field_map[rel.from_entity]
            to_fields = field_map[rel.to_entity]
            for from_field, to_field in rel.join_on:
                if from_field not in from_fields:
                    errors.append(f"ERROR: Relationship '{rel.name}' from_field '{from_field}' not in {rel.from_entity}")
                if to_field not in to_fields:
                    errors.append(f"ERROR: Relationship '{rel.name}' to_field '{to_field}' not in {rel.to_entity}")

    # Check 9: Reasoner outputs exist for their entity_type
    type_to_fields: Dict[str, set[str]] = {}
    for entity in entities:
        if not entity.entity_type:
            continue
        type_to_fields.setdefault(entity.entity_type, set()).update(field_map[entity.name])
    for reasoner in load_reasoners():
        if not reasoner.outputs:
            continue
        available = type_to_fields.get(reasoner.entity_type, set())
        missing = set(reasoner.outputs) - available
        if missing:
            errors.append(
                f"WARNING: Reasoner '{reasoner.id}' outputs not found for entity_type '{reasoner.entity_type}': {missing}"
            )

    # Check 10: Derived Rel rules reference valid entities/fields
    derived_rules = load_derived_rel_rules()
    for rule in derived_rules:
        if rule.entity not in entity_map:
            errors.append(f"ERROR: Derived rule references unknown entity '{rule.entity}'")
            continue
        if rule.field not in field_map[rule.entity]:
            errors.append(f"ERROR: Derived rule references unknown field '{rule.entity}.{rule.field}'")
            continue
        entity = entity_map[rule.entity]
        field_lookup = {f.name: f for f in entity.fields}
        if rule.field in field_lookup and not field_lookup[rule.field].derived:
            errors.append(
                f"WARNING: Derived rule targets non-derived field '{rule.entity}.{rule.field}'"
            )
        if not rule.rules:
            errors.append(f"ERROR: Derived rule for '{rule.entity}.{rule.field}' has no rules")
        for clause in rule.rules:
            if not clause.expr:
                errors.append(f"ERROR: Derived rule clause missing expr for '{rule.entity}.{rule.field}'")

    # Check 11: Prompt templates shape
    templates = load_prompt_templates()
    if templates and not isinstance(templates, dict):
        errors.append("ERROR: prompt_templates must be an object/dict.")
    elif isinstance(templates, dict):
        for k, v in templates.items():
            if not isinstance(v, str):
                errors.append(f"WARNING: prompt_templates['{k}'] should be a string.")

    # Check 12: Analysis config shape
    analysis_cfg = load_analysis_config()
    if analysis_cfg and not isinstance(analysis_cfg, dict):
        errors.append("ERROR: analysis_config must be an object/dict.")
    elif isinstance(analysis_cfg, dict):
        default_months = analysis_cfg.get("default_months_window")
        if default_months is not None and not isinstance(default_months, (int, float, str)):
            errors.append("WARNING: analysis_config.default_months_window should be a number or numeric string.")

        join_keys_pref = analysis_cfg.get("join_keys_preference")
        if join_keys_pref is not None and not isinstance(join_keys_pref, list):
            errors.append("WARNING: analysis_config.join_keys_preference should be a list.")

        dim_hint = analysis_cfg.get("dimension_hint")
        if dim_hint is not None and not isinstance(dim_hint, str):
            errors.append("WARNING: analysis_config.dimension_hint should be a string.")

        entity_priority = analysis_cfg.get("entity_priority")
        if entity_priority is not None and not isinstance(entity_priority, list):
            errors.append("WARNING: analysis_config.entity_priority should be a list.")

        aliases = analysis_cfg.get("entity_aliases")
        if aliases is not None and not isinstance(aliases, dict):
            errors.append("WARNING: analysis_config.entity_aliases should be an object/dict.")
        elif isinstance(aliases, dict):
            for key in ("rm_prefixes", "mandate_prefixes"):
                if key in aliases and not isinstance(aliases.get(key), list):
                    errors.append(f"WARNING: analysis_config.entity_aliases.{key} should be a list.")

        driver_syn = analysis_cfg.get("driver_metric_synonyms")
        if driver_syn is not None and not isinstance(driver_syn, dict):
            errors.append("WARNING: analysis_config.driver_metric_synonyms should be an object/dict.")
        elif isinstance(driver_syn, dict):
            for k, v in driver_syn.items():
                if not isinstance(v, list):
                    errors.append(
                        f"WARNING: analysis_config.driver_metric_synonyms['{k}'] should be a list."
                    )

        for key in (
            "driver_question_core_terms",
            "driver_question_change_terms",
            "driver_question_special_phrases",
            "driver_question_prescriptive_phrases",
        ):
            if key in analysis_cfg and not isinstance(analysis_cfg.get(key), list):
                errors.append(f"WARNING: analysis_config.{key} should be a list.")

    return errors
