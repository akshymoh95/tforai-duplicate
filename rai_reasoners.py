"""
RAI-Native Reasoner Stub (DEPRECATED)

NOTE: All reasoning logic has been migrated to RAI semantic layer.
This stub file remains for backward compatibility and documentation only.

MIGRATION STATUS:
✅ Trend computation: Now in SQL window functions (aum_trend, revenue_trend, etc)
✅ Calculated metrics: Now in SQL (profit_margin, cost_to_revenue, etc)
✅ Signal detection: Now in SQL boolean columns (signal_aum_decline, etc)
✅ Risk scoring: Now in SQL computed fields (mandate_risk_score, churn_risk_score, etc)
✅ Condition classification: Now in SQL (mandate_risk_condition, churn_risk_condition, etc)

ORIGINAL PYTHON REASONERS (NOW DEPRECATED):
- MandateRiskReasoner → mandate_risk_score, mandate_risk_condition columns
- RMPerformanceReasoner → rm_performance_condition column
- TrendReasoner → aum_trend, revenue_trend, profit_trend columns
- ChurnRiskReasoner → churn_risk_score, churn_risk_condition columns
- CostDriverReasoner → cost_driver_impact_score, cost_driver_condition columns
- AllocationReasoner → allocation_priority, allocation_composite_score columns

All reasoner output columns are now available directly in query results from RAI.
The LLM receives pre-computed reasoning without expensive Python post-processing.

See RAI_NATIVE_REASONER_MIGRATION_PLAN.md for complete migration details.
See rai_mandate_profitability_semantic_model.yaml for SQL reasoner definitions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rai_semantic_registry import load_reasoners, load_registry

@dataclass(frozen=True)
class RiskSignal:
    """A single risk indicator (DEPRECATED - now in SQL)"""
    name: str
    description: str
    metric_field: str
    threshold: float
    direction: str  # "below" or "above"
    weight: float = 1.0


@dataclass(frozen=True)
class ReasonerResult:
    """Output from a reasoner (DEPRECATED - now returned as columns)"""
    entity_id: str
    condition: str  # e.g., "at_risk", "healthy", "declining"
    risk_score: float  # 0.0 to 1.0
    signals_triggered: List[str]
    factors: Dict[str, Any]
    recommendations: List[str]
    confidence: float


class BaseReasoner:
    """
    DEPRECATED: Reasoner logic moved to RAI semantic layer.
    
    This class remains for:
    - Documentation of reasoner specs
    - Backward compatibility with any client code
    - Future extensibility
    
    Actual reasoning computation happens in YAML semantic models
    as SQL computed columns. No Python post-processing.
    
    Migration: See RAI_NATIVE_REASONER_MIGRATION_PLAN.md
    """
    
    def __init__(self, name: str = "", description: str = "", signals: List[RiskSignal] = None):
        self.name = name
        self.description = description
        self.signals = signals or []
    
    @staticmethod
    def from_spec(spec: Any) -> BaseReasoner:
        """Factory method: Create reasoner from registry specification."""
        return BaseReasoner(name=spec.name, description=spec.description)
    
    def get_required_metrics(self) -> List[str]:
        """Return list of metric fields this reasoner needs."""
        # These are now pre-computed in RAI and returned in query results
        return []
    
    def get_dimension_fields(self) -> List[str]:
        """Return list of dimension fields this reasoner needs for grouping."""
        # Return common dimension fields (mandate_id, rm_id, etc)
        # These are used for group_by in queries
        return []
    
    def analyze(self, df: Any = None, context: Dict[str, Any] = None) -> List[ReasonerResult]:
        """DEPRECATED: Analysis now happens in RAI SQL layer, not Python."""
        return []
    
    def get_prompt_context(self) -> str:
        """DEPRECATED: Reasoning data now in DataFrame columns."""
        return ""


# Stub implementations - all logic moved to RAI
class MandateRiskReasoner(BaseReasoner):
    """DEPRECATED - Use mandate_risk_score and mandate_risk_condition columns from RAI."""
    def __init__(self):
        super().__init__(
            name="Mandate Risk Reasoner (DEPRECATED)",
            description="Use mandate_risk_score and mandate_risk_condition columns from RAI semantic layer"
        )


class RMPerformanceReasoner(BaseReasoner):
    """DEPRECATED - Use rm_performance_condition column from RAI."""
    def __init__(self):
        super().__init__(
            name="RM Performance Reasoner (DEPRECATED)",
            description="Use rm_performance_condition column from RAI semantic layer"
        )


class TrendReasoner(BaseReasoner):
    """DEPRECATED - Use aum_trend, revenue_trend, profit_trend columns from RAI."""
    def __init__(self):
        super().__init__(
            name="Trend Reasoner (DEPRECATED)",
            description="Use aum_trend, revenue_trend, profit_trend computed columns from RAI semantic layer"
        )


class ChurnRiskReasoner(BaseReasoner):
    """DEPRECATED - Use churn_risk_score and churn_risk_condition columns from RAI."""
    def __init__(self):
        super().__init__(
            name="Churn Risk Reasoner (DEPRECATED)",
            description="Use churn_risk_score and churn_risk_condition columns from RAI semantic layer"
        )


class CostDriverReasoner(BaseReasoner):
    """DEPRECATED - Use cost_driver_impact_score and cost_driver_condition columns from RAI."""
    def __init__(self):
        super().__init__(
            name="Cost Driver Reasoner (DEPRECATED)",
            description="Use cost_driver_impact_score and cost_driver_condition columns from RAI semantic layer"
        )


class AllocationReasoner(BaseReasoner):
    """DEPRECATED - Use allocation_priority and allocation_composite_score columns from RAI."""
    def __init__(self):
        super().__init__(
            name="Allocation Reasoner (DEPRECATED)",
            description="Use allocation_priority and allocation_composite_score columns from RAI semantic layer"
        )


# Minimal registry for backward compatibility
_REASONER_REGISTRY: Dict[str, BaseReasoner] = {}


def _initialize_reasoner_registry():
    """Stub: Registry no longer needed - reasoners defined in RAI semantic layer."""
    global _REASONER_REGISTRY
    _REASONER_REGISTRY = {
        "mandate_risk": MandateRiskReasoner(),
        "rm_performance": RMPerformanceReasoner(),
        "trend": TrendReasoner(),
        "churn_risk": ChurnRiskReasoner(),
        "cost_driver": CostDriverReasoner(),
        "allocation": AllocationReasoner(),
    }


def register_reasoner(reasoner_id: str, reasoner: BaseReasoner) -> None:
    """Stub: Reasoners now defined in RAI, not registered here."""
    pass


def get_reasoner(reasoner_id: str) -> Optional[BaseReasoner]:
    """Stub: Reasoners defined in RAI semantic layer."""
    if not _REASONER_REGISTRY:
        _initialize_reasoner_registry()
    return _REASONER_REGISTRY.get(reasoner_id)


def list_reasoners() -> Dict[str, str]:
    """List all available reasoners with descriptions."""
    if not _REASONER_REGISTRY:
        _initialize_reasoner_registry()
    return {
        rid: r.description
        for rid, r in _REASONER_REGISTRY.items()
    }


def get_reasoners_for_entity(entity_name: str) -> List[str]:
    """
    DEPRECATED: Discover which reasoners apply to an entity.
    
    NOW: All reasoners always apply - they're in the semantic layer.
    This function returns all reasoner IDs for backward compatibility.
    
    Args:
        entity_name: Entity name (e.g., "MandateMonthlySummary")
    
    Returns:
        List of reasoner IDs that apply to this entity
    """
    # Prefer registry-defined applicable_reasoners if present.
    try:
        entities = load_registry()
        for entity in entities:
            if entity.name == entity_name:
                if entity.applicable_reasoners:
                    return list(entity.applicable_reasoners)
                break
    except Exception:
        pass

    if not _REASONER_REGISTRY:
        _initialize_reasoner_registry()
    return list(_REASONER_REGISTRY.keys())


def format_rai_reasoner_columns_for_llm(df, reasoner_id: str) -> str:
    """
    DEPRECATED: Format reasoner results for LLM.
    
    NOW: Reasoning data already in DataFrame columns, no formatting needed.
    
    This function maintains backward compatibility by returning a summary
    of the relevant columns for the given reasoner.
    
    Args:
        df: DataFrame with reasoner columns
        reasoner_id: Which reasoner's columns to summarize
    
    Returns:
        String summary for LLM context (for backward compatibility)
    """
    if df is None or df.empty:
        return ""
    
    try:
        import pandas as pd
    except ImportError:
        return ""
    
    # Map reasoner IDs to their column names (legacy)
    reasoner_columns = {
        "mandate_risk": ["mandate_risk_score", "mandate_risk_condition", "signal_aum_decline", "signal_revenue_decline"],
        "churn_risk": ["churn_risk_score", "churn_risk_condition", "signal_churn_risk"],
        "trend": ["aum_trend", "revenue_trend", "profit_trend"],
        "cost_driver": ["cost_driver_impact_score", "cost_driver_condition", "signal_high_cost"],
        "allocation": ["allocation_priority", "allocation_composite_score"],
        "rm_performance": ["rm_performance_condition"],
    }

    cols = reasoner_columns.get(reasoner_id, [])
    if not cols:
        try:
            reasoners = load_reasoners()
            reasoner_map = {r.id: list(r.outputs or []) for r in reasoners if r.id}
            cols = reasoner_map.get(reasoner_id, [])
        except Exception:
            cols = []
    relevant_cols = [c for c in cols if c in df.columns]
    
    if not relevant_cols:
        return ""
    
    # Just return the column names for documentation
    # The actual data is already in the DataFrame
    return f"[RAI-NATIVE] {reasoner_id}: {', '.join(relevant_cols)}"


def apply_reasoner_to_query_spec(
    spec: Dict[str, Any],
    reasoner_id: str,
    builder: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Enhance query spec to include reasoner output columns.
    
    Maps reasoner ID to the list of computed columns it produces,
    then adds those columns to the select list so they're returned.
    
    For aggregated queries, avoid grouping by metric-like reasoner fields.
    Instead, add reasonable aggregations (avg/max) where possible.
    
    Args:
        spec: Dynamic query spec (from dynamic query engine)
        reasoner_id: Which reasoner's columns to add
        builder: Optional KGBuilder (ignored in RAI-native mode)
    
    Returns:
        Enhanced spec with reasoner columns added to select list and group_by
    """
    if not spec or not reasoner_id:
        return spec

    def _dict_list(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [v for v in value if isinstance(v, dict)]
    
    reasoner_map = {r.id: r for r in load_reasoners()}
    reasoner = reasoner_map.get(reasoner_id)
    cols_to_add = list(reasoner.outputs or []) if reasoner else []
    if not cols_to_add:
        return spec
    
    # Get the alias from the bind (usually "m" for mandate)
    alias = None
    if spec.get("bind"):
        alias = spec["bind"][0].get("alias", "m")
    
    if not alias:
        return spec
    
    entity_name = None
    if spec.get("bind"):
        for b in spec["bind"]:
            if b.get("alias") == alias:
                entity_name = b.get("entity")
                break
        if not entity_name:
            entity_name = spec["bind"][0].get("entity")

    entity_map = {e.name: e for e in load_registry()}
    field_map = {
        f.name: f
        for f in (entity_map.get(entity_name).fields if entity_name in entity_map else [])
    }

    # Only inject reasoner outputs that actually exist on the bound entity.
    cols_to_add = [col for col in cols_to_add if col in field_map]
    if not cols_to_add:
        return spec

    def _field_role(col: str) -> str:
        field = field_map.get(col)
        if field:
            return field.role
        col = (col or "").lower()
        if col.startswith("signal_") or col.endswith("_score") or col.endswith("_trend"):
            return "metric"
        if col.endswith("_condition") or col.endswith("_priority"):
            return "dimension"
        return "metric"

    def _field_default_agg(col: str) -> str:
        field = field_map.get(col)
        if field and field.default_agg:
            return field.default_agg
        col = (col or "").lower()
        if col.startswith("signal_"):
            return "max"
        if col.endswith("_score") or col.endswith("_trend"):
            return "avg"
        return "avg"

    original_select = list(spec.get("select") or [])
    spec["select"] = _dict_list(spec.get("select"))
    
    for col in cols_to_add:
        # Check if already in select
        already_selected = any(
            s.get("prop") == col and s.get("alias") == alias
            for s in spec["select"]
        )
        
        if not already_selected:
            spec["select"].append({
                "alias": alias,
                "prop": col,
                "as": col,
            })
    
    if spec.get("aggregations"):
        spec["group_by"] = _dict_list(spec.get("group_by"))
        spec["aggregations"] = _dict_list(spec.get("aggregations"))

        aggregations = list(spec.get("aggregations") or [])
        agg_terms = {
            (a.get("term", {}).get("alias"), a.get("term", {}).get("prop"))
            for a in aggregations
            if isinstance(a, dict) and isinstance(a.get("term"), dict)
        }

        # Remove metric-like reasoner fields from group_by to avoid shadowed variables.
        spec["group_by"] = [
            g for g in spec["group_by"]
            if not (
                isinstance(g, dict)
                and g.get("alias") == alias
                and _field_role(g.get("prop")) == "metric"
            )
        ]

        # Add aggregations for metric-like reasoner fields when missing.
        for col in cols_to_add:
            if _field_role(col) != "metric":
                continue
            key = (alias, col)
            if key in agg_terms:
                continue
            op = _field_default_agg(col)
            if op not in ("sum", "avg", "min", "max"):
                op = "max" if col.lower().startswith("signal_") else "avg"
            aggregations.append({
                "op": op,
                "term": {"alias": alias, "prop": col},
                "as": f"{op}_{col}",
            })
            agg_terms.add(key)

        spec["aggregations"] = aggregations

        # Drop dimension-like reasoner fields from select for aggregated queries
        # unless they were explicitly included in the original spec.
        original_props = {(s.get("alias"), s.get("prop")) for s in original_select if isinstance(s, dict)}
        spec["select"] = [
            s for s in spec["select"]
            if not (
                isinstance(s, dict)
                and s.get("alias") == alias
                and _field_role(s.get("prop")) == "dimension"
                and (s.get("alias"), s.get("prop")) not in original_props
            )
        ]

    return spec
