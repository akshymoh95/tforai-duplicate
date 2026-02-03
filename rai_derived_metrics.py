"""
Derived Metrics Computation Module

Computes derived metrics from base metrics:
- profit_margin = profit_amount / revenue_amount
- cost_to_revenue = cost / revenue
- revenue_per_cost = revenue / cost
- aum_per_cost = aum / cost
- month_date = date from year + month

Can be applied either:
1. As QB-level computed properties (via define rules)
2. Post-query as dataframe transformations
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from rai_semantic_registry import load_registry


def get_derived_field_definition(field_name: str) -> Optional[Dict[str, any]]:
    """Get the definition (expression, dependencies) for a derived field"""
    entities = {e.name: e for e in load_registry()}
    
    for entity in entities.values():
        for field in entity.fields:
            if field.name == field_name and field.derived:
                return {
                    "name": field.name,
                    "expr": field.expr,
                    "depends_on": field.depends_on,
                    "dtype": field.dtype,
                    "default_agg": field.default_agg,
                }
    return None


def get_all_derived_fields_for_entity(entity_name: str) -> List[Dict[str, any]]:
    """Get all derived field definitions for an entity"""
    entities = {e.name: e for e in load_registry()}
    entity = entities.get(entity_name)
    if not entity:
        return []
    
    return [
        {
            "name": field.name,
            "expr": field.expr,
            "depends_on": field.depends_on,
            "dtype": field.dtype,
            "default_agg": field.default_agg,
        }
        for field in entity.fields
        if field.derived
    ]


def get_applicable_derived_metrics_for_entity(entity_name: str) -> List[str]:
    """
    Get the list of derived metrics applicable to this entity.
    
    Derived from registry's EntitySpec.applicable_derived_metrics
    This allows clients to declare:
    EntitySpec(..., applicable_derived_metrics=["profit_margin", "cost_to_revenue"])
    
    Then this function returns those metric names automatically.
    """
    entities = {e.name: e for e in load_registry()}
    entity = entities.get(entity_name)
    if not entity:
        return []
    
    return list(entity.applicable_derived_metrics or [])


def add_applicable_derived_metrics_to_spec(
    spec: Dict[str, any],
    entity_name: str,
) -> None:
    """
    DEPRECATED: Don't manually add derived metric columns to spec.
    
    Derived metrics should be computed by the reasoner rules or post-processing.
    This function is a no-op now to avoid type resolution errors in RAI compiler.
    """
    pass


def _safe_divide(numerator, denominator, fill_value=None):
    """Safely divide, handling division by zero and type mismatches (Decimal/float)"""
    from decimal import Decimal
    
    # Convert to float to handle Decimal types from Snowflake
    try:
        numerator = pd.to_numeric(numerator, errors='coerce')
        denominator = pd.to_numeric(denominator, errors='coerce')
    except Exception:
        pass
    
    # Use numpy where with explicit type handling
    numerator_vals = np.asarray(numerator, dtype=float)
    denominator_vals = np.asarray(denominator, dtype=float)
    
    # Suppress divide by zero warning - we handle it with np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            denominator_vals == 0,
            fill_value,
            numerator_vals / denominator_vals
        )
    return result


def _compute_profit_margin(df: pd.DataFrame) -> pd.Series:
    """Compute profit_margin = profit_amount / revenue_amount"""
    if "profit_amount" not in df.columns or "revenue_amount" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    profit = pd.to_numeric(df["profit_amount"], errors='coerce').fillna(0)
    revenue = pd.to_numeric(df["revenue_amount"], errors='coerce').fillna(0)
    return _safe_divide(profit, revenue, np.nan)


def _compute_cost_to_revenue(df: pd.DataFrame) -> pd.Series:
    """Compute cost_to_revenue = rm_cost_for_each_mandate / revenue_amount"""
    if "rm_cost_for_each_mandate" not in df.columns or "revenue_amount" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    cost = pd.to_numeric(df["rm_cost_for_each_mandate"], errors='coerce').fillna(0)
    revenue = pd.to_numeric(df["revenue_amount"], errors='coerce').fillna(0)
    return _safe_divide(cost, revenue, np.nan)


def _compute_revenue_per_cost(df: pd.DataFrame) -> pd.Series:
    """Compute revenue_per_cost = revenue_amount / rm_cost_for_each_mandate"""
    if "revenue_amount" not in df.columns or "rm_cost_for_each_mandate" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    revenue = pd.to_numeric(df["revenue_amount"], errors='coerce').fillna(0)
    cost = pd.to_numeric(df["rm_cost_for_each_mandate"], errors='coerce').fillna(0)
    return _safe_divide(revenue, cost, np.nan)


def _compute_aum_per_cost(df: pd.DataFrame) -> pd.Series:
    """Compute aum_per_cost = total_aum / rm_cost_for_each_mandate"""
    if "total_aum" not in df.columns or "rm_cost_for_each_mandate" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    aum = pd.to_numeric(df["total_aum"], errors='coerce').fillna(0)
    cost = pd.to_numeric(df["rm_cost_for_each_mandate"], errors='coerce').fillna(0)
    return _safe_divide(aum, cost, np.nan)


def _compute_month_date(df: pd.DataFrame) -> pd.Series:
    """Compute month_date from posyear and posmon"""
    if "posyear" not in df.columns or "posmon" not in df.columns:
        return pd.Series(index=df.index, dtype='datetime64[ns]')
    
    try:
        return pd.to_datetime(
            df["posyear"].astype(str) + "-" + df["posmon"].astype(str).str.zfill(2) + "-01",
            format="%Y-%m-%d",
            errors="coerce"
        )
    except Exception:
        return pd.Series(index=df.index, dtype='datetime64[ns]')


def _compute_aum_trend(df: pd.DataFrame) -> pd.Series:
    """
    Compute aum_trend = (current_aum - previous_aum) / previous_aum
    Requires data sorted by date with mandate grouping
    """
    if "total_aum" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    # Try to compute period-over-period change
    # If there's a month_date or time dimension, group by it
    if "mandateid" in df.columns and ("posmon" in df.columns or "month_date" in df.columns):
        # Sort by mandate and date
        sort_cols = ["mandateid"]
        if "posyear" in df.columns and "posmon" in df.columns:
            sort_cols.extend(["posyear", "posmon"])
        
        try:
            sorted_df = df.sort_values(sort_cols)
            trend = sorted_df.groupby("mandateid")["total_aum"].pct_change()
            return trend.reset_index(drop=True)
        except Exception:
            return pd.Series(index=df.index, dtype=float)
    
    return pd.Series(index=df.index, dtype=float)


def _compute_revenue_trend(df: pd.DataFrame) -> pd.Series:
    """
    Compute revenue_trend = (current_revenue - previous_revenue) / previous_revenue
    """
    if "revenue_amount" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    
    if "mandateid" in df.columns and ("posmon" in df.columns or "month_date" in df.columns):
        sort_cols = ["mandateid"]
        if "posyear" in df.columns and "posmon" in df.columns:
            sort_cols.extend(["posyear", "posmon"])
        
        try:
            sorted_df = df.sort_values(sort_cols)
            trend = sorted_df.groupby("mandateid")["revenue_amount"].pct_change()
            return trend.reset_index(drop=True)
        except Exception:
            return pd.Series(index=df.index, dtype=float)
    
    return pd.Series(index=df.index, dtype=float)


# Registry of compute functions
_COMPUTE_FUNCTIONS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "profit_margin": _compute_profit_margin,
    "cost_to_revenue": _compute_cost_to_revenue,
    "revenue_per_cost": _compute_revenue_per_cost,
    "aum_per_cost": _compute_aum_per_cost,
    "month_date": _compute_month_date,
    "aum_trend": _compute_aum_trend,
    "revenue_trend": _compute_revenue_trend,
}


def compute_derived_metric(df: pd.DataFrame, metric_name: str) -> Optional[pd.Series]:
    """
    Compute a single derived metric for a dataframe.
    
    Args:
        df: DataFrame with base metrics
        metric_name: Name of the derived metric to compute
    
    Returns:
        pd.Series with computed values, or None if not available
    """
    compute_fn = _COMPUTE_FUNCTIONS.get(metric_name)
    if compute_fn is None:
        return None
    
    try:
        return compute_fn(df)
    except Exception as e:
        print(f"[WARN] Failed to compute {metric_name}: {e}")
        return None


def compute_all_derived_metrics(
    df: pd.DataFrame,
    include_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute all available derived metrics and add them to the dataframe.
    
    Args:
        df: DataFrame with base metrics
        include_metrics: If provided, only compute these metrics
    
    Returns:
        DataFrame with new columns for derived metrics
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    metrics_to_compute = include_metrics or list(_COMPUTE_FUNCTIONS.keys())
    
    for metric_name in metrics_to_compute:
        if metric_name in result.columns:
            continue  # Already present
        
        computed = compute_derived_metric(result, metric_name)
        if computed is not None:
            result[metric_name] = computed
    
    return result


def enrich_dataframe_with_derived_metrics(
    df: pd.DataFrame,
    entity_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Intelligently add derived metrics based on available columns.
    
    If entity_name is provided, only compute derived fields defined for that entity.
    Otherwise, compute all available metrics that have their dependencies met.
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if entity_name:
        # Only compute derived fields for this entity
        derived_defs = get_all_derived_fields_for_entity(entity_name)
        metrics = [d["name"] for d in derived_defs]
    else:
        metrics = list(_COMPUTE_FUNCTIONS.keys())
    
    for metric_name in metrics:
        if metric_name in result.columns:
            continue
        
        # Check if dependencies are available
        defn = get_derived_field_definition(metric_name)
        if defn and defn.get("depends_on"):
            deps = defn["depends_on"]
            if not all(dep in result.columns for dep in deps):
                continue  # Skip if dependencies not available
        
        computed = compute_derived_metric(result, metric_name)
        if computed is not None:
            result[metric_name] = computed
    
    return result


def get_derived_field_sql_expr(field_name: str) -> Optional[str]:
    """Get the SQL expression for a derived field"""
    defn = get_derived_field_definition(field_name)
    return defn["expr"] if defn else None
