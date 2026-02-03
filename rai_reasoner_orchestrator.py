"""
Reasoner orchestrator for RAI AI-Insights.

Key design:
- Base dynamic spec should stay SAFE (no fanout joins across event tables).
- "Multi-hop" explanation questions are answered by running additional drilldown
  queries (small limits, no aggregations) after the base dataframe exists.

This module therefore supports:
1) Optional spec enhancement for "single-query" reasoner columns (legacy path)
2) True multi-hop orchestration for reasoners that require evidence drilldowns:
   - seed (base df) -> worst entities -> worst runs -> events within run windows -> audits
"""

from __future__ import annotations

import json
import os
import re
import sys
import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rai_dynamic_query import run_dynamic_query
from rai_derived_metrics import enrich_dataframe_with_derived_metrics
from rai_reasoners import (
    apply_reasoner_to_query_spec,
    format_rai_reasoner_columns_for_llm,
    get_reasoner,
    get_reasoners_for_entity,
    list_reasoners,
)
from relationalai.semantics.rel.rel_utils import sanitize_identifier
from reasoner_selection_intelligence import infer_reasoners_from_question


# -----------------------------
# Registry loading (optional)
# -----------------------------

_REGISTRY_CACHE: Optional[Dict[str, Any]] = None
_DEBUG_REASONERS = os.environ.get("AI_INSIGHTS_DEBUG_REASONERS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def _dbg(msg: str) -> None:
    if _DEBUG_REASONERS:
        print(f"[DEBUG][reasoners] {msg}", file=sys.stderr)


def _load_semantic_registry() -> Optional[Dict[str, Any]]:
    """
    Best-effort local registry load. This is optional.
    If not found, orchestration still works using built-in heuristics.
    """
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE

    candidates = [
        os.environ.get("RAI_SEMANTIC_REGISTRY_PATH", "").strip(),
        "semantic_registry.json",
        os.path.join(os.getcwd(), "semantic_registry.json"),
        os.path.join(os.getcwd(), "registry", "semantic_registry.json"),
    ]
    candidates = [c for c in candidates if c]

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _REGISTRY_CACHE = json.load(f)
                return _REGISTRY_CACHE
        except Exception:
            continue

    _REGISTRY_CACHE = None
    return None


def _registry_reasoner_outputs(reasoner_id: str) -> List[str]:
    """Get outputs for a reasoner, handling both dict and object types."""
    reg = _load_semantic_registry()
    if not reg:
        return []
    for r in reg.get("reasoners", []) or []:
        # Handle both dict and dataclass object
        rid = r.get("id") if isinstance(r, dict) else getattr(r, "id", None)
        if rid == reasoner_id:
            outputs = r.get("outputs") if isinstance(r, dict) else getattr(r, "outputs", [])
            return list(outputs or [])
    return []


def _registry_reasoner_plan(reasoner_id: str) -> Dict[str, Any]:
    """Get drilldown plan for a reasoner, handling both dict and object types."""
    reg = _load_semantic_registry()
    if not reg:
        return {}
    for r in reg.get("reasoners", []) or []:
        # Handle both dict and dataclass object
        rid = r.get("id") if isinstance(r, dict) else getattr(r, "id", None)
        if rid == reasoner_id:
            plan = r.get("drilldown_plan") if isinstance(r, dict) else getattr(r, "drilldown_plan", None)
            return dict(plan or {})
    return {}


def _registry_reasoner_type(reasoner_id: str) -> str:
    """Get type for a reasoner, handling both dict and object types."""
    reg = _load_semantic_registry()
    if not reg:
        return ""
    for r in reg.get("reasoners", []) or []:
        # Handle both dict and dataclass object
        rid = r.get("id") if isinstance(r, dict) else getattr(r, "id", None)
        if rid == reasoner_id:
            rtype = r.get("type") if isinstance(r, dict) else getattr(r, "type", None)
            return str(rtype or "")
    return ""


def _is_graph_reasoner(reasoner_id: str) -> bool:
    r_type = _registry_reasoner_type(reasoner_id).strip().lower()
    return r_type == "graph_reasoner"


def _reasoner_step_concept_name(reasoner_id: str, step_id: str) -> str:
    base = f"Reasoner_{reasoner_id}_{step_id}"
    return sanitize_identifier(base)


def _build_alias_field_map(spec: Dict[str, Any]) -> Dict[str, List[str]]:
    binds = spec.get("bind") or []
    alias_to_fields: Dict[str, List[str]] = {}
    reg = _load_semantic_registry()
    if not reg:
        return alias_to_fields
    entities = {e.get("name"): e for e in reg.get("entities", []) if isinstance(e, dict)}
    for b in binds:
        if not isinstance(b, dict):
            continue
        alias = b.get("alias")
        ent_name = b.get("entity")
        if not alias or not ent_name:
            continue
        ent = entities.get(ent_name)
        if not ent:
            continue
        fields = [f.get("name") for f in (ent.get("fields") or []) if isinstance(f, dict) and f.get("name")]
        alias_to_fields[alias] = fields
    return alias_to_fields


def _find_alias_for_field(alias_to_fields: Dict[str, List[str]], field_name: str) -> Optional[str]:
    if not field_name:
        return None
    for alias, fields in alias_to_fields.items():
        if field_name in fields:
            return alias
    return None


def _build_select_as_map(spec: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for term in spec.get("select") or []:
        if not isinstance(term, dict):
            continue
        alias = term.get("alias")
        prop = term.get("prop")
        as_name = term.get("as")
        if alias and prop and as_name:
            out[as_name] = (alias, prop)
    return out


def inject_reasoner_relations_into_spec(
    spec: Dict[str, Any],
    reasoner_ids: List[str],
) -> Dict[str, Any]:
    if not isinstance(spec, dict) or not reasoner_ids:
        return spec
    reasoner_ids = [r for r in reasoner_ids if not _is_graph_reasoner(r)]
    if not reasoner_ids:
        return spec

    alias_to_fields = _build_alias_field_map(spec)
    select_as_map = _build_select_as_map(spec)
    if not alias_to_fields:
        return spec

    window_start, window_end = _extract_seed_window(spec)
    spec = json.loads(json.dumps(spec, default=str))
    binds = spec.get("bind") or []
    wh = spec.get("where") or []
    if not isinstance(wh, list):
        wh = [wh]
    select = spec.get("select") or []
    if not isinstance(select, list):
        select = []

    for reasoner_id in reasoner_ids:
        plan = _registry_reasoner_plan(reasoner_id)
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            if (step.get("from") or "base") != "base":
                continue
            step_id = str(step.get("id") or "").strip()
            if not step_id:
                continue
            concept_name = _reasoner_step_concept_name(reasoner_id, step_id)
            rel_alias = sanitize_identifier(f"r_{reasoner_id}_{step_id}")
            binds.append({"alias": rel_alias, "entity": concept_name})

            inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
            for key, source in (inputs or {}).items():
                prop = _normalize_prop_name(key)
                if isinstance(source, str):
                    if source.startswith("row."):
                        field = source[4:]
                        base_alias = _find_alias_for_field(alias_to_fields, field)
                        base_prop = field
                        if not base_alias:
                            mapped = select_as_map.get(field)
                            if mapped:
                                base_alias, base_prop = mapped
                        if base_alias:
                            wh.append(
                                {
                                    "op": "==",
                                    "left": {"alias": base_alias, "prop": base_prop},
                                    "right": {"alias": rel_alias, "prop": prop},
                                }
                            )
                    elif source.startswith("context."):
                        ctx_key = source[8:]
                        if ctx_key == "window_start" and window_start:
                            ws = _coerce_value_for_dtype(window_start, "DateTime")
                            wh.append(
                                {
                                    "op": "==",
                                    "left": {"alias": rel_alias, "prop": prop},
                                    "right": {"value": ws},
                                }
                            )
                        elif ctx_key == "window_end" and window_end:
                            we = _coerce_value_for_dtype(window_end, "DateTime")
                            wh.append(
                                {
                                    "op": "==",
                                    "left": {"alias": rel_alias, "prop": prop},
                                    "right": {"value": we},
                                }
                            )
                    elif source.startswith("literal:"):
                        wh.append(
                            {
                                "op": "==",
                                "left": {"alias": rel_alias, "prop": prop},
                                "right": {"value": source[len("literal:") :]},
                            }
                        )

            q = step.get("query") if isinstance(step.get("query"), dict) else {}
            for term in q.get("select") or []:
                if not isinstance(term, dict):
                    continue
                prop = term.get("as") or term.get("prop")
                if not prop:
                    continue
                as_name = sanitize_identifier(f"{reasoner_id}_{step_id}_{prop}")
                select.append({"alias": rel_alias, "prop": _normalize_prop_name(prop), "as": as_name})

    spec["bind"] = binds
    spec["where"] = wh
    spec["select"] = select
    return spec


def seed_spec_for_reasoners(spec: Dict[str, Any], reasoner_ids: List[str]) -> Dict[str, Any]:
    if not isinstance(spec, dict) or not reasoner_ids:
        return spec
    reasoner_ids = [r for r in reasoner_ids if not _is_graph_reasoner(r)]
    if not reasoner_ids:
        return spec

    # If already non-aggregated, keep as-is.
    if not spec.get("aggregations") and not spec.get("group_by"):
        return spec

    alias_to_fields = _build_alias_field_map(spec)
    select_as_map = _build_select_as_map(spec)

    seed = json.loads(json.dumps(spec, default=str))
    seed.pop("aggregations", None)
    seed.pop("group_by", None)
    seed.pop("order_by", None)

    selects = seed.get("select") or []
    if not isinstance(selects, list):
        selects = []

    existing = {(t.get("alias"), t.get("prop")) for t in selects if isinstance(t, dict)}

    # Ensure required row.<field> inputs are selected
    needed_fields: List[Tuple[str, str]] = []
    for reasoner_id in reasoner_ids:
        plan = _registry_reasoner_plan(reasoner_id)
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            if (step.get("from") or "base") != "base":
                continue
            inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
            for _key, source in (inputs or {}).items():
                if isinstance(source, str) and source.startswith("row."):
                    field = source[4:]
                    base_alias = _find_alias_for_field(alias_to_fields, field)
                    base_prop = field
                    if not base_alias:
                        mapped = select_as_map.get(field)
                        if mapped:
                            base_alias, base_prop = mapped
                    if base_alias and (base_alias, base_prop) not in existing:
                        needed_fields.append((base_alias, base_prop))
                        existing.add((base_alias, base_prop))

    for alias, prop in needed_fields:
        selects.append({"alias": alias, "prop": prop, "as": prop})

    seed["select"] = selects
    return seed


def _reasoner_step_concept_name(reasoner_id: str, step_id: str) -> str:
    return sanitize_identifier(f"Reasoner_{reasoner_id}_{step_id}")


def _normalize_prop_name(name: str) -> str:
    return sanitize_identifier(str(name or "").lower())


# -----------------------------
# Helpers: detect time window in spec
# -----------------------------

def _find_where_range(spec: Dict[str, Any], alias: str, prop: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a simple >= and <= range from spec['where'] for alias.prop.
    Returns (low, high) timestamp strings if present.
    Handles nested AND/OR trees.
    """
    low = None
    high = None

    def _walk(node: Any) -> None:
        nonlocal low, high
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return
        if "and" in node:
            _walk(node.get("and"))
            return
        if "or" in node:
            _walk(node.get("or"))
            return
        if "op" not in node:
            return
        op = node.get("op")
        left = node.get("left") or {}
        right = node.get("right") or {}
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return
        if left.get("alias") != alias or left.get("prop") != prop:
            return
        v = right.get("value")
        if not isinstance(v, str):
            return
        if op == ">=":
            low = v
        elif op == "<=":
            high = v

    _walk(spec.get("where") or [])
    return low, high


def _extract_seed_window(spec: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Best-effort window extraction. Prefers operation_start constraints on any alias.
    """
    meta_window = (spec.get("meta") or {}).get("time_window") or {}
    if isinstance(meta_window, dict):
        start = meta_window.get("start")
        end = meta_window.get("end")
        if isinstance(start, str) or isinstance(end, str):
            return (start if isinstance(start, str) else None, end if isinstance(end, str) else None)
    binds = spec.get("bind") or []
    if isinstance(binds, list):
        for b in binds:
            if isinstance(b, dict) and b.get("alias"):
                a = b["alias"]
                lo, hi = _find_where_range(spec, a, "operation_start")
                if lo or hi:
                    return lo, hi
    return None, None


# -----------------------------
# Safe drilldown query builders
# -----------------------------

def _drill_spec_worst_ops(
    pu_id: int,
    window_start: Optional[str],
    window_end: Optional[str],
    limit: int = 3,
) -> Dict[str, Any]:
    where = [
        {"op": "==", "left": {"alias": "om", "prop": "pu_id"}, "right": {"value": pu_id}},
    ]
    if window_start:
        where.append({"op": ">=", "left": {"alias": "om", "prop": "operation_start"}, "right": {"value": window_start}})
    if window_end:
        where.append({"op": "<=", "left": {"alias": "om", "prop": "operation_start"}, "right": {"value": window_end}})

    return {
        "bind": [{"alias": "om", "entity": "dt_operation_metrics"}],
        "where": where,
        "select": [
            {"alias": "om", "prop": "start_id", "as": "start_id"},
            {"alias": "om", "prop": "prod_id", "as": "prod_id"},
            {"alias": "om", "prop": "operation_start", "as": "operation_start"},
            {"alias": "om", "prop": "operation_end", "as": "operation_end"},
            {"alias": "om", "prop": "oee_pct", "as": "oee_pct"},
            {"alias": "om", "prop": "availability_pct", "as": "availability_pct"},
            {"alias": "om", "prop": "performance_pct", "as": "performance_pct"},
            {"alias": "om", "prop": "quality_rate_pct", "as": "quality_rate_pct"},
            {"alias": "om", "prop": "dt_minutes", "as": "dt_minutes"},
            {"alias": "om", "prop": "waste_excl_setup_conv", "as": "waste_excl_setup_conv"},
            {"alias": "om", "prop": "target_speed", "as": "target_speed"},
            {"alias": "om", "prop": "rateunit_value", "as": "rateunit_value"},
            {"alias": "om", "prop": "ideal_produce", "as": "ideal_produce"},
            {"alias": "om", "prop": "actual_produce_exec", "as": "actual_produce_exec"},
            {"alias": "om", "prop": "actual_produce_adjusted", "as": "actual_produce_adjusted"},
            {"alias": "om", "prop": "gross_produce", "as": "gross_produce"},
            {"alias": "om", "prop": "net_produce", "as": "net_produce"},
            {"alias": "om", "prop": "waste_cf", "as": "waste_cf"},
        ],
        "order_by": [{"term": {"alias": "om", "prop": "oee_pct", "as": "oee_pct"}, "dir": "asc"}],
        "limit": int(limit),
    }


def _drill_spec_timed_events(
    pu_id: int,
    run_start: str,
    run_end: str,
    limit: int = 10,
) -> Dict[str, Any]:
    return {
        "bind": [{"alias": "te", "entity": "dt_timed_event_denorm"}],
        "where": [
            {"op": "==", "left": {"alias": "te", "prop": "pu_id"}, "right": {"value": pu_id}},
            {"op": ">=", "left": {"alias": "te", "prop": "start_time"}, "right": {"value": run_start}},
            {"op": "<=", "left": {"alias": "te", "prop": "start_time"}, "right": {"value": run_end}},
        ],
        "select": [
            {"alias": "te", "prop": "start_time", "as": "start_time"},
            {"alias": "te", "prop": "end_time", "as": "end_time"},
            {"alias": "te", "prop": "duration_minutes_eff", "as": "duration_minutes_eff"},
            {"alias": "te", "prop": "erc_id", "as": "erc_id"},
            {"alias": "te", "prop": "erc_desc", "as": "erc_desc"},
            {"alias": "te", "prop": "reason1", "as": "reason1"},
            {"alias": "te", "prop": "reason2", "as": "reason2"},
            {"alias": "te", "prop": "reason3", "as": "reason3"},
            {"alias": "te", "prop": "reason4", "as": "reason4"},
            {"alias": "te", "prop": "tefault_name", "as": "tefault_name"},
            {"alias": "te", "prop": "remarks", "as": "remarks"},
        ],
        "order_by": [{"term": {"alias": "te", "prop": "duration_minutes_eff", "as": "duration_minutes_eff"}, "dir": "desc"}],
        "limit": int(limit),
    }


def _drill_spec_waste_events(
    pu_id: int,
    run_start: str,
    run_end: str,
    limit: int = 10,
) -> Dict[str, Any]:
    return {
        "bind": [{"alias": "we", "entity": "dt_waste_event_denorm"}],
        "where": [
            {"op": "==", "left": {"alias": "we", "prop": "pu_id"}, "right": {"value": pu_id}},
            {"op": ">=", "left": {"alias": "we", "prop": "entry_on"}, "right": {"value": run_start}},
            {"op": "<=", "left": {"alias": "we", "prop": "entry_on"}, "right": {"value": run_end}},
        ],
        "select": [
            {"alias": "we", "prop": "entry_on", "as": "entry_on"},
            {"alias": "we", "prop": "amount", "as": "amount"},
            {"alias": "we", "prop": "erc_id", "as": "erc_id"},
            {"alias": "we", "prop": "erc_desc", "as": "erc_desc"},
            {"alias": "we", "prop": "reason1", "as": "reason1"},
            {"alias": "we", "prop": "reason2", "as": "reason2"},
            {"alias": "we", "prop": "reason3", "as": "reason3"},
            {"alias": "we", "prop": "reason4", "as": "reason4"},
            {"alias": "we", "prop": "wefault_name", "as": "wefault_name"},
        ],
        "order_by": [{"term": {"alias": "we", "prop": "amount", "as": "amount"}, "dir": "desc"}],
        "limit": int(limit),
    }


def _drill_spec_target_speed_audit(
    pu_id: int,
    prod_id: int,
    as_of_time: str,
    limit: int = 1,
) -> Dict[str, Any]:
    # Effective dating: pick latest record with effective_date <= as_of_time.
    return {
        "bind": [{"alias": "ts", "entity": "dt_target_speed"}],
        "where": [
            {"op": "==", "left": {"alias": "ts", "prop": "pu_id"}, "right": {"value": pu_id}},
            {"op": "==", "left": {"alias": "ts", "prop": "prod_id"}, "right": {"value": prod_id}},
            {"op": "<=", "left": {"alias": "ts", "prop": "effective_date"}, "right": {"value": as_of_time}},
        ],
        "select": [
            {"alias": "ts", "prop": "effective_date", "as": "effective_date"},
            {"alias": "ts", "prop": "expiration_date", "as": "expiration_date"},
            {"alias": "ts", "prop": "target_speed", "as": "target_speed"},
        ],
        "order_by": [{"term": {"alias": "ts", "prop": "effective_date", "as": "effective_date"}, "dir": "desc"}],
        "limit": int(limit),
    }


def _run_drilldown(builder: Any, spec: Dict[str, Any]) -> pd.DataFrame:
    # enforce sane limits for drilldowns
    lim = spec.get("limit", 50)
    try:
        lim_i = int(lim)
    except Exception:
        lim_i = 50
    spec["limit"] = max(1, min(lim_i, 200))
    return run_dynamic_query(builder, spec)


def _substitute_placeholders(value: Any, env: Dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_substitute_placeholders(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _substitute_placeholders(v, env) for k, v in value.items()}
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        return env.get(key)
    return value


def _can_batch_inputs(inputs: Dict[str, Any], batch_key: str) -> bool:
    for key, source in (inputs or {}).items():
        if key == batch_key:
            continue
        if isinstance(source, str) and source.startswith("row."):
            return False
    return True


def _rewrite_eq_list_to_in(node: Any) -> Any:
    if isinstance(node, list):
        return [_rewrite_eq_list_to_in(x) for x in node]
    if not isinstance(node, dict):
        return node
    if "and" in node:
        node["and"] = _rewrite_eq_list_to_in(node.get("and"))
        return node
    if "or" in node:
        node["or"] = _rewrite_eq_list_to_in(node.get("or"))
        return node
    if "op" in node:
        op = node.get("op")
        left = node.get("left")
        right = node.get("right")
        if op == "==" and isinstance(left, dict) and isinstance(right, dict):
            rv = right.get("value")
            if isinstance(rv, list):
                return {"in": {"left": left, "right": {"value": rv}}}
        return node
    # other predicate shapes
    for k, v in node.items():
        node[k] = _rewrite_eq_list_to_in(v)
    return node


def _spec_has_missing_values(spec: Dict[str, Any]) -> bool:
    missing = False

    def walk(node: Any) -> None:
        nonlocal missing
        if missing:
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if "right" in node and isinstance(node.get("right"), dict):
            if node["right"].get("value") is None:
                missing = True
                return
        for key in ("left", "right", "and", "or", "between", "in", "not"):
            if key in node:
                walk(node.get(key))

    walk(spec)
    return missing


def _lookup_registry_field_dtype(field_name: str) -> str:
    if not field_name:
        return ""
    reg = _load_semantic_registry()
    if not reg:
        return ""
    target = str(field_name).lower()
    for ent in reg.get("entities", []) or []:
        if not isinstance(ent, dict):
            continue
        for f in ent.get("fields") or []:
            if not isinstance(f, dict):
                continue
            if str(f.get("name") or "").lower() == target:
                return str(f.get("dtype") or "")
    return ""


def _coerce_value_for_dtype(value: Any, dtype: str) -> Any:
    if value is None:
        return None
    dtype_norm = str(dtype or "").lower()
    try:
        if any(t in dtype_norm for t in ("number", "numeric", "decimal", "int")):
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str):
                return int(float(value.strip()))
        if any(t in dtype_norm for t in ("float", "double", "real")):
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.strip())
        if any(t in dtype_norm for t in ("bool", "boolean")):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "1", "yes", "y"):
                    return True
                if v in ("false", "0", "no", "n"):
                    return False
        if any(t in dtype_norm for t in ("timestamp", "datetime", "date")):
            if isinstance(value, (datetime.datetime, datetime.date)):
                return value
            if isinstance(value, str):
                v = value.strip()
                try:
                    if "t" in v.lower() or ":" in v:
                        return datetime.datetime.fromisoformat(v)
                    return datetime.date.fromisoformat(v)
                except Exception:
                    return value
    except Exception:
        return value
    return value


def _resolve_inputs(
    inputs: Dict[str, Any],
    row: Dict[str, Any],
    context: Dict[str, Any],
    alias_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    def _coerce_numeric_str(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        s = value.strip()
        if not s:
            return value
        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except Exception:
                return value
        if re.fullmatch(r"-?\d+\.\d+", s):
            try:
                return float(s)
            except Exception:
                return value
        return value

    env = dict(context or {})
    env.update(row or {})
    for key, source in (inputs or {}).items():
        if isinstance(source, str):
            if source.startswith("row."):
                field = source[4:]
                value = row.get(field)
                if value is None and alias_map and field in alias_map:
                    value = row.get(alias_map[field])
                env[key] = _coerce_numeric_str(value)
            elif source.startswith("context."):
                env[key] = _coerce_numeric_str(context.get(source[8:]))
            elif source.startswith("literal:"):
                env[key] = _coerce_numeric_str(source[len("literal:") :])
            else:
                value = row.get(source)
                if value is None and alias_map and source in alias_map:
                    value = row.get(alias_map[source])
                env[key] = _coerce_numeric_str(value if value is not None else source)
        else:
            env[key] = _coerce_numeric_str(source)
    return env


def _build_alias_map_from_spec(spec: Dict[str, Any]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for section in ("select", "group_by"):
        for item in (spec.get(section) or []):
            if not isinstance(item, dict):
                continue
            prop = item.get("prop")
            alias = item.get("as") or prop
            if not prop or not alias:
                continue
            if prop not in alias_map:
                alias_map[prop] = alias
    return alias_map


def _execute_drilldown_plan(
    builder: Any,
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list) or not steps:
        return {}

    window_start, window_end = _extract_seed_window(spec)
    context = {"window_start": window_start, "window_end": window_end}
    alias_map = _build_alias_map_from_spec(spec)

    step_records: Dict[str, List[Dict[str, Any]]] = {}
    step_outputs: Dict[str, Any] = {}

    base_rows = base_df.to_dict(orient="records") if base_df is not None else []
    max_sources = int(os.environ.get("AI_INSIGHTS_REASONER_MAX_SOURCES", "50") or 50)

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "").strip() or f"step_{len(step_outputs)+1}"
        from_ref = step.get("from") or "base"
        source_rows = base_rows if from_ref == "base" else step_records.get(str(from_ref), [])
        try:
            limit = int(step.get("limit") or 0)
        except Exception:
            limit = 0
        if max_sources > 0:
            source_rows = source_rows[:max_sources]
        if limit > 0:
            source_rows = source_rows[:limit]

        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        query_template = step.get("query") if isinstance(step.get("query"), dict) else {}
        if not query_template:
            continue

        collected: List[Dict[str, Any]] = []
        per_source: List[Dict[str, Any]] = []

        batch_key = (step.get("batch_on") or step.get("batch_key") or "").strip()
        if batch_key and _can_batch_inputs(inputs, batch_key):
            values: List[Any] = []
            for row in source_rows:
                row_dict = dict(row or {})
                env = _resolve_inputs(inputs, row_dict, context, alias_map=alias_map)
                v = env.get(batch_key)
                if v is not None:
                    values.append(v)
            if values:
                # de-dup while preserving order
                values = list(dict.fromkeys(values))
                env = _resolve_inputs(inputs, {}, context, alias_map=alias_map)
                env[batch_key] = values
                query_spec = _substitute_placeholders(query_template, env)
                if isinstance(query_spec, dict):
                    query_spec = _rewrite_eq_list_to_in(query_spec)
                if isinstance(query_spec, dict) and not query_spec.get("aggregations") and not _spec_has_missing_values(query_spec):
                    try:
                        df = _run_drilldown(builder, query_spec)
                        records = df.to_dict(orient="records") if df is not None else []
                    except Exception:
                        records = []
                    per_source.append({"source": {"batch_on": batch_key, "values": values}, "records": records})
                    if records:
                        collected.extend(records)
                    step_outputs[step_id] = {"rows": per_source}
                    step_records[step_id] = collected
                    _dbg(f"drilldown step '{step_id}': sources={len(source_rows)}, records={len(collected)} (batched)")
                    continue

        for row in source_rows:
            row_dict = dict(row or {})
            env = _resolve_inputs(inputs, row_dict, context, alias_map=alias_map)
            query_spec = _substitute_placeholders(query_template, env)
            if not isinstance(query_spec, dict):
                continue
            if query_spec.get("aggregations"):
                continue
            if _spec_has_missing_values(query_spec):
                continue
            try:
                df = _run_drilldown(builder, query_spec)
                records = df.to_dict(orient="records") if df is not None else []
            except Exception:
                records = []
            per_source.append({"source": row_dict, "records": records})
            if records:
                collected.extend(records)

        step_outputs[step_id] = {"rows": per_source}
        step_records[step_id] = collected
        _dbg(f"drilldown step '{step_id}': sources={len(source_rows)}, records={len(collected)}")

    return {"steps": step_outputs}


def _execute_drilldown_plan_native(
    builder: Any,
    base_df: pd.DataFrame,
    spec: Dict[str, Any],
    plan: Dict[str, Any],
    reasoner_id: str,
) -> Dict[str, Any]:
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list) or not steps:
        return {}

    window_start, window_end = _extract_seed_window(spec)
    context = {"window_start": window_start, "window_end": window_end}

    step_records: Dict[str, List[Dict[str, Any]]] = {}
    step_outputs: Dict[str, Any] = {}

    base_rows = base_df.to_dict(orient="records") if base_df is not None else []

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "").strip() or f"step_{len(step_outputs)+1}"
        from_ref = step.get("from") or "base"
        source_rows = base_rows if from_ref == "base" else step_records.get(str(from_ref), [])
        try:
            limit = int(step.get("limit") or 0)
        except Exception:
            limit = 0
        if limit > 0:
            source_rows = source_rows[:limit]

        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
        query_template = step.get("query") if isinstance(step.get("query"), dict) else {}
        if not query_template:
            continue

        concept_name = _reasoner_step_concept_name(reasoner_id, step_id)
        alias = "r"
        where_list: List[Dict[str, Any]] = []

        value_map: Dict[str, Any] = {}
        for key, source in (inputs or {}).items():
            if isinstance(source, str):
                if source.startswith("row."):
                    col = source[4:]
                    values = [row.get(col) for row in source_rows if col in row]
                    values = [v for v in values if v is not None]
                    if values:
                        value_map[key] = sorted({str(v) if isinstance(v, str) else v for v in values})
                elif source.startswith("context."):
                    value_map[key] = context.get(source[8:])
                elif source.startswith("literal:"):
                    value_map[key] = source[len("literal:") :]
                else:
                    value_map[key] = source
            else:
                value_map[key] = source

        for key, val in list(value_map.items()):
            dtype = _lookup_registry_field_dtype(key)
            if not dtype:
                continue
            if isinstance(val, list):
                coerced = [_coerce_value_for_dtype(v, dtype) for v in val]
                value_map[key] = [v for v in coerced if v is not None]
            else:
                value_map[key] = _coerce_value_for_dtype(val, dtype)

        for key, val in value_map.items():
            prop = _normalize_prop_name(key)
            if isinstance(val, list):
                where_list.append(
                    {
                        "in": {
                            "left": {"alias": alias, "prop": prop},
                            "right": {"value": val},
                        }
                    }
                )
            elif val is not None:
                where_list.append(
                    {
                        "op": "==",
                        "left": {"alias": alias, "prop": prop},
                        "right": {"value": val},
                    }
                )

        select_list = []
        for term in query_template.get("select") or []:
            if not isinstance(term, dict):
                continue
            prop = term.get("as") or term.get("prop")
            if not prop:
                continue
            select_list.append({"alias": alias, "prop": _normalize_prop_name(prop), "as": prop})

        if not select_list:
            select_list = [
                {"alias": alias, "prop": _normalize_prop_name(k), "as": k}
                for k in value_map.keys()
            ]

        native_spec = {
            "bind": [{"alias": alias, "entity": concept_name}],
            "where": where_list,
            "select": select_list,
            "limit": int(step.get("limit") or 200),
        }

        try:
            _dbg(f"native relation queried: {concept_name}")
            df = run_dynamic_query(builder, native_spec)
            records = df.to_dict(orient="records") if df is not None else []
        except Exception:
            records = []

        step_outputs[step_id] = {"rows": [{"records": records}]}
        step_records[step_id] = records

    return {"steps": step_outputs}


# -----------------------------
# Legacy: spec enhancement helpers
# -----------------------------

def enhance_spec_with_needed_reasoners(
    spec: Dict[str, Any],
    question: str,
    builder: Any = None,
) -> Dict[str, Any]:
    """
    Legacy helper: mutates the base spec to include columns required by inferred reasoners.
    This is NOT multi-hop. Prefer apply_all_relevant_reasoners() which runs drilldowns.
    """
    needed_reasoners = infer_reasoners_from_question(question) or []
    enhanced = spec
    for reasoner_id in needed_reasoners:
        enhanced = apply_reasoner_to_query_spec(enhanced, reasoner_id, builder=builder)
    return enhanced


def enhance_spec_for_reasoner(
    spec: Dict[str, Any],
    reasoner_id: str,
    builder: Any = None,
) -> Dict[str, Any]:
    return apply_reasoner_to_query_spec(spec, reasoner_id, builder=builder)


# -----------------------------
# Multi-hop orchestrator
# -----------------------------

def apply_all_relevant_reasoners(
    builder: Any,
    spec: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    reasoner_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Multi-hop reasoner orchestrator.

    - Ensures a base df exists (either provided or executed from spec)
    - Runs reasoner.analyze(df) for "summary" insights
    - For selected reasoners, runs additional drilldown queries to collect evidence:
      (worst runs, timed events, waste events, target speed audit)

    Returns:
      {
        "dataframe": <base df (enriched if possible)>,
        "reasoner_results": {reasoner_id: {"analysis":..., "drilldowns":...}},
        "reasoning_context": "<text>"
      }
    """
    if not isinstance(spec, dict):
        print(f"[WARN] Reasoner orchestration skipped: spec is not a dict ({type(spec).__name__}).")
        return {"dataframe": df, "reasoner_results": {}, "reasoning_context": ""}

    # Identify base entity (first bind)
    binds = spec.get("bind") or []
    entity_name = ""
    if isinstance(binds, list) and binds and isinstance(binds[0], dict):
        entity_name = binds[0].get("entity", "") or ""

    # Resolve reasoners
    if reasoner_ids is None:
        reasoner_ids = get_reasoners_for_entity(entity_name) if entity_name else []
    else:
        # If caller provided reasoners, keep only those applicable to this entity (if registry supports it)
        if entity_name:
            allowed = set(get_reasoners_for_entity(entity_name))
            reasoner_ids = [r for r in reasoner_ids if r in allowed]
    reasoner_ids = [r for r in (reasoner_ids or []) if not _is_graph_reasoner(r)]

    _dbg(f"entity={entity_name or '<unknown>'} reasoner_ids={reasoner_ids}")

    if not reasoner_ids:
        return {"dataframe": df, "reasoner_results": {}, "reasoning_context": ""}

    # Ensure base df exists
    if df is None:
        df = run_dynamic_query(builder, spec)

    try:
        _dbg(f"base df rows={len(df)} cols={len(df.columns)}")
    except Exception:
        pass

    # Enrich derived metrics when possible (does NOT join to event tables)
    try:
        if entity_name:
            df = enrich_dataframe_with_derived_metrics(df, entity_name)
    except Exception:
        pass

    reasoner_results: Dict[str, Any] = {}
    context_lines: List[str] = ["=== REASONER ANALYSIS ===\n"]
    native_enabled = os.environ.get("AI_INSIGHTS_REASONER_NATIVE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if native_enabled:
        _dbg("native reasoner traversal enabled")

    for r_id in reasoner_ids:
        reasoner = get_reasoner(r_id)

        entry: Dict[str, Any] = {}
        drilldowns: Dict[str, Any] = {}

        # 1) Analysis on the base df (lightweight)
        if reasoner:
            try:
                entry["analysis"] = reasoner.analyze(df)
            except Exception as exc:
                entry["analysis_error"] = str(exc)

        # 2) Evidence drilldowns (multi-hop) driven by registry plan
        plan = _registry_reasoner_plan(r_id)
        if plan:
            step_count = len(plan.get("steps") or []) if isinstance(plan, dict) else 0
            _dbg(f"{r_id}: registry drilldown plan found (steps={step_count})")
        if plan:
            if native_enabled:
                drilldowns = _execute_drilldown_plan_native(builder, df, spec, plan, r_id)
            else:
                drilldowns = _execute_drilldown_plan(builder, df, spec, plan)
        elif r_id in ("oee_explainer", "downtime_driver", "waste_impact"):
            _dbg(f"{r_id}: using legacy drilldown fallback")
            # Legacy fallback (explicit plan for known reasoners)
            # (Kept to avoid fanout joins when no registry plan exists.)
            window_start, window_end = _extract_seed_window(spec)
            possible_machine_cols = ["pu_id", "machine_id"]
            machine_col = next((c for c in possible_machine_cols if c in df.columns), None)
            machines: List[int] = []

            if machine_col:
                try:
                    if "avg_oee_pct" in df.columns:
                        sub = df[[machine_col, "avg_oee_pct"]].dropna()
                        sub = sub.sort_values("avg_oee_pct", ascending=True)
                        machines = [int(x) for x in sub[machine_col].head(3).tolist()]
                    else:
                        machines = [int(x) for x in df[machine_col].dropna().unique().tolist()[:3]]
                except Exception:
                    machines = []

            drill_machine_blocks = []
            for pu in machines:
                # Worst operations for this machine in the seed window
                ops_df = _run_drilldown(builder, _drill_spec_worst_ops(pu, window_start, window_end, limit=3))

                ops_records = []
                for _, row in ops_df.iterrows():
                    prod_id = row.get("prod_id")
                    op_start = row.get("operation_start")
                    op_end = row.get("operation_end")

                    op_block: Dict[str, Any] = {"operation": row.to_dict()}

                    # Evidence within op window
                    if isinstance(op_start, str) and isinstance(op_end, str):
                        if r_id in ("oee_explainer", "downtime_driver"):
                            te_df = _run_drilldown(builder, _drill_spec_timed_events(pu, op_start, op_end, limit=10))
                            op_block["timed_events"] = te_df.to_dict(orient="records")

                        if r_id in ("oee_explainer", "waste_impact"):
                            we_df = _run_drilldown(builder, _drill_spec_waste_events(pu, op_start, op_end, limit=10))
                            op_block["waste_events"] = we_df.to_dict(orient="records")

                        # Optional: include target speed audit for explainers if prod_id present
                        if r_id == "oee_explainer" and prod_id is not None:
                            try:
                                prod_int = int(prod_id)
                                ts_df = _run_drilldown(
                                    builder, _drill_spec_target_speed_audit(pu, prod_int, op_start, limit=1)
                                )
                                op_block["target_speed_audit"] = ts_df.to_dict(orient="records")
                            except Exception:
                                pass

                    ops_records.append(op_block)

                drill_machine_blocks.append({"pu_id": pu, "worst_operations": ops_records})

            drilldowns["machines"] = drill_machine_blocks

        if r_id == "target_speed_audit":
            _dbg(f"{r_id}: running audit fallback")
            # If base df already has pu_id+prod_id+operation_start, audit first few rows
            audits = []
            if {"pu_id", "prod_id", "operation_start"}.issubset(set(df.columns)):
                for _, row in df.head(5).iterrows():
                    try:
                        pu = int(row["pu_id"])
                        prod = int(row["prod_id"])
                        as_of = row["operation_start"]
                        if isinstance(as_of, str):
                            ts_df = _run_drilldown(builder, _drill_spec_target_speed_audit(pu, prod, as_of, limit=1))
                            audits.append(
                                {
                                    "pu_id": pu,
                                    "prod_id": prod,
                                    "as_of_time": as_of,
                                    "records": ts_df.to_dict(orient="records"),
                                }
                            )
                    except Exception:
                        continue
            drilldowns["audits"] = audits

        if drilldowns:
            entry["drilldowns"] = drilldowns
            if "steps" in drilldowns:
                step_summary = {
                    step_id: len((payload or {}).get("rows") or [])
                    for step_id, payload in (drilldowns.get("steps") or {}).items()
                }
                _dbg(f"{r_id}: drilldown step rows={step_summary}")
            elif "machines" in drilldowns:
                _dbg(f"{r_id}: drilldown machines={len(drilldowns.get('machines') or [])}")
            elif "audits" in drilldowns:
                _dbg(f"{r_id}: drilldown audits={len(drilldowns.get('audits') or [])}")

        reasoner_results[r_id] = entry

        # 3) LLM-facing context
        try:
            txt = format_rai_reasoner_columns_for_llm(df, r_id)
            if txt:
                context_lines.append(txt)
        except Exception:
            pass

        # Add compact drilldown summary to context
        if drilldowns:
            context_lines.append(f"[{r_id}] drilldown summary:")
            if "steps" in drilldowns:
                for step_id, payload in (drilldowns.get("steps") or {}).items():
                    rows = (payload or {}).get("rows") or []
                    context_lines.append(f"- {step_id}: {len(rows)} source rows")
            elif "machines" in drilldowns:
                for mblk in (drilldowns.get("machines") or [])[:3]:
                    pu = mblk.get("pu_id")
                    ops = mblk.get("worst_operations") or []
                    context_lines.append(f"- pu_id={pu}: {len(ops)} worst operations")
            elif "audits" in drilldowns:
                context_lines.append(f"- audits: {len(drilldowns.get('audits') or [])}")
            context_lines.append("")

    reasoning_context = "\n".join(context_lines).strip()

    return {
        "dataframe": df,
        "reasoner_results": reasoner_results,
        "reasoning_context": reasoning_context,
    }


def get_reasoner_context_for_llm(reasoner_id: str, results: Optional[Any] = None) -> str:
    """
    Backward-compatible helper expected by ai_insights.py.
    Returns any static prompt context a reasoner provides.
    """
    reasoner = get_reasoner(reasoner_id)
    if not reasoner:
        return ""
    try:
        return reasoner.get_prompt_context() or ""
    except Exception:
        return ""
