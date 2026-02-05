# streamlit_app_pro_cols_docfix_v4_noaggrid.py
"""
AI Planner → Cortex Analyst (Pro) — v4
- ID-safe charts (IDs are categorical; force category x-axis)
- Auto-switch line→bar for ID X axes
- Smart aggregation defaults (AUM=last, Score=avg, Revenue/Topup/Profit=sum)
- KPI cards stacked vertically
- NEW: Multi-view auto-join — combine datasets from different semantic views on shared keys (RMID / MANDATEID / RM_NAME / MONTH / MONTH_DATE / MEET_YEAR)
"""

import json
import os
import requests
import uuid
import re
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from rai_ai_insights_ontology import (
    build_ai_insights_builder,
    ensure_rai_config,
    infer_join_keys,
    load_ai_insights_specs,
    load_ai_insights_relationships,
    render_ai_insights_ontology_text,
)
from rai_dynamic_query import run_dynamic_query
from rai_dynamic_reasoner import generate_dynamic_query_spec
from rai_reasoner_orchestrator import inject_reasoner_relations_into_spec
# Reasoner orchestration removed (RAI-native model relations only)
from rai_derived_metrics import enrich_dataframe_with_derived_metrics
from rai_semantic_registry import load_prompt_templates, load_analysis_config
from rai_semantic_registry import load_kg_spec, load_reasoners
import relationalai.semantics as qb
from relationalai.semantics.reasoners.graph import Graph
from relationalai.semantics.rel.rel_utils import sanitize_identifier

# --- Final LLM token budget helpers -----------------------------------------
FINAL_CTX_TOKENS = 128_000          # model window
FINAL_RESERVE_TOKENS = 6_000        # leave space for model's reply
FINAL_INPUT_LIMIT_TOKENS = 90_000
FINAL_INPUT_LIMIT_CHARS  = 320_000   # extra safety: ~3.5 chars ≈ 1 token (rough)


def _approx_tokens(s: str) -> int:
    # ~4 chars ≈ 1 token (safe approximation)
    return 0 if not s else max(1, len(s) // 4)

def _hard_slice_to_limit(s: str, limit_tokens: int, safety: int = 1_000) -> str:
    # final guard: slice by chars to fit token budget
    max_chars = max(4_000, (limit_tokens - safety) * 4)
    return (s[:max_chars] + "\n...[prompt truncated to fit model window]") if len(s) > max_chars else s


# --- Graph reasoner routing -------------------------------------------------
def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (text or "").lower()))


def _select_graph_for_question(question: str, graphs: list[dict]) -> dict | None:
    if not question or not graphs:
        return None
    q_tokens = _tokenize(question)
    best = None
    best_score = 0
    for g in graphs:
        if not isinstance(g, dict):
            continue
        gid = str(g.get("id") or "").strip()
        if not gid:
            continue
        text = f"{gid} {g.get('description') or ''}".lower()
        score = len(q_tokens & _tokenize(text))
        if score > best_score:
            best_score = score
            best = g
    return best


def _select_graph_for_question_llm(
    question: str,
    graphs: list[dict],
    cortex_complete: callable,
) -> tuple[dict | None, str]:
    if not question or not graphs or not callable(cortex_complete):
        return None, ""
    candidates = []
    for g in graphs:
        if not isinstance(g, dict):
            continue
        gid = str(g.get("id") or "").strip()
        if not gid:
            continue
        candidates.append(
            {
                "id": gid,
                "description": str(g.get("description") or "").strip(),
            }
        )
    if not candidates:
        return None, ""

    prompt = (
        "You are a graph router. Choose the single most relevant graph for the QUESTION.\n"
        "Return JSON only.\n\n"
        "RULES:\n"
        "- If none are relevant, return {\"graph_id\": null, \"reasoning\": \"\"}.\n"
        "- Only select from the provided graph ids.\n"
        "- Be concise.\n\n"
        f"GRAPHS:\n{json.dumps(candidates, indent=2)}\n\n"
        f"QUESTION:\n{question}\n"
    )
    try:
        raw = cortex_complete(prompt)
        parsed = safe_json_loads(raw if isinstance(raw, str) else "")
        if not isinstance(parsed, dict):
            return None, ""
        graph_id = parsed.get("graph_id")
        if not graph_id:
            return None, str(parsed.get("reasoning") or "")
        graph_id = str(graph_id).strip()
        for g in graphs:
            if isinstance(g, dict) and str(g.get("id") or "").strip() == graph_id:
                return g, str(parsed.get("reasoning") or "")
    except Exception:
        return None, ""
    return None, ""


def _graph_reasoners_from_ids(reasoner_ids: list[str]) -> list:
    try:
        reasoners = {r.id: r for r in load_reasoners() if getattr(r, "id", None)}
    except Exception:
        return []
    graph_reasoners = []
    for rid in reasoner_ids or []:
        spec = reasoners.get(rid)
        if not spec:
            continue
        if str(getattr(spec, "type", "")).strip().lower() != "graph_reasoner":
            continue
        graph_id = str(getattr(spec, "graph_id", "") or "").strip()
        if graph_id:
            graph_reasoners.append(spec)
    return graph_reasoners


def _graph_algo_for_question(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["cluster", "community", "segment", "louvain"]):
        return "louvain"
    if any(k in q for k in ["influence", "rank", "pagerank"]):
        return "pagerank"
    if any(k in q for k in ["degree", "connected", "hub"]):
        return "degree"
    return "degree"


def _is_graph_intent(question: str) -> bool:
    q = (question or "").lower()
    keywords = [
        "graph",
        "network",
        "connected",
        "connection",
        "cluster",
        "community",
        "central",
        "influence",
        "pagerank",
        "louvain",
        "recurring",
        "shared",
        "common",
        "root cause",
        "root-cause",
        "driver",
        "cause",
        "hotspot",
        "path",
        "traverse",
    ]
    return any(k in q for k in keywords)


def _extract_time_window_from_spec(spec: dict) -> tuple[str | None, str | None]:
    if not isinstance(spec, dict):
        return None, None
    meta = spec.get("meta") or {}
    if isinstance(meta, dict):
        tw = meta.get("time_window") or {}
        if isinstance(tw, dict):
            start = tw.get("start")
            end = tw.get("end")
            if isinstance(start, (datetime.date, datetime.datetime)) or isinstance(end, (datetime.date, datetime.datetime)):
                start_val = start.isoformat() if isinstance(start, (datetime.date, datetime.datetime)) else None
                end_val = end.isoformat() if isinstance(end, (datetime.date, datetime.datetime)) else None
                return (start_val, end_val)
            if isinstance(start, str) or isinstance(end, str):
                return (start if isinstance(start, str) else None, end if isinstance(end, str) else None)
    if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
        try:
            print(f"[GRAPH_BUILD] time_window meta missing or invalid meta={meta}")
        except Exception:
            pass

    lows: list[str] = []
    highs: list[str] = []
    time_props = {
        "operation_start",
        "operation_end",
        "start_time",
        "end_time",
        "entry_on",
        "effective_date",
        "event_time",
        "timestamp",
        "date",
    }

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if "and" in node:
            walk(node.get("and"))
        if "or" in node:
            walk(node.get("or"))
        op = node.get("op")
        left = node.get("left") or {}
        right = node.get("right") or {}
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return
        prop = left.get("prop")
        if not isinstance(prop, str):
            return
        if prop not in time_props:
            return
        val = right.get("value")
        if not isinstance(val, str):
            return
        if op in (">=", ">"):
            lows.append(val)
        elif op in ("<=", "<"):
            highs.append(val)

    walk(spec.get("where") or [])
    low = max(lows) if lows else None
    high = min(highs) if highs else None
    return low, high


def _coerce_datetime(value: Any) -> datetime.datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time.min)
    if isinstance(value, str):
        v = value.strip()
        try:
            if "t" in v.lower() or ":" in v:
                return datetime.datetime.fromisoformat(v)
            return datetime.datetime.fromisoformat(f"{v}T00:00:00")
        except Exception:
            return None
    return None


def _extract_filter_values_from_spec(spec: dict, props: list[str]) -> Dict[str, List[Any]]:
    if not isinstance(spec, dict) or not props:
        return {}
    wanted = {p: [] for p in props}

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if "and" in node:
            walk(node.get("and"))
        if "or" in node:
            walk(node.get("or"))
        op = node.get("op")
        if op != "==":
            return
        left = node.get("left") or {}
        right = node.get("right") or {}
        if not (isinstance(left, dict) and isinstance(right, dict)):
            return
        prop = left.get("prop")
        if not isinstance(prop, str) or prop not in wanted:
            return
        if "value" in right:
            wanted[prop].append(right.get("value"))

    walk(spec.get("where") or [])
    return {k: v for k, v in wanted.items() if v}


def _graph_builder_registry():
    return {
        "machine_to_operations": _graph_build_machine_to_operations,
        "operation_to_downtime_faults": _graph_build_operation_to_timed_events,
        "operation_to_waste_faults": _graph_build_operation_to_waste_events,
        "operation_to_target_speed": _graph_build_operation_to_target_speed,
    }


def _get_ref_prop(ref: object, prop: str):
    if not prop:
        return None
    try:
        if hasattr(ref, prop):
            return getattr(ref, prop)
        p2 = sanitize_identifier(str(prop).lower())
        if hasattr(ref, p2):
            return getattr(ref, p2)
    except Exception:
        return None
    return None


def _resolve_graph_operand(operand: Any, refs: Dict[str, object], params: Dict[str, Any]):
    if operand is None:
        return None
    if isinstance(operand, dict):
        if "param" in operand:
            return params.get(str(operand.get("param") or "").strip())
        if "value" in operand:
            val = operand.get("value")
            if isinstance(val, str) and val.startswith("$"):
                return params.get(val[1:], val)
            return val
        ent = operand.get("entity")
        prop = operand.get("prop")
        if isinstance(ent, str) and isinstance(prop, str):
            ref = refs.get(ent)
            if ref is None:
                return None
            return _get_ref_prop(ref, prop)
    return operand


def _apply_edge_set(builder, g, edge_set: Dict[str, Any], params: Dict[str, Any]) -> None:
    if not isinstance(edge_set, dict):
        return
    label = str(edge_set.get("label") or "").strip() or "edge"
    if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
        try:
            ws = params.get("window_start")
            we = params.get("window_end")
            has_overlap = bool(edge_set.get("overlap"))
            has_eff = bool(edge_set.get("effective_dated"))
            where_count = len(edge_set.get("where") or []) if isinstance(edge_set.get("where"), list) else 0
            print(
                f"[GRAPH_BUILD] edge_set={label} window_start={ws} window_end={we} "
                f"overlap={has_overlap} effective_dated={has_eff} where_terms={where_count}"
            )
        except Exception:
            pass
    src = edge_set.get("src") if isinstance(edge_set.get("src"), dict) else {}
    dst = edge_set.get("dst") if isinstance(edge_set.get("dst"), dict) else {}
    src_entity = str(src.get("entity") or "").strip()
    dst_entity = str(dst.get("entity") or "").strip()
    if not src_entity or not dst_entity:
        return

    src_concept = builder.get_concept(src_entity)
    dst_concept = builder.get_concept(dst_entity)
    if src_concept is None or dst_concept is None:
        return
    a = src_concept.ref()
    b = dst_concept.ref()
    refs = {src_entity: a, dst_entity: b}

    wh = [a, b]

    # Optional generic interval overlap constraint:
    # overlap iff left.start <= right.end AND left.end >= right.start
    overlap = edge_set.get("overlap") if isinstance(edge_set.get("overlap"), dict) else None
    if overlap:
        left = overlap.get("left") if isinstance(overlap.get("left"), dict) else {}
        right = overlap.get("right") if isinstance(overlap.get("right"), dict) else {}
        le = refs.get(str(left.get("entity") or "").strip())
        re = refs.get(str(right.get("entity") or "").strip())
        if le is not None and re is not None:
            ls = _get_ref_prop(le, str(left.get("start") or ""))
            le_ = _get_ref_prop(le, str(left.get("end") or ""))
            rs = _get_ref_prop(re, str(right.get("start") or ""))
            re_ = _get_ref_prop(re, str(right.get("end") or ""))
            if ls is not None and le_ is not None and rs is not None and re_ is not None:
                wh.append(ls <= re_)
                wh.append(le_ >= rs)

    # Optional effective-dated constraint:
    # effective <= as_of AND expiration >= as_of
    eff = edge_set.get("effective_dated") if isinstance(edge_set.get("effective_dated"), dict) else None
    if eff:
        as_of = eff.get("as_of") if isinstance(eff.get("as_of"), dict) else {}
        effective = eff.get("effective") if isinstance(eff.get("effective"), dict) else {}
        expiration = eff.get("expiration") if isinstance(eff.get("expiration"), dict) else {}
        as_ref = refs.get(str(as_of.get("entity") or "").strip())
        eff_ref = refs.get(str(effective.get("entity") or "").strip())
        exp_ref = refs.get(str(expiration.get("entity") or "").strip())
        if as_ref is not None and eff_ref is not None and exp_ref is not None:
            as_term = _get_ref_prop(as_ref, str(as_of.get("prop") or ""))
            eff_term = _get_ref_prop(eff_ref, str(effective.get("prop") or ""))
            exp_term = _get_ref_prop(exp_ref, str(expiration.get("prop") or ""))
            if as_term is not None and eff_term is not None and exp_term is not None:
                wh.append(eff_term <= as_term)
                wh.append(exp_term >= as_term)

    where_terms = edge_set.get("where") or []
    if isinstance(where_terms, list):
        for t in where_terms:
            if not isinstance(t, dict):
                continue
            op = str(t.get("op") or "").strip()
            left = _resolve_graph_operand(t.get("left"), refs, params)
            right = _resolve_graph_operand(t.get("right"), refs, params)
            if left is None or right is None:
                continue
            if op == "==":
                wh.append(left == right)
            elif op == "<=":
                wh.append(left <= right)
            elif op == ">=":
                wh.append(left >= right)
            elif op == "<":
                wh.append(left < right)
            elif op == ">":
                wh.append(left > right)

    def _node_id(spec: Dict[str, Any], ref: object):
        ident = spec.get("id")
        if ident == "ref":
            return ref
        if isinstance(ident, dict) and "prop" in ident:
            return _get_ref_prop(ref, str(ident.get("prop") or ""))
        return ref

    src_id = _node_id(src, a)
    dst_id = _node_id(dst, b)
    if src_id is None or dst_id is None:
        return

    qb.where(*wh).define(
        na := g.Node.new(id=src_id),
        nb := g.Node.new(id=dst_id),
        g.Edge.new(src=na, dst=nb, label=label),
    )


def _graph_build_machine_to_operations(builder, g, window_start, window_end, filters, args):
    machine_entity = str((args or {}).get("machine_entity") or "dt_unit_config")
    operation_entity = str((args or {}).get("operation_entity") or "dt_operation_metrics")
    relation = str((args or {}).get("relation") or "").strip()
    edge_label = str((args or {}).get("edge_label") or "has_operation")

    machine = builder.get_concept(machine_entity)
    operation = builder.get_concept(operation_entity)
    if machine is None or operation is None:
        return
    m = machine.ref()
    o = operation.ref()

    wh = [m, o]
    rel = getattr(machine, relation, None) if relation else None
    if rel is not None:
        wh.append(rel(m, o))
    elif hasattr(m, "pu_id") and hasattr(o, "pu_id"):
        wh.append(m.pu_id == o.pu_id)

    ws = _coerce_datetime(window_start)
    we = _coerce_datetime(window_end)
    if ws and hasattr(o, "operation_end"):
        wh.append(getattr(o, "operation_end") >= ws)
    if we and hasattr(o, "operation_start"):
        wh.append(getattr(o, "operation_start") <= we)

    if filters:
        pu_vals = filters.get("pu_id") or []
        if pu_vals and hasattr(o, "pu_id"):
            wh.append(getattr(o, "pu_id") == pu_vals[0])
        prod_vals = filters.get("prod_id") or []
        if prod_vals and hasattr(o, "prod_id"):
            wh.append(getattr(o, "prod_id") == prod_vals[0])

    qb.where(*wh).define(
        na := g.Node.new(id=m),
        nb := g.Node.new(id=o),
        g.Edge.new(src=na, dst=nb, label=edge_label),
    )


def _graph_build_operation_to_timed_events(builder, g, window_start, window_end, filters, args):
    operation_entity = str((args or {}).get("operation_entity") or "dt_operation_metrics")
    event_entity = str((args or {}).get("event_entity") or "dt_timed_event_denorm")
    node_field = str((args or {}).get("node_field") or "tefault_name")
    edge_label = str((args or {}).get("edge_label") or "downtime_fault")
    event_start_field = str((args or {}).get("event_start_field") or "start_time")
    event_end_field = str((args or {}).get("event_end_field") or "end_time")

    operation = builder.get_concept(operation_entity)
    event = builder.get_concept(event_entity)
    if operation is None or event is None:
        return
    o = operation.ref()
    e = event.ref()

    wh = [o, e]
    if hasattr(o, "pu_id") and hasattr(e, "pu_id"):
        wh.append(getattr(o, "pu_id") == getattr(e, "pu_id"))

    ws = _coerce_datetime(window_start)
    we = _coerce_datetime(window_end)
    if ws and hasattr(o, "operation_end"):
        wh.append(getattr(o, "operation_end") >= ws)
    if we and hasattr(o, "operation_start"):
        wh.append(getattr(o, "operation_start") <= we)

    if ws and hasattr(e, event_end_field):
        wh.append(getattr(e, event_end_field) >= ws)
    if we and hasattr(e, event_start_field):
        wh.append(getattr(e, event_start_field) <= we)

    if hasattr(e, event_start_field) and hasattr(o, "operation_end"):
        wh.append(getattr(e, event_start_field) <= getattr(o, "operation_end"))
    if hasattr(e, event_end_field) and hasattr(o, "operation_start"):
        wh.append(getattr(e, event_end_field) >= getattr(o, "operation_start"))

    if filters:
        pu_vals = filters.get("pu_id") or []
        if pu_vals and hasattr(o, "pu_id"):
            wh.append(getattr(o, "pu_id") == pu_vals[0])
        prod_vals = filters.get("prod_id") or []
        if prod_vals and hasattr(o, "prod_id"):
            wh.append(getattr(o, "prod_id") == prod_vals[0])

    node_id = getattr(e, node_field, None) if hasattr(e, node_field) else None
    if node_id is None:
        node_id = getattr(e, "tefault_name", None)
    if node_id is None:
        node_id = e

    qb.where(*wh).define(
        na := g.Node.new(id=o),
        nb := g.Node.new(id=node_id),
        g.Edge.new(src=na, dst=nb, label=edge_label),
    )


def _graph_build_operation_to_waste_events(builder, g, window_start, window_end, filters, args):
    operation_entity = str((args or {}).get("operation_entity") or "dt_operation_metrics")
    event_entity = str((args or {}).get("event_entity") or "dt_waste_event_denorm")
    node_field = str((args or {}).get("node_field") or "wefault_name")
    edge_label = str((args or {}).get("edge_label") or "waste_fault")
    event_time_field = str((args or {}).get("event_time_field") or "entry_on")

    operation = builder.get_concept(operation_entity)
    event = builder.get_concept(event_entity)
    if operation is None or event is None:
        return
    o = operation.ref()
    e = event.ref()

    wh = [o, e]
    if hasattr(o, "pu_id") and hasattr(e, "pu_id"):
        wh.append(getattr(o, "pu_id") == getattr(e, "pu_id"))

    ws = _coerce_datetime(window_start)
    we = _coerce_datetime(window_end)
    if ws and hasattr(o, "operation_end"):
        wh.append(getattr(o, "operation_end") >= ws)
    if we and hasattr(o, "operation_start"):
        wh.append(getattr(o, "operation_start") <= we)

    if hasattr(e, event_time_field):
        if ws:
            wh.append(getattr(e, event_time_field) >= ws)
        if we:
            wh.append(getattr(e, event_time_field) <= we)
        if hasattr(o, "operation_start"):
            wh.append(getattr(e, event_time_field) >= getattr(o, "operation_start"))
        if hasattr(o, "operation_end"):
            wh.append(getattr(e, event_time_field) <= getattr(o, "operation_end"))

    if filters:
        pu_vals = filters.get("pu_id") or []
        if pu_vals and hasattr(o, "pu_id"):
            wh.append(getattr(o, "pu_id") == pu_vals[0])
        prod_vals = filters.get("prod_id") or []
        if prod_vals and hasattr(o, "prod_id"):
            wh.append(getattr(o, "prod_id") == prod_vals[0])

    node_id = getattr(e, node_field, None) if hasattr(e, node_field) else None
    if node_id is None:
        node_id = getattr(e, "wefault_name", None)
    if node_id is None:
        node_id = e

    qb.where(*wh).define(
        na := g.Node.new(id=o),
        nb := g.Node.new(id=node_id),
        g.Edge.new(src=na, dst=nb, label=edge_label),
    )


def _graph_build_operation_to_target_speed(builder, g, window_start, window_end, filters, args):
    operation_entity = str((args or {}).get("operation_entity") or "dt_operation_metrics")
    target_entity = str((args or {}).get("target_entity") or "dt_target_speed")
    edge_label = str((args or {}).get("edge_label") or "target_speed")

    operation = builder.get_concept(operation_entity)
    target = builder.get_concept(target_entity)
    if operation is None or target is None:
        return
    o = operation.ref()
    t = target.ref()

    wh = [o, t]
    if hasattr(o, "pu_id") and hasattr(t, "pu_id"):
        wh.append(getattr(o, "pu_id") == getattr(t, "pu_id"))
    if hasattr(o, "prod_id") and hasattr(t, "prod_id"):
        wh.append(getattr(o, "prod_id") == getattr(t, "prod_id"))

    if hasattr(o, "operation_start") and hasattr(t, "effective_date"):
        wh.append(getattr(t, "effective_date") <= getattr(o, "operation_start"))
    if hasattr(o, "operation_start") and hasattr(t, "expiration_date"):
        wh.append(getattr(t, "expiration_date") >= getattr(o, "operation_start"))

    ws = _coerce_datetime(window_start)
    we = _coerce_datetime(window_end)
    if ws and hasattr(o, "operation_end"):
        wh.append(getattr(o, "operation_end") >= ws)
    if we and hasattr(o, "operation_start"):
        wh.append(getattr(o, "operation_start") <= we)

    if filters:
        pu_vals = filters.get("pu_id") or []
        if pu_vals and hasattr(o, "pu_id"):
            wh.append(getattr(o, "pu_id") == pu_vals[0])
        prod_vals = filters.get("prod_id") or []
        if prod_vals and hasattr(o, "prod_id"):
            wh.append(getattr(o, "prod_id") == prod_vals[0])

    qb.where(*wh).define(
        na := g.Node.new(id=o),
        nb := g.Node.new(id=t),
        g.Edge.new(src=na, dst=nb, label=edge_label),
    )


def _build_graph_for_reasoner(builder, graph_spec, reasoner_spec, window_start, window_end, filters):
    graph_build = getattr(reasoner_spec, "graph_build", None) or {}
    if not isinstance(graph_build, dict):
        return None
    directed = graph_build.get("directed")
    weighted = graph_build.get("weighted")
    if directed is None:
        directed = bool(graph_spec.get("directed", True)) if isinstance(graph_spec, dict) else True
    if weighted is None:
        weighted = bool(graph_spec.get("weighted", False)) if isinstance(graph_spec, dict) else False
    g = Graph(builder._model, directed=bool(directed), weighted=bool(weighted))

    # Prefer domain-agnostic declarative edge_sets if present.
    params = {
        "window_start": _coerce_datetime(window_start),
        "window_end": _coerce_datetime(window_end),
    }
    for k, v in (filters or {}).items():
        if v:
            params[k] = v[0]
    edge_sets = graph_build.get("edge_sets")
    if isinstance(edge_sets, list) and edge_sets:
        if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
            try:
                ws = params.get("window_start")
                we = params.get("window_end")
                print(
                    f"[GRAPH_BUILD] graph={graph_spec.get('id')} reasoner={getattr(reasoner_spec, 'id', '')} "
                    f"edge_sets={len(edge_sets)} window_start={ws} window_end={we} filters={filters or {}}"
                )
            except Exception:
                pass
        for es in edge_sets:
            try:
                _apply_edge_set(builder, g, es, params)
            except Exception:
                if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
                    try:
                        print(f"[GRAPH_BUILD] edge_set failed label={es.get('label')}")
                    except Exception:
                        pass
                continue
        return g

    # Back-compat: older builder-id based graph_build.
    builders = graph_build.get("builders") or []
    if isinstance(builders, list) and builders:
        registry = _graph_builder_registry()
        if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
            try:
                print(
                    f"[GRAPH_BUILD] graph={graph_spec.get('id')} reasoner={getattr(reasoner_spec, 'id', '')} "
                    f"builders={len(builders)} window_start={window_start} window_end={window_end} filters={filters or {}}"
                )
            except Exception:
                pass
        for b in builders:
            if not isinstance(b, dict):
                continue
            bid = str(b.get("id") or b.get("builder_id") or "").strip()
            if not bid:
                continue
            fn = registry.get(bid)
            if not fn:
                continue
            args = b.get("args") or {}
            try:
                fn(builder, g, window_start, window_end, filters, args)
            except Exception:
                if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
                    try:
                        print(f"[GRAPH_BUILD] builder failed id={bid}")
                    except Exception:
                        pass
                continue
    return g


def _run_graph_reasoner_query(
    builder,
    graph_spec: dict,
    reasoner_spec,
    question: str,
    limit: int = 10,
    window_start: str | None = None,
    window_end: str | None = None,
    filters: Optional[Dict[str, List[Any]]] = None,
):
    graph_id = str(graph_spec.get("id") or "").strip()
    if not graph_id:
        return None
    graph = None
    if reasoner_spec is not None and getattr(reasoner_spec, "graph_build", None):
        graph = _build_graph_for_reasoner(
            builder,
            graph_spec,
            reasoner_spec,
            window_start,
            window_end,
            filters or {},
        )
    if graph is None:
        graph = builder.get_graph(graph_id)
    if graph is None:
        return None
    algo = _graph_algo_for_question(question)
    node = graph.Node.ref()
    if algo == "pagerank":
        rel = graph.pagerank()
        score = qb.Float.ref()
    elif algo == "louvain":
        try:
            rel = graph.louvain()
        except Exception:
            rel = graph.degree()
        score = qb.Integer.ref()
    else:
        rel = graph.degree()
        score = qb.Integer.ref()
    label_term = None
    try:
        node_id = getattr(node, "id", None)
        if node_id is not None:
            label_term = getattr(node_id, "tefault_name", None) or getattr(node_id, "erc_desc", None)
        if label_term is None:
            label_term = node_id
    except Exception:
        label_term = None
    if label_term is None:
        label_term = node
    where_args = [rel(node, score)]
    # NOTE: graph node properties are currently untyped in this pipeline,
    # so applying time window filters can produce unresolved overloads.
    # We skip time filtering here until graph node typing is explicit.
    q = qb.where(*where_args)
    df = q.select(label_term.alias("label"), score.alias(algo)).to_df()
    if not df.empty and algo in df.columns:
        df = df.sort_values(by=[algo], ascending=False)
    return df.head(int(limit))


# --- Helper: robust JSON extraction from LLM responses ---
import json as _json
import re as _re

def _strip_code_fences(s: str) -> str:
    """Remove markdown code fences and surrounding backticks from an LLM reply."""
    if s is None:
        return s
    # remove leading/trailing whitespace
    s = s.strip()
    # remove ```json ... ``` or ``` ... ``` blocks by extracting inner content if it looks like JSON
    m = _re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', s, _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # remove single backtick blocks `...`
    m2 = _re.search(r'`([^`{].*?)`', s, _re.DOTALL)
    if m2 and ('{' in m2.group(1) or ':' in m2.group(1)):
        return m2.group(1).strip()
    # remove common leading/trailing phrases
    # e.g., "```json\n{...}\n```" or "Here is the JSON:\n{...}"
    # fallback to return original
    return s

def _find_balanced_json(s: str):
    """Find the first balanced JSON object substring in s. Returns substring or None."""
    if not s or '{' not in s:
        return None
    start = s.find('{')
    stack = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == '{':
            stack += 1
        elif c == '}':
            stack -= 1
            if stack == 0:
                return s[start:i+1]
    return None


def safe_json_loads(text: str):
    """Robustly parse JSON or SQL-like outputs from LLMs.

    Strategy (best-effort):
    1. Strip markdown / code fences.
    2. Try a strict json.loads call.
    3. Try to extract a balanced {...} or [...] substring and parse it.
    4. Attempt common fixes: convert NULL/None -> null, single->double quotes,
       remove trailing commas, quote unquoted keys, then json.loads.
    5. If the text looks like SQL, return {"sql": "<original SQL>"}.
    6. As a last resort, try a simple "key: value" line parser into a dict.
    Raises ValueError if nothing parseable is found.
    """
    if text is None:
        raise ValueError("No text to parse JSON from.")
    s = _strip_code_fences(text)
    s_stripped = s.strip()

    # 1) Strict JSON first
    try:
        return _json.loads(s_stripped)
    except Exception:
        pass

    # 2) Balanced JSON extraction (first {} or [] block)
    candidate = _find_balanced_json(s_stripped)
    if candidate:
        for cand in (candidate, candidate.replace("'", '"'), candidate.replace('NULL', 'null').replace('None', 'null')):
            try:
                return _json.loads(cand)
            except Exception:
                continue

    # 3) Heuristic fixes
    fixed = s_stripped
    # Normalize common tokens
    fixed = _re.sub(r'\bNULL\b', 'null', fixed, flags=_re.IGNORECASE)
    fixed = fixed.replace("None", "null")
    # Replace single quotes with double quotes (simple heuristic)
    fixed = fixed.replace("'", '"')
    # Remove trailing commas before } or ]
    fixed = _re.sub(r',\s*}', '}', fixed)
    fixed = _re.sub(r',\s*]', ']', fixed)
    # Quote unquoted keys (very conservative: keys that appear at object starts)
    fixed = _re.sub(r'([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', fixed)
    try:
        return _json.loads(fixed)
    except Exception:
        pass

    # 4) If it looks like SQL, return as SQL payload
    if _re.match(r'^(SELECT|WITH|INSERT|UPDATE|DELETE)\b', s_stripped, _re.IGNORECASE):
        return {"sql": s_stripped}

    # 5) Try to extract a "sql": "..." by regex
    m_sql = _re.search(r'"sql"\s*[:=]\s*"([^"]+)"', s_stripped, _re.IGNORECASE)
    if m_sql:
        return {"sql": m_sql.group(1)}

    # 6) Last resort: simple key: value lines into dict
    lines = [ln.strip() for ln in s_stripped.splitlines() if ln.strip()]
    simple = {}
    for ln in lines:
        if ':' in ln:
            k, v = ln.split(':', 1)
            k = k.strip().strip('"').strip("'")
            v = v.strip().strip('"').strip("'")
            # Try to coerce numbers
            if re.match(r'^-?\d+\.?\d*$', v):
                try:
                    if '.' in v:
                        v_parsed = float(v)
                    else:
                        v_parsed = int(v)
                    simple[k] = v_parsed
                    continue
                except Exception:
                    pass
            # booleans/null
            if v.lower() in ('null', 'none'):
                simple[k] = None; continue
            if v.lower() in ('true', 'false'):
                simple[k] = True if v.lower()=='true' else False; continue
            simple[k] = v
    if simple:
        return simple

    raise ValueError("Could not parse JSON or SQL from LLM response.")

# --- end helper ---

# --- Prompt template override (registry-driven) ---
def _get_prompt_template(key: str, default: str) -> str:
    try:
        templates = load_prompt_templates()
        if isinstance(templates, dict):
            value = templates.get(key)
            if isinstance(value, str) and value.strip():
                return value
    except Exception:
        pass
    return default

_DEFAULT_DRIVER_METRIC_SYNONYMS = {
    "aum": ["aum", "current_aum", "assets", "assets under management", "total_aum", "nav"],
    "revenue": ["revenue", "income", "fee", "fees", "topline", "buy", "buy_amount", "purchases"],
    "profit": ["profit", "pnl", "p&l", "net_income", "netincome", "margin"],
    "cost": ["cost", "rm_cost", "expense", "expenses", "spend", "spending", "sell", "redemption"],
    "topup": ["topup", "top_up", "contribution", "deposit", "inflow", "inflows", "fundin", "cash_in"],
    "withdrawal": ["withdrawal", "outflow", "outflows", "redemption", "redeem", "payout", "fundout", "sell"],
    "nnm": ["nnm", "netnewmoney", "net new money", "total_nnm"],
    "commitment": ["commitment", "commitments", "firstcommitment", "commitmenttarget", "topupcommitment"],
    "sentiment": ["sentiment", "nps", "csat", "satisfaction", "sentiment_score", "positive_cnt", "negative_cnt"],
    "score": ["score", "rating", "grade", "index", "performance_score"],
    "meeting": ["meeting", "meetings", "touch", "touches", "interaction", "meeting_target"],
    "pipeline": ["pipeline", "opportunity", "deal"],
    "balance": ["balance", "bal"],
    "performance": ["performance", "perf", "return", "yield"],
    "clients": ["client", "clients", "clienttarget", "total_clients", "mandate_count"],
    "duration": ["duration", "tenure", "relationship_duration"],
}

_DEFAULT_DRIVER_QUESTION_CORE = [
    "why", "cause", "reason", "driver", "drivers", "because",
    "due to", "explain", "reason behind", "root cause", "driving",
    "justify", "justified", "justifies", "justifying",
    "justification", "rationale", "rationalize", "rationalized", "rationalizing",
]

_DEFAULT_DRIVER_QUESTION_CHANGE = [
    "increase", "increasing", "rise", "rising", "growth", "growing",
    "decrease", "decreasing", "decline", "dropping", "drop",
    "fall", "falling", "down", "up", "trend", "trending", "fluctuation", "fluctuating",
    "improve", "improving", "improvement", "boost", "boosting",
    "enhance", "optimise", "optimize", "better",
]

_DEFAULT_DRIVER_QUESTION_SPECIAL = [
    "what drove", "what is driving", "drivers of",
    "driver behind", "cause of", "reasons for",
    "justify the", "justification for", "rationale for", "explain why",
]

_DEFAULT_DRIVER_QUESTION_PRESCRIPTIVE = [
    "what can", "how can", "what should", "how should", "what could", "how could",
]


def _merge_driver_synonyms(defaults: dict, overrides: object) -> dict:
    merged = {k: list(v) for k, v in defaults.items()}
    if not isinstance(overrides, dict):
        return merged
    for key, values in overrides.items():
        if not key:
            continue
        if isinstance(values, list):
            merged[str(key)] = [str(v) for v in values if v]
        elif isinstance(values, str) and values.strip():
            merged[str(key)] = [values.strip()]
    return merged


def _load_analysis_overrides() -> dict:
    cfg = {}
    try:
        cfg = load_analysis_config()
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}

    def _list(key: str, default: list[str]) -> list[str]:
        vals = cfg.get(key)
        if isinstance(vals, list) and vals:
            return [str(v) for v in vals if v]
        return list(default)

    return {
        "driver_metric_synonyms": _merge_driver_synonyms(
            _DEFAULT_DRIVER_METRIC_SYNONYMS,
            cfg.get("driver_metric_synonyms"),
        ),
        "driver_question_core_terms": _list("driver_question_core_terms", _DEFAULT_DRIVER_QUESTION_CORE),
        "driver_question_change_terms": _list("driver_question_change_terms", _DEFAULT_DRIVER_QUESTION_CHANGE),
        "driver_question_special_phrases": _list("driver_question_special_phrases", _DEFAULT_DRIVER_QUESTION_SPECIAL),
        "driver_question_prescriptive_phrases": _list(
            "driver_question_prescriptive_phrases",
            _DEFAULT_DRIVER_QUESTION_PRESCRIPTIVE,
        ),
    }


_ANALYSIS_CONFIG = _load_analysis_overrides()
_DRIVER_METRIC_SYNONYMS = _ANALYSIS_CONFIG["driver_metric_synonyms"]


def _analysis_default_int(key: str, default: int) -> int:
    try:
        val = _ANALYSIS_CONFIG.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
        if isinstance(val, str) and val.strip().isdigit():
            return int(val.strip())
    except Exception:
        pass
    return default


def _analysis_list(key: str, default: list[str]) -> list[str]:
    try:
        vals = _ANALYSIS_CONFIG.get(key)
        if isinstance(vals, list) and vals:
            return [str(v) for v in vals if v]
    except Exception:
        pass
    return list(default)


def _analysis_dimension_hint(default: str = "RMID") -> str:
    try:
        hint = _ANALYSIS_CONFIG.get("dimension_hint")
        if isinstance(hint, str) and hint.strip():
            return hint.strip()
        priority = _ANALYSIS_CONFIG.get("entity_priority")
        if isinstance(priority, list) and priority:
            mapping = {"rm": "RMID", "mandate": "MANDATEID", "client": "CLIENTID"}
            for item in priority:
                key = str(item or "").strip().lower()
                if key in mapping:
                    return mapping[key]
    except Exception:
        pass
    return default


DEFAULT_MONTHS_WINDOW = _analysis_default_int("default_months_window", 12)
_DRIVER_METRIC_TOKENS = set()
for _aliases in _DRIVER_METRIC_SYNONYMS.values():
    for _alias in _aliases:
        _alias_l = _alias.lower()
        _DRIVER_METRIC_TOKENS.add(_alias_l)
        _DRIVER_METRIC_TOKENS.update(re.findall(r"[a-z0-9]+", _alias_l))

def _looks_like_driver_question(text: Optional[str]) -> bool:
    if not text:
        return False
    lc = text.lower()
    cfg = _ANALYSIS_CONFIG
    core = tuple(cfg.get("driver_question_core_terms") or [])
    change = tuple(cfg.get("driver_question_change_terms") or [])
    special = tuple(cfg.get("driver_question_special_phrases") or [])
    prescriptive = tuple(cfg.get("driver_question_prescriptive_phrases") or [])
    tokens = set(re.findall(r"[a-z0-9]+", lc))
    if any(phrase in lc for phrase in special):
        return True
    if any(word in lc for word in core) and any(word in lc for word in change):
        return True
    if any(word in lc for word in core):
        if tokens & _DRIVER_METRIC_TOKENS:
            return True
    if any(phrase in lc for phrase in prescriptive):
        if tokens & _DRIVER_METRIC_TOKENS and any(word in lc for word in change):
            return True
    return False

def _run_driver_analysis(question: str,
                         frames: Dict[str, 'pd.DataFrame']) -> Optional[Dict[str, Any]]:
    import pandas as pd
    import numpy as np
    import math
    import re
    import datetime as dt

    try:
        q_text = (question or "").strip()
        if not q_text:
            return None
        q_lower = q_text.lower()

        TIME_PRIORITY = [
            "MONTH_DATE", "MONTH_DT", "MONTHKEY", "MONTH_KEY", "MONTH_ID",
            "MEET_MONTH", "MEET_MON", "MONTH", "PERIOD", "YEARMONTH",
            "YEAR_MONTH", "REPORTING_MONTH", "AS_OF_MONTH", "DATE",
            "AS_OF_DATE", "SNAP_DATE", "RUN_DATE", "EOM_DATE",
            "BUSINESS_DATE", "MONTHEND", "MONTH_END"
        ]

        MAX_ROWS = 2000
        MAX_SERIES = 36
        MIN_POINTS = 4
        DRIVER_MAX_RESULTS = 7
        DRIVER_SUMMARY_LIMIT = 5

        def _trim_pair_df(df_pair: 'pd.DataFrame', a_col: str, b_col: str, p: float = 0.01,
                           min_points: int = MIN_POINTS) -> 'pd.DataFrame':
            try:
                if df_pair is None or df_pair.empty:
                    return df_pair
                a = pd.to_numeric(df_pair[a_col], errors="coerce")
                b = pd.to_numeric(df_pair[b_col], errors="coerce")
                df_local = pd.DataFrame({a_col: a, b_col: b}).dropna()
                if df_local.shape[0] < max(min_points, 6):
                    return df_pair
                qa_lo, qa_hi = df_local[a_col].quantile(p), df_local[a_col].quantile(1 - p)
                qb_lo, qb_hi = df_local[b_col].quantile(p), df_local[b_col].quantile(1 - p)
                mask = (
                    (df_pair[a_col] >= qa_lo) & (df_pair[a_col] <= qa_hi)
                    & (df_pair[b_col] >= qb_lo) & (df_pair[b_col] <= qb_hi)
                )
                trimmed = df_pair[mask].dropna(subset=[a_col, b_col])
                if trimmed.shape[0] >= min_points:
                    return trimmed
                return df_pair.dropna(subset=[a_col, b_col])
            except Exception:
                return df_pair

        def _trim_arrays(a: 'np.ndarray', b: 'np.ndarray', p: float = 0.01,
                          min_points: int = MIN_POINTS) -> tuple['np.ndarray','np.ndarray']:
            try:
                a = np.asarray(a, dtype=float)
                b = np.asarray(b, dtype=float)
                mask = np.isfinite(a) & np.isfinite(b)
                a = a[mask]
                b = b[mask]
                if a.size < max(min_points, 6):
                    return a, b
                qa_lo, qa_hi = np.quantile(a, [p, 1 - p])
                qb_lo, qb_hi = np.quantile(b, [p, 1 - p])
                keep = (a >= qa_lo) & (a <= qa_hi) & (b >= qb_lo) & (b <= qb_hi)
                a2 = a[keep]; b2 = b[keep]
                if a2.size >= min_points:
                    return a2, b2
                return a, b
            except Exception:
                return a, b

        tokens = set(re.findall(r"[a-z0-9]+", q_lower))
        if not _looks_like_driver_question(q_text):
            return None

        def _normalize_text(value: Any) -> str:
            text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
            return re.sub(r"\s+", " ", text).strip()

        def _extract_rm_target(question_raw: str) -> Dict[str, Any]:
            result = {
                "rmid_variants": set(),
                "rmid_display": None,
                "rm_name": None,
                "rm_name_norm": None,
            }
            if not question_raw:
                return result

            aliases = _ANALYSIS_CONFIG.get("entity_aliases", {}) if isinstance(_ANALYSIS_CONFIG, dict) else {}
            rm_prefixes = aliases.get("rm_prefixes") if isinstance(aliases, dict) else None
            if not isinstance(rm_prefixes, list) or not rm_prefixes:
                rm_prefixes = ["RM"]
            prefix_pattern = "|".join([re.escape(str(p)) for p in rm_prefixes if p])
            if not prefix_pattern:
                prefix_pattern = "RM"

            try:
                id_matches = re.findall(
                    rf"\b(?:{prefix_pattern})[\s\-]*(\d{{2,}})\b",
                    question_raw,
                    flags=re.IGNORECASE,
                )
            except Exception:
                id_matches = []
            for match in id_matches:
                cleaned = match.strip()
                if not cleaned:
                    continue
                upper_val = cleaned.upper()
                trimmed = cleaned.lstrip("0") or cleaned
                result["rmid_variants"].update({upper_val, trimmed.upper()})
                if result["rmid_display"] is None:
                    result["rmid_display"] = cleaned

            stop_tokens = {
                "do", "does", "did", "to", "for", "in", "on", "over", "within",
                "across", "during", "during", "next", "increase", "improve", "boost",
                "his", "her", "their", "the", "this", "that", "score", "scores",
                "help", "can", "could", "should", "would", "will"
            }
            try:
                name_match = re.search(
                    rf"\b(?:{prefix_pattern})[\s\-]+([A-Za-z][A-Za-z\s]{{0,60}})",
                    question_raw,
                    flags=re.IGNORECASE,
                )
            except Exception:
                name_match = None
            if name_match:
                tail = name_match.group(1).strip()
                tokens_name = tail.split()
                captured: list[str] = []
                for token in tokens_name:
                    token_clean = token.strip(" ,.-")
                    if not token_clean:
                        continue
                    if token_clean.lower() in stop_tokens:
                        break
                    captured.append(token_clean)
                name_text = " ".join(captured).strip()
                if name_text:
                    result["rm_name"] = name_text
                    try:
                        result["rm_name_norm"] = _normalize_text(name_text)
                    except Exception:
                        result["rm_name_norm"] = name_text.lower()

            return result

        def _extract_years(question_raw: str) -> list[int]:
            years: set[int] = set()
            if not question_raw:
                return []
            try:
                for match in re.findall(r"\b(20\d{2})\b", question_raw):
                    try:
                        years.add(int(match))
                    except Exception:
                        continue
            except Exception:
                pass
            return sorted(years)

        rm_target = _extract_rm_target(q_text)
        question_years = _extract_years(q_text)

        def _metric_family(name: Optional[str]) -> Optional[str]:
            if not name:
                return None
            text = _normalize_text(name)
            for fam, synonyms in _DRIVER_METRIC_SYNONYMS.items():
                for syn in synonyms:
                    syn_norm = _normalize_text(syn)
                    if syn_norm and (syn_norm in text or text in syn_norm):
                        return fam
            return None

        METRIC_NAME_STOPWORDS = {
            "total", "totals", "amount", "amt", "value", "sum", "net", "gross",
            "overall", "actual", "current", "latest", "figure", "score", "index",
            "count", "number", "num", "pct", "percent", "percentage", "ratio",
            "metric", "measure", "worth", "aggregate", "aggregation", "avg", "average",
            "per", "each", "every",
            "what", "which", "who", "when", "where", "why", "how",
            "are", "is", "was", "were", "do", "does", "did", "can",
            "could", "should", "would", "please", "show", "give", "tell",
            "explain", "drivers", "driver", "analysis", "data", "year", "years",
            "month", "months", "period", "periods", "trend", "trends", "question"
        }

        def _metric_tokens(text: Optional[str]) -> set[str]:
            if not text:
                return set()
            norm = _normalize_text(text)
            if not norm:
                return set()
            tokens = {tok for tok in norm.split() if tok}
            return {
                tok for tok in tokens
                if len(tok) > 2 and tok not in METRIC_NAME_STOPWORDS and not tok.isdigit()
            }

        def _is_same_metric_name(
            src_name: Optional[str],
            target_norm_val: str,
            target_tokens_primary: set[str],
            fallback_tokens: set[str],
        ) -> bool:
            if not src_name:
                return False
            src_norm = _normalize_text(src_name)
            if not src_norm:
                return False
            if target_norm_val and (
                src_norm == target_norm_val
                or src_norm in target_norm_val
                or target_norm_val in src_norm
            ):
                return True

            src_tokens = _metric_tokens(src_name)
            if target_tokens_primary:
                if src_tokens == target_tokens_primary:
                    return True
                overlap = len(src_tokens & target_tokens_primary)
                if len(target_tokens_primary) > 1 and overlap and overlap >= min(len(src_tokens), len(target_tokens_primary)):
                    return True
                union = len(src_tokens | target_tokens_primary)
                if union and (overlap / union) >= 0.8:
                    return True
            elif fallback_tokens:
                if src_tokens == fallback_tokens:
                    return True
                overlap_fb = len(src_tokens & fallback_tokens)
                if len(fallback_tokens) > 1 and overlap_fb and overlap_fb >= min(len(src_tokens), len(fallback_tokens)):
                    return True
                union_fb = len(src_tokens | fallback_tokens)
                if union_fb and (overlap_fb / union_fb) >= 0.8:
                    return True
            return False

        def _looks_like_near_duplicate(
            target_values: "np.ndarray",
            driver_values: "np.ndarray",
            corr_hint: Optional[float] = None,
        ) -> bool:
            try:
                mask = np.isfinite(target_values) & np.isfinite(driver_values)
            except Exception:
                return False
            if mask.sum() < max(MIN_POINTS, 6):
                return False
            tv = target_values[mask]
            dv = driver_values[mask]
            if tv.size < max(MIN_POINTS, 6) or dv.size < max(MIN_POINTS, 6):
                return False
            try:
                tv_std = float(np.std(tv))
                dv_std = float(np.std(dv))
            except Exception:
                return False
            if not math.isfinite(tv_std) or not math.isfinite(dv_std):
                return False
            if tv_std == 0.0 or dv_std == 0.0:
                return True
            try:
                corr_val = (
                    float(corr_hint)
                    if corr_hint is not None and math.isfinite(float(corr_hint))
                    else float(np.corrcoef(tv, dv)[0, 1])
                )
            except Exception:
                return False
            if abs(corr_val) < 0.97:
                return False
            try:
                A = np.vstack([dv, np.ones_like(dv)]).T
                slope, intercept = np.linalg.lstsq(A, tv, rcond=None)[0]
                residual = tv - (slope * dv + intercept)
                resid_std = float(np.std(residual))
            except Exception:
                return False
            if not math.isfinite(resid_std):
                return False
            if resid_std <= 0.05 * tv_std:
                return True
            if resid_std <= 0.1 * tv_std and abs(abs(slope) - 1.0) <= 0.05:
                return True
            return False

        def _safe_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value) if math.isfinite(float(value)) else None
                out = float(value)
                return out if math.isfinite(out) else None
            except Exception:
                return None

        def _format_period(value: Any) -> str:
            try:
                if isinstance(value, (dt.datetime, dt.date)):
                    if isinstance(value, dt.datetime) and value.day == 1:
                        return value.strftime("%Y-%m")
                    return value.strftime("%Y-%m-%d")
            except Exception:
                pass
            try:
                if isinstance(value, pd.Timestamp):
                    if value.day == 1:
                        return value.strftime("%Y-%m")
                    return value.strftime("%Y-%m-%d")
            except Exception:
                pass
            return str(value)

        def _lag_corr(target_values: np.ndarray, driver_values: np.ndarray, lag: int) -> Optional[float]:
            if lag > 0:
                vt = target_values[lag:]
                vd = driver_values[:-lag]
            elif lag < 0:
                lag_abs = abs(lag)
                vt = target_values[:-lag_abs]
                vd = driver_values[lag_abs:]
            else:
                vt = target_values
                vd = driver_values
            if len(vt) < MIN_POINTS or len(vd) < MIN_POINTS:
                return None
            mask = np.isfinite(vt) & np.isfinite(vd)
            if mask.sum() < MIN_POINTS:
                return None
            # Trim 1% tails on both arrays before computing correlation
            vt2, vd2 = _trim_arrays(vt[mask], vd[mask], p=0.01, min_points=MIN_POINTS)
            if len(vt2) < MIN_POINTS:
                vt2, vd2 = vt[mask], vd[mask]
            try:
                return float(np.corrcoef(vt2, vd2)[0, 1])
            except Exception:
                return None

        def _metric_score(col_name: str,
                          series_numeric: 'pd.Series',
                          keyword_tokens: set[str],
                          question_lower: str) -> float:
            name = str(col_name or "")
            name_lower = name.lower()
            score = 0.0
            if "id" in name_lower:
                score -= 3.0

            coverage = series_numeric.notna().mean()
            score += min(1.5, coverage * 2.0)

            try:
                std_val = float(series_numeric.std(skipna=True))
                if math.isfinite(std_val) and std_val > 0:
                    score += min(4.0, math.log10(std_val + 1.0))
            except Exception:
                pass

            for token in keyword_tokens:
                if len(token) <= 2:
                    continue
                if token in name_lower:
                    score += 1.2

            for base, synonyms in _DRIVER_METRIC_SYNONYMS.items():
                if base in question_lower:
                    for syn in synonyms:
                        if syn in name_lower:
                            score += 3.5
                else:
                    for syn in synonyms:
                        if syn in name_lower and len(syn) > 3:
                            score += 0.6

            if name_lower in ("value", "amount") and any(tok in question_lower for tok in ("value", "amount", "sum")):
                score += 0.5

        return score

        def _coerce_datetime(df_local: 'pd.DataFrame', col_name: str) -> Optional['pd.Series']:
            try:
                series = df_local[col_name]
            except Exception:
                return None

            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    converted = pd.to_datetime(series, errors="coerce")
                    if converted.notna().sum() >= MIN_POINTS:
                        return converted
            except Exception:
                pass

            try:
                converted = pd.to_datetime(series, errors="coerce")
                if converted.notna().sum() >= max(MIN_POINTS, int(len(series) * 0.4)):
                    return converted
            except Exception:
                pass

            try:
                converted = pd.to_datetime(series.astype(str), format="%Y-%m", errors="coerce")
                if converted.notna().sum() >= max(MIN_POINTS, int(len(series) * 0.4)):
                    return converted
            except Exception:
                pass

            try:
                converted = pd.to_datetime(series.astype(str), format="%Y%m", errors="coerce")
                if converted.notna().sum() >= max(MIN_POINTS, int(len(series) * 0.4)):
                    return converted
            except Exception:
                pass

            name_upper = str(col_name).upper()
            if name_upper in ("MONTH", "MEET_MON", "MEET_MONTH"):
                for year_col in ("MEET_YEAR", "YEAR", "CAL_YEAR"):
                    if year_col in df_local.columns:
                        y = pd.to_numeric(df_local.get(year_col), errors="coerce")
                        m = pd.to_numeric(series, errors="coerce")
                        combo = pd.to_datetime({"year": y, "month": m, "day": 1}, errors="coerce")
                        if combo.notna().sum() >= max(MIN_POINTS, int(len(series) * 0.4)):
                            return combo

            return None

        def _normalize_month_index(series: 'pd.Series') -> 'pd.Series':
            try:
                if series is None:
                    return series
                series_dt = pd.to_datetime(series, errors="coerce")
                if series_dt.notna().sum() == 0:
                    return series_dt
                periods = series_dt.dt.to_period("M")
                normalized = periods.dt.to_timestamp()
                try:
                    normalized = normalized.dt.tz_localize(None)
                except Exception:
                    pass
                return normalized
            except Exception:
                return series

        def _aggregate_by_time(df_in: 'pd.DataFrame',
                                time_col: str,
                                cols: list[str],
                                agg_map: Dict[str, str],
                                output_col: str = "__time__") -> 'pd.DataFrame':
            if not cols or df_in.empty or time_col not in df_in.columns:
                return pd.DataFrame(columns=[output_col] + list(cols))
            df_work = df_in[[time_col] + cols].dropna(subset=[time_col])
            if df_work.empty:
                return pd.DataFrame(columns=[output_col] + list(cols))
            df_work = df_work.sort_values(time_col)
            grouped = df_work.groupby(time_col)
            series_list = []
            for col in cols:
                mode = (agg_map.get(col) or "avg").lower()
                if mode == "sum":
                    agg_series = grouped[col].sum(min_count=1)
                elif mode == "count":
                    agg_series = grouped[col].count()
                elif mode == "last":
                    agg_series = grouped[col].apply(
                        lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan
                    )
                else:
                    agg_series = grouped[col].mean()
                agg_series.name = col
                series_list.append(agg_series)
            if not series_list:
                return pd.DataFrame(columns=[output_col] + list(cols))
            agg_df = pd.concat(series_list, axis=1)
            agg_df = agg_df.sort_index()
            agg_df.index.name = output_col
            agg_df = agg_df.reset_index()
            return agg_df

        def _select_entity(df_local: 'pd.DataFrame',
                           numeric_cols: list[str],
                           time_col: str,
                           question_norm: str) -> Dict[str, Any]:
            mask = pd.Series(True, index=df_local.index)
            best = {"column": None, "value": None, "matched": False, "mask": mask}
            entity_cols = []
            for col in df_local.columns:
                if col in numeric_cols or col in ("__time__", time_col):
                    continue
                series = df_local[col]
                try:
                    if pd.api.types.is_numeric_dtype(series):
                        continue
                except Exception:
                    continue
                uniques = series.dropna().astype(str).unique()
                if len(uniques) == 0 or len(uniques) > 50:
                    continue
                entity_cols.append((col, uniques))

            for col, values in entity_cols:
                for val in values:
                    norm_val = _normalize_text(val)
                    if norm_val and norm_val in question_norm:
                        col_norm = df_local[col].astype(str).apply(_normalize_text)
                        mask = col_norm == norm_val
                        if mask.any():
                            return {
                                "column": col,
                                "value": str(val),
                                "matched": True,
                                "mask": mask
                            }

            for col, values in entity_cols:
                if len(values) == 1:
                    val = values[0]
                    col_norm = df_local[col].astype(str).apply(_normalize_text)
                    mask = col_norm == _normalize_text(val)
                    return {
                        "column": col,
                        "value": str(val),
                        "matched": False,
                        "mask": mask
                    }

            return best

        candidate: Optional[Dict[str, Any]] = None

        for step, df in (frames or {}).items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            df_local = df.copy()
            filter_applied = False

            # Apply RMID filters if the question referenced a specific RM id
            if rm_target.get("rmid_variants"):
                rmid_cols = [c for c in df_local.columns if "RMID" in str(c).upper()]
                matched_rmid = False
                for col in rmid_cols:
                    try:
                        col_series = df_local[col].astype(str).str.upper().str.strip()
                    except Exception:
                        continue
                    mask = col_series.isin(rm_target["rmid_variants"])
                    if mask.any():
                        df_local = df_local[mask]
                        filter_applied = True
                        matched_rmid = True
                        break
                if not matched_rmid:
                    # If an RMID was requested but this frame doesn't contain it, skip
                    continue

            if df_local.empty:
                continue

            # Apply RM name filters (if supplied and still necessary)
            if rm_target.get("rm_name_norm"):
                name_cols = [
                    c for c in df_local.columns
                    if any(tag in str(c).upper() for tag in ("RM_NAME", "RELATIONSHIP_MANAGER", "RELATIONSHIPMANAGER"))
                ]
                matched_name = False
                for col in name_cols:
                    try:
                        col_norm = df_local[col].astype(str).apply(_normalize_text)
                    except Exception:
                        continue
                    mask = col_norm == rm_target["rm_name_norm"]
                    if mask.any():
                        df_local = df_local[mask]
                        filter_applied = True
                        matched_name = True
                        break
                if name_cols and not matched_name and not rm_target.get("rmid_variants"):
                    # Only a name was provided and we couldn't find it here
                    continue

            if df_local.empty:
                continue

            if question_years:
                year_cols = [
                    c for c in df_local.columns
                    if str(c).upper() in ("MEET_YEAR", "YEAR", "CAL_YEAR", "FISCAL_YEAR", "POSYEAR")
                ]
                year_filtered = False
                for col in year_cols:
                    try:
                        col_numeric = pd.to_numeric(df_local[col], errors="coerce").astype("Int64")
                        mask_year = col_numeric.isin(question_years)
                    except Exception:
                        continue
                    if mask_year.any():
                        df_local = df_local[mask_year]
                        filter_applied = True
                        year_filtered = True
                        break
                if year_cols and not year_filtered:
                    continue

            if df_local.empty:
                continue

            if len(df_local) > MAX_ROWS:
                df_local = df_local.tail(MAX_ROWS)

            time_col = None
            time_series = None
            for col in TIME_PRIORITY:
                if col in df_local.columns:
                    tmp = _coerce_datetime(df_local, col)
                    if tmp is not None:
                        tmp = _normalize_month_index(tmp)
                        if tmp.notna().sum() >= MIN_POINTS:
                            time_col = col
                            time_series = tmp
                            break

            if time_series is None:
                for col in df_local.columns:
                    tmp = _coerce_datetime(df_local, col)
                    if tmp is not None:
                        tmp = _normalize_month_index(tmp)
                        if tmp.notna().sum() >= MIN_POINTS:
                            time_col = col
                            time_series = tmp
                            break

            if time_series is None:
                continue

            time_series = _normalize_month_index(time_series)
            df_local = df_local.assign(__time__=time_series)
            df_local = df_local[df_local["__time__"].notna()]
            if df_local.empty:
                continue

            numeric_cols: Dict[str, 'pd.Series'] = {}
            for col in df_local.columns:
                if col in ("__time__", time_col):
                    continue
                series_num = pd.to_numeric(df_local[col], errors="coerce")
                if series_num.notna().sum() < max(MIN_POINTS, int(len(series_num) * 0.3)):
                    continue
                if series_num.nunique(dropna=True) <= 1:
                    continue
                numeric_cols[col] = series_num

            if not numeric_cols:
                continue

            metric_best = None
            for col_name, series_num in numeric_cols.items():
                score = _metric_score(col_name, series_num, tokens, q_lower)
                if metric_best is None or score > metric_best["score"]:
                    metric_best = {"column": col_name, "score": score}

            if metric_best is None:
                continue

            priority = metric_best["score"]
            if priority < 1.5:
                continue

            candidate_entry = {
                "step": step,
                "df": df_local,
                "time_col": time_col,
                "metric_col": metric_best["column"],
                "metric_score": priority,
                "numeric_cols": list(numeric_cols.keys()),
                "filter_applied": filter_applied,
            }

            if candidate is None or priority > candidate["metric_score"]:
                candidate = candidate_entry

        if not candidate:
            return None

        df_scope_full = candidate["df"].copy()
        filter_applied_candidate = bool(candidate.get("filter_applied"))
        question_norm = _normalize_text(q_text)
        entity = _select_entity(df_scope_full, candidate["numeric_cols"], candidate["time_col"], question_norm)
        try:
            df_scope = df_scope_full[entity["mask"]].copy()
        except Exception:
            df_scope = df_scope_full.copy()

        if df_scope.empty:
            return None

        if filter_applied_candidate and not entity.get("column"):
            if rm_target.get("rm_name") and rm_target.get("rm_name_norm"):
                entity["column"] = next(
                    (c for c in df_scope_full.columns if any(tag in str(c).upper() for tag in ("RM_NAME", "RELATIONSHIP_MANAGER", "RELATIONSHIPMANAGER"))),
                    None
                )
                entity["value"] = rm_target["rm_name"]
                entity["matched"] = True
            elif rm_target.get("rmid_display"):
                entity["column"] = next(
                    (c for c in df_scope_full.columns if "RMID" in str(c).upper()),
                    None
                )
                entity["value"] = rm_target["rmid_display"]
                entity["matched"] = True

        def _canonical_col(name: Any) -> str:
            return re.sub(r"[^A-Z0-9]+", "", str(name or "").upper())

        ENTITY_GROUPS = [
            {"RMID", "RM", "RELATIONSHIPMANAGERID"},
            {"RMNAME", "RELATIONSHIPMANAGER", "RELATIONSHIPMANAGERNAME"},
            {"MANDATEID", "ACCOUNTID", "PORTFOLIOID", "PORTFOLIO"},
            {"MANDATENAME", "ACCOUNTNAME", "PORTFOLIONAME", "MANDATE"},
            {"CLIENTID", "CLIENT"},
            {"CLIENTNAME", "CLIENT_FULLNAME"},
        ]

        def _entity_group(canon: str) -> Optional[int]:
            for idx, group in enumerate(ENTITY_GROUPS):
                if canon in group:
                    return idx
            return None

        entity_lookup = {}
        if not df_scope.empty:
            try:
                sample_row = df_scope.iloc[0]
                for col in df_scope.columns:
                    canon = _canonical_col(col)
                    if not canon:
                        continue
                    if any(tag in canon for tag in ("ID", "NAME", "RM", "MANDATE", "CLIENT", "ACCOUNT")):
                        val = sample_row.get(col)
                        if val is not None and val == val:
                            entity_lookup[canon] = _normalize_text(val)
            except Exception:
                entity_lookup = {}

        numeric_valid: list[str] = []
        for col in candidate["numeric_cols"]:
            series_num = pd.to_numeric(df_scope[col], errors="coerce")
            if series_num.notna().sum() < MIN_POINTS:
                continue
            try:
                std_val = float(series_num.std(skipna=True))
                if not math.isfinite(std_val) or std_val == 0:
                    continue
            except Exception:
                continue
            df_scope[col] = series_num
            numeric_valid.append(col)

        if candidate["metric_col"] not in numeric_valid:
            metric_series = pd.to_numeric(df_scope[candidate["metric_col"]], errors="coerce")
            if metric_series.notna().sum() >= MIN_POINTS:
                df_scope[candidate["metric_col"]] = metric_series
                numeric_valid.append(candidate["metric_col"])

        numeric_valid = list(dict.fromkeys(numeric_valid))
        if candidate["metric_col"] not in numeric_valid:
            return None

        agg_map_base = {col: _viz_default_agg(col) for col in numeric_valid}
        grouped = _aggregate_by_time(df_scope, "__time__", numeric_valid, agg_map_base)
        target_col = candidate["metric_col"]
        if target_col in grouped.columns:
            grouped = grouped[grouped[target_col].notna()]
        grouped = grouped.dropna(how="all", subset=[col for col in grouped.columns if col != "__time__"])
        if len(grouped) >= MAX_SERIES:
            grouped = grouped.tail(MAX_SERIES)
        grouped = grouped.sort_values("__time__").reset_index(drop=True)
        feature_df = grouped.copy()

        feature_meta: Dict[str, Dict[str, Any]] = {}
        for col in numeric_valid:
            feature_meta[col] = {
                "step": candidate["step"],
                "column": col,
                "label": f"{candidate['step']}.{col}"
            }

        entity_key = _canonical_col(entity["column"]) if entity.get("column") else None
        entity_group = _entity_group(entity_key) if entity_key else None
        entity_value_norm = _normalize_text(entity.get("value")) if entity.get("value") else None

        feature_cap = 36

        def _current_feature_count() -> int:
            return len([c for c in feature_df.columns if c not in ("__time__", target_col)])

        for step_name, df_other in (frames or {}).items():
            if step_name == candidate["step"]:
                continue
            if not isinstance(df_other, pd.DataFrame) or df_other.empty:
                continue
            if _current_feature_count() >= feature_cap:
                break

            df_other_local = df_other.copy()
            if len(df_other_local) > MAX_ROWS:
                df_other_local = df_other_local.tail(MAX_ROWS)

            time_other = None
            for col in TIME_PRIORITY:
                if col in df_other_local.columns:
                    tmp = _coerce_datetime(df_other_local, col)
                    if tmp is not None:
                        tmp = _normalize_month_index(tmp)
                        if tmp.notna().sum() >= MIN_POINTS:
                            time_other = col
                            df_other_local = df_other_local.assign(__time_other__=tmp)
                            break
            if time_other is None:
                for col in df_other_local.columns:
                    tmp = _coerce_datetime(df_other_local, col)
                    if tmp is not None:
                        tmp = _normalize_month_index(tmp)
                        if tmp.notna().sum() >= MIN_POINTS:
                            time_other = col
                            df_other_local = df_other_local.assign(__time_other__=tmp)
                            break
            if time_other is None:
                continue

            df_other_local["__time_other__"] = _normalize_month_index(df_other_local["__time_other__"])
            df_other_local = df_other_local[df_other_local["__time_other__"].notna()]
            if df_other_local.empty:
                continue

            match_col = None
            if entity_key:
                for col in df_other_local.columns:
                    canon = _canonical_col(col)
                    if canon == entity_key or (entity_group is not None and _entity_group(canon) == entity_group):
                        match_col = col
                        break
            if match_col:
                search_value = entity_value_norm or entity_lookup.get(_canonical_col(match_col))
                if search_value:
                    try:
                        col_norm = df_other_local[match_col].astype(str).apply(_normalize_text)
                        df_other_local = df_other_local[col_norm == search_value]
                    except Exception:
                        pass
                    if df_other_local.empty:
                        continue

            numeric_cols_other: list[str] = []
            for col in df_other_local.columns:
                if col in ("__time_other__", time_other):
                    continue
                series_num = pd.to_numeric(df_other_local[col], errors="coerce")
                if series_num.notna().sum() < max(MIN_POINTS, int(len(series_num) * 0.3)):
                    continue
                try:
                    std_val = float(series_num.std(skipna=True))
                    if not math.isfinite(std_val) or std_val == 0:
                        continue
                except Exception:
                    continue
                df_other_local[col] = series_num
                numeric_cols_other.append(col)

            if not numeric_cols_other:
                continue

            remaining_slots = feature_cap - _current_feature_count()
            if remaining_slots <= 0:
                break
            stats = []
            for col in numeric_cols_other:
                try:
                    std_val = float(df_other_local[col].std(skipna=True))
                except Exception:
                    std_val = None
                if std_val is not None and math.isfinite(std_val):
                    stats.append((std_val, col))
            stats.sort(reverse=True, key=lambda x: x[0])
            keep_cols = [col for _, col in stats[:max(1, remaining_slots)]]
            if not keep_cols:
                continue

            agg_map_other = {col: _viz_default_agg(col) for col in keep_cols}
            grouped_other = _aggregate_by_time(
                df_other_local,
                "__time_other__",
                keep_cols,
                agg_map_other,
                output_col="__time__",
            )
            if grouped_other.empty:
                continue
            grouped_other = grouped_other.dropna(
                how="all",
                subset=[col for col in keep_cols if col in grouped_other.columns],
            )
            if grouped_other.empty:
                continue

            rename_map = {}
            for col in keep_cols:
                new_name = f"{step_name}__{col}"
                base_name = new_name
                idx = 2
                while new_name in feature_df.columns or new_name in rename_map.values():
                    new_name = f"{base_name}_{idx}"
                    idx += 1
                rename_map[col] = new_name
                feature_meta[new_name] = {
                    "step": step_name,
                    "column": col,
                    "label": f"{step_name}.{col}"
                }
            grouped_other = grouped_other.rename(columns=rename_map)
            feature_df = feature_df.merge(grouped_other, on="__time__", how="left")

        feature_df = feature_df.sort_values("__time__").reset_index(drop=True)

        metric_series = feature_df[target_col].dropna()
        if metric_series.shape[0] < MIN_POINTS:
            return None

        driver_candidates = [
            col for col in feature_df.columns
            if col not in ("__time__", target_col) and feature_df[col].notna().sum() >= MIN_POINTS
        ]
        drivers_output = []
        target_family = _metric_family(candidate["metric_col"])
        target_norm = _normalize_text(candidate["metric_col"])
        if target_family is None and question_norm:
            for fam, synonyms in _DRIVER_METRIC_SYNONYMS.items():
                matched_family = False
                for syn in synonyms:
                    syn_norm = _normalize_text(syn)
                    if syn_norm and syn_norm in question_norm:
                        target_family = fam
                        matched_family = True
                        break
                if matched_family:
                    break
        target_tokens_primary = _metric_tokens(candidate["metric_col"])
        family_tokens = set()
        if target_family:
            for syn in _DRIVER_METRIC_SYNONYMS.get(target_family, []):
                family_tokens.update(_metric_tokens(syn))
        question_metric_tokens = {
            tok for tok in _metric_tokens(q_text)
            if tok in _DRIVER_METRIC_TOKENS
        }
        fallback_tokens = family_tokens | question_metric_tokens
        skipped_same_metric = 0
        skipped_duplicate_series = 0

        for col in driver_candidates:
            paired = feature_df[["__time__", target_col, col]].dropna()
            if len(paired) < MIN_POINTS:
                continue

            source_info = feature_meta.get(col, {"step": candidate["step"], "column": col})
            source_column = source_info.get("column") or col
            source_norm = _normalize_text(source_column)
            if source_norm == target_norm:
                skipped_same_metric += 1
                continue
            if target_family and _metric_family(source_column) == target_family:
                skipped_same_metric += 1
                continue
            if _is_same_metric_name(source_column, target_norm, target_tokens_primary, fallback_tokens):
                skipped_same_metric += 1
                continue

            # 1% outlier trimming for robust correlation
            paired_trim = _trim_pair_df(paired, target_col, col, p=0.01, min_points=MIN_POINTS)
            vals_target = paired_trim[target_col].to_numpy(dtype=float)
            vals_driver = paired_trim[col].to_numpy(dtype=float)

            corr_main = paired_trim[target_col].corr(paired_trim[col])
            if corr_main is not None and not math.isfinite(corr_main):
                corr_main = None

            # First-difference correlation with trimming
            diff = paired_trim[[target_col, col]].diff().dropna()
            corr_delta = None
            if len(diff) >= MIN_POINTS - 1:
                diff_trim = _trim_pair_df(diff, target_col, col, p=0.01, min_points=MIN_POINTS - 1)
                tmp = diff_trim[target_col].corr(diff_trim[col])
                if tmp is not None and math.isfinite(tmp):
                    corr_delta = float(tmp)

            lag_best = 0
            lag_corr = corr_main
            for lag in range(-3, 4):
                if lag == 0:
                    continue
                lc = _lag_corr(vals_target, vals_driver, lag)
                if lc is None:
                    continue
                if lag_corr is None or abs(lc) > abs(lag_corr):
                    lag_corr = lc
                    lag_best = lag

            corr_abs = max(
                abs(corr_main or 0.0),
                abs(corr_delta or 0.0),
                abs(lag_corr or 0.0)
            )
            if corr_abs < 0.3:
                continue
            if corr_abs > 0.995:
                skipped_duplicate_series += 1
                continue
            if _looks_like_near_duplicate(vals_target, vals_driver, corr_hint=corr_main):
                skipped_duplicate_series += 1
                continue

            diff_full = paired[[target_col, col]].diff()
            events = []
            if diff_full[target_col].notna().any():
                idx_neg = diff_full[target_col].idxmin()
                if idx_neg is not None and pd.notna(diff_full.loc[idx_neg, target_col]):
                    events.append({
                        "period": _format_period(paired.loc[idx_neg, "__time__"]),
                        "target_change": _safe_float(diff_full.loc[idx_neg, target_col]),
                        "driver_change": _safe_float(diff_full.loc[idx_neg, col])
                    })
                idx_pos = diff_full[target_col].idxmax()
                if idx_pos is not None and pd.notna(diff_full.loc[idx_pos, target_col]) and idx_pos != idx_neg:
                    events.append({
                        "period": _format_period(paired.loc[idx_pos, "__time__"]),
                        "target_change": _safe_float(diff_full.loc[idx_pos, target_col]),
                        "driver_change": _safe_float(diff_full.loc[idx_pos, col])
                    })

            drivers_output.append({
                "column": col,
                "label": source_info.get("label", col),
                "source_step": source_info.get("step"),
                "source_column": source_info.get("column"),
                "corr": _safe_float(corr_main),
                "corr_delta": _safe_float(corr_delta),
                "lag": int(lag_best),
                "lag_corr": _safe_float(lag_corr),
                "support": int(len(paired)),
                "score": round(corr_abs, 4),
                "events": events[:2],
                "latest": _safe_float(paired.iloc[-1][col]),
                "earliest": _safe_float(paired.iloc[0][col])
            })

        drivers_output.sort(key=lambda x: x["score"], reverse=True)
        drivers_output = drivers_output[:DRIVER_MAX_RESULTS]

        series_values = feature_df[["__time__", target_col]].dropna()
        trend_direction = "flat"
        target_summary = {"first": None, "last": None, "delta": None, "pct_change": None}
        if not series_values.empty:
            first_val = _safe_float(series_values[target_col].iloc[0])
            last_val = _safe_float(series_values[target_col].iloc[-1])
            if first_val is not None:
                target_summary["first"] = first_val
            if last_val is not None:
                target_summary["last"] = last_val
            if first_val is not None and last_val is not None:
                delta_val = last_val - first_val
                target_summary["delta"] = _safe_float(delta_val)
                if abs(delta_val) > 1e-9:
                    trend_direction = "up" if delta_val > 0 else "down"
                if first_val != 0:
                    try:
                        pct = (delta_val / first_val) * 100.0
                        target_summary["pct_change"] = _safe_float(pct)
                    except Exception:
                        target_summary["pct_change"] = None

        series_records = [
            {
                "period": _format_period(row["__time__"]),
                "value": _safe_float(row[target_col])
            }
            for _, row in series_values.tail(MAX_SERIES).iterrows()
        ]

        horizon_text = None
        try:
            horizon_match = re.search(r"next\s+(\d+)\s+(month|months|quarter|quarters)", q_lower)
        except Exception:
            horizon_match = None
        if horizon_match:
            horizon_text = f"over the next {horizon_match.group(1)} {horizon_match.group(2)}"
        else:
            horizon_text = "in the coming period"

        recommendations: list[str] = []

        if drivers_output:
            summary_chunks: list[str] = []
            for driver in drivers_output[:min(DRIVER_SUMMARY_LIMIT, len(drivers_output))]:
                corr_primary = driver.get("lag_corr")
                if corr_primary is None or not math.isfinite(corr_primary):
                    corr_primary = driver.get("corr")
                if corr_primary is None or not math.isfinite(corr_primary):
                    continue
                label = driver.get("label", driver.get("column"))
                direction = "aligns" if corr_primary >= 0 else "moves opposite"
                lag_val = driver.get("lag")
                lag_text = f", lag={lag_val:+d}" if isinstance(lag_val, int) and lag_val else ""
                summary_chunks.append(
                    f"{label} ({direction}, corr={corr_primary:+.2f}, n={driver.get('support')}{lag_text})"
                )
            if summary_chunks:
                summary_value = f"{candidate['metric_col']} drivers: " + "; ".join(summary_chunks)
            else:
                summary_value = f"No strong drivers detected for {candidate['metric_col']}."
        else:
            summary_value = f"No strong drivers detected for {candidate['metric_col']}."

        for driver in drivers_output:
            corr_primary = driver.get("lag_corr")
            if corr_primary is None or not math.isfinite(corr_primary):
                corr_primary = driver.get("corr")
            if corr_primary is None or not math.isfinite(corr_primary):
                continue
            if abs(corr_primary) < 0.3:
                continue
            label = driver.get("label", driver.get("column"))
            action = "Increase" if corr_primary > 0 else "Reduce"
            impact = "boost" if corr_primary > 0 else "protect"
            lag_val = driver.get("lag")
            lag_text = f", lag={lag_val:+d}" if isinstance(lag_val, int) and lag_val else ""
            rec = f"{action} {label} {horizon_text} to {impact} {candidate['metric_col']} (corr={corr_primary:+.2f}, n={driver.get('support')}{lag_text})."
            if driver.get("events"):
                evt = driver["events"][0]
                if evt and evt.get("period"):
                    tgt_change = evt.get("target_change")
                    drv_change = evt.get("driver_change")
                    if tgt_change is not None and drv_change is not None:
                        tgt_dir = "increase" if tgt_change > 0 else "decline"
                        drv_dir = "rise" if drv_change > 0 else "drop"
                        rec += f" Notably, a {drv_dir} in {label} during {evt.get('period')} aligned with a {tgt_dir} in {candidate['metric_col']}."
            recommendations.append(rec)

        recommendations = recommendations[:3]

        spec_filters = {}
        if entity["column"] and entity["value"] is not None:
            spec_filters[entity["column"]] = entity["value"]

        analysis_notes = []
        driver_feature_count = len(driver_candidates)
        if driver_feature_count:
            analysis_notes.append(
                f"Evaluated {driver_feature_count} potential drivers across {max(0, len(feature_meta) - 1)} metrics."
            )
        if skipped_same_metric:
            noun = "metric" if skipped_same_metric == 1 else "metrics"
            analysis_notes.append(
                f"Skipped {skipped_same_metric} near-duplicate {noun} that matched the target naming."
            )
        if skipped_duplicate_series:
            analysis_notes.append(
                f"Skipped {skipped_duplicate_series} near-identical series to avoid self-drivers."
            )
        if drivers_output and len(drivers_output) > DRIVER_SUMMARY_LIMIT:
            analysis_notes.append(
                f"Identified {len(drivers_output)} drivers; narrative highlights the top {DRIVER_SUMMARY_LIMIT}."
            )
        if not drivers_output:
            analysis_notes.append("No driver exceeded correlation threshold (absolute 0.30).")

        return {
            "spec": {
                "tool": "driver_analysis",
                "step": candidate["step"],
                "column": candidate["metric_col"],
                "group_by": [candidate["time_col"]],
                "filters": spec_filters or None,
                "metric_score": float(candidate["metric_score"])
            },
            "result": {
                "ok": True,
                "value": summary_value,
                "metric": candidate["metric_col"],
                "time_col": candidate["time_col"],
                "trend": trend_direction,
                "entity": {
                    "column": entity["column"],
                    "value": entity["value"],
                    "matched": bool(entity["column"] and entity["matched"])
                },
                "question_target": {
                    "rmid": rm_target.get("rmid_display"),
                    "rm_name": rm_target.get("rm_name")
                },
                "n_periods": int(len(series_values)),
                "target_summary": target_summary,
                "drivers": drivers_output,
                "recommendations": recommendations,
                "series": series_records,
                "notes": analysis_notes
            }
        }

    except Exception as exc:
        try:
            trace("driver_analysis_exception", str(exc)[:MAX_TRACE_PAYLOAD])
        except Exception:
            pass
        return None


def _augment_plan_for_driver(plan: Dict[str, Any],
                             catalog: Dict[str, Any]) -> Dict[str, Any]:
    try:
        if not isinstance(plan, dict):
            return plan
        aug = copy.deepcopy(plan)
        steps = aug.get("steps")
        if not isinstance(steps, list):
            return plan

        try:
            synonyms = build_schema_synonyms(catalog)
        except Exception:
            synonyms = {}

        month_fields = {
            "MEET_MON", "MEET_MONTH", "MONTH", "MONTH_NO", "MONTH_NUM",
            "CAL_MONTH", "FISCAL_MONTH", "MONTH_ID"
        }
        numeric_keywords = {
            "AMOUNT", "AMT", "AUM", "ASSET", "REVENUE", "FEE", "INCOME",
            "PROFIT", "MARGIN", "TOPUP", "TOP_UP", "CONTRIBUTION", "INFLOW",
            "WITHDRAW", "OUTFLOW", "REDEMPT", "PAYOUT", "NNM", "NETNEW",
            "BALANCE", "SENTIMENT", "SCORE", "RETURN", "YIELD", "DELTA",
            "COST", "EXPENSE", "SPEND"
        }
        component_keywords = {"commitment", "positive", "negative", "meeting", "ratio", "achievement", "mandate", "target", "component", "count"}

        def _norm_filters(filters):
            if isinstance(filters, list):
                out = []
                for f in filters:
                    if isinstance(f, dict):
                        out.append({
                            "field": f.get("field"),
                            "op": f.get("op"),
                            "value": f.get("value")
                        })
                return out
            if isinstance(filters, dict):
                out = []
                for k, v in filters.items():
                    if isinstance(v, dict):
                        for op, val in v.items():
                            out.append({"field": k, "op": op, "value": val})
                    else:
                        out.append({"field": k, "op": "=", "value": v})
                return out
            return []

        def _is_numeric_field(field: str, ftypes: Dict[str, str]) -> bool:
            t = str(ftypes.get(field) or ftypes.get(field.upper()) or ftypes.get(field.lower(), "")).upper()
            if any(tok in t for tok in ("NUMBER", "DECIMAL", "NUMERIC", "INT", "FLOAT", "DOUBLE", "REAL")):
                return True
            fu = field.upper()
            return any(tok in fu for tok in numeric_keywords)

        def _normalize_filters_list(filters: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            seen = set()
            for f in (filters or []):
                if not isinstance(f, dict):
                    continue
                fld = f.get("field")
                op = f.get("op")
                val = f.get("value")
                key = (str(fld).upper(), str(op).upper(), json.dumps(val, sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"field": fld, "op": op, "value": val})
            return out

        def _merge_filters(existing: Optional[List[Dict[str, Any]]],
                           additions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            merged = _normalize_filters_list(existing)
            merged.extend(additions)
            return _normalize_filters_list(merged)

        def _slug(text: str) -> str:
            return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')

        view_meta: Dict[str, Dict[str, Any]] = {}
        for view_name, meta in catalog.items():
            fields = meta.get("fields") or []
            types = {str(k): str(v) for k, v in (meta.get("types") or {}).items()}
            view_syn = synonyms.get(view_name, {}) or {}
            alias_map = {}
            if isinstance(view_syn, dict):
                alias_map.update(view_syn.get("__aliases__", {}) or {})
                for key, value in view_syn.items():
                    if key == "__aliases__":
                        continue
                    if isinstance(value, str) and value:
                        alias_map.setdefault(_normalize_token(key), value)
                        alias_map.setdefault(key.lower(), value)
                        alias_map.setdefault(_slug(key), value)
            for field in fields:
                alias_map.setdefault(_normalize_token(field), field)
                alias_map.setdefault(field.lower(), field)
                alias_map.setdefault(_slug(field), field)
            for field, syns in (meta.get("synonyms") or {}).items():
                for syn_val in syns or []:
                    alias_map.setdefault(_normalize_token(syn_val), field)
                    alias_map.setdefault(syn_val.lower(), field)
                    alias_map.setdefault(_slug(syn_val), field)

            join_keys: List[str] = []
            for alias in ("rmid", "rm_name", "mandateid", "clientid", "client_name"):
                actual = alias_map.get(alias) or view_syn.get(alias)
                if actual and actual in fields and actual not in join_keys:
                    join_keys.append(actual)
            if not join_keys:
                for fallback in ("RMID", "RM_NAME", "MANDATEID", "CLIENTID", "CLIENT_NAME"):
                    if fallback in fields and fallback not in join_keys:
                        join_keys.append(fallback)
            time_candidates: List[str] = []
            for alias in ("month_date", "meeting_date", "any_date", "pos_date"):
                actual = alias_map.get(alias) or view_syn.get(alias)
                if actual and actual in fields and actual not in time_candidates:
                    time_candidates.append(actual)
            for fallback in ("MONTH_DATE", "MEETING_DATE", "MEETINGDATE", "AS_OF_DATE", "POS_DATE", "DATE"):
                if fallback in fields and fallback not in time_candidates:
                    time_candidates.append(fallback)
            numeric_fields: List[str] = []
            for field in fields:
                try:
                    if _is_numeric_field(field, types):
                        numeric_fields.append(field)
                except Exception:
                    continue
            view_meta[view_name] = {
                "fields": fields,
                "types": types,
                "syn": view_syn,
                "alias_map": alias_map,
                "join_keys": join_keys,
                "time_candidates": time_candidates,
                "numeric_fields": numeric_fields,
            }

        def _normalize_filters_list(filters: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            seen = set()
            for f in filters or []:
                if not isinstance(f, dict):
                    continue
                fld = f.get("field")
                op = f.get("op")
                val = f.get("value")
                key = (str(fld).upper(), str(op).upper(), json.dumps(val, sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"field": fld, "op": op, "value": val})
            return out

        def _ensure_select_also(step_dict: Dict[str, Any],
                                meta: Dict[str, Any],
                                metric_field: Optional[str]) -> None:
            select_also = step_dict.get("select_also")
            if isinstance(select_also, list):
                extras = []
                seen = set()
                for item in select_also:
                    up = str(item).upper()
                    if up in seen:
                        continue
                    extras.append(item)
                    seen.add(up)
            elif select_also is None:
                extras = []
                seen = set()
            else:
                extras = [select_also]
                seen = {str(select_also).upper()}
            if metric_field and metric_field not in extras:
                extras.append(metric_field)
                seen.add(metric_field.upper())
            for key in meta.get("join_keys", []):
                if key.upper() not in seen:
                    extras.append(key)
                    seen.add(key.upper())
            for candidate in ("MEET_YEAR", "YEAR", "MEET_MON"):
                if candidate in meta.get("fields", []) and candidate.upper() not in seen:
                    extras.append(candidate)
                    seen.add(candidate.upper())
            step_dict["select_also"] = extras

        def _expand_numeric_select(step_dict: Dict[str, Any],
                                   meta: Dict[str, Any],
                                   target: int = 50) -> None:
            select_also = list(step_dict.get("select_also") or [])
            seen = {str(item).upper() for item in select_also}
            for field in meta.get("numeric_fields", []):
                if len(select_also) >= target:
                    break
                fu = field.upper()
                if fu in seen:
                    continue
                select_also.append(field)
                seen.add(fu)
            step_dict["select_also"] = select_also

        existing_signatures = set()
        context_filters: List[Dict[str, Any]] = []
        processed_steps: List[Dict[str, Any]] = []

        for step in steps:
            if not isinstance(step, dict):
                continue
            step = copy.deepcopy(step)
            view = step.get("view")
            meta = view_meta.get(view, {})
            fields = meta.get("fields", [])
            types = meta.get("types", {})

            filters = _norm_filters(step.get("filters"))
            new_filters = []
            for filt in filters:
                field = str(filt.get("field") or "")
                op = str(filt.get("op") or "").strip().lower()
                if field.upper() in month_fields and op in ("=", "=="):
                    continue
                new_filters.append(filt)
            if new_filters:
                step["filters"] = new_filters
            else:
                step.pop("filters", None)

            normalized_filters = _normalize_filters_list(new_filters)
            for filt in normalized_filters:
                field_u = str((filt or {}).get("field") or "").upper()
                if field_u in {"RMID", "RM_ID", "RM_NAME", "MANDATEID", "MANDATE_ID", "CLIENTID", "CLIENT_ID", "CLIENT_NAME"}:
                    context_filters.append(filt)

            time_field = (step.get("time") or {}).get("field")
            if not time_field:
                for candidate in meta.get("time_candidates", []):
                    time_field = candidate
                    break
            if time_field:
                step["time"] = {"field": time_field, "grain": "month"}

            metric_field = (step.get("metric") or {}).get("field")
            _ensure_select_also(step, meta, metric_field if isinstance(metric_field, str) else None)
            _expand_numeric_select(step, meta)

            if view and isinstance(metric_field, str):
                existing_signatures.add((view, metric_field.upper()))
            processed_steps.append(step)

        steps[:] = processed_steps
        context_filters = _normalize_filters_list(context_filters)

        DRIVER_EXTRA_METRICS = [
            ("topup", "sum"),
            ("withdrawal", "sum"),
            ("nnm", "sum"),
            ("revenue", "sum"),
            ("profit", "sum"),
            ("sentiment", "avg"),
            ("score", "avg"),
            ("meetings", "sum"),
            ("transactions", "sum"),
            ("cost", "sum"),
            ("commitment", "sum"),
            ("clients", "sum"),
            ("duration", "avg"),
        ]

        MAX_DRIVER_STEPS = 16
        base_steps = list(steps)
        base_views = {s.get("view") for s in base_steps if isinstance(s, dict)}

        def _add_step(template: Optional[Dict[str, Any]],
                      view_name: Optional[str],
                      metric_field: str,
                      agg_mode: Optional[str],
                      suffix: str) -> None:
            if not view_name or view_name not in view_meta:
                return
            meta = view_meta[view_name]
            fields = meta.get("fields", [])
            types = meta.get("types", {})
            if metric_field not in fields:
                return
            signature = (view_name, metric_field.upper())
            if signature in existing_signatures:
                return
            try:
                if not _is_numeric_field(metric_field, types):
                    return
            except Exception:
                return

            if template:
                new_step = copy.deepcopy(template)
                base_id = template.get("id") or view_name.replace(".", "_")
            else:
                new_step = {"view": view_name}
                base_id = view_name.replace(".", "_")

            suffix_clean = re.sub(r"[^A-Za-z0-9_]+", "_", suffix or metric_field)
            new_step["id"] = f"{base_id}__{suffix_clean}"

            metric_cfg = new_step.get("metric") if isinstance(new_step.get("metric"), dict) else {}
            new_step["metric"] = metric_cfg
            metric_cfg["field"] = metric_field
            metric_cfg["mode"] = agg_mode or smart_default_agg_for_metric(metric_field)

            if not template:
                join_keys = meta.get("join_keys", [])
                if join_keys:
                    new_step["dim"] = {"field": join_keys[0]}
                else:
                    new_step.pop("dim", None)
                time_candidates = meta.get("time_candidates", [])
                if time_candidates:
                    new_step["time"] = {"field": time_candidates[0], "grain": "month"}

            _ensure_select_also(new_step, meta, metric_field)
            _expand_numeric_select(new_step, meta)

            if context_filters:
                new_step["filters"] = _merge_filters(new_step.get("filters"), context_filters)

            steps.append(new_step)
            existing_signatures.add(signature)

        for base_step in base_steps:
            if len(steps) >= MAX_DRIVER_STEPS:
                break
            if not isinstance(base_step, dict):
                continue
            view = base_step.get("view")
            meta = view_meta.get(view, {})
            alias_map = meta.get("alias_map", {})
            syn_map = meta.get("syn", {})
            for canon, mode in DRIVER_EXTRA_METRICS:
                field = syn_map.get(canon)
                if not field:
                    key_norm = _normalize_token(canon)
                    field = alias_map.get(key_norm) or alias_map.get(canon.lower())
                if field:
                    _add_step(base_step, view, field, mode, canon)
                if len(steps) >= MAX_DRIVER_STEPS:
                    break
            if len(steps) >= MAX_DRIVER_STEPS:
                break
            metric_field = (base_step.get("metric") or {}).get("field")
            alias_items = list(alias_map.items())
            for alias, field in alias_items:
                if len(steps) >= MAX_DRIVER_STEPS:
                    break
                if not field or field == metric_field:
                    continue
                alias_norm = alias.lower()
                if len(alias_norm) < 4:
                    continue
                if not any(tok in alias_norm for tok in component_keywords):
                    continue
                agg_guess = "avg" if any(tok in alias_norm for tok in ("ratio", "score", "achievement", "target", "component")) else smart_default_agg_for_metric(field)
                _add_step(base_step, view, field, agg_guess, alias)

        for base_step in base_steps:
            if len(steps) >= MAX_DRIVER_STEPS:
                break
            if not isinstance(base_step, dict):
                continue
            view = base_step.get("view")
            meta = view_meta.get(view, {})
            metric_field = (base_step.get("metric") or {}).get("field")
            dim_field = (base_step.get("dim") or {}).get("field")
            fallback_candidates = []
            for field in meta.get("numeric_fields", []):
                if field == metric_field or field == dim_field:
                    continue
                signature = (view, field.upper())
                if signature in existing_signatures:
                    continue
                fallback_candidates.append(field)
            for field in fallback_candidates[:6]:
                if len(steps) >= MAX_DRIVER_STEPS:
                    break
                _add_step(base_step, view, field, smart_default_agg_for_metric(field), f"auto_{field}")

        for view_name, meta in view_meta.items():
            if len(steps) >= MAX_DRIVER_STEPS:
                break
            if view_name in base_views:
                continue
            added = False
            alias_map = meta.get("alias_map", {})
            syn_map = meta.get("syn", {})
            for canon, mode in DRIVER_EXTRA_METRICS:
                field = syn_map.get(canon)
                if not field:
                    key_norm = _normalize_token(canon)
                    field = alias_map.get(key_norm) or alias_map.get(canon.lower())
                if not field:
                    continue
                _add_step(None, view_name, field, mode, canon)
                added = True
                if len(steps) >= MAX_DRIVER_STEPS:
                    break
            if len(steps) >= MAX_DRIVER_STEPS:
                break
            if len(steps) < MAX_DRIVER_STEPS:
                for alias, field in alias_map.items():
                    if len(steps) >= MAX_DRIVER_STEPS:
                        break
                    if not field:
                        continue
                    alias_norm = alias.lower()
                    if len(alias_norm) < 4:
                        continue
                    if not any(tok in alias_norm for tok in component_keywords):
                        continue
                    agg_guess = "avg" if any(tok in alias_norm for tok in ("ratio", "score", "achievement", "target", "component")) else smart_default_agg_for_metric(field)
                    _add_step(None, view_name, field, agg_guess, alias)
                    added = True
            if len(steps) >= MAX_DRIVER_STEPS:
                break
            if not added:
                for field in meta.get("numeric_fields", [])[:5]:
                    if len(steps) >= MAX_DRIVER_STEPS:
                        break
                    _add_step(None, view_name, field, smart_default_agg_for_metric(field), f"auto_{field}")

        if len(steps) > MAX_DRIVER_STEPS:
            steps[:] = steps[:MAX_DRIVER_STEPS]

        return aug
    except Exception:
        return plan
import re
import traceback



# Recognize explicit 4-digit years and relative time phrases
# Recognize explicit 4-digit years and relative time phrases (keep your regexes above)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
RELATIVE_TIME_RE = re.compile(r"\b(last|past|previous|ytd|year to date|last\s+\d+\s+months|last\s+year|this\s+year)\b", re.I)

# NEW month patterns
MONTH_NAME_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+(\d{4})\b", re.I)
MONTH_ONLY_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b", re.I)
YM_RE = re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b")       # 2025-07, 2025/7
MY_RE = re.compile(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b")       # 07/2025, 7-2025

_MONTHS = {m:i for i,m in enumerate(
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], start=1)}

def _extract_time_hints(q: str) -> dict:
    """Return explicit years, and a single-month hint if present."""
    q = q or ""
    years = sorted({int(y) for y in YEAR_RE.findall(q)})

    m_y = None
    month_only = None
    m = MONTH_NAME_RE.search(q)
    if m:
        mon = _MONTHS[m.group(1)[:3].lower()]
        yr = int(m.group(2))
        m_y = {"month": mon, "year": yr}
    else:
        m = YM_RE.search(q) or MY_RE.search(q)
        if m:
            # unify capture groups
            if m.re is YM_RE:
                yr, mon = int(m.group(1)), int(m.group(2))
            else:
                mon, yr = int(m.group(1)), int(m.group(2))
            m_y = {"month": mon, "year": yr}
        else:
            m2 = MONTH_ONLY_RE.search(q)
            if m2:
                mon = _MONTHS[m2.group(1)[:3].lower()]
                month_only = {"month": mon}

    return {
        "explicit_years": years,
        "min_year": (min(years) if years else None),
        "max_year": (max(years) if years else None),
        "single_month": m_y,      # {"month": 7, "year": 2025} or None
        "month_only": month_only, # {"month": 11} or None
    }

def _extract_category_filters(q: str) -> tuple[str|None, str|None]:
    """
    Returns (field, value) for a single detected category.
    Field: CURRENT_MANDATE_CATEGORY by default; OUTPUT_LABEL if 'forecast/next month' is in the question.
    """
    if not q: return (None, None)
    ql = q.lower()
    cats = {
        "low": "LOW", "medium": "MEDIUM", "high": "HIGH",
        "upgrade": "UPGRADE", "downgrade": "DOWNGRADE",
        "no change": "NO CHANGE", "no-change": "NO CHANGE", "nochange": "NO CHANGE",
        "stable": "NO CHANGE"
    }
    chosen = None
    for k, v in cats.items():
        if k in ql:
            # Avoid capturing a *list* of categories; we optimize for the common case (one)
            if chosen and chosen != v: 
                return (None, None)
            chosen = v
    if not chosen:
        return (None, None)
    field = "OUTPUT_LABEL" if ("forecast" in ql or "next month" in ql) else "CURRENT_MANDATE_CATEGORY"
    return (field, chosen)


from datetime import datetime
import time


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

@st.cache_resource
def _rai_builder_and_specs(rebuild_token: str = ""):
    debug_init = os.environ.get("AI_INSIGHTS_DEBUG_INIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] ai_insights _rai_builder_and_specs: start\n")
        except Exception:
            debug_init = False
    t0 = time.perf_counter() if debug_init else 0.0
    ensure_rai_config()
    force_rebuild = os.environ.get("AI_INSIGHTS_FORCE_KG_REBUILD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if force_rebuild:
        try:
            load_ai_insights_specs.cache_clear()
        except Exception:
            pass
        try:
            build_ai_insights_builder.cache_clear()
        except Exception:
            pass
    specs = load_ai_insights_specs()
    if specs and not getattr(specs[0], "field_exprs", None):
        load_ai_insights_specs.cache_clear()
        specs = load_ai_insights_specs()
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write(f"[INIT] ai_insights _rai_builder_and_specs: specs={len(specs)}\n")
        except Exception:
            pass
    builder = build_ai_insights_builder()
    if debug_init:
        try:
            import sys as _sys
            elapsed = time.perf_counter() - t0
            _sys.stderr.write(f"[INIT] ai_insights _rai_builder_and_specs: done in {elapsed:.2f}s\n")
        except Exception:
            pass
    return builder, specs


def _is_lqp_internal_error(exc: Exception) -> bool:
    """Heuristic detection for LQP internal errors that benefit from legacy fallback."""
    text_parts = [str(exc)]
    for attr in ("content", "raw_content"):
        val = getattr(exc, attr, None)
        if val:
            text_parts.append(str(val))
    problem = getattr(exc, "problem", None)
    if isinstance(problem, dict):
        report = problem.get("report")
        message = problem.get("message")
        if report:
            text_parts.append(str(report))
        if message:
            text_parts.append(str(message))
    text = " ".join([t for t in text_parts if t]).lower()
    return "internal system error" in text


def _ensure_lqp_fallback_suffix() -> str:
    suffix = os.environ.get("AI_INSIGHTS_MODEL_NAME_SUFFIX", "").strip()
    if not suffix:
        suffix = uuid.uuid4().hex[:8]
        os.environ["AI_INSIGHTS_MODEL_NAME_SUFFIX"] = suffix
    return suffix


def _switch_builder_to_legacy(builder: Any, log_items: Optional[List[str]] = None) -> tuple[Any, bool]:
    if os.environ.get("RAI_USE_LQP", "").strip().lower() in ("0", "false", "no"):
        return builder, False
    suffix = _ensure_lqp_fallback_suffix()
    try:
        new_builder = build_ai_insights_builder(use_lqp=False, model_suffix=suffix)
    except Exception:
        return builder, False
    if log_items is not None:
        log_items.append("LQP internal error detected; retrying with legacy backend (use_lqp=False).")
    return new_builder, True

# ── RBAC helpers (mirrors RM Performance / Client Profitability) ─────────────
RBAC_TABLE = "vw_user_mapping"

def _current_rmid_from_session() -> str | None:
    try:
        from shared import session as shared_session
        ident = shared_session._identity_info() or {}
    except Exception:
        ident = {}
    who = (
        ident.get("header_user")
        or ident.get("caller_user")
        or (st.session_state.get("user", {}).get("email") if isinstance(st.session_state.get("user"), dict) else None)
        or ident.get("sf_user")
        or None
    )
    if not who:
        return None
    u = str(who)
    if "@" in u:
        u = u.split("@", 1)[0]
    return u.strip().upper()

def _role_and_scope_from_mapping(sess, user_rmid: str):
    if not user_rmid:
        return "OTHER", [], []
    try:
        user_num = int(str(user_rmid).strip())
    except Exception:
        return "OTHER", [], []
    try:
        m = sess.sql(f"""
            select
              RMID::number            as RMID,
              REPORTS_TO_RMID::number as REPORTS_TO_RMID
            from {RBAC_TABLE}
        """).to_pandas()
    except Exception:
        return "OTHER", [], []
    if m.empty:
        return "OTHER", [], []

    present_in_rmid      = (m["RMID"] == user_num).any()
    present_in_reportsTo = (m["REPORTS_TO_RMID"] == user_num).any()
    if not present_in_rmid and not present_in_reportsTo:
        return "OTHER", [], []
    role = "SRM" if present_in_reportsTo else "RM"

    from collections import defaultdict, deque
    children = defaultdict(list)
    for rmid, mgr in zip(m["RMID"].tolist(), m["REPORTS_TO_RMID"].tolist()):
        if mgr is not None:
            try:
                children[int(mgr)].append(int(rmid))
            except Exception:
                continue

    if role == "RM":
        scope_nums = {user_num}
    else:
        seeds = [user_num] if present_in_rmid else children.get(user_num, [])
        scope_nums = set()
        q = deque(seeds)
        while q:
            x = q.popleft()
            if x in scope_nums:
                continue
            scope_nums.add(x)
            q.extend(children.get(x, []))
        scope_nums.add(user_num)

    scope_rmids = [str(int(x)) for x in sorted(scope_nums)]
    return role, scope_rmids, []

def _mandates_from_scope(sess, scope_rmids: list[str]) -> list[str]:
    if not scope_rmids:
        return []
    try:
        scope_nums = [int(s) for s in scope_rmids]
    except Exception:
        scope_nums = []
    q = """
        WITH latest AS (
          SELECT *
          FROM tfo.tfo_schema.rmtomandate
          WHERE create_date = (SELECT max(create_date) FROM tfo.tfo_schema.rmtomandate)
        )
        SELECT RMID::number as RMID, MANDATEID::string as MANDATEID
        FROM latest
    """
    df_map = sess.sql(q).to_pandas()
    if df_map.empty:
        return []
    if scope_nums:
        df_map = df_map[df_map["RMID"].isin(scope_nums)]
    else:
        df_map = df_map[df_map["RMID"].astype(str).isin(scope_rmids)]
    return sorted(df_map["MANDATEID"].astype(str).dropna().unique().tolist())
# ─────────────────────────────────────────────────────────────────────────────


# --- Prefer shared session helpers (use shared/session.py when available) ---
try:
    from shared import session as shared_session
    get_session = getattr(shared_session, "get_session", None)
    render_login = getattr(shared_session, "render_login", lambda *a, **k: None)
    render_identity_banner = getattr(shared_session, "render_identity_banner", lambda *a, **k: None)
except Exception:
    # graceful fallback if shared.session can't be imported in this environment
    shared_session = None
    get_session = None
    render_login = lambda *a, **k: None
    render_identity_banner = lambda *a, **k: None

# removed st_aggrid usage per request

# Snowflake helpers
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session

# ──────────────────────────────────────────────────────────────────────────────
# Config
def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [v.strip() for v in raw.split(",") if v.strip()]

DEFAULT_VIEWS = _env_list(
    "AI_INSIGHTS_VIEWS",
    [
        "TFO_TEST.ML_TEST.RAI_RM_MONTHLY_SUMMARY_MODEL",
        "TFO_TEST.ML_TEST.RAI_MANDATE_MONTHLY_SUMMARY_MODEL",
        "TFO_TEST.ML_TEST.RAI_MEETING_DETAIL_MODEL",
    ],
)

def _semantic_kind_for_name(name: str) -> str:
    forced = os.environ.get("AI_INSIGHTS_SEMANTIC_KIND", "").strip().lower()
    name_u = (name or "").upper()
    if forced.startswith("model"):
        return "model"
    if forced.startswith("view"):
        if name_u.endswith("_MODEL"):
            return "model"
        return "view"
    if name_u.endswith("_MODEL"):
        return "model"
    return "view"

def _describe_semantic_sql_variants(name: str, quoted: bool) -> List[str]:
    kinds: List[str] = []
    forced = os.environ.get("AI_INSIGHTS_SEMANTIC_KIND", "").strip().lower()
    if forced.startswith("model"):
        kinds = ["model"]
    elif forced.startswith("view"):
        kinds = ["view"]
    else:
        kinds = ["view", "model"]

    name_u = (name or "").upper()
    if name_u.endswith("_MODEL") and "model" not in kinds:
        kinds.append("model")
    if name_u.endswith("_MODEL"):
        kinds = ["model"] + [k for k in kinds if k != "model"]

    ident = name
    if quoted:
        ident = '"' + name.replace('.', '"."') + '"'

    stmts: List[str] = []
    for kind in kinds:
        keyword = "SEMANTIC MODEL" if kind == "model" else "SEMANTIC VIEW"
        stmts.append(f"DESCRIBE {keyword} {ident}")
    return stmts

def _describe_semantic_df(session, name: str):
    last_exc = None
    for quoted in (False, True):
        for stmt in _describe_semantic_sql_variants(name, quoted):
            try:
                return session.sql(stmt).to_pandas()
            except Exception as exc:
                last_exc = exc
                continue
    if last_exc:
        raise last_exc

def _describe_semantic_rows(session, name: str):
    last_exc = None
    for quoted in (False, True):
        for stmt in _describe_semantic_sql_variants(name, quoted):
            try:
                return session.sql(stmt).collect()
            except Exception as exc:
                last_exc = exc
                continue
    if last_exc:
        raise last_exc
CORTEX_LLM_MODEL = "openai-gpt-5-chat"
RM_LOOKUP_LIMIT = 5000
MAX_TRACE_PAYLOAD = 2400

# Data quality & follow-ups
MIN_ROWS_THRESHOLD = 30
MIN_UNIQUE_GROUPS = 6
FOLLOWUP_ALLOWED_ROUNDS = 1

# Join keys preference order
_DEFAULT_JOIN_KEYS_PREF = ["RMID", "MANDATEID", "MANDATEID_STR", "RM_NAME", "MONTH_DATE", "MEET_YEAR"]
JOIN_KEYS_PREF = _analysis_list("join_keys_preference", _DEFAULT_JOIN_KEYS_PREF)

# Planner: base synonyms
BASE_SYNONYMS = {
    "aum": ["aum", "assets under management", "current_aum", "current aum", "snap_current_aum"],
    "performance": ["performance", "RM score", "score"],
    "revenue": ["revenue", "income", "fee", "fees"],
    "profit": ["profit", "margin", "net income"],
    "topup": ["topup", "inflow", "contribution", "deposit"],
    "meetings": ["meeting", "meetings", "mtgs", "appointments", "calls"],
    "sentiment": ["sentiment", "tone", "positive", "negative", "score"],
    "transactions": ["transaction", "txn", "txns", "purchase", "sale", "count"],
}
DEFAULT_MODE = {
    "aum": "snapshot",
    "performance": "avg",
    "revenue": "sum",
    "profit": "sum",
    "topup": "sum",
    "meetings": "count",
    "sentiment": "avg",
    "transactions": "count",
}

def _viz_default_agg(col: str) -> str:
    u = (col or "").upper()
    if any(k in u for k in ("AUM", "SNAP")):
        return "last"
    if any(k in u for k in ("PERFORMANCE", "SENTIMENT", "SCORE", "RATE", "AVG")):
        return "avg"
    if any(k in u for k in ("REVEN", "PROFIT", "AMOUNT", "AMT", "FEE", "TOPUP", "INFLOW", "OUTFLOW", "COST", "EXPENSE", "SPEND")):
        return "sum"
    if any(k in u for k in ("COUNT", "NUM")):
        return "count"
    return "sum"

# ──────────────────────────────────────────────────────────────────────────────
PROJECTION_METRIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "meetings": (
        "meeting", "meetings", "client meeting", "rm meeting", "touch", "touchpoint", "touchpoints",
        "meeting target", "meeting count target", "total meetings"
    ),
    "aum": (
        "aum", "asset under management", "assets under management", "aum balance", "aum value",
        "total aum", "avg monthly aum"
    ),
    "revenue": (
        "revenue", "revenues", "income", "fee", "fees", "turnover", "sales", "buy amount", "purchases"
    ),
    "profit": (
        "profit", "profits", "profitability", "net profit", "net income", "margin", "margins", "p&l", "earnings",
        "profit percent", "profit percentage"
    ),
    "cost": (
        "cost", "costs", "rm cost", "rm costs", "relationship manager cost", "expense", "expenses", "spend", "spending",
        "sell amount", "redemptions", "withdrawal"
    ),
    "headcount": (
        "headcount", "fte", "full time", "staff", "employee", "employees", "rm count", "team size"
    ),
    "volume": (
        "volume", "volumes", "transactions", "transaction count", "deal count", "topup count", "commitment count"
    ),
    "mandates": (
        "mandates", "mandate count", "total mandates", "mandates this month", "monthly mandates"
    ),
    "topup": (
        "topup", "top-up", "topups", "top-up amount", "topup amount", "cash in", "cash_in", "cash inflow", "inflows",
        "contribution", "deposit", "topupcommitment", "top up commitment", "top-up commitment"
    ),
    "nnm": (
        "nnm", "total nnm", "net new money", "net-new money", "nnm target", "nnmtarget"
    ),
    "performance": (
        "performance", "performance score", "rm score", "score", "rm performance", "composite score"
    ),
    "sentiment": (
        "sentiment", "sentiment score", "meeting sentiment", "positive meetings", "negative meetings", "neutral meetings"
    ),
    "commitment": (
        "commitment", "commitments", "first commitment", "topup commitment", "commitment target", "commitment amount"
    ),
    "clients": (
        "client", "clients", "client target", "clienttarget", "total clients", "mandates per month", "dealname count"
    ),
    "duration": (
        "duration", "relationship duration", "total duration", "tenure", "duration in months"
    ),
}

PROJECTION_METRIC_KEYWORDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "meetings": (
        ("MEETING",),
        ("VISIT",),
        ("TOUCH",),
        ("MEETING_COUNT_TARGET",),
        ("TOTAL_MEETINGS",),
        ("MEETING", "TARGET"),
    ),
    "aum": (
        ("AUM",),
        ("ASSET", "MANAGEMENT"),
        ("SNAP", "AUM"),
        ("TOTAL_AUM",),
        ("AVG", "AUM"),
    ),
    "revenue": (
        ("REVENUE",),
        ("INCOME",),
        ("FEE",),
        ("BUY",),
        ("PURCHASE",),
    ),
    "profit": (
        ("PROFIT",),
        ("MARGIN",),
        ("NET", "INCOME"),
        ("PROFIT_PERCENT",),
    ),
    "cost": (
        ("COST",),
        ("EXPENSE",),
        ("SPEND",),
        ("RM", "COST"),
        ("SELL",),
        ("REDEMPTION",),
    ),
    "headcount": (
        ("HEADCOUNT",),
        ("FTE",),
        ("STAFF",),
        ("EMPLOYEE",),
    ),
    "volume": (
        ("VOLUME",),
        ("TRANSACTION",),
        ("COUNT",),
    ),
    "mandates": (
        ("MANDATE",),
        ("CLIENTTARGET",),
        ("TOTAL_CLIENTS",),
        ("DEALNAME",),
    ),
    "topup": (
        ("TOPUP",),
        ("TOP", "UP"),
        ("CASH_IN",),
        ("CASH", "IN"),
        ("INFLOW",),
        ("DEPOSIT",),
        ("CONTRIBUTION",),
    ),
    "nnm": (
        ("NNM",),
        ("NET", "NEW", "MONEY"),
    ),
    "performance": (
        ("PERFORMANCE",),
        ("SCORE",),
        ("RM", "SCORE"),
    ),
    "sentiment": (
        ("SENTIMENT",),
        ("POSITIVE", "CNT"),
        ("NEGATIVE", "CNT"),
        ("NEUTRAL", "CNT"),
    ),
    "commitment": (
        ("COMMITMENT",),
        ("FIRST", "COMMITMENT"),
        ("COMMITMENTTARGET",),
        ("TOPUPCOMMITMENT",),
    ),
    "clients": (
        ("CLIENTTARGET",),
        ("TOTAL_CLIENTS",),
        ("CLIENT",),
        ("DEALNAME", "COUNT"),
    ),
    "duration": (
        ("DURATION",),
        ("TENURE",),
        ("MONTHS", "DURATION"),
    ),
}

METRIC_DISPLAY_NAMES: dict[str, str] = {
    "meetings": "meetings",
    "aum": "AUM",
    "revenue": "revenue",
    "profit": "profit",
    "cost": "RM cost",
    "headcount": "headcount",
    "volume": "transaction volume",
    "mandates": "mandate count",
    "topup": "top-up amount",
    "nnm": "net new money",
    "performance": "performance score",
    "sentiment": "sentiment balance",
    "commitment": "commitment amount",
    "clients": "client count",
    "duration": "relationship duration",
}

_PROJECTION_INCREASE_WORDS = {
    "increase", "increases", "increased", "grow", "grows", "growth",
    "raise", "raises", "boost", "boosts", "add", "adds", "higher"
}
_PROJECTION_DECREASE_WORDS = {
    "decrease", "decreases", "decreased", "reduce", "reduces", "reduced",
    "lower", "drop", "drops", "decline", "declines", "cut", "cuts"
}
_PROJECTION_IMPACT_WORDS = {
    "impact", "impacts", "affect", "affects", "effect", "effects",
    "influence", "influences", "result", "results"
}


def _metric_display_name(canonical: str) -> str:
    return METRIC_DISPLAY_NAMES.get(canonical, canonical.replace("_", " "))


def _safe_float(val):
    try:
        import math
        if val is None:
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None

def _best_metric_column_general(df, keyword_groups: tuple[tuple[str, ...], ...]):
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    best_col = None
    best_score = float("-inf")
    for col in df.columns:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
        if series.notna().sum() < 3:
            continue
        name = str(col).upper()
        score = 0.0
        if any(bad in name for bad in ("PCT", "PERCENT", "RATIO", "RATE")):
            score -= 1.0
        for group in keyword_groups:
            if all(token in name for token in group):
                score += 2.0 * len(group)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col



def _time_key_series(df):
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    upper_map = {str(c).upper(): c for c in df.columns}
    datetime_candidates = (
        # generic
        "MONTH_DATE",
        "MEET_MONTH_DATE",
        "MEETING_MONTH",
        "MEETING_DATE",
        "MEETINGDATE",
        "MONTH_DT",
        "MONTHDATE",
        "MONTHDAY",
        "MONTH",
        "MONTH_START",
        "MONTHBEGIN",
        "MONTH_END",
        "MONTHEND",
        "MONTH_KEY",
        "MONTHKEY",
        "MONTH_YEAR",
        "YEAR_MONTH",
        "YEARMONTH",
        "PERIOD",
        "DATE",
        "AS_OF_DATE",
        # POS variants
        "POS_DATE",
        "POSDATE",
        "POS_DT",
    )
    for cand in datetime_candidates:
        col = upper_map.get(cand)
        if not col:
            continue
        try:
            raw = df[col]
        except Exception:
            raw = None
        if raw is None:
            continue
        try:
            if pd.api.types.is_period_dtype(raw):
                series = raw.astype("period[M]").to_timestamp()
                if series.notna().sum() >= max(3, series.shape[0] // 2 or 1):
                    return series.dt.to_period("M").astype(str)
        except Exception:
            pass
        try:
            series = pd.to_datetime(raw, errors="coerce")
            if series.notna().sum() >= max(3, series.shape[0] // 2 or 1):
                return series.dt.to_period("M").astype(str)
        except Exception:
            pass
        try:
            as_str = pd.Series(raw, copy=False).astype(str)
            for fmt in ("%Y-%m", "%Y/%m", "%Y%m", "%b-%Y", "%b %Y"):
                series = pd.to_datetime(as_str, format=fmt, errors="coerce")
                if series.notna().sum() >= max(3, series.shape[0] // 2 or 1):
                    return series.dt.to_period("M").astype(str)
        except Exception:
            pass
        try:
            numeric_series = pd.Series(raw, copy=False)
            numeric = pd.to_numeric(numeric_series, errors="coerce")
            valid_numeric = numeric.notna().sum()
            if valid_numeric >= max(3, len(numeric) // 2 or 1):
                series = pd.to_datetime(numeric.astype("Int64"), format="%Y%m", errors="coerce")
                if series.notna().sum() >= max(3, series.shape[0] // 2 or 1):
                    return series.dt.to_period("M").astype(str)
        except Exception:
            pass
        except Exception:
            continue

    year_col = None
    month_col = None
    for key in ("MEET_YEAR", "YEAR", "CAL_YEAR", "FISCAL_YEAR", "POS_YEAR"):
        if key in upper_map:
            year_col = upper_map[key]
            break
    for key in ("MEET_MON", "MEETING_MONTH", "MONTH", "CAL_MONTH", "FISCAL_MONTH", "POS_MON", "POS_MONTH"):
        if key in upper_map:
            month_col = upper_map[key]
            break
    if year_col and month_col:
        try:
            year_series = pd.to_numeric(df[year_col], errors="coerce")
            month_series = pd.to_numeric(df[month_col], errors="coerce")
            series = pd.to_datetime({
                "year": year_series.astype("Int64"),
                "month": month_series.astype("Int64"),
                "day": 1,
            }, errors="coerce")
            if series.notna().any():
                return series.dt.to_period("M").astype(str)
        except Exception:
            pass
    return None


def _parse_projection_question(question: str) -> dict[str, object]:
    q = (question or "").lower()
    intent_terms = (
        "what if",
        "scenario",
        "projection",
        "project",
        "impact",
        "affect",
        "effect",
        "sensitivity",
        "would",
        "if ",
    )
    if "%" not in q and not any(term in q for term in intent_terms):
        return {"drivers": [], "targets": [], "delta_pct": 0.10}
    mentioned: list[tuple[str, int]] = []
    drivers: set[str] = set()
    targets: set[str] = set()
    decreased = False

    def _window(text: str, start: int, end: int, radius: int = 20) -> tuple[str, str]:
        before = text[max(0, start - radius):start]
        after = text[end:end + radius]
        return before, after

    for canonical, synonyms in PROJECTION_METRIC_SYNONYMS.items():
        for syn in synonyms:
            pattern = r"\b" + re.escape(syn) + r"s?\b"
            for match in re.finditer(pattern, q):
                mentioned.append((canonical, match.start()))
                before, after = _window(q, match.start(), match.end())
                context = before + " " + after
                if any(word in context for word in _PROJECTION_INCREASE_WORDS):
                    drivers.add(canonical)
                if any(word in context for word in _PROJECTION_DECREASE_WORDS):
                    drivers.add(canonical)
                    decreased = True
                if any(word in context for word in _PROJECTION_IMPACT_WORDS):
                    targets.add(canonical)
                if "profit" in syn or "profit" in context or "profitability" in context:
                    targets.add("profit")

    if "profitability" in q or "profit" in q:
        targets.add("profit")
    if "margin" in q:
        targets.add("profit")

    percent = None
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", q):
        try:
            percent = float(match.group(1)) / 100.0
        except Exception:
            continue
        before, after = _window(q, match.start(), match.end())
        context = before + " " + after
        if any(word in context for word in _PROJECTION_DECREASE_WORDS):
            percent = -abs(percent)
        elif any(word in context for word in _PROJECTION_INCREASE_WORDS):
            percent = abs(percent)
        break
    if percent is None:
        percent = 0.10
    if decreased and percent > 0:
        percent = -percent

    mentioned_sorted = sorted(mentioned, key=lambda x: x[1])
    if any(word in q for word in _PROJECTION_INCREASE_WORDS | _PROJECTION_DECREASE_WORDS):
        adjustable_metrics = {"meetings", "aum", "cost", "headcount", "volume", "mandates"}
        for metric, _pos in mentioned_sorted:
            if metric in adjustable_metrics:
                drivers.add(metric)

    if not drivers and mentioned_sorted:
        drivers.add(mentioned_sorted[0][0])
    if not targets:
        remaining = [m for m, _ in mentioned_sorted if m not in drivers]
        if remaining:
            targets.update(remaining)
    if not targets:
        targets.add("profit")
    if any(d in targets for d in list(drivers)):
        targets = {t for t in targets if t not in drivers} or {"profit"}

    # Preserve mention order for drivers/targets
    drivers_ordered: list[str] = []
    for metric, _ in mentioned_sorted:
        if metric in drivers and metric not in drivers_ordered:
            drivers_ordered.append(metric)
    if not drivers_ordered:
        drivers_ordered = list(drivers)

    targets_ordered: list[str] = []
    for metric, _ in mentioned_sorted:
        if metric in targets and metric not in targets_ordered:
            targets_ordered.append(metric)
    if not targets_ordered:
        targets_ordered = list(targets)

    drivers_final = [d for d in drivers_ordered if d not in targets_ordered]
    if not drivers_final:
        fallback = [m for m, _ in mentioned_sorted if m not in targets_ordered]
        if fallback:
            drivers_final = [fallback[0]]
        elif drivers_ordered:
            drivers_final = [drivers_ordered[0]]
        else:
            drivers_final = list(drivers)[:1]

    if not targets_ordered:
        targets_ordered = list(targets) or ["profit"]

    return {
        "drivers": drivers_final,
        "targets": targets_ordered,
        "delta_pct": percent,
    }


def _aggregate_monthly_simple(df_in, value_col: str, agg_mode: str) -> "pd.DataFrame | None":
    """Return monthly aggregated series for a single numeric column without entity joins.

    Output columns: ["__PERIOD__", value_col]
    """
    try:
        import pandas as pd  # type: ignore
        import numpy as np   # type: ignore
    except Exception:
        return None
    try:
        if df_in is None or value_col not in getattr(df_in, "columns", []):
            return None
        df = df_in.copy()
        period = _time_key_series(df)
        if period is None:
            return None
        ser = pd.to_numeric(df[value_col], errors="coerce")
        out = pd.DataFrame({"__PERIOD__": period, value_col: ser})
        out = out.dropna(subset=["__PERIOD__"]).copy()
        if out.empty:
            return None
        # group by month label
        if agg_mode == "sum":
            g = out.groupby("__PERIOD__")[value_col].sum(min_count=1)
        elif agg_mode == "count":
            g = out.groupby("__PERIOD__")[value_col].count()
        elif agg_mode == "last":
            # best-effort: last non-null per month
            g = (
                out.sort_index()
                   .groupby("__PERIOD__")[value_col]
                   .apply(lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan)
            )
        elif agg_mode == "median":
            g = out.groupby("__PERIOD__")[value_col].median()
        else:
            g = out.groupby("__PERIOD__")[value_col].mean()
        g = g.dropna()
        if g.empty:
            return None
        return g.reset_index()
    except Exception:
        return None


def _build_monthly_totals_nojoin(frames: dict, metrics: set[str]) -> "pd.DataFrame | None":
    """Compose a monthly totals frame across datasets without entity joins.

    Strategy:
    - For each canonical metric, scan frames to find the best column via keyword matching.
    - Aggregate that column by month using sensible defaults (DEFAULT_MODE).
    - Keep the single best source per metric (longest monthly span).
    - Inner-join metrics on __PERIOD__ to form a paired monthly dataframe.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    if not frames or not metrics:
        return None

    # Choose best source per metric
    chosen: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        best_df = None
        best_rows = -1
        agg_mode = DEFAULT_MODE.get(metric, "avg") if 'DEFAULT_MODE' in globals() else "avg"
        for _, df in (frames or {}).items():
            if df is None:
                continue
            col = _find_metric_column_general(df, metric)
            if not col:
                continue
            agg = _aggregate_monthly_simple(df, col, agg_mode)
            if agg is None or agg.empty or "__PERIOD__" not in agg.columns:
                continue
            # rename to canonical metric name
            agg = agg.rename(columns={col: metric}) if col in agg.columns else agg
            # score by number of months with data
            nrows = int(agg[metric].dropna().shape[0])
            if nrows > best_rows:
                best_rows = nrows
                best_df = agg[["__PERIOD__", metric]].copy()
        if best_df is not None:
            chosen[metric] = best_df

    if not chosen:
        return None

    # Merge on period across chosen metrics
    merged = None
    for metric, dfm in chosen.items():
        if merged is None:
            merged = dfm.copy()
        else:
            merged = merged.merge(dfm, on="__PERIOD__", how="inner")
            if merged.empty:
                return None
    if merged is not None and not merged.empty:
        # sanitize periods
        merged["__PERIOD__"] = merged["__PERIOD__"].astype(str)
    return merged


def _find_metric_column_general(df, canonical: str) -> str | None:
    groups = PROJECTION_METRIC_KEYWORDS.get(canonical)
    if not groups:
        return None
    col = _best_metric_column_general(df, groups)
    if col:
        return col
    upper = {str(c).upper(): c for c in df.columns}
    for c_upper, col_name in upper.items():
        if any(all(token in c_upper for token in group) for group in groups):
            return col_name
    return None


def _entity_column(df: "pd.DataFrame") -> str | None:
    if not hasattr(df, "columns"):
        return None
    candidates = [
        "RM_NAME", "RMID", "RM_ID", "RELATIONSHIP_MANAGER", "RELATIONSHIP_MANAGER_NAME",
        "REL_MANAGER", "MANAGER", "ADVISOR", "ENTITY", "MANDATEID", "MANDATE_ID", "MANDATE",
        "CLIENT", "CLIENT_NAME", "ACCOUNT", "PORTFOLIO", "X"
    ]
    upper_map = {str(c).upper(): c for c in df.columns}
    for cand in candidates:
        if cand in upper_map:
            return upper_map[cand]
    for c in df.columns:
        try:
            import pandas as pd
            if not pd.api.types.is_numeric_dtype(df[c]):
                return c
        except Exception:
            continue
    return None


_ALIGNMENT_ID_FIELDS = [
    "RMID", "RM_ID", "RELATIONSHIP_MANAGER_ID",
    "MANDATEID", "MANDATE_ID",
    "CLIENTID", "CLIENT_ID",
    "PORTFOLIOID", "PORTFOLIO_ID",
]

_ALIGNMENT_NAME_FIELDS = [
    "RM_NAME", "RELATIONSHIP_MANAGER", "RELATIONSHIP_MANAGER_NAME",
    "MANDATE", "MANDATE_NAME",
    "CLIENT", "CLIENT_NAME",
    "PORTFOLIO", "PORTFOLIO_NAME",
]


def _select_alignment_columns(df: "pd.DataFrame") -> list[str]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    upper_map = {str(c).upper(): c for c in getattr(df, "columns", [])}

    for field in _ALIGNMENT_ID_FIELDS:
        if field in upper_map:
            return [upper_map[field]]

    for field in _ALIGNMENT_NAME_FIELDS:
        if field in upper_map:
            return [upper_map[field]]

    if pd is not None:
        for col in getattr(df, "columns", []):
            try:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    continue
                nunique = series.nunique(dropna=True)
                if 1 < nunique <= 100:
                    return [col]
            except Exception:
                continue

    return []


def _frame_with_metrics(df, metrics: set[str]) -> tuple["pd.DataFrame", dict[str, str]] | tuple[None, None]:
    import pandas as pd
    import numpy as np

    if not metrics:
        return None, None

    metrics = set(metrics)
    mapping: dict[str, str] = {}
    agg_modes: dict[str, str] = {}

    for metric in metrics:
        col = _find_metric_column_general(df, metric)
        if not col:
            return None, None
        mapping[metric] = col
        agg_modes[metric] = (_viz_default_agg(col) or "avg").lower()

    if not mapping:
        return None, None

    try:
        time_key = _time_key_series(df)
    except Exception:
        time_key = None

    time_series = None
    if time_key is not None:
        try:
            time_series = pd.to_datetime(time_key, errors="coerce")
        except Exception:
            try:
                time_series = pd.to_datetime(pd.Series(time_key, index=df.index).astype(str), errors="coerce")
            except Exception:
                time_series = None
        if time_series is not None:
            time_series = time_series.dt.to_period("M").dt.to_timestamp()
            if time_series.notna().sum() < max(4, int(len(df) * 0.3)):
                time_series = None

    key_cols = _select_alignment_columns(df)
    if not key_cols:
        key_cols = []
        df_local = df.copy()
        df_local["__GLOBAL__"] = "__ALL__"
        key_cols.append("__GLOBAL__")
    else:
        df_local = df.copy()

    group_cols = list(key_cols)
    if time_series is not None:
        df_local["__TIME__"] = time_series
        df_local = df_local[df_local["__TIME__"].notna()]
        if df_local.empty:
            return None, None
        df_local["__TIME__"] = df_local["__TIME__"].dt.to_period("M").dt.to_timestamp()
        group_cols.append("__TIME__")
    elif "__TIME__" in df_local.columns:
        df_local = df_local.drop(columns=["__TIME__"])
    else:
        # No explicit time; try to derive a yearly time index if year columns exist
        for ycol in ("MEET_YEAR", "YEAR", "CAL_YEAR", "FISCAL_YEAR"):
            if ycol in df_local.columns:
                try:
                    ynum = pd.to_numeric(df_local[ycol], errors="coerce").astype("Int64")
                    df_local["__TIME__"] = pd.to_datetime({"year": ynum, "month": 1, "day": 1}, errors="coerce")
                    group_cols.append("__TIME__")
                    break
                except Exception:
                    pass

    for key in key_cols:
        df_local[key] = df_local[key].astype(str)

    data_cols = group_cols + [mapping[m] for m in metrics]
    subset = df_local[data_cols].copy()
    for metric, col in mapping.items():
        subset[metric] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset.drop(columns=[mapping[m] for m in metrics])
    subset = subset.dropna(subset=list(metrics), how="all")
    if subset.empty:
        return None, None

    grouped = subset.groupby(group_cols, dropna=False)
    aggregated_parts = []
    for metric in metrics:
        series = grouped[metric]
        mode = agg_modes.get(metric, "avg")
        if mode == "sum":
            agg_series = series.sum(min_count=1)
        elif mode == "count":
            agg_series = series.count()
        elif mode == "last":
            agg_series = series.apply(
                lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan
            )
        elif mode == "median":
            agg_series = series.median()
        else:
            agg_series = series.mean()
        aggregated_parts.append(agg_series.rename(metric))

    aggregated = pd.concat(aggregated_parts, axis=1).reset_index()
    aggregated = aggregated.dropna(subset=list(metrics), how="all")
    if aggregated.empty:
        return None, None

    if "__TIME__" in aggregated.columns:
        aggregated["__TIME__"] = pd.to_datetime(aggregated["__TIME__"], errors="coerce")
        aggregated["__PERIOD__"] = aggregated["__TIME__"].dt.to_period("M").astype(str)
    else:
        aggregated["__PERIOD__"] = None

    def _compose_entity(row):
        parts = []
        for col in key_cols:
            val = row.get(col)
            val_str = "" if val is None else str(val).strip()
            if not val_str or val_str.lower() in {"nan", "none"}:
                return None
            parts.append(f"{col}={val_str}")
        time_label = row.get("__PERIOD__")
        if isinstance(time_label, str) and time_label:
            parts.append(f"TIME={time_label}")
        if not parts:
            return None
        return "|".join(parts)

    aggregated["__ENTITY__"] = aggregated.apply(_compose_entity, axis=1)
    aggregated = aggregated.dropna(subset=["__ENTITY__"])
    if aggregated.empty:
        return None, None

    result_cols = ["__ENTITY__", "__PERIOD__"] + [metric for metric in metrics]
    subset_out = aggregated[result_cols].copy()
    return subset_out, mapping



def _build_combined_metrics_general(frames: dict, metrics: set[str]):
    import pandas as pd
    import numpy as np

    metric_frames: dict[str, list[tuple[pd.DataFrame, str]]] = {m: [] for m in metrics}
    for _, df in (frames or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for metric in metrics:
            subset, mapping = _frame_with_metrics(df, {metric})
            if subset is None or metric not in subset.columns:
                continue
            col_name = mapping.get(metric, metric)
            agg_mode = (_viz_default_agg(col_name) or "avg").lower()
            keep_cols = [c for c in ["__ENTITY__", "__PERIOD__", metric] if c in subset.columns]
            metric_frames[metric].append((subset[keep_cols].copy(), agg_mode))

    if any(len(v) == 0 for v in metric_frames.values()):
        return None

    merged = None
    for metric, items in metric_frames.items():
        dfs = [df for df, _ in items]
        modes = [mode for _, mode in items]
        primary_mode = modes[0] if modes else "avg"
        mdf = pd.concat(dfs, axis=0, ignore_index=True)
        need_cols = [c for c in [metric, "__ENTITY__", "__PERIOD__"] if c in mdf.columns]
        mdf = mdf.dropna(subset=need_cols)
        if mdf.empty:
            return None
        mdf["__ENTITY__"] = mdf["__ENTITY__"].astype(str)
        # Base entity without TIME for joining across metrics
        try:
            mdf["__EBASE__"] = mdf["__ENTITY__"].str.replace(r"\|?TIME=[^|]+", "", regex=True).str.strip("|")
        except Exception:
            mdf["__EBASE__"] = mdf["__ENTITY__"].astype(str)
        # Ensure monthly period label
        if "__PERIOD__" not in mdf.columns:
            mdf["__PERIOD__"] = None
        mdf["__PERIOD__"] = mdf["__PERIOD__"].astype(str)
        # Expand annual-only series to monthly by replicating within year when needed
        try:
            mdf["__YEAR__"] = mdf["__PERIOD__"].str.slice(0, 4)
            def _expand_group(g):
                months = sorted(set(p[5:7] for p in g["__PERIOD__"].astype(str) if isinstance(p, str) and len(p) >= 7))
                if len(months) >= 3:
                    return g
                year = str(g["__YEAR__"].iloc[0]) if not g.empty else None
                if not year:
                    return g
                val = g[metric].astype(float).mean() if metric in g.columns else None
                if val is None:
                    return g
                rows = []
                for mm in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
                    rows.append({
                        "__EBASE__": g["__EBASE__"].iloc[0],
                        "__PERIOD__": f"{year}-{mm}",
                        metric: float(val),
                    })
                return pd.DataFrame(rows)
            mdf = mdf.groupby(["__EBASE__", "__YEAR__"], dropna=False, as_index=False, group_keys=False).apply(_expand_group)
        except Exception:
            pass
        # Aggregate by (entity, month)
        by = ["__EBASE__", "__PERIOD__"]
        if primary_mode == "sum":
            mdf = mdf.groupby(by, as_index=False)[metric].sum(min_count=1)
        elif primary_mode == "count":
            mdf = mdf.groupby(by, as_index=False)[metric].count()
        elif primary_mode == "last":
            mdf = mdf.sort_values(by).groupby(by).tail(1)[by + [metric]]
        elif primary_mode == "median":
            mdf = mdf.groupby(by, as_index=False)[metric].median()
        else:
            mdf = mdf.groupby(by, as_index=False)[metric].mean()
        if merged is None:
            merged = mdf
        else:
            merged = merged.merge(mdf, on=["__EBASE__", "__PERIOD__"], how="inner")
            if merged.empty:
                return None
    if merged is not None:
        merged = merged.dropna()
        # return period + metrics for modeling
        keep_cols = ["__PERIOD__"] + [m for m in metrics if m in merged.columns]
        merged = merged[keep_cols]
    return merged







def _projection_blueprint_via_llm(question_text: str, drivers_can: list[str], targets_can: list[str], frames_dict: dict[str, 'pd.DataFrame']):
    import json as _json
    import pandas as pd

    def _col_metadata(col_name: str, series: 'pd.Series') -> dict:
        """Build lightweight metadata describing the series so the LLM can reason about semantics."""
        meta: dict[str, object] = {"dtype": str(series.dtype)}
        try:
            non_null = int(series.count())
            meta["non_null"] = non_null
        except Exception:
            non_null = None
        try:
            unique_count = int(series.nunique(dropna=True))
            meta["unique"] = unique_count
        except Exception:
            unique_count = None

        semantics: list[str] = []
        name_up = str(col_name).upper()
        sample_values: list[str] = []
        try:
            for v in series.dropna().head(5).tolist():
                sample_values.append(str(v))
        except Exception:
            pass
        if sample_values:
            meta["samples"] = sample_values

        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                semantics.append("date")
            else:
                parsed = pd.to_datetime(series.dropna().head(20), errors="coerce")
                if parsed.notna().mean() >= 0.6:
                    semantics.append("date")
        except Exception:
            pass

        numeric_ratio = 0.0
        numeric_series = None
        try:
            numeric_series = pd.to_numeric(series, errors="coerce")
            if non_null:
                numeric_ratio = numeric_series.notna().sum() / max(1, non_null)
        except Exception:
            numeric_series = None

        if numeric_series is not None and (pd.api.types.is_numeric_dtype(series) or numeric_ratio >= 0.6):
            semantics.append("numeric")
            try:
                numeric_clean = numeric_series.dropna()
                if not numeric_clean.empty:
                    meta["range"] = [
                        float(numeric_clean.min()),
                        float(numeric_clean.max()),
                    ]
                    as_int = numeric_clean[abs(numeric_clean - numeric_clean.round()) < 1e-6].astype("Int64")
                    if len(as_int) == len(numeric_clean):
                        semantics.append("integer")
                        if name_up.endswith("_ID") or "ID" in name_up:
                            semantics.append("id")
                        if (1 <= as_int.min() <= 12 and 1 <= as_int.max() <= 12) or ("MONTH" in name_up or "MON" in name_up):
                            semantics.append("month_number")
                        if (1900 <= as_int.min() <= 9999 and 1900 <= as_int.max() <= 9999) or "YEAR" in name_up or name_up.endswith("_YR"):
                            semantics.append("year")
            except Exception:
                pass
        else:
            if unique_count is not None and non_null:
                unique_ratio = unique_count / max(1, non_null)
                if unique_ratio <= 0.3:
                    semantics.append("categorical")

        if ("DATE" in name_up or name_up.endswith("_DT") or name_up.endswith("_DATE")) and "date" not in semantics:
            semantics.append("date")
        if ("RMID" in name_up or "RM_ID" in name_up or name_up.endswith("_ID")) and "id" not in semantics:
            semantics.append("id")

        if semantics:
            meta["semantic_hints"] = sorted(dict.fromkeys(semantics))
        return meta

    sess = globals().get('_GLOBAL_SNOWFLAKE_SESSION')
    if sess is None:
        return None, None
    schema = []
    for fname, fdf in (frames_dict or {}).items():
        if isinstance(fdf, pd.DataFrame) and not fdf.empty:
            cols = [str(c) for c in fdf.columns]
            sample = fdf[cols].head(5).astype(str).to_dict('records')
            col_hints = {}
            for c in cols:
                try:
                    col_hints[c] = _col_metadata(c, fdf[c])
                except Exception:
                    col_hints[c] = {"dtype": str(fdf[c].dtype)}
            schema.append({'frame': fname, 'columns': cols, 'column_hints': col_hints, 'sample': sample})
    if not schema:
        return None, None
    payload = _json.dumps({
        'question': question_text,
        'drivers': drivers_can,
        'targets': targets_can,
        'schema': schema
    }, ensure_ascii=False, indent=2)
    prompt = f"""You are a senior analytics engineer. Determine how to build a MONTHLY modeling table to project targets from drivers.
The table must align drivers and targets by time and entity (prefer RMID).

QUESTION: {question_text}
DRIVERS: {drivers_can}
TARGETS: {targets_can}

SCHEMA_JSON:
{payload}

Each frame includes column_hints with:
- dtype (pandas dtype)
- semantic_hints (e.g., date, month_number, year, id, numeric, categorical)
- samples, non_null, unique counts and numeric ranges when available

Return JSON ONLY with this structure:
{{
  "panel": [
    {{
      "metric": "<canonical metric>",
      "frame": "<frame name>",
      "column": "<column name>",
      "aggregation": "sum|avg|median|last|count",
      "entity": "<entity column or null>",
      "time": {{"column": "<date column>"}} or {{"columns": ["YEAR_COLUMN", "MONTH_COLUMN"]}}
    }}
  ]
}}
Rules:
- Use only frame and column names provided.
- Prefer monthly grain (convert year/month pairs if needed).
- Do not sum AUM metrics; use "last" for AUM.
- Use column_hints to detect time columns (date/month/year), entities/IDs, and numeric vs categorical data when choosing joins.
- Provide entries for every driver and target listed exactly once.
Return JSON only."""
    try:
        query = "select snowflake.cortex.complete('" + str(CORTEX_LLM_MODEL) + "', $$" + prompt.replace('$$', '$ $') + "$$) as response"
        df_resp = sess.sql(query).to_pandas()
        if df_resp is None or df_resp.empty:
            return None, None
        resp = df_resp.iloc[0, 0]
        plan = safe_json_loads(resp) if isinstance(resp, str) else None
        if not isinstance(plan, dict):
            return None, None
    except Exception:
        return None, None
    panel_specs = plan.get('panel') or []
    if not isinstance(panel_specs, list) or not panel_specs:
        return None, None

    def _detect_rmid(df_loc: pd.DataFrame) -> str | None:
        up = {str(c).upper(): c for c in df_loc.columns}
        for key in ('RMID', 'RM_ID'):
            if key in up:
                return up[key]
        return None

    def _infer_period(df_loc: pd.DataFrame, info: dict) -> pd.Series | None:
        try:
            if isinstance(info.get('columns'), list) and len(info['columns']) >= 2:
                ycol, mcol = info['columns'][:2]
                if ycol in df_loc.columns and mcol in df_loc.columns:
                    return pd.to_datetime({
                        'year': pd.to_numeric(df_loc[ycol], errors='coerce').astype('Int64'),
                        'month': pd.to_numeric(df_loc[mcol], errors='coerce').astype('Int64'),
                        'day': 1
                    }, errors='coerce').dt.to_period('M').astype(str)
            col = info.get('column')
            if isinstance(col, str) and col in df_loc.columns:
                return pd.to_datetime(df_loc[col], errors='coerce').dt.to_period('M').astype(str)
        except Exception:
            pass
        for candidate in ('POS_DATE', 'MONTH_DATE', 'MEETING_DATE', 'MEETINGDATE', 'AS_OF_DATE', 'DATE'):
            if candidate in df_loc.columns:
                try:
                    s = pd.to_datetime(df_loc[candidate], errors='coerce').dt.to_period('M').astype(str)
                    if s.notna().any():
                        return s
                except Exception:
                    pass
        for ycol, mcol in (('MEET_YEAR', 'MEET_MON'), ('POSYEAR', 'POSMON'), ('YEAR', 'MONTH')):
            if ycol in df_loc.columns and mcol in df_loc.columns:
                try:
                    s = pd.to_datetime({
                        'year': pd.to_numeric(df_loc[ycol], errors='coerce').astype('Int64'),
                        'month': pd.to_numeric(df_loc[mcol], errors='coerce').astype('Int64'),
                        'day': 1
                    }, errors='coerce').dt.to_period('M').astype(str)
                    if s.notna().any():
                        return s
                except Exception:
                    pass
        for candidate in ('PERIOD', 'MEET_MONTH'):
            if candidate in df_loc.columns:
                try:
                    s = pd.to_datetime(df_loc[candidate].astype(str), format='%Y-%m', errors='coerce').dt.to_period('M').astype(str)
                    if s.notna().any():
                        return s
                except Exception:
                    pass
        return None

    def _aggregate(df_loc: pd.DataFrame, group_cols: list[str], value_col: str, agg: str) -> pd.DataFrame:
        if agg in ('avg', 'mean'):
            return df_loc.groupby(group_cols, as_index=False)[value_col].mean()
        if agg == 'median':
            return df_loc.groupby(group_cols, as_index=False)[value_col].median()
        if agg == 'last':
            df_sorted = df_loc.sort_values(group_cols)
            return df_sorted.groupby(group_cols, as_index=False).tail(1)[group_cols + [value_col]]
        if agg == 'count':
            return df_loc.groupby(group_cols, as_index=False)[value_col].count()
        return df_loc.groupby(group_cols, as_index=False)[value_col].sum()

    parts: dict[str, list[tuple[pd.DataFrame, str]]] = {}
    rm_possible = True
    metrics_order = list(dict.fromkeys(list(drivers_can) + list(targets_can)))
    # Map lowercase -> canonical metric so LLM case differences do not break lookup
    metrics_lookup = {str(m).strip().lower(): str(m) for m in metrics_order}

    for spec in panel_specs:
        metric_raw = spec.get('metric')
        if not metric_raw:
            continue
        metric_key = str(metric_raw).strip().lower()
        canonical_metric = metrics_lookup.get(metric_key)
        if not canonical_metric:
            continue
        frame_name = spec.get('frame')
        column = spec.get('column')
        agg = str(spec.get('aggregation') or 'sum').lower()
        time_info = spec.get('time') or {}
        entity_hint = spec.get('entity')
        if frame_name not in frames_dict:
            continue
        df_src = frames_dict.get(frame_name)
        if not isinstance(df_src, pd.DataFrame) or df_src.empty or column not in df_src.columns:
            continue
        if 'aum' in metric_key and agg == 'sum':
            agg = 'last'
        per_series = _infer_period(df_src, time_info)
        if per_series is None:
            continue
        rmid_col = entity_hint if isinstance(entity_hint, str) and entity_hint in df_src.columns else _detect_rmid(df_src)
        if rmid_col is None:
            rm_possible = False
        d = df_src[[column]].copy()
        d['__PERIOD__'] = per_series
        if rmid_col:
            d['__RM__'] = df_src[rmid_col].astype(str)
        d = d.dropna(subset=['__PERIOD__'])
        parts.setdefault(canonical_metric, []).append((d, agg))

    if not parts:
        return None, None

    def _materialize(parts_dict: dict[str, list[tuple[pd.DataFrame, str]]], require_rm: bool) -> pd.DataFrame | None:
        merged = None
        for metric in metrics_order:
            entries = parts_dict.get(metric)
            if not entries:
                return None
            frames_agg = []
            for d, agg in entries:
                cols = ['__PERIOD__'] + (["__RM__"] if '__RM__' in d.columns else []) + [d.columns[0]]
                dfc = d[cols].dropna(subset=['__PERIOD__', d.columns[0]])
                if dfc.empty:
                    continue
                if require_rm and '__RM__' not in dfc.columns:
                    return None
                group_cols = ['__PERIOD__'] + (['__RM__'] if '__RM__' in dfc.columns else [])
                agg_df = _aggregate(dfc, group_cols, d.columns[0], agg)
                frames_agg.append(agg_df.rename(columns={d.columns[0]: metric}))
            if not frames_agg:
                return None
            df_metric = frames_agg[0]
            for g in frames_agg[1:]:
                on = [c for c in ['__PERIOD__', '__RM__'] if c in g.columns and c in df_metric.columns]
                df_metric = df_metric.merge(g, on=on, how='outer')
            if merged is None:
                merged = df_metric
            else:
                on = [c for c in ['__PERIOD__', '__RM__'] if c in df_metric.columns and c in merged.columns]
                merged = merged.merge(df_metric, on=on, how='inner')
                if merged.empty:
                    return None
        return merged

    panel_rm = _materialize(parts, True) if rm_possible else None
    if isinstance(panel_rm, pd.DataFrame) and not panel_rm.empty:
        keep = ['__PERIOD__', '__RM__'] + [m for m in metrics_order if m in panel_rm.columns]
        return '__panel_rm_month_llm__', panel_rm[keep].dropna()
    panel_month = _materialize(parts, False)
    if isinstance(panel_month, pd.DataFrame) and not panel_month.empty:
        keep = ['__PERIOD__'] + [m for m in metrics_order if m in panel_month.columns]
        return '__panel_month_llm__', panel_month[keep].dropna()
    return None, None

def _scenario_projection_payload(question: str, frames: dict):
    """
    Estimate target metric impact from percentage change in driver metrics using
    a simple linear regression fit. Falls back to proportional ratios when
    regression is not feasible.
    """
    try:
        import pandas as pd  # type: ignore
        import numpy as np   # type: ignore
        import json
    except Exception:
        return [], None

    parsed = _parse_projection_question(question)
    drivers: list[str] = parsed.get("drivers") or []
    targets: list[str] = parsed.get("targets") or []
    delta_pct: float = parsed.get("delta_pct") or 0.10

    if not drivers or not targets:
        return [], None

    metrics_needed = set(drivers) | set(targets)

    panel_name_llm, panel_df_llm = _projection_blueprint_via_llm(question, drivers, targets, frames)

    combined = _build_combined_metrics_general(frames, metrics_needed)
    simple_monthly = _build_monthly_totals_nojoin(frames, metrics_needed)
    frame_candidates: list[tuple[str, pd.DataFrame, dict[str, str]]] = []
    if simple_monthly is not None and hasattr(simple_monthly, 'empty') and not simple_monthly.empty:
        frame_candidates.append(("__monthly_totals_nojoin__", simple_monthly, {m: m for m in metrics_needed}))
    if panel_df_llm is not None and isinstance(panel_df_llm, pd.DataFrame) and not panel_df_llm.empty:
        frame_candidates.append((panel_name_llm or "__panel_llm__", panel_df_llm, {m: m for m in metrics_needed}))
    if combined is not None and not combined.empty:
        frame_candidates.append(("__combined_metrics__", combined, {m: m for m in metrics_needed}))

    for frame_name, df in (frames or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        subset, mapping = _frame_with_metrics(df, metrics_needed)
        if subset is not None and mapping is not None:
            frame_candidates.append((frame_name, subset, mapping))

    if not frame_candidates:
        return [], None

    best_payload = None
    best_notes: list[str] = []
    best_score = float('-inf')
    fallback_candidate: tuple[str, pd.DataFrame, dict[str, str]] | None = None

    def _fmt_slope(val: float | None) -> str:
        if val is None:
            return "n/a"
        slope_val = float(val)
        abs_val = abs(slope_val)
        if abs_val >= 1_000_000:
            text = f"${abs_val/1_000_000:,.2f}M"
        elif abs_val >= 1_000:
            text = f"${abs_val/1_000:,.2f}K"
        elif abs_val >= 1:
            text = f"${abs_val:,.0f}"
        elif abs_val >= 0.01:
            text = f"${abs_val:,.2f}"
        else:
            text = f"{slope_val:.4g}"
        if slope_val < 0 and not text.startswith("-"):
            text = "-" + text
        return text

    def _fmt_money_val(val):
        val = _safe_float(val)
        if val is None:
            return "n/a"
        return f"${val:,.0f}"

    def _fmt_pct_val(val):
        val = _safe_float(val)
        if val is None:
            return "n/a"
        return f"{val*100:+.1f}%"

    def _trim_df_quantile(df_in: 'pd.DataFrame', cols: list[str], p: float = 0.01,
                          min_rows: int = 10) -> 'pd.DataFrame':
        try:
            df = df_in.copy()
            if df.empty:
                return df
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=cols)
            if df.shape[0] < max(min_rows, 6):
                return df_in
            bounds = {}
            for c in cols:
                lo = df[c].quantile(p)
                hi = df[c].quantile(1 - p)
                bounds[c] = (lo, hi)
            mask = pd.Series(True, index=df.index)
            for c, (lo, hi) in bounds.items():
                mask &= (df[c] >= lo) & (df[c] <= hi)
            trimmed = df[mask]
            if trimmed.shape[0] >= min_rows:
                return trimmed
            return df
        except Exception:
            return df_in

    def _ensure_period(df_in: 'pd.DataFrame') -> 'pd.DataFrame':
        df = df_in.copy()
        if "__PERIOD__" in df.columns:
            return df
        # Try to parse from __ENTITY__
        if "__ENTITY__" in df.columns:
            try:
                period = df["__ENTITY__"].astype(str).str.extract(r"TIME=([^|]+)", expand=False)
                df["__PERIOD__"] = period
                return df
            except Exception:
                pass
        df["__PERIOD__"] = None
        return df

    def _pick_time_grain(df: 'pd.DataFrame') -> tuple[str, str]:
        # returns (time_grain, latest_period_label)
        if "__PERIOD__" not in df.columns:
            return ("unknown", None)
        periods = df["__PERIOD__"].dropna().astype(str)
        if periods.empty:
            return ("unknown", None)
        # Assume YYYY-MM indicates month; if others exist, fall back to max
        latest = sorted(periods.unique())[-1]
        grain = "month" if len(latest) >= 7 and "-" in latest else "year" if len(latest) == 4 else "unknown"
        return (grain, latest)

    def _parse_lag_policy(text: str) -> int:
        try:
            import re as _re
            if not text:
                return 0
            m = _re.search(r"lag\s*[:=]?\s*(\d+)", text.lower())
            if m:
                return max(0, int(m.group(1)))
            m2 = _re.search(r"t\s*\+\s*(\d+)", text.lower())
            if m2:
                return max(0, int(m2.group(1)))
        except Exception:
            pass
        return 0

    def _apply_lag(df_in: 'pd.DataFrame', driver_cols: list[str], lag: int) -> 'pd.DataFrame':
        if lag <= 0:
            return df_in
        df = df_in.copy()
        try:
            # create sortable period index
            per = pd.PeriodIndex(pd.to_datetime(df["__PERIOD__"], errors="coerce").dt.to_period("M"), freq="M")
            df["__PERIOD_IDX__"] = per
            if "__ENTITY__" not in df.columns:
                df["__ENTITY__"] = "__ALL__"
            df = df.sort_values(["__ENTITY__", "__PERIOD_IDX__"])
            for col in driver_cols:
                df[col] = df.groupby("__ENTITY__")[col].shift(lag)
            df = df.drop(columns=["__PERIOD_IDX__"])
            return df
        except Exception:
            return df_in

    def _parse_weights_policy(text: str, df: 'pd.DataFrame') -> tuple[str, str | None]:
        # returns (policy, weight_col)
        tl = (text or "").lower()
        up = {c.upper(): c for c in df.columns}
        # explicit hints
        if "duration" in tl and ("DURATION" in up or "TENOR" in up):
            c = up.get("DURATION") or up.get("TENOR")
            return ("duration_weight", c)
        if "exposure" in tl and ("EXPOSURE" in up or "WEIGHT" in up or "WT" in up):
            for k in ("EXPOSURE", "WEIGHT", "WT"):
                if k in up:
                    return ("exposure_weight", up[k])
        # auto-detect
        for k in ("EXPOSURE", "WEIGHT", "WT"):
            if k in up:
                return ("exposure_weight", up[k])
        for k in ("DURATION", "TENOR"):
            if k in up:
                return ("duration_weight", up[k])
        return ("none", None)

    def _extract_weights(df_in: 'pd.DataFrame', col: str) -> 'np.ndarray | None':
        if not col or col not in df_in.columns:
            return None
        try:
            weights = pd.to_numeric(df_in[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            weights = weights.clip(lower=0.0)
            if weights.sum() <= 0 or int((weights > 0).sum()) < 2:
                return None
            return weights.to_numpy()
        except Exception:
            return None

    def _weighted_mean(values: 'pd.Series | np.ndarray', weights: 'np.ndarray | None') -> float | None:
        arr = np.asarray(values, dtype=float)
        mask = np.isfinite(arr)
        if mask.sum() == 0:
            return None
        arr = arr[mask]
        if weights is None:
            return float(arr.mean()) if arr.size > 0 else None
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != mask.shape[0]:
            return float(arr.mean()) if arr.size > 0 else None
        w = w[mask]
        w = np.clip(w, 0.0, None)
        w_sum = float(w.sum())
        if w_sum <= 0:
            return float(arr.mean()) if arr.size > 0 else None
        return float(np.sum(arr * w) / w_sum)

    def _weighted_corr(x_vals: 'np.ndarray', y_vals: 'np.ndarray', weights: 'np.ndarray | None') -> float | None:
        if weights is None:
            return None
        x = np.asarray(x_vals, dtype=float)
        y = np.asarray(y_vals, dtype=float)
        w = np.asarray(weights, dtype=float)
        if x.shape != y.shape or x.shape != w.shape:
            return None
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        x = x[mask]
        y = y[mask]
        w = np.clip(w[mask], 0.0, None)
        if x.size < 3:
            return None
        w_sum = float(w.sum())
        if w_sum <= 0:
            return None
        mu_x = float(np.sum(w * x) / w_sum)
        mu_y = float(np.sum(w * y) / w_sum)
        cov_xy = float(np.sum(w * (x - mu_x) * (y - mu_y)))
        var_x = float(np.sum(w * (x - mu_x) ** 2))
        var_y = float(np.sum(w * (y - mu_y) ** 2))
        if var_x <= 0 or var_y <= 0:
            return None
        return float(cov_xy / np.sqrt(var_x * var_y))

    for frame_name, df_model, mapping in frame_candidates:
        df_work = _ensure_period(df_model)
        # numeric casting & hygiene
        try:
            for metric in metrics_needed:
                df_work[metric] = pd.to_numeric(df_work[metric], errors="coerce")
        except Exception:
            continue
        df_work = df_work.replace([np.inf, -np.inf], np.nan).dropna(subset=list(metrics_needed))
        if df_work.empty or len(df_work) < max(10, len(drivers) + len(targets) + 2):
            if fallback_candidate is None:
                fallback_candidate = (frame_name, df_model, mapping)
            continue
        if len(df_work) > 20000:
            df_work = df_work.sample(20000, random_state=42)

        # Freeze cohort/time
        time_grain, latest_period = _pick_time_grain(df_work)
        # Policies
        lag_k = _parse_lag_policy(question)
        driver_cols = list(drivers)
        target_cols = list(targets)
        # Apply lag to drivers consistently if requested
        df_work = _apply_lag(df_work, driver_cols, lag_k)
        slice_latest = df_work[df_work["__PERIOD__"].astype(str) == str(latest_period)] if latest_period else df_work
        # Weights policy
        weights_policy, wcol = _parse_weights_policy(question, df_work)
        # Global baselines (cohort-wide means for diagnostics)
        weights_global = _extract_weights(df_work, wcol) if wcol and wcol in df_work.columns else None
        baseline_means: dict[str, float | None] = {}
        for metric in metrics_needed:
            try:
                val = _weighted_mean(df_work[metric], weights_global)
                if val is None:
                    val = float(pd.to_numeric(df_work[metric], errors="coerce").dropna().mean())
                baseline_means[metric] = val
            except Exception:
                baseline_means[metric] = None
        betas: dict[str, np.ndarray] = {}
        r2_scores: dict[str, float | None] = {}
        fitinfo: dict[str, dict[str, 'np.ndarray' | float]] = {}
        baselines_by_target: dict[str, dict[str, float]] = {}
        corr_by_target_driver: dict[tuple[str, str], float | None] = {}
        try:
            for target in target_cols:
                cols_now = driver_cols + [target]
                cols_for_fit = ["__PERIOD__"] + cols_now
                if "__ENTITY__" in df_work.columns:
                    cols_for_fit = ["__ENTITY__"] + cols_for_fit
                if wcol and wcol in df_work.columns:
                    cols_for_fit = cols_for_fit + [wcol]
                df_pair = df_work[cols_for_fit].dropna()
                if df_pair.shape[0] > 5000:
                    df_pair = df_pair.sample(5000, random_state=42)
                df_pair = _trim_df_quantile(df_pair, cols_now, p=0.01, min_rows=max(10, len(driver_cols) + 3))
                if df_pair.shape[0] < max(10, len(driver_cols) + 3):
                    # fallback to untrimmed if too small
                    df_pair = df_work[cols_for_fit].dropna()
                X_mat = df_pair[driver_cols].astype(float).values
                y_vec = df_pair[target].astype(float).values
                w_vec = _extract_weights(df_pair, wcol) if wcol and wcol in df_pair.columns else None

                # baselines & correlations (used both for regression diagnostics and fallback)
                try:
                    if w_vec is not None:
                        baseline_target_means = {}
                        for m in driver_cols + [target]:
                            val = _weighted_mean(df_pair[m], w_vec)
                            if val is None:
                                val = float(pd.to_numeric(df_pair[m], errors="coerce").dropna().mean())
                            baseline_target_means[m] = val
                    else:
                        baseline_target_means = {m: float(pd.to_numeric(df_pair[m], errors="coerce").dropna().mean())
                                                 for m in driver_cols + [target]}
                except Exception:
                    baseline_target_means = {m: None for m in driver_cols + [target]}
                baselines_by_target[target] = baseline_target_means
                for drv in driver_cols:
                    try:
                        xv = df_pair[drv].astype(float).values
                        yv = y_vec
                        if w_vec is not None:
                            corr_val = _weighted_corr(xv, yv, w_vec)
                        else:
                            if xv.size >= 3 and np.nanstd(xv) > 1e-8 and np.nanstd(y_vec) > 1e-8:
                                corr_val = float(np.corrcoef(xv, y_vec)[0, 1])
                            else:
                                corr_val = None
                        corr_by_target_driver[(target, drv)] = corr_val
                    except Exception:
                        corr_by_target_driver[(target, drv)] = None

                if X_mat.shape[0] <= X_mat.shape[1] + 1:
                    continue
                if any(np.nanstd(X_mat[:, i]) < 1e-8 for i in range(X_mat.shape[1])):
                    continue
                if np.nanstd(y_vec) < 1e-8:
                    continue
                ones = np.ones((X_mat.shape[0], 1))
                design = np.hstack([ones, X_mat])
                Xw = None
                sw = None
                if w_vec is not None:
                    sw = np.sqrt(np.clip(w_vec, 0.0, None))
                    Xw = design * sw[:, None]
                    yw = y_vec * sw
                    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
                else:
                    beta, _, _, _ = np.linalg.lstsq(design, y_vec, rcond=None)
                betas[target] = beta
                pred = design @ beta
                resid = y_vec - pred
                if w_vec is not None and w_vec.sum() > 0:
                    y_mean_w = float(np.sum(w_vec * y_vec) / w_vec.sum())
                    ss_tot = float(np.sum(w_vec * (y_vec - y_mean_w) ** 2))
                    ss_res = float(np.sum(w_vec * resid ** 2))
                else:
                    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
                    ss_res = float(np.sum(resid ** 2))
                if ss_tot > 1e-9:
                    r2_scores[target] = float(max(-1.0, min(1.0, 1 - ss_res / ss_tot)))
                else:
                    r2_scores[target] = None
                # store OLS variance info for CI (linearization)
                try:
                    if w_vec is not None and Xw is not None and sw is not None:
                        xtx = Xw.T @ Xw
                        resid_w = resid * sw
                        dof = max(1, design.shape[0] - design.shape[1])
                        sigma2 = float(np.sum(resid_w ** 2) / dof)
                    else:
                        xtx = design.T @ design
                        dof = max(1, design.shape[0] - design.shape[1])
                        sigma2 = float(np.sum(resid ** 2) / dof)
                    inv_xtx = np.linalg.pinv(xtx)
                    fitinfo[target] = {"inv_xtx": inv_xtx, "sigma2": sigma2}
                except Exception:
                    fitinfo[target] = {"inv_xtx": None, "sigma2": None}
        except Exception:
            if fallback_candidate is None:
                fallback_candidate = (frame_name, df_model, mapping)
            continue

        relationships_struct: dict[str, dict[str, object]] = {}
        scenarios_struct: dict[str, dict[str, object]] = {}
        notes: list[str] = []

        for driver_idx, driver in enumerate(driver_cols):
            rel_targets: dict[str, dict[str, float | None]] = {}
            scen_targets: dict[str, dict[str, float | None]] = {}

            for target in target_cols:
                beta = betas.get(target)
                if beta is None:
                    # Fallback: ratio-based projection on trimmed means when regression not available
                    base_means_fb = baselines_by_target.get(target) or {}
                    drv_mean_fb = base_means_fb.get(driver)
                    tgt_mean_fb = base_means_fb.get(target)
                    corr_fb = corr_by_target_driver.get((target, driver))
                    weights_fb = None
                    if drv_mean_fb is None or tgt_mean_fb is None or corr_fb is None:
                        cols_fb = [driver, target]
                        if wcol and wcol in df_work.columns:
                            cols_fb.append(wcol)
                        df_fb = df_work[cols_fb].dropna(subset=[driver, target])
                        if df_fb.shape[0] < 3:
                            continue
                        df_fb = _trim_df_quantile(df_fb, [driver, target], p=0.01, min_rows=3)
                        weights_fb = _extract_weights(df_fb, wcol) if wcol and wcol in df_fb.columns else None
                        drv_vals_fb = df_fb[driver].astype(float).values
                        tgt_vals_fb = df_fb[target].astype(float).values
                        if drv_vals_fb.size < 3 or np.nanstd(drv_vals_fb) < 1e-8 or np.nanstd(tgt_vals_fb) < 1e-8:
                            continue
                        drv_mean_fb = _weighted_mean(drv_vals_fb, weights_fb)
                        tgt_mean_fb = _weighted_mean(tgt_vals_fb, weights_fb)
                        if drv_mean_fb is None or tgt_mean_fb is None:
                            continue
                        if not base_means_fb:
                            base_means_fb = {}
                        base_means_fb[driver] = drv_mean_fb
                        base_means_fb[target] = tgt_mean_fb
                        baselines_by_target[target] = base_means_fb
                        if weights_fb is not None:
                            corr_fb = _weighted_corr(drv_vals_fb, tgt_vals_fb, weights_fb)
                        else:
                            corr_fb = float(np.corrcoef(drv_vals_fb, tgt_vals_fb)[0, 1]) if np.nanstd(drv_vals_fb) > 1e-8 and np.nanstd(tgt_vals_fb) > 1e-8 else None
                        corr_by_target_driver[(target, driver)] = corr_fb
                    if drv_mean_fb is None or tgt_mean_fb is None:
                        continue
                    slope_fb = None
                    if abs(drv_mean_fb) > 1e-8:
                        slope_fb = tgt_mean_fb / drv_mean_fb
                    delta_value_fb = slope_fb * drv_mean_fb * delta_pct if slope_fb is not None else tgt_mean_fb * delta_pct
                    delta_pct_target_fb = None
                    if abs(tgt_mean_fb) > 1e-8:
                        delta_pct_target_fb = delta_value_fb / tgt_mean_fb
                    # Latest-period corr for fallback case
                    try:
                        cols_latest = [driver, target]
                        if wcol and wcol in slice_latest.columns:
                            cols_latest.append(wcol)
                        df_latest_pair = slice_latest[cols_latest].copy().dropna(subset=[driver, target])
                        if not df_latest_pair.empty:
                            df_latest_pair = _trim_df_quantile(df_latest_pair, [driver, target], p=0.01, min_rows=3)
                            xv_l = df_latest_pair[driver].astype(float).values
                            yv_l = df_latest_pair[target].astype(float).values
                            weights_latest = _extract_weights(df_latest_pair, wcol) if wcol and wcol in df_latest_pair.columns else None
                            if weights_latest is not None:
                                corr_latest = _weighted_corr(xv_l, yv_l, weights_latest)
                            elif xv_l.size >= 3 and np.nanstd(xv_l) > 0 and np.nanstd(yv_l) > 0:
                                corr_latest = float(np.corrcoef(xv_l, yv_l)[0, 1])
                            else:
                                corr_latest = None
                        else:
                            corr_latest = None
                    except Exception:
                        corr_latest = None
                    rel_targets[target] = {
                        "correlation": corr_fb,
                        "corr_scope": "cohort_trimmed",
                        "corr_latest_period": corr_latest,
                        "slope": slope_fb,
                    }
                    scen_targets[target] = {
                        "projected": tgt_mean_fb + delta_value_fb,
                        "delta": delta_value_fb,
                        "delta_pct": delta_pct_target_fb,
                    }
                    corr_text = "n/a" if corr_fb is None else f"{corr_fb:.2f}"
                    slope_text = _fmt_slope(slope_fb)
                    notes.append(
                        f"[RELATIONSHIP] {frame_name}: {_metric_display_name(driver)} â†’ {_metric_display_name(target)} corr={corr_text}; slope â‰ˆ {slope_text} per unit."
                    )
                    delta_text = "n/a" if delta_value_fb is None else f"{delta_value_fb:,.0f}"
                    target_pct_text = "n/a" if delta_pct_target_fb is None else f"{delta_pct_target_fb*100:+.1f}%"
                    notes.append(
                        f"[SCENARIO] {frame_name}: {_metric_display_name(driver)} {delta_pct*100:+.1f}% â†’ {_metric_display_name(target)} Î” â‰ˆ {delta_text} ({target_pct_text})."
                    )
                    continue
                base_means = baselines_by_target.get(target) or {}
                driver_mean = base_means.get(driver)
                if driver_mean is None:
                    continue
                base_features = np.array([1.0] + [base_means.get(d, np.nan) for d in driver_cols], dtype=float)
                if np.isnan(base_features).any():
                    continue
                driver_features = base_features.copy()
                driver_features[1 + driver_idx] = driver_mean * (1 + delta_pct)
                corr = corr_by_target_driver.get((target, driver))
                slope = float(beta[1 + driver_idx]) if len(beta) > 1 + driver_idx else None
                # Latest-period correlation (context-only)
                try:
                    cols_latest = [driver, target]
                    if wcol and wcol in latest_rows.columns:
                        cols_latest.append(wcol)
                    df_latest_pair = latest_rows[cols_latest].copy().dropna(subset=[driver, target])
                    if not df_latest_pair.empty:
                        df_latest_pair = _trim_df_quantile(df_latest_pair, [driver, target], p=0.01, min_rows=3)
                        xv_l = df_latest_pair[driver].astype(float).values
                        yv_l = df_latest_pair[target].astype(float).values
                        weights_latest = _extract_weights(df_latest_pair, wcol) if wcol and wcol in df_latest_pair.columns else None
                        if weights_latest is not None:
                            corr_latest = _weighted_corr(xv_l, yv_l, weights_latest)
                        elif xv_l.size >= 3 and np.nanstd(xv_l) > 0 and np.nanstd(yv_l) > 0:
                            corr_latest = float(np.corrcoef(xv_l, yv_l)[0, 1])
                        else:
                            corr_latest = None
                    else:
                        corr_latest = None
                except Exception:
                    corr_latest = None
                rel_targets[target] = {
                    "correlation": corr,
                    "corr_scope": "cohort_trimmed",
                    "corr_latest_period": corr_latest,
                    "slope": slope,
                }

                # Bump-test: compute row-wise predictions for latest period slice
                # Using per-target beta and holding other drivers constant
                sel_cols = driver_cols + [target] + ([wcol] if wcol and wcol in slice_latest.columns else [])
                latest_rows = slice_latest[sel_cols].dropna()
                if latest_rows.empty:
                    continue
                X_latest = latest_rows[driver_cols].astype(float).values
                ones_latest = np.ones((X_latest.shape[0], 1))
                des_latest = np.hstack([ones_latest, X_latest])
                pred_base_rows = des_latest @ beta
                X_bumped = X_latest.copy()
                X_bumped[:, driver_idx] = X_bumped[:, driver_idx] * (1 + delta_pct)
                des_bumped = np.hstack([ones_latest, X_bumped])
                pred_bumped_rows = des_bumped @ beta
                diffs = pred_bumped_rows - pred_base_rows
                if wcol and wcol in latest_rows.columns:
                    w_latest = pd.to_numeric(latest_rows[wcol], errors="coerce").fillna(0.0).values.astype(float)
                    delta_value = float(np.nansum(diffs * w_latest))
                else:
                    delta_value = float(np.nansum(diffs))
                baseline_target = base_means.get(target)
                # Baseline total for latest period slice
                if wcol and wcol in latest_rows.columns:
                    baseline_total = float(np.nansum(pd.to_numeric(latest_rows[target], errors="coerce").values.astype(float) * w_latest))
                else:
                    baseline_total = float(latest_rows[target].sum()) if target in latest_rows.columns else None
                delta_pct_target = None
                if baseline_total and abs(baseline_total) > 1e-8:
                    delta_pct_target = delta_value / baseline_total
                # CI via linearization: g^T Var(beta) g, where g has only driver component
                ci_low = None; ci_high = None; ci_method = None
                try:
                    info = fitinfo.get(target) or {}
                    inv_xtx = info.get("inv_xtx")
                    sigma2 = info.get("sigma2")
                    if inv_xtx is not None and sigma2 is not None:
                        g = np.zeros(inv_xtx.shape[0])
                        # delta in driver coefficient direction: sum(Δ x_d)
                        sum_dx = float(np.nansum(X_latest[:, driver_idx] * delta_pct))
                        g[1 + driver_idx] = sum_dx
                        var_delta = float(sigma2 * (g.T @ inv_xtx @ g))
                        se = float(np.sqrt(max(0.0, var_delta)))
                        ci_low = delta_value - 1.96 * se
                        ci_high = delta_value + 1.96 * se
                        ci_method = "ols_delta_linearization"
                except Exception:
                    pass
                scen_targets[target] = {
                    "baseline_total": baseline_total,
                    "projected": None if baseline_total is None else baseline_total + delta_value,
                    "delta": delta_value,
                    "delta_pct": delta_pct_target,
                    "ci": [ci_low, ci_high] if (ci_low is not None and ci_high is not None) else None,
                    "ci_method": ci_method,
                }

                corr_text = "n/a" if corr is None else f"{corr:.2f}"
                slope_text = _fmt_slope(slope)
                notes.append(
                    f"[RELATIONSHIP] {frame_name}: {_metric_display_name(driver)} → {_metric_display_name(target)} corr={corr_text}; slope ≈ {slope_text} per unit."
                )
                delta_text = "n/a" if delta_value is None else f"{delta_value:,.0f}"
                target_pct_text = "n/a" if delta_pct_target is None else f"{delta_pct_target*100:+.1f}%"
                notes.append(
                    f"[SCENARIO] {frame_name}: {_metric_display_name(driver)} {delta_pct*100:+.1f}% → {_metric_display_name(target)} Δ ≈ {delta_text} ({target_pct_text})."
                )

            relationships_struct[driver] = {
                "mean": driver_mean,
                "targets": rel_targets,
            }
            scenarios_struct[driver] = {
                "targets": scen_targets,
            }

        # Joint scenario: bump all drivers together by the same delta_pct
        if len(driver_cols) > 1 and target_cols:
            joint_targets: dict[str, dict[str, float | None]] = {}
            try:
                headline_beta = betas.get(target_cols[0])
                if headline_beta is not None:
                    latest_rows_all = slice_latest[driver_cols + [target_cols[0]] + ([wcol] if wcol and wcol in slice_latest.columns else [])].dropna()
                    if not latest_rows_all.empty:
                        X_latest = latest_rows_all[driver_cols].astype(float).values
                        ones_latest = np.ones((X_latest.shape[0], 1))
                        des_latest = np.hstack([ones_latest, X_latest])
                        for target in target_cols:
                            beta_t = betas.get(target)
                            if beta_t is None:
                                continue
                            pred_base_rows = des_latest @ beta_t
                            X_bumped = X_latest * (1 + delta_pct)
                            des_bumped = np.hstack([ones_latest, X_bumped])
                            pred_bumped_rows = des_bumped @ beta_t
                            diffs = pred_bumped_rows - pred_base_rows
                            if wcol and wcol in latest_rows_all.columns:
                                w_latest = pd.to_numeric(latest_rows_all[wcol], errors="coerce").fillna(0.0).values.astype(float)
                                delta_value = float(np.nansum(diffs * w_latest))
                                baseline_total = float(np.nansum(pd.to_numeric(latest_rows_all[target], errors="coerce").values.astype(float) * w_latest))
                            else:
                                delta_value = float(np.nansum(diffs))
                                baseline_total = float(latest_rows_all[target].sum())
                            delta_pct_target = None
                            if baseline_total and abs(baseline_total) > 1e-8:
                                delta_pct_target = delta_value / baseline_total
                            joint_targets[target] = {
                                "projected": baseline_total + delta_value,
                                "delta": delta_value,
                                "delta_pct": delta_pct_target,
                            }
                scenarios_struct["joint"] = {"targets": joint_targets}
            except Exception:
                pass

        notes.append(
            "[SCENARIO_BASELINE] " + json.dumps({
                "frame": frame_name,
                "baseline": {metric: _safe_float(baseline_means.get(metric)) for metric in metrics_needed},
                "rows_used": int(len(df_work)),
                "mode": "regression",
            }, ensure_ascii=False)
        )

        # Prefer the no-join monthly aggregation approach when available
        score = len(df_work) + (10000 if frame_name == "__monthly_totals_nojoin__" else 0)
        for target in target_cols:
            r2 = r2_scores.get(target)
            if r2 is not None:
                score += max(0.0, r2) * 100

        if score <= best_score:
            continue

        best_score = score
        # Headline totals for the first driver-target
        headline_target = target_cols[0] if target_cols else None
        headline_driver = driver_cols[0] if driver_cols else None
        baseline_total = None; delta_total = None; projected_total = None; delta_pct_total = None
        corr_head = None
        if headline_target and headline_driver:
            try:
                # Baseline total in latest period slice (weighted if applicable)
                sel_cols = [headline_target] + driver_cols + ([wcol] if wcol and wcol in slice_latest.columns else [])
                latest_for_target = slice_latest[sel_cols].dropna()
                if wcol and wcol in latest_for_target.columns:
                    baseline_total = float(np.nansum(pd.to_numeric(latest_for_target[headline_target], errors="coerce").values.astype(float) * pd.to_numeric(latest_for_target[wcol], errors="coerce").fillna(0.0).values.astype(float)))
                else:
                    baseline_total = float(latest_for_target[headline_target].sum())
                head_entry = scenarios_struct.get(headline_driver, {}).get("targets", {}).get(headline_target, {})
                delta_total = _safe_float(head_entry.get("delta"))
                if delta_total is not None:
                    projected_total = baseline_total + delta_total
                    if abs(baseline_total) > 1e-8:
                        delta_pct_total = delta_total / baseline_total
                corr_head = _safe_float(corr_by_target_driver.get((headline_target, headline_driver)))
            except Exception:
                pass

        # Sensitivity over multiple shocks using same beta for headline target, if available
        sensitivity_summary = None
        try:
            if headline_target and headline_driver and headline_target in betas:
                beta = betas[headline_target]
                latest_rows = slice_latest[driver_cols + [headline_target]].dropna()
                if not latest_rows.empty:
                    X_latest = latest_rows[driver_cols].astype(float).values
                    ones_latest = np.ones((X_latest.shape[0], 1))
                    des_latest = np.hstack([ones_latest, X_latest])
                    pred_base_rows = des_latest @ beta
                    Xd = X_latest[:, driver_cols.index(headline_driver)]
                    shocks = [0.01, 0.03, 0.05, 0.10]
                    out = {}
                    for s in shocks:
                        X_b = X_latest.copy(); X_b[:, driver_cols.index(headline_driver)] = Xd * (1 + s)
                        des_b = np.hstack([ones_latest, X_b])
                        pred_b = des_b @ beta
                        d = float(np.nansum(pred_b - pred_base_rows))
                        out[f"{int(s*100)}%"] = {
                            "delta_total": d,
                            "projected_total": None if baseline_total is None else baseline_total + d,
                        }
                    sensitivity_summary = out
        except Exception:
            pass

        # Structural break diagnostics (latest vs previous period)
        drift_flag = False
        representativeness_note = None
        try:
            if latest_period is not None:
                periods = sorted(df_work["__PERIOD__"].dropna().astype(str).unique().tolist())
                if len(periods) >= 2:
                    prev_period = periods[-2]
                    latest_slice_all = df_work[df_work["__PERIOD__"].astype(str) == str(latest_period)]
                    prev_slice = df_work[df_work["__PERIOD__"].astype(str) == str(prev_period)]
                    if headline_target and headline_target in latest_slice_all.columns and headline_target in prev_slice.columns:
                        n_latest = int(latest_slice_all.shape[0]); n_prev = int(prev_slice.shape[0])
                        mu_latest = float(pd.to_numeric(latest_slice_all[headline_target], errors="coerce").mean())
                        mu_prev = float(pd.to_numeric(prev_slice[headline_target], errors="coerce").mean())
                        if (n_prev > 0) and (n_latest < max(20, n_prev // 4) or (mu_prev and (mu_latest > 2*mu_prev or mu_latest < 0.5*mu_prev))):
                            drift_flag = True
                            representativeness_note = f"Latest period {latest_period} may be unrepresentative (rows={n_latest} vs {n_prev}, mean={mu_latest:.2f} vs {mu_prev:.2f})."
        except Exception:
            pass

        leakage_check = "passed"

        # Simple frame hash for provenance
        try:
            import hashlib
            hash_cols = [c for c in (["__PERIOD__"] + driver_cols + target_cols) if c in df_work.columns]
            sample_csv = df_work[hash_cols].head(10000).to_csv(index=False)
            frame_hash = hashlib.sha1(sample_csv.encode("utf-8")).hexdigest()
        except Exception:
            frame_hash = None

        # Log a concise diagnostic line
        try:
            notes.append(
                f"[RUN] mode=regression, grain={time_grain}, period={latest_period}, rows={len(df_work)}, "
                f"delta_pct={delta_pct:.3f}, baseline_total={baseline_total}, delta_total={delta_total}, "
                f"corr={corr_head}, model=ols_trim_1pct, frame_hash={frame_hash}"
            )
        except Exception:
            pass

        best_payload = {
            "spec": {
                "tool": "scenario_projection",
                "step": frame_name,
                "column": None,
                "description": "projection via OLS bump-test (ceteris paribus)",
            },
            "result": {
                "ok": True,
                "value": {
                    "frame": frame_name,
                    "rows_used": int(len(df_work)),
                    "delta_percent": delta_pct,
                    "drivers": driver_cols,
                    "targets": target_cols,
                    "baseline": {metric: _safe_float(baseline_means.get(metric)) for metric in metrics_needed},
                    "relationships": relationships_struct,
                    "scenarios": scenarios_struct,
                    "fit": {target: _safe_float(r2_scores.get(target)) for target in target_cols},
                    "columns": mapping,
                    "mode": "regression",
                    "time_grain": time_grain,
                    "period": latest_period,
                    "latest_rows_n": int(slice_latest.shape[0]) if isinstance(slice_latest, pd.DataFrame) else None,
                    # headline totals binding for UI
                    "baseline_total": baseline_total,
                    "projected_total": projected_total,
                    "delta_total": delta_total,
                    "delta_pct_total": delta_pct_total,
                    "corr_headline": corr_head,
                    "sensitivity_summary": sensitivity_summary,
                    # provenance & policies (placeholders/defaults where not available)
                    "cohort_filter_json": None,
                    "frame_hash": frame_hash,
                    "model_spec": {
                        "engine": "ols_trim_1pct",
                        "features": driver_cols,
                        "intercept": True,
                    },
                    "weights_policy": weights_policy,
                    "lag_policy": (f"lag_{lag_k}" if lag_k else "none"),
                    "currency": None,
                    "scale": "units",
                    "price_level": "nominal",
                    "drivers_changed": [headline_driver] if headline_driver else [],
                    "engine_version": "scenario_v3",
                    "run_id": str(uuid.uuid4()),
                    "ci_delta_total": None,
                    "ci_method": None,
                    "drift_flag": drift_flag,
                    "leakage_check": leakage_check,
                    "representativeness_note": representativeness_note,
                },
            },
        }
        # propagate CI for headline into top-level if available
        try:
            if headline_driver and headline_target:
                head_entry = scenarios_struct.get(headline_driver, {}).get("targets", {}).get(headline_target, {})
                head_ci = head_entry.get("ci")
                head_ci_method = head_entry.get("ci_method")
                if head_ci:
                    best_payload["result"]["value"]["ci_delta_total"] = head_ci
                    best_payload["result"]["value"]["ci_method"] = head_ci_method or "ols_delta_linearization"
        except Exception:
            pass
        best_notes = notes
    if best_payload is None:
        # Strict temporal OLS only � do not fall back to ratios/cross-sectional
        frame_name = None
        months = None
        rows = None
        try:
            if fallback_candidate is not None:
                frame_name, df_model, _ = fallback_candidate
                rows = int(len(df_model)) if df_model is not None else None
                if df_model is not None and "__PERIOD__" in df_model.columns:
                    months = int(df_model["__PERIOD__"].astype(str).nunique())
        except Exception:
            pass
        note = {
            "message": "Temporal OLS skipped: insufficient temporal coverage or constant series.",
            "frame": frame_name,
            "distinct_periods": months,
            "rows": rows,
            "requirement": ">= 6 distinct months with non-constant driver/target",
            "action": "Widen the window (e.g., last 12-24 months) to enable temporal OLS."
        }
        return ["[SCENARIO_SKIPPED] " + json.dumps(note, ensure_ascii=False)], None

    return best_notes, best_payload



# --- end helper ---

# Page

def load_catalog(views: List[str]) -> Dict[str, Dict[str, Any]]:
    catalog = {}
    for v in views:
        tried = [v]
        try:
            try:
                df_raw = _describe_semantic_df(session, v)
            except Exception:
                qv = '"' + v.replace('.', '"."') + '"'
                tried.append(qv)
                df_raw = _describe_semantic_df(session, qv)
            parsed = parse_describe_df(df_raw)
            catalog[v] = {"describe_ok": parsed["ok"], "fields": parsed["fields"], "types": parsed["types"], "raw": parsed["raw"], "tried": tried}
            trace("catalog", {v: {"fields": len(parsed["fields"]), "ok": parsed["ok"]}})
        except Exception as e:
            trace("catalog_error", {v: str(e)})
            catalog[v] = {"describe_ok": False, "fields": [], "types": {}, "raw": None, "tried": tried}
    return catalog

# --- Sidebar and main render wrappers (refactored) ---

def render_sidebar(container, state=None, shared=None):
    """Render sidebar controls into provided `container` (e.g., st.sidebar).
    Preserves original sidebar contents from the uploaded app.
    """
    

    try:
        with container:

            # ensure user_prefs exist to avoid Streamlit session_state KeyError on first run
            try:
                st.session_state.setdefault(
                    "user_prefs",
                    {
                        "widen_time": True,
                        "dev_mode": False,
                        "months_window": DEFAULT_MONTHS_WINDOW,
                        "fit_on_load": True,
                    }
                )
            except Exception:
                # In rare cases session_state may not be available yet; ignore
                pass
            #st.subheader("Layout")
            #st.markdown("---") 
            #st.subheader("Behavior")
            #st.session_state["user_prefs"]["months_window"] = st.number_input("Default time window (months)", 1, 60, 12)
            st.session_state.setdefault("user_prefs", {})["months_window"] = DEFAULT_MONTHS_WINDOW

            #st.session_state["user_prefs"]["widen_time"] = st.toggle("Auto-widen to 36 months if data is thin", value=True)
            st.session_state.setdefault("user_prefs", {})["widen_time"] = True
            #st.session_state["user_prefs"]["fit_on_load"] = st.toggle("Fit columns on load", value=True)
            st.session_state.setdefault("user_prefs", {})["fit_on_load"] = True
            #st.session_state["user_prefs"]["dev_mode"] = st.toggle("Developer mode", value=False)
            st.session_state.setdefault("user_prefs", {})["dev_mode"] = False

            res = st.session_state.get("last_result") or {}
            rr = res.get("reasoner_results") if isinstance(res, dict) else None
            if rr:
                st.markdown("### Reasoner Evidence")
                for step_id, reasoners in (rr or {}).items():
                    if not isinstance(reasoners, dict):
                        continue
                    with st.expander(f"{step_id} ({len(reasoners)} reasoner(s))", expanded=False):
                        for rid, entry in reasoners.items():
                            st.markdown(f"**{rid}**")
                            drill = (entry or {}).get("drilldowns") or {}
                            if "steps" in drill:
                                for sid, payload in (drill.get("steps") or {}).items():
                                    rows = (payload or {}).get("rows") or []
                                    st.caption(f"{sid}: {len(rows)} source rows")
                                    sample_records = []
                                    for row_block in rows[:2]:
                                        recs = (row_block or {}).get("records") or []
                                        if recs:
                                            sample_records.extend(recs[:3])
                                    if sample_records:
                                        try:
                                            st.dataframe(pd.DataFrame(sample_records), height=180)
                                        except Exception:
                                            st.json(sample_records)
                                    else:
                                        st.write("No drilldown records.")
                            elif "machines" in drill:
                                st.caption(f"machines: {len(drill.get('machines') or [])}")
                                st.json(drill.get("machines"))
                            elif "audits" in drill:
                                st.caption(f"audits: {len(drill.get('audits') or [])}")
                                st.json(drill.get("audits"))

            #st.markdown("---")
            #if st.button("Reload catalog"): load_catalog.clear(); build_schema_synonyms.clear(); st.rerun()
            
    except Exception as e:
        st.error('Sidebar render failed: ' + str(e))
# ---------- New helpers for chart selection & display ----------

# --- ID/name detection used by chart + comparison helpers ---
def is_identifier_name(name: str) -> bool:
    """
    Return True if the column name looks like an identifier (e.g., RMID, *_ID).
    This is a *name-based* check (no DataFrame needed), so it can be used in helpers.
    """
    if not isinstance(name, str):
        return False
    u = name.upper().strip()
    return (
        u in ("RMID", "RM_ID", "MANDATEID", "MANDATE_ID", "RELATIONSHIP_MANAGER_ID")
        or u.endswith("_ID")
        or u.endswith("ID")
    )
# --- /ID/name detection ---

def determine_possible_chart_types(df: pd.DataFrame, x: str, y: str) -> list[str]:
    """Return allowed chart types for (x,y) on df."""
    out = []
    try:
        if x in df.columns and y in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                out.append("line")
            if pd.api.types.is_numeric_dtype(df[x]) and pd.api.types.is_numeric_dtype(df[y]):
                out.append("scatter")
            if (pd.api.types.is_string_dtype(df[x]) or is_identifier_name(x) or not pd.api.types.is_numeric_dtype(df[x])) and pd.api.types.is_numeric_dtype(df[y]):
                out.append("bar")
        # If nothing matched, always allow table as fallback
    except Exception:
        pass
    if not out:
        out = ["table"]
    return out

def determine_best_chart_type(df: pd.DataFrame, x: str, y: str) -> str:
    """Pick a single best chart from allowed types."""
    allowed = determine_possible_chart_types(df, x, y)
    # Preference order: line -> scatter -> bar -> table
    for prefer in ("line", "scatter", "bar", "table"):
        if prefer in allowed:
            return prefer
    return allowed[0]

def prefer_display_dimension(df: pd.DataFrame, x: str, rm_lookup: Optional[pd.DataFrame]=None) -> tuple[str,str]:
    """
    If x is an identifier like RMID and we can map to a name column (RM_NAME), return
    (display_col, underlying_col). display_col is used in chart axis labels and title;
    underlying_col is the column used for grouping/aggregation. If no mapping, return (x,x).
    """
    # prefer RM_NAME mapping if available
    try:
        if is_identifier_name(x):
            # if df already has RM_NAME, use that for display
            if "RM_NAME" in df.columns:
                return ("RM_NAME", x)
            # otherwise, if rm_lookup provided, create an RM_NAME column by merging (done earlier ideally)
            if rm_lookup is not None:
                # caller should merge before calling plot; return a hint
                return ("RM_NAME", x)
    except Exception:
        pass
    return (x, x)


# ---------- robust comparison + mapping helpers (N-way, any dimension) ----------
import re



from typing import Iterable

def _pretty_metric_name(col: str) -> str:
    """
    Turn S1__PERFORMANCE__AVG -> "Performance (avg)"
    Works for ...__SUM|AVG|MEDIAN|MIN|MAX|COUNT|LAST
    """
    if not isinstance(col, str):
        return "Value"
    m = re.match(r"^[A-Z0-9]+__([A-Z0-9_]+)__(SUM|AVG|MEDIAN|MIN|MAX|COUNT|LAST)$", col, re.I)
    if m:
        field = m.group(1).replace("_", " ").title()
        agg   = m.group(2).lower()
        return f"{field} ({agg})"
    return col.replace("_", " ").title()

def _unique_sorted(values: Iterable):
    try:
        s = sorted(set([v for v in values if v is not None]))
    except Exception:
        s = []
    return s

def _year_range_str(years: list[int]) -> str:
    if not years:
        return ""
    if len(years) == 1:
        return str(years[0])
    # If few unique values (<=4), list them; else show range
    if len(years) <= 4:
        return ", ".join(str(y) for y in years)
    return f"{years[0]}–{years[-1]}"

def _month_name(m: int) -> str:
    # 1-based month → short name
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    if isinstance(m, int) and 1 <= m <= 12:
        return names[m-1]
    return str(m)

def _month_range_str(years: list[int], months: list[int]) -> str:
    # When both year & month present, build a compact range like "Jan 2025–Mar 2025"
    try:
        pairs = _unique_sorted(list(zip(years, months)))
    except Exception:
        pairs = []
    if not pairs:
        return ""
    if len(pairs) == 1:
        y, m = pairs[0]
        return f"{_month_name(int(m))} {int(y)}"
    y1, m1 = pairs[0]
    y2, m2 = pairs[-1]
    # If few points (<=4), list each; else show range
    if len(pairs) <= 4:
        return ", ".join(f"{_month_name(int(m))} {int(y)}" for y, m in pairs)
    return f"{_month_name(int(m1))} {int(y1)}–{_month_name(int(m2))} {int(y2)}"

def _monthdate_range_str(vals: Iterable[str]) -> str:
    """
    Handle YYYY-MM or pandas Period('YYYY-MM') columns like MONTH_DATE.
    """
    cleaned = []
    for v in vals:
        if v is None: 
            continue
        s = str(v)
        # Keep only YYYY-MM prefix if present
        m = re.match(r"^(\d{4})-(\d{2})", s)
        if m:
            cleaned.append((int(m.group(1)), int(m.group(2))))
    cleaned = _unique_sorted(cleaned)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        y, m = cleaned[0]
        return f"{_month_name(m)} {y}"
    if len(cleaned) <= 4:
        return ", ".join(f"{_month_name(m)} {y}" for y, m in cleaned)
    (y1, m1), (y2, m2) = cleaned[0], cleaned[-1]
    return f"{_month_name(m1)} {y1}–{_month_name(m2)} {y2}"

def _time_suffix(df) -> str:
    """
    Derive a compact time window suffix from common time columns:
    - (MEET_YEAR, MEET_MON) or (POSYEAR, POSMON): uses the actual (year, month) pairs per row
    - Single year columns: MEET_YEAR / POSYEAR / YEAR
    - Period-like month columns: MONTH_DATE, MONTH, PERIOD, MEET_MONTH (expects YYYY-MM)
    - Date-like columns: DATE / MEET_DATE / TX_DATE (falls back to year)
    Returns like: " — 2025", " — 2024–2025", " — Jan–Mar 2025", etc.
    """
    try:
        cols = list(getattr(df, "columns", []))
    except Exception:
        cols = []

    # --- Prefer explicit YEAR + MONTH pairs ---
    for y_col, m_col in [("MEET_YEAR", "MEET_MON"), ("POSYEAR", "POSMON")]:
        if y_col in cols and m_col in cols:
            try:
                # Build (year, month) from actual rows, not zip(unique years, unique months)
                pairs = []
                for y, m in df[[y_col, m_col]].dropna().itertuples(index=False):
                    try:
                        pairs.append((int(y), int(m)))
                    except Exception:
                        pass
                pairs = _unique_sorted(pairs)
            except Exception:
                pairs = []

            if not pairs:
                break

            years  = _unique_sorted([y for y, _ in pairs])
            if len(years) == 1:
                # Single year; decide how to show months
                y = years[0]
                months = _unique_sorted([m for _, m in pairs])
                if len(months) == 12:
                    return f" — {y}"
                # Contiguous helper
                def _is_contiguous(seq):
                    return all(b - a == 1 for a, b in zip(seq, seq[1:]))

                if len(months) <= 4:
                    return " — " + ", ".join(_month_name(m) for m in months) + f" {y}"
                if _is_contiguous(months):
                    return f" — {_month_name(months[0])} {y}–{_month_name(months[-1])} {y}"
                # Non-contiguous and many points → show range
                return f" — {_month_name(months[0])} {y}–{_month_name(months[-1])} {y}"

            # Multiple years
            y_first, m_first = pairs[0]
            y_last,  m_last  = pairs[-1]

            # If every year in the span has all 12 months, show just year range
            full_years = {}
            for y, m in pairs:
                full_years.setdefault(y, set()).add(m)
            if all(len(ms) == 12 for ms in full_years.values()):
                if len(years) == 2 and years[0] == years[1]:
                    return f" — {years[0]}"
                return f" — {years[0]}–{years[-1]}"

            # Otherwise show month+year range
            return f" — {_month_name(m_first)} {y_first}–{_month_name(m_last)} {y_last}"

    # --- Single YEAR column ---
    for y_col in ["MEET_YEAR", "POSYEAR", "YEAR"]:
        if y_col in cols:
            try:
                years = _unique_sorted([int(y) for y in df[y_col].dropna().tolist()])
                if not years:
                    break
                if len(years) == 1:
                    return f" — {years[0]}"
                return f" — {years[0]}–{years[-1]}"
            except Exception:
                pass

    # --- Period-like month column (YYYY-MM) ---
    for mcol in ["MONTH_DATE", "MONTH", "PERIOD", "MEET_MONTH"]:
        if mcol in cols:
            s = _monthdate_range_str(df[mcol].dropna().tolist())
            if s:
                return f" — {s}"

    # --- Datelike columns: fall back to year(s) ---
    for dcol in ["DATE", "MEET_DATE", "TX_DATE"]:
        if dcol in cols:
            try:
                years = _unique_sorted([int(str(v)[:4]) for v in df[dcol].dropna().tolist()])
                if not years:
                    continue
                if len(years) == 1:
                    return f" — {years[0]}"
                return f" — {years[0]}–{years[-1]}"
            except Exception:
                pass

    # Nothing detected
    return ""

def frame_label_for(fid: str, df, aliases: dict | None) -> str:
    """
    Produce a human-friendly label for a frame id (fid).
    Priority:
      1) explicit alias from _execute_plan() (e.g., S1__PROFIT_AMOUNT__SUM)
      2) df.attrs["label"] (we set this for compare_auto_any)
      3) infer from columns + attrs (series_label/value_label/x_label) + time suffix
      4) fallback to fid
    """
    # 1) alias from planner/steps
    if aliases and fid in aliases and aliases[fid]:
        base = _pretty_metric_name(aliases[fid])
        return f"{base}{_time_suffix(df)}"

    # 2) explicit DataFrame label
    try:
        frame_attrs = getattr(df, "attrs", {}) or {}
    except Exception:
        frame_attrs = {}
    t = frame_attrs.get("label")
    if t:
        return f"{t}{_time_suffix(df)}"

    # 3) infer from columns + attrs
    try:
        cols = list(getattr(df, "columns", []))
    except Exception:
        cols = []

    series_label = frame_attrs.get("series_label", "Series")
    value_label  = frame_attrs.get("value_label", "Value")
    x_label      = frame_attrs.get("x_label", None)

    # detect metric style columns like S1__FOO__AVG
    metric_cols = [c for c in cols if isinstance(c, str) and re.search(r"__[A-Z]+$", c)]
    if metric_cols:
        metric = _pretty_metric_name(metric_cols[0])
        if not x_label:
            if "RM_NAME" in cols: x_label = "RM Name"
            elif "RMID" in cols:  x_label = "RMID"
            elif "MEET_YEAR" in cols and "MEET_MON" in cols: x_label = "Month"
            elif "MEET_YEAR" in cols: x_label = "Year"
            else: x_label = "Category"
        if "SERIES" in cols and "VALUE" in cols:
            return f"{metric} by {x_label} — {series_label} comparison{_time_suffix(df)}"
        return f"{metric} by {x_label}{_time_suffix(df)}"

    # 4) fallback
    return f"{fid}{_time_suffix(df)}"


YEAR_RE = re.compile(r"\b(20\d{2})\b", re.I)
VS_SPLIT_RE = re.compile(r"\b(?:vs\.?|versus|compared\s+to|against)\b", re.I)
MONTH_TOKEN_RE = re.compile(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
                            r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
                            r"dec(?:ember)?)\b", re.I)

_MONTHS = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}

def _extract_compare_terms(q: str) -> dict:
    """Detect explicit values user wants to compare (years, months, category tokens)."""
    q = q or ""
    years = [int(y) for y in YEAR_RE.findall(q)]
    # months → numbers (keep order, unique)
    months = []
    for m in MONTH_TOKEN_RE.findall(q):
        mm = _MONTHS[m[:3].lower()]
        if mm not in months: months.append(mm)

    # generic tokens split by 'vs/versus/compared to'
    tokens = []
    if VS_SPLIT_RE.search(q):
        parts = VS_SPLIT_RE.split(q)
        # split each side by commas or "and"
        for p in parts:
            for t in re.split(r"[,\s]+and\s+|,|/", p):
                tt = t.strip()
                if tt and len(tt) <= 32 and tt.lower() not in ("in","for","by","top","best","most","least"):
                    tokens.append(tt)
    # unique preserving order
    seen=set(); tokens=[t for t in tokens if not (t in seen or seen.add(t))]
    return {"years": years, "months": months, "tokens": tokens}

def _find_compare_column(df: pd.DataFrame, terms: dict) -> tuple[str|None, list]:
    """
    Pick a compare column (YEAR / MONTH / any low-cardinality category) and the target values.
    Returns (compare_col, values). Values are normalized to df's dtype where possible.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return (None, [])
    up = {c.upper(): c for c in df.columns}
    # 1) explicit years
    if terms["years"]:
        for k in ("MEET_YEAR","YEAR","CAL_YEAR","FISCAL_YEAR","POSYEAR"):
            if k in up: return (up[k], terms["years"])
        # if only a datetime exists, we still treat it as year
        for c in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    return (c, terms["years"])
            except Exception:
                pass
    # 2) explicit months (integers 1..12)
    if terms["months"]:
        for k in ("MEET_MON","MONTH","CAL_MONTH","FISCAL_MONTH","POSMON"):
            if k in up: return (up[k], terms["months"])
        for c in df.columns:
            if "MONTH" in c.upper(): return (c, terms["months"])
    # 3) general tokens → find a low-cardinality text column that contains them
    cand_text = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    for c in cand_text:
        try:
            cats = set(str(x).strip().lower() for x in df[c].dropna().unique().tolist() if str(x).strip())
            asked = [t for t in terms["tokens"] if t.lower() in cats]
            if len(asked) >= 2:
                return (c, asked)
        except Exception:
            continue
    return (None, [])

def _derive_compare_series(df: pd.DataFrame, compare_col: str) -> pd.Series:
    """Normalize compare column: datetime→year/month, numeric/text as is."""
    s = df[compare_col]
    if pd.api.types.is_datetime64_any_dtype(s):
        # Prefer year unless month-only was asked; the top-K stage doesn’t need exact month name
        return pd.to_datetime(s, errors="coerce").dt.year
    return s

def _choose_entity_column(df: pd.DataFrame) -> tuple[str, str]:
    """
    Return (x_display, x_id). Prefer RM_NAME for display with RMID as ID, else any non-numeric ID-like col.
    """
    up = {c.upper(): c for c in df.columns}
    rmid = up.get("RMID") or up.get("RM_ID") or up.get("RELATIONSHIP_MANAGER_ID")
    rmname = up.get("RM_NAME")
    if rmid and rmname:
        return ("RM_NAME", rmid)
    # fallbacks: any ID-like
    for c in df.columns:
        if is_identifier_name(str(c)):
            return (c, c)
    # last resort: first non-numeric col
    for c in df.columns:
        try:
            if not pd.api.types.is_numeric_dtype(df[c]): return (c, c)
        except Exception:
            pass
    return (df.columns[0], df.columns[0])

def _pick_metric_col(df: pd.DataFrame, prefer: list[str] = None) -> str | None:
    prefer = prefer or []
    up = {c.upper(): c for c in df.columns}
    for p in prefer:
        if p.upper() in up: return up[p.upper()]
    for c in df.columns:
        u=str(c).upper()
        if pd.api.types.is_numeric_dtype(df[c]) and any(k in u for k in ("PERFORMANCE","SCORE","AUM","REVENUE","PROFIT","AMOUNT","VALUE","AVG")):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]): return c
    return None

def _rm_lookup_from_frames(frames: dict) -> pd.DataFrame|None:
    """If any frame contains both RMID and RM_NAME, return a 2-col lookup."""
    for _, d in (frames or {}).items():
        if not isinstance(d, pd.DataFrame) or d.empty: continue
        up = {c.upper(): c for c in d.columns}
        rmid = up.get("RMID") or up.get("RM_ID") or up.get("RELATIONSHIP_MANAGER_ID")
        rmname = up.get("RM_NAME")
        if rmid and rmname:
            return d[[rmid, rmname]].dropna().drop_duplicates().rename(columns={rmid:"RMID", rmname:"RM_NAME"})
    return None

def make_multi_compare_frame_any(df: pd.DataFrame,
                                 *,
                                 x_id: str,
                                 metric_col: str,
                                 compare_col: str,
                                 compare_values: list,
                                 top_k: int = 5,
                                 agg: str = "avg",
                                 rm_lookup: pd.DataFrame|None = None) -> pd.DataFrame:
    """
    Return long: X_ID | (RM_NAME?) | SERIES | VALUE
    SERIES is normalized from compare_col; Top-K chosen by the **latest** value (if sortable) else by first.
    """
    d = df.copy()
    # Normalize SERIES
    ser = _derive_compare_series(d, compare_col)
    d["__SERIES__"] = ser
    # Filter to requested values if any
    if compare_values:
        # try to coerce values to dtype
        vs = []
        for v in compare_values:
            try:
                vs.append(type(ser.dropna().iloc[0])(v))
            except Exception:
                vs.append(v)
        d = d[ d["__SERIES__"].astype(str).isin([str(v) for v in vs]) ]
    # choose X_ID and display
    x_disp, x_col = _choose_entity_column(d if "RM_NAME" in d.columns else df)
    # aggregate
    keep = [x_col, "__SERIES__", metric_col]
    keep = [c for c in keep if c in d.columns]
    g = d[keep].dropna(subset=[x_col, metric_col]).copy()
    if is_identifier_name(x_col):
        g[x_col] = g[x_col].astype(str)
    if agg == "sum":
        ga = g.groupby([x_col, "__SERIES__"], dropna=False)[metric_col].sum()
    elif agg in ("avg","mean"):
        ga = g.groupby([x_col, "__SERIES__"], dropna=False)[metric_col].mean()
    elif agg == "median":
        ga = g.groupby([x_col, "__SERIES__"], dropna=False)[metric_col].median()
    elif agg == "count":
        ga = g.groupby([x_col, "__SERIES__"], dropna=False)[metric_col].count()
    elif agg == "last":
        # noop here; for snapshot metrics we expect a pre-filtered row per series
        ga = g.groupby([x_col, "__SERIES__"], dropna=False)[metric_col].last()
    else:
        ga = g.groupby([x_col, "__SERIES__"], dropna=False)[metric_col].sum()

    out = ga.reset_index().rename(columns={x_col:"X_ID","__SERIES__":"SERIES", metric_col:"VALUE"})

    # Top-K: pick by the latest (if sortable), else first SERIES
    try:
        latest = sorted(out["SERIES"].dropna().unique().tolist())[-1]
    except Exception:
        latest = out["SERIES"].dropna().unique().tolist()[0]
    tops = (out[out["SERIES"]==latest].sort_values("VALUE", ascending=False).head(top_k)["X_ID"]
            .astype(str).tolist())
    out = out[out["X_ID"].astype(str).isin(tops)]

    # Attach RM_NAME if available
    if rm_lookup is not None and not rm_lookup.empty:
        try:
            out["X_ID"] = out["X_ID"].astype(str)
        except Exception:
            pass
        lu = rm_lookup.copy()
        if "RMID" in lu.columns:
            try:
                lu["RMID"] = lu["RMID"].astype(str)
            except Exception:
                pass
        out = out.merge(lu, left_on="X_ID", right_on="RMID", how="left")
    return out  # columns: X_ID, SERIES, VALUE, (RMID, RM_NAME?)
# ---------- /robust comparison + mapping helpers ----------



# ---------- end new helpers ----------

# Local Snowflake session helpers
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val

_LAST_SNOWFLAKE_SESSION_ERROR: Optional[str] = None

def _build_local_snowpark_session() -> Optional[Any]:
    global _LAST_SNOWFLAKE_SESSION_ERROR
    account = _env("SNOWFLAKE_ACCOUNT")
    user = _env("SNOWFLAKE_USER")
    if not account or not user:
        _LAST_SNOWFLAKE_SESSION_ERROR = "Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER"
        return None

    cfg: Dict[str, Any] = {
        "account": account,
        "user": user,
    }

    # CRITICAL FIX: Use username_password_mfa by default (non-blocking)
    # Do NOT use "externalbrowser" in API/Streamlit context (causes infinite hang on MFA)
    authenticator = _env("SNOWFLAKE_AUTHENTICATOR")
    if not authenticator:
        # If password is available, use password or username_password_mfa
        # Do NOT default to "externalbrowser" (causes hang in non-interactive context)
        if _env("SNOWFLAKE_PASSWORD"):
            authenticator = "username_password_mfa"  # Use MFA but with passcode
        else:
            authenticator = None
    
    if authenticator:
        cfg["authenticator"] = authenticator

    password = _env("SNOWFLAKE_PASSWORD")
    if password:
        cfg["password"] = password

    passcode = _env("SNOWFLAKE_PASSCODE")
    if passcode:
        cfg["passcode"] = passcode

    passcode_in_password = _env("SNOWFLAKE_PASSCODE_IN_PASSWORD")
    if passcode_in_password is not None:
        cfg["passcode_in_password"] = passcode_in_password.lower() in ("1", "true", "yes", "y")

    role = _env("SNOWFLAKE_ROLE")
    if role:
        cfg["role"] = role
    warehouse = _env("SNOWFLAKE_WAREHOUSE")
    if warehouse:
        cfg["warehouse"] = warehouse
    database = _env("SNOWFLAKE_DATABASE")
    if database:
        cfg["database"] = database
    schema = _env("SNOWFLAKE_SCHEMA")
    if schema:
        cfg["schema"] = schema

    try:
        # Add connection timeout to prevent indefinite hangs
        # Default is ~30s, but explicitly set here for clarity
        print("[DEBUG] Creating Snowflake session with config:", {k: v if k not in ("password", "passcode") else "***" for k, v in cfg.items()})
        import sys
        sys.stdout.flush()
        
        session = Session.builder.configs(cfg).create()
        print("[DEBUG] Snowflake session created successfully")
        sys.stdout.flush()
        return session
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        _LAST_SNOWFLAKE_SESSION_ERROR = error_msg
        print(f"[ERROR] Failed to create Snowflake session: {error_msg}", file=sys.stderr)
        import sys
        sys.stderr.flush()
        return None

# Session bootstrap (mirror of Client Profitability style)
def init_snowflake_session(shared=None):
    """
    Try shared.session.get_session() first, then fall back to Snowpark's get_active_session(),
    then build a local Snowpark session from env vars.
    Returns a Snowpark Session or None.
    """
    # shared launcher (if passed in)
    try:
        if shared is not None and hasattr(shared, "get_session"):
            sess = shared.get_session()
            if sess is not None:
                return sess
    except Exception:
        pass

    # module-level `get_session` imported from shared/session.py (if available)
    try:
        if callable(get_session):
            sess = get_session()
            if sess is not None:
                return sess
    except Exception:
        pass

    # final fallback: active Snowpark session (if running inside Snowflake)
    try:
        return get_active_session()
    except Exception:
        pass

    # local MFA-friendly fallback (externalbrowser or passcode)
    return _build_local_snowpark_session()


def render_main(state=None, shared=None):
    """Render the main Streamlit UI. Pass `shared` if your launcher provides shared.get_session().
    This function contains the original app UI (excluding sidebar) and preserves original control flow.
    """
    # Provide `shared` in module globals for backward compatibility if needed
    globals()['shared'] = shared
    
    # Early export of orchestrate function to module level (skip UI if already done)
    _export_orchestrate_once = globals().get('_orchestrate_exported', False)
    if _export_orchestrate_once:
        # Already exported; skip heavy UI initialization when called from orchestrator
        return
    
    # Mark that we're going to export (will be set after orchestrate is defined)
    globals()['_orchestrate_will_export'] = True

    # Must be the first Streamlit call
    try:
        st.set_page_config(page_title="AI Analyst", layout="wide")
    except Exception:
        pass

    # Ensure user preferences exist before any access
    st.session_state.setdefault("user_prefs", {})
    st.session_state["user_prefs"].setdefault("widen_time", True)
    st.session_state["user_prefs"].setdefault("dev_mode", False)
    st.session_state["user_prefs"].setdefault("months_window", DEFAULT_MONTHS_WINDOW)
    st.session_state["user_prefs"].setdefault("fit_on_load", True)


    # Paste this entire block into your AI Insights page (e.g., right after `st.set_page_config(...)` in render_main).
# IMPORTANT: To avoid duplicate headings, remove or comment out any existing `st.title(...)` / `st.caption(...)`
# lines in that file (or keep them removed) before using this block.
    

    try:
        prefs = st.session_state.get('user_prefs', {})
        data_w = st.sidebar.slider("Data panel width", 20, 60, int(prefs.get('data_w', 26)))
        insights_w = st.sidebar.slider("Insights panel width", 20, 60, int(prefs.get('insights_w', 46)))
        charts_w = max(10, 100 - data_w - insights_w)
        st.sidebar.caption(f"Charts auto width: {charts_w}%")
        # persist preferences
        st.session_state.setdefault('user_prefs', {})
        st.session_state['user_prefs']['data_w'] = int(data_w)
        st.session_state['user_prefs']['insights_w'] = int(insights_w)
    except Exception:
        # fallback defaults if sidebar not available
        data_w = st.session_state.get('user_prefs', {}).get('data_w', 34)
        insights_w = st.session_state.get('user_prefs', {}).get('insights_w', 33)
        charts_w = max(10, 100 - int(data_w) - int(insights_w))
    # prepare ratios for layout
    ratios = [int(data_w), int(insights_w), int(charts_w)]

    try:
        st.markdown(r"""
<style id="ai-analyst-ui-v4">
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  background:#f6f8fb !important;
  color:#12141a !important;
}
/* ───────────────────────────── General ───────────────────────────── */
.block-container, main, .stApp { overflow: visible !important; } /* never clip floating elements */

/* ──────────────────────── Full-width blue header ─────────────────── */
.dashboard-top-card{
  width:100% !important;
  box-sizing:border-box !important;
  margin:16px 0 8px 0 !important;
  padding:28px 36px !important;
  background:#0133cc !important;
  color:#fff !important;
  border-radius:16px !important;
  box-shadow:0 18px 60px rgba(11,59,154,.10) !important;
}
.dashboard-top-card .dashboard-title{
  margin:0 0 6px 0 !important;
  font-size:2.7rem !important;
  font-weight:850 !important;
  letter-spacing:-.5px !important;
  line-height:1.02 !important;
  color:#fff !important;
}
.dashboard-top-card .dashboard-subcaption{
  margin:0 !important;
  font-size:.95rem !important;
  color:rgba(255,255,255,.95) !important;
}

/* ───────────── Chat/Text input pill + floating submit button ─────── */
div[data-testid="stChatInput"],
div[data-testid="stTextInput"]{
  position: relative !important;
  overflow: visible !important;                 /* avoid cropped button */
  background: linear-gradient(180deg,#fbfdff,#f7fbff) !important;
  border: 1px solid rgba(11,59,154,.10) !important;
  border-radius: 28px !important;
  padding: 8px 16px !important;
  box-shadow: 0 8px 30px rgba(11,59,154,.035) !important;
  display: flex !important;
  align-items: center !important;
  outline: none !important;
}

/* INNER WRAPPER (this is what was drawing the thin inner line) */
div[data-testid="stChatInput"] > div,
div[data-testid="stTextInput"] > div{
  width: 100% !important;
  display: flex !important;
  align-items: center !important;
  padding-right: 112px !important;              /* space for 56px btn */
  box-sizing: border-box !important;
  border: 0 !important;                          /* ← kill inner border */
  background: transparent !important;
  box-shadow: none !important;                   /* ← kill inner shadow */
}
div[data-testid="stChatInput"] > div::after,
div[data-testid="stTextInput"] > div::after{     /* BaseWeb focus ring */
  content: none !important;
}

/* Also nuke any BaseWeb descendants that add inset rings */
div[data-testid="stChatInput"] [data-baseweb],
div[data-testid="stTextInput"] [data-baseweb]{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
div[data-testid="stChatInput"] *:focus,
div[data-testid="stTextInput"] *:focus{
  outline: none !important;
  box-shadow: none !important;
}

/* Clean outer-pill focus (single outline only) */
div[data-testid="stChatInput"]:focus-within,
div[data-testid="stTextInput"]:focus-within{
  border-color: #2e5cff !important;
  box-shadow: 0 8px 30px rgba(11,59,154,.06) !important;  /* no inner ring */
}

/* Input element */
div[data-testid="stChatInput"] textarea,
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] textarea{
  background: transparent !important;
  border: none !important;
  outline: none !important;
  width: 100% !important;
  height: 48px !important;
  min-height: 44px !important;
  padding: 12px 14px !important;
  font-size: 16px !important;
  line-height: 22px !important;
  color: #0b1b2b !important;
  box-sizing: border-box !important;
}

/* Floating round submit button (never clipped) */
div[data-testid="stChatInput"] button,
div[data-testid="stTextInput"] button{
  position: absolute !important;
  right: 12px !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  width: 56px !important;
  height: 56px !important;
  min-width: 56px !important;
  border-radius: 999px !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: linear-gradient(180deg,#eaf1ff,#dfeaff) !important;
  border: 1px solid rgba(11,59,154,.12) !important;
  box-shadow: 0 12px 28px rgba(11,59,154,.10) !important;
  color: #344154 !important;
  z-index: 10 !important;
  cursor: pointer !important;
}

/* Mobile tweaks */
@media (max-width: 1100px){
  .dashboard-top-card{ padding:20px 18px !important; border-radius:12px !important; }
  div[data-testid="stChatInput"] > div,
  div[data-testid="stTextInput"] > div{ padding-right: 96px !important; }
  div[data-testid="stChatInput"] button,
  div[data-testid="stTextInput"] button{ width: 48px !important; height: 48px !important; right: 8px !important; }
}
</style>

<!-- Header card -->
<div class="dashboard-top-card">
  <h1 class="dashboard-title">Your AI Analyst</h1>
  <p class="dashboard-subcaption">Now smarter across Data</p>
</div>
        """, unsafe_allow_html=True)


        
        # Session + trace
        def _now():
            return datetime.now().strftime("%H:%M:%S")
        
        def trace(kind: str, payload: Any):
            if 'trace' not in st.session_state: st.session_state['trace'] = []
            st.session_state['trace'].append({'when': _now(), 'kind': kind, 'payload': payload if isinstance(payload, (dict, list)) else str(payload)})
            if len(st.session_state['trace']) > 500:
                st.session_state['trace'] = st.session_state['trace'][-500:]
        
        # Resolve Snowflake session (container/shared -> Snowsight -> local MFA)
        session = init_snowflake_session(shared=shared)
        if session is None:
            detail = _LAST_SNOWFLAKE_SESSION_ERROR or "Unknown error (check env vars and Snowflake connectivity)."
            st.error(
                "Could not get a Snowflake session. Run inside Snowsight or set "
                "SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER (plus "
                "SNOWFLAKE_AUTHENTICATOR=externalbrowser for MFA). "
                f"Details: {detail}"
            )
            st.stop()
        globals()["_GLOBAL_SNOWFLAKE_SESSION"] = session
        globals()["session"] = session

        # Utils
        def json_extract(s: str):
            if not isinstance(s, str): return None
            for lo, hi in ((s.find("{"), s.rfind("}")), (s.find("["), s.rfind("]"))):
                if lo != -1 and hi != -1 and hi > lo:
                    try: return safe_json_loads(s[lo:hi+1])
                    except Exception: pass
            return None
        
        def clean_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None: return df
            d = df.copy()
            d.columns = [str(c).strip() for c in d.columns]
            return d

        def _coerce_int128_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if not isinstance(df, pd.DataFrame):
                return df
            import numpy as np
            d = df.copy()
            for c in d.columns:
                try:
                    s = d[c]
                    dtype_str = str(getattr(s, "dtype", "")).lower()
                    if "int128" in dtype_str or "uint128" in dtype_str:
                        d[c] = s.astype(str)
                        continue
                    if s.dtype == object:
                        sample = s.dropna().head(20)
                        if sample.map(lambda v: isinstance(v, (np.int128, np.uint128))).any():
                            d[c] = s.astype(str)
                except Exception:
                    continue
            return d
        
        def parse_dates(df):
            if df is None: 
                return df
            d = df.copy()
            hard_exclude = {
                "MEET_YEAR","MEET_MON","YEAR","MONTH",
                "CAL_YEAR","CAL_MONTH","FISCAL_YEAR","FISCAL_MONTH",
                "POSYEAR","POSMON"
            }
            def _looks_like_calendar_date(series) -> bool:
                s = series.astype(str)
                # Only treat strings with separators and a 4-digit token as dates
                return s.str.contains(r"(\d{4}.*[-/])|([-/].*\d{4})", regex=True, na=False).any()

            for c in d.columns:
                try:
                    if d[c].dtype == object and c.upper() not in hard_exclude and _looks_like_calendar_date(d[c]):
                        d[c] = pd.to_datetime(d[c], errors="ignore")
                except Exception:
                    pass
            return _coerce_int128_cols(d)

        def _normalize_rai_spec(question_text: str, spec: dict, specs_list: list) -> dict:
            if not isinstance(spec, dict):
                return spec
            alias_map = {"rm_score": "performance"}
            binds = spec.get("bind") or []
            alias_to_entity = {}
            for b in binds:
                if isinstance(b, dict) and b.get("alias") and b.get("entity"):
                    alias_to_entity[b["alias"]] = b["entity"]
            fields_by_entity = {}
            roles_by_entity = {}
            aggs_by_entity = {}
            types_by_entity = {}
            field_name_by_entity = {}
            derived_by_entity = {}
            exprs_by_entity = {}
            expr_to_field_by_entity = {}
            for s in specs_list or []:
                try:
                    fields = list(s.fields or [])
                    derived = list(getattr(s, "derived_fields", []) or [])
                    fields_by_entity[s.name] = set(fields)
                    derived_by_entity[s.name] = set(derived)
                    field_name_by_entity[s.name] = {str(f).lower(): f for f in (fields + derived)}
                    roles_by_entity[s.name] = dict(getattr(s, "field_roles", {}) or {})
                    aggs_by_entity[s.name] = dict(getattr(s, "field_aggs", {}) or {})
                    types_by_entity[s.name] = dict(getattr(s, "field_types", {}) or {})
                    exprs = dict(getattr(s, "field_exprs", {}) or {})
                    exprs_by_entity[s.name] = exprs
                    expr_to_field_by_entity[s.name] = {
                        str(expr).lower(): field for field, expr in (exprs or {}).items() if expr
                    }
                except Exception:
                    continue

            def _rewrite_props(node):
                if isinstance(node, dict):
                    prop = node.get("prop")
                    prop_norm = prop.lower() if isinstance(prop, str) else prop
                    if prop_norm in alias_map:
                        node = dict(node)
                        node["prop"] = alias_map[prop_norm]
                    if "alias" in node and "prop" in node:
                        alias = node.get("alias")
                        prop = node.get("prop")
                        prop_norm = prop.lower() if isinstance(prop, str) else prop
                        ent = alias_to_entity.get(alias)
                        canonical = (field_name_by_entity.get(ent, {}) or {}).get(prop_norm)
                        if not canonical:
                            canonical = (expr_to_field_by_entity.get(ent, {}) or {}).get(prop_norm)
                        mapped = canonical or prop
                        expr = (exprs_by_entity.get(ent, {}) or {}).get(canonical) if canonical else None
                        if expr and re.match(r"^[A-Za-z0-9_]+$", str(expr)):
                            mapped = str(expr).lower()
                        if mapped and mapped != prop:
                            node = dict(node)
                            node["prop"] = mapped
                    return {k: _rewrite_props(v) for k, v in node.items()}
                if isinstance(node, list):
                    return [_rewrite_props(v) for v in node]
                return node

            spec = _rewrite_props(spec)

            meeting_entities = {"MeetingSentiment", "MeetingDetail"}
            mandate_entities = {"MandateMonthlySummary", "MandateProfitability"}
            where_list = spec.get("where") or []
            fixed_where = []
            for pred in where_list:
                if not isinstance(pred, dict):
                    fixed_where.append(pred)
                    continue
                if pred.get("op") == "==" and isinstance(pred.get("left"), dict) and isinstance(pred.get("right"), dict):
                    left = dict(pred["left"])
                    right = dict(pred["right"])
                    if str(left.get("prop") or "").lower() == "mandateid" and str(right.get("prop") or "").lower() == "mandateid":
                        left_ent = alias_to_entity.get(left.get("alias", ""))
                        right_ent = alias_to_entity.get(right.get("alias", ""))
                        if (left_ent in meeting_entities and right_ent in mandate_entities) or (
                            right_ent in meeting_entities and left_ent in mandate_entities
                        ):
                            left_type = (types_by_entity.get(left_ent, {}) or {}).get("mandateid", "")
                            right_type = (types_by_entity.get(right_ent, {}) or {}).get("mandateid", "")
                            if left_type and right_type and str(left_type).lower() != str(right_type).lower():
                                if "rmid" in fields_by_entity.get(left_ent, set()) and "rmid" in fields_by_entity.get(
                                    right_ent, set()
                                ):
                                    left["prop"] = "rmid"
                                    right["prop"] = "rmid"
                                    pred = {"op": "==", "left": left, "right": right}
                fixed_where.append(pred)
            def _date_type_for(alias: str | None, prop: str | None) -> str:
                ent = alias_to_entity.get(alias or "")
                prop_norm = str(prop or "").lower()
                canonical = (field_name_by_entity.get(ent, {}) or {}).get(prop_norm)
                if not canonical:
                    canonical = (expr_to_field_by_entity.get(ent, {}) or {}).get(prop_norm)
                if not canonical:
                    return ""
                dtype = (types_by_entity.get(ent, {}) or {}).get(canonical)
                return str(dtype or "").lower()

            ym_match = re.compile(r"^(\d{4})-(\d{2})$")
            date_fixed_where = []
            for pred in fixed_where:
                if not isinstance(pred, dict) or pred.get("op") != "==":
                    date_fixed_where.append(pred)
                    continue
                left = pred.get("left")
                right = pred.get("right")
                if not isinstance(left, dict) or not isinstance(right, dict) or "value" not in right:
                    date_fixed_where.append(pred)
                    continue
                dtype = _date_type_for(left.get("alias"), left.get("prop"))
                if dtype not in ("date", "datetime", "timestamp"):
                    date_fixed_where.append(pred)
                    continue
                raw_val = right.get("value")
                if not isinstance(raw_val, str):
                    date_fixed_where.append(pred)
                    continue
                raw_val = raw_val.strip()
                m = ym_match.match(raw_val)
                if m:
                    try:
                        import calendar
                        year = int(m.group(1))
                        month = int(m.group(2))
                        last_day = calendar.monthrange(year, month)[1]
                        low_val = f"{year:04d}-{month:02d}-01"
                        high_val = f"{year:04d}-{month:02d}-{last_day:02d}"
                        if dtype in ("datetime", "timestamp"):
                            high_val = f"{high_val}T23:59:59"
                        pred = {
                            "between": {
                                "left": left,
                                "low": {"value": low_val},
                                "high": {"value": high_val},
                            }
                        }
                    except Exception:
                        pass
                    date_fixed_where.append(pred)
                    continue
                date_fixed_where.append(pred)

            spec["where"] = date_fixed_where

            def _canonical_field(ent: str | None, prop: str | None) -> str:
                if not ent or not prop:
                    return ""
                prop_norm = str(prop).lower()
                canonical = (field_name_by_entity.get(ent, {}) or {}).get(prop_norm)
                if canonical:
                    return canonical
                derived = (derived_by_entity.get(ent, set()) or set())
                if prop_norm in derived:
                    return prop_norm
                return (expr_to_field_by_entity.get(ent, {}) or {}).get(prop_norm, "")

            def _is_metric_like(ent: str | None, prop: str | None) -> bool:
                if not ent or not prop:
                    return False
                prop_norm = str(prop).lower()
                canonical = _canonical_field(ent, prop)
                role = (roles_by_entity.get(ent, {}) or {}).get(canonical) or (roles_by_entity.get(ent, {}) or {}).get(prop_norm)
                agg = (aggs_by_entity.get(ent, {}) or {}).get(canonical) or (aggs_by_entity.get(ent, {}) or {}).get(prop_norm)
                return role == "metric" or bool(agg)

            def _raw_field(ent: str | None, canonical: str) -> str:
                if not ent or not canonical:
                    return canonical
                expr = (exprs_by_entity.get(ent, {}) or {}).get(canonical)
                if expr and re.match(r"^[A-Za-z0-9_]+$", str(expr)):
                    return str(expr).lower()
                return canonical

            def _spec_uses_time_fields() -> bool:
                time_fields = {"meet_year", "meet_mon", "posyear", "posmon", "month_date", "meeting_date", "positiondate"}
                buckets = [spec.get("select"), spec.get("group_by"), spec.get("order_by"), spec.get("aggregations")]
                for bucket in buckets:
                    for item in bucket or []:
                        if not isinstance(item, dict):
                            continue
                        term = item.get("term") if "term" in item else item
                        if isinstance(term, dict):
                            alias = term.get("alias")
                            prop = term.get("prop")
                            ent = alias_to_entity.get(alias)
                            canonical = _canonical_field(ent, prop)
                            if canonical in time_fields:
                                return True
                for pred in spec.get("where") or []:
                    if not isinstance(pred, dict):
                        continue
                    for side_key in ("left", "right"):
                        side = pred.get(side_key)
                        if isinstance(side, dict):
                            alias = side.get("alias")
                            prop = side.get("prop")
                            ent = alias_to_entity.get(alias)
                            canonical = _canonical_field(ent, prop)
                            if canonical in time_fields:
                                return True
                return False

            def _is_join_pred(node: dict) -> bool:
                if not isinstance(node, dict):
                    return False
                if node.get("op") != "==":
                    return False
                left = node.get("left")
                right = node.get("right")
                if not isinstance(left, dict) or not isinstance(right, dict):
                    return False
                if "value" in left or "value" in right:
                    return False
                return bool(left.get("alias") and right.get("alias"))

            def _join_sig(left: dict, right: dict) -> tuple | None:
                if not isinstance(left, dict) or not isinstance(right, dict):
                    return None
                la = left.get("alias")
                ra = right.get("alias")
                lp = left.get("prop")
                rp = right.get("prop")
                if not la or not ra or not lp or not rp:
                    return None
                if str(la) <= str(ra):
                    return (la, lp, ra, rp)
                return (ra, rp, la, lp)

            def _collect_join_sigs(node: object, out: set[tuple]) -> None:
                if isinstance(node, list):
                    for item in node:
                        _collect_join_sigs(item, out)
                    return
                if not isinstance(node, dict):
                    return
                if _is_join_pred(node):
                    sig = _join_sig(node.get("left"), node.get("right"))
                    if sig:
                        out.add(sig)
                    return
                if "and" in node and isinstance(node["and"], list):
                    for item in node["and"]:
                        _collect_join_sigs(item, out)
                if "or" in node and isinstance(node["or"], list):
                    for item in node["or"]:
                        _collect_join_sigs(item, out)
                if "not" in node:
                    _collect_join_sigs(node["not"], out)

            def _add_join_predicates(spec: dict) -> None:
                relationships = load_ai_insights_relationships() or []
                if not relationships:
                    return
                rel_map: Dict[tuple[str, str], list] = {}
                for rel in relationships:
                    rel_map.setdefault((rel.from_entity, rel.to_entity), []).append(rel)

                binds = [b for b in spec.get("bind") or [] if isinstance(b, dict) and b.get("alias") and b.get("entity")]
                if len(binds) < 2:
                    return

                where_list = list(spec.get("where") or [])
                existing_join_sigs: set[tuple] = set()
                _collect_join_sigs(where_list, existing_join_sigs)
                use_time_join = _spec_uses_time_fields()

                for i in range(len(binds)):
                    for j in range(i + 1, len(binds)):
                        a = binds[i]
                        b = binds[j]
                        alias_a = a["alias"]
                        alias_b = b["alias"]
                        ent_a = a["entity"]
                        ent_b = b["entity"]
                        rels = rel_map.get((ent_a, ent_b)) or rel_map.get((ent_b, ent_a)) or []
                        if not rels:
                            continue
                        rels = list(rels)
                        rels.sort(key=lambda r: (0 if use_time_join and len(r.join_on) > 1 else 1, len(r.join_on)))
                        rel = rels[0]

                        join_preds = []
                        for left_key, right_key in rel.join_on or []:
                            left_ent = ent_a if rel.from_entity == ent_a else ent_b
                            right_ent = ent_b if left_ent == ent_a else ent_a
                            left_alias = alias_a if left_ent == ent_a else alias_b
                            right_alias = alias_b if left_alias == alias_a else alias_a
                            left_prop = _raw_field(left_ent, left_key)
                            right_prop = _raw_field(right_ent, right_key)
                            candidate = {
                                "op": "==",
                                "left": {"alias": left_alias, "prop": left_prop},
                                "right": {"alias": right_alias, "prop": right_prop},
                            }
                            sig = _join_sig(candidate["left"], candidate["right"])
                            if sig and sig in existing_join_sigs:
                                continue
                            join_preds.append(candidate)
                            if sig:
                                existing_join_sigs.add(sig)

                        # Remove existing join predicates between the pair
                        existing = [
                            p for p in where_list
                            if _is_join_pred(p)
                            and {p["left"].get("alias"), p["right"].get("alias")} == {alias_a, alias_b}
                        ]
                        for p in existing:
                            if p in where_list:
                                where_list.remove(p)
                        where_list.extend(join_preds)

                spec["where"] = where_list

            if len(spec.get("bind") or []) > 1:
                _add_join_predicates(spec)

            def _prune_unused_binds(spec: dict) -> None:
                binds = [b for b in spec.get("bind") or [] if isinstance(b, dict)]
                if not binds:
                    return

                used_aliases: set[str] = set()

                def _mark_term(t: dict):
                    if not isinstance(t, dict):
                        return
                    alias = t.get("alias")
                    if alias:
                        used_aliases.add(alias)

                def _walk_pred(node: object):
                    if isinstance(node, list):
                        for item in node:
                            _walk_pred(item)
                        return
                    if not isinstance(node, dict):
                        return
                    if "op" in node:
                        _mark_term(node.get("left", {}))
                        _mark_term(node.get("right", {}))
                        return
                    if "between" in node:
                        spec_b = node.get("between") or {}
                        _mark_term(spec_b.get("left", {}))
                        _mark_term(spec_b.get("low", {}))
                        _mark_term(spec_b.get("high", {}))
                        return
                    if "in" in node:
                        spec_i = node.get("in") or {}
                        _mark_term(spec_i.get("left", {}))
                        _mark_term(spec_i.get("right", {}))
                        return
                    if "not" in node:
                        _walk_pred(node.get("not"))
                        return
                    if "and" in node:
                        _walk_pred(node.get("and"))
                        return
                    if "or" in node:
                        _walk_pred(node.get("or"))
                        return

                for item in spec.get("select") or []:
                    if isinstance(item, dict):
                        _mark_term(item)
                for item in spec.get("group_by") or []:
                    if isinstance(item, dict):
                        _mark_term(item)
                for item in spec.get("order_by") or []:
                    if isinstance(item, dict):
                        term = item.get("term")
                        if isinstance(term, dict):
                            _mark_term(term)
                for item in spec.get("aggregations") or []:
                    if isinstance(item, dict):
                        term = item.get("term")
                        if isinstance(term, dict):
                            _mark_term(term)
                _walk_pred(spec.get("where") or [])

                keep_aliases = {b.get("alias") for b in binds if b.get("alias") in used_aliases}
                if not keep_aliases:
                    return

                spec["bind"] = [b for b in binds if b.get("alias") in keep_aliases]

                def _filter_pred(node: object) -> object | None:
                    if isinstance(node, list):
                        filtered = [_filter_pred(item) for item in node]
                        filtered = [item for item in filtered if item is not None]
                        return filtered
                    if not isinstance(node, dict):
                        return node
                    if "op" in node:
                        left = node.get("left", {})
                        right = node.get("right", {})
                        l_alias = left.get("alias") if isinstance(left, dict) else None
                        r_alias = right.get("alias") if isinstance(right, dict) else None
                        if (l_alias and l_alias not in keep_aliases) or (r_alias and r_alias not in keep_aliases):
                            return None
                        return node
                    if "between" in node:
                        spec_b = node.get("between") or {}
                        for side in ("left", "low", "high"):
                            term = spec_b.get(side, {})
                            alias = term.get("alias") if isinstance(term, dict) else None
                            if alias and alias not in keep_aliases:
                                return None
                        return node
                    if "in" in node:
                        spec_i = node.get("in") or {}
                        left = spec_i.get("left", {})
                        right = spec_i.get("right", {})
                        for term in (left, right):
                            alias = term.get("alias") if isinstance(term, dict) else None
                            if alias and alias not in keep_aliases:
                                return None
                        return node
                    if "not" in node:
                        inner = _filter_pred(node.get("not"))
                        return {"not": inner} if inner is not None else None
                    if "and" in node:
                        inner = _filter_pred(node.get("and"))
                        if not inner:
                            return None
                        return {"and": inner}
                    if "or" in node:
                        inner = _filter_pred(node.get("or"))
                        if not inner:
                            return None
                        return {"or": inner}
                    return node

                if "where" in spec:
                    filtered_where = _filter_pred(spec.get("where") or [])
                    spec["where"] = filtered_where or []

            _prune_unused_binds(spec)

            group_by = list(spec.get("group_by") or [])
            group_keys = {(g.get("alias"), g.get("prop")) for g in group_by if isinstance(g, dict)}
            aggregations = list(spec.get("aggregations") or [])
            new_select = []
            updated_group_by = []

            for g in group_by:
                if not isinstance(g, dict):
                    continue
                alias = g.get("alias")
                prop = g.get("prop")
                prop_norm = prop.lower() if isinstance(prop, str) else prop
                ent = alias_to_entity.get(alias)
                canonical = _canonical_field(ent, prop)
                role = (roles_by_entity.get(ent, {}) or {}).get(canonical) or (roles_by_entity.get(ent, {}) or {}).get(prop_norm)
                agg = (aggs_by_entity.get(ent, {}) or {}).get(canonical) or (aggs_by_entity.get(ent, {}) or {}).get(prop_norm)
                if role == "metric" or agg:
                    agg_op = agg or "avg"
                    as_name = g.get("as") or prop
                    aggregations.append({"op": agg_op, "term": {"alias": alias, "prop": prop}, "as": as_name})
                else:
                    updated_group_by.append(g)

            group_by = updated_group_by
            group_keys = {(g.get("alias"), g.get("prop")) for g in group_by if isinstance(g, dict)}

            for sel in list(spec.get("select") or []):
                if not isinstance(sel, dict):
                    continue
                alias = sel.get("alias")
                prop = sel.get("prop")
                prop_norm = prop.lower() if isinstance(prop, str) else prop
                ent = alias_to_entity.get(alias)
                canonical = _canonical_field(ent, prop)
                role = (roles_by_entity.get(ent, {}) or {}).get(canonical) or (roles_by_entity.get(ent, {}) or {}).get(prop_norm)
                agg = (aggs_by_entity.get(ent, {}) or {}).get(canonical) or (aggs_by_entity.get(ent, {}) or {}).get(prop_norm)

                if group_by:
                    if role == "metric" or agg:
                        agg_op = agg or "avg"
                        as_name = sel.get("as") or prop
                        aggregations.append({"op": agg_op, "term": {"alias": alias, "prop": prop}, "as": as_name})
                    else:
                        new_select.append(sel)
                        key = (alias, prop)
                        if alias and prop and key not in group_keys:
                            group_by.append({"alias": alias, "prop": prop, "as": sel.get("as") or prop})
                            group_keys.add(key)
                else:
                    new_select.append(sel)

            def _fix_metric_aggregations(aggs: list) -> list:
                fixed = []
                for a in aggs:
                    if not isinstance(a, dict):
                        fixed.append(a)
                        continue
                    term = a.get("term")
                    if not isinstance(term, dict):
                        fixed.append(a)
                        continue
                    alias = term.get("alias")
                    prop = term.get("prop")
                    ent = alias_to_entity.get(alias)
                    canonical = _canonical_field(ent, prop)
                    if canonical in {"meeting_id", "rmid", "mandateid"}:
                        a = dict(a)
                        a["op"] = "count"
                    fixed.append(a)
                return fixed

            aggregations = _fix_metric_aggregations(aggregations)

            def _dedupe_aggregations(aggs: list) -> tuple[list, dict]:
                seen: dict[tuple, int] = {}
                alias_map: dict[str, str] = {}
                out: list = []

                def _pref(as_name: str | None, prop_name: str | None) -> int:
                    if not as_name:
                        return 0
                    as_lower = str(as_name).lower()
                    if "avg" in as_lower:
                        return 3
                    if prop_name and as_lower == str(prop_name).lower():
                        return 1
                    return 2

                for a in aggs:
                    if not isinstance(a, dict):
                        out.append(a)
                        continue
                    term = a.get("term")
                    if not isinstance(term, dict):
                        out.append(a)
                        continue
                    key = (a.get("op"), term.get("alias"), term.get("prop"))
                    as_name = a.get("as") or term.get("prop")
                    if key in seen:
                        idx = seen[key]
                        existing = out[idx]
                        existing_term = existing.get("term") if isinstance(existing, dict) else None
                        existing_prop = existing_term.get("prop") if isinstance(existing_term, dict) else None
                        existing_as = existing.get("as") if isinstance(existing, dict) else None
                        if _pref(as_name, term.get("prop")) > _pref(existing_as, existing_prop):
                            if existing_as and as_name:
                                alias_map[existing_as] = as_name
                            out[idx] = a
                        else:
                            if as_name and existing_as:
                                alias_map[as_name] = existing_as
                        continue
                    seen[key] = len(out)
                    out.append(a)
                return out, alias_map

            aggregations, agg_alias_map = _dedupe_aggregations(aggregations)

            def _normalize_order_by(aggs: list) -> None:
                alias_by_term: dict[tuple, set[str]] = {}
                for a in aggs:
                    if not isinstance(a, dict):
                        continue
                    term = a.get("term")
                    if not isinstance(term, dict):
                        continue
                    key = (term.get("alias"), term.get("prop"))
                    alias_by_term.setdefault(key, set()).add(a.get("as") or term.get("prop"))

                for ob in spec.get("order_by") or []:
                    term = ob.get("term")
                    if not isinstance(term, dict):
                        continue
                    key = (term.get("alias"), term.get("prop"))
                    aliases = alias_by_term.get(key)
                    if aliases and len(aliases) == 1:
                        term["as"] = next(iter(aliases))

            def _is_meeting_id_count(a: dict) -> bool:
                term = a.get("term") if isinstance(a, dict) else None
                if not isinstance(term, dict):
                    return False
                alias = term.get("alias")
                prop = term.get("prop")
                ent = alias_to_entity.get(alias)
                canonical = _canonical_field(ent, prop)
                return canonical == "meeting_id" and a.get("op") == "count"

            if any(_is_meeting_id_count(a) for a in aggregations):
                group_by = [
                    g for g in group_by
                    if not isinstance(g, dict)
                    or _canonical_field(alias_to_entity.get(g.get("alias")), g.get("prop")) != "meeting_id"
                ]
                new_select = [
                    s for s in new_select
                    if not isinstance(s, dict)
                    or _canonical_field(alias_to_entity.get(s.get("alias")), s.get("prop")) != "meeting_id"
                ]

            if group_by:
                if agg_alias_map:
                    for ob in spec.get("order_by") or []:
                        term = ob.get("term")
                        if isinstance(term, dict):
                            as_name = term.get("as")
                            if as_name in agg_alias_map:
                                term["as"] = agg_alias_map[as_name]

                _normalize_order_by(aggregations)
                def _question_mentions_month(text: str) -> bool:
                    q = text.lower()
                    if "month" in q or "monthly" in q or "mo" in q:
                        return True
                    month_tokens = [
                        "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may", "jun",
                        "june", "jul", "july", "aug", "august", "sep", "sept", "september", "oct",
                        "october", "nov", "november", "dec", "december",
                    ]
                    return any(tok in q for tok in month_tokens)

            def _iter_pred_props(node):
                if isinstance(node, dict):
                    if "alias" in node and "prop" in node:
                        yield node.get("alias"), node.get("prop")
                    for v in node.values():
                        yield from _iter_pred_props(v)
                elif isinstance(node, list):
                    for v in node:
                        yield from _iter_pred_props(v)

            def _spec_has_year_filter() -> bool:
                year_fields = {"meet_year", "posyear"}
                for pred in spec.get("where") or []:
                    for alias, prop in _iter_pred_props(pred):
                        ent = alias_to_entity.get(alias)
                        canonical = _canonical_field(ent, prop)
                        if canonical in year_fields:
                            return True
                return False

                q_lower = (question_text or "").lower()
                if any(t in q_lower for t in (" top ", "top ", " best ", "best ", " highest ", "highest ")):
                    if not _question_mentions_month(question_text) and _spec_has_year_filter():
                        drop_time = {"meet_mon", "posmon"}
                        group_by = [
                            g for g in group_by
                            if not isinstance(g, dict)
                            or _canonical_field(alias_to_entity.get(g.get("alias")), g.get("prop")) not in drop_time
                        ]
                        new_select = [
                            s for s in new_select
                            if not isinstance(s, dict)
                            or _canonical_field(alias_to_entity.get(s.get("alias")), s.get("prop")) not in drop_time
                        ]

                    def _is_derived_dimension(item: dict) -> bool:
                        if not isinstance(item, dict):
                            return False
                        alias = item.get("alias")
                        prop = item.get("prop")
                        ent = alias_to_entity.get(alias)
                        canonical = _canonical_field(ent, prop)
                        if canonical and canonical in (derived_by_entity.get(ent, set()) or set()):
                            role = (roles_by_entity.get(ent, {}) or {}).get(canonical, "")
                            return role == "dimension"
                        return False

                    def _question_mentions_field(text: str, name: str) -> bool:
                        return name and name.lower() in (text or "").lower()

                    group_by = [
                        g for g in group_by
                        if not _is_derived_dimension(g)
                        or _question_mentions_field(question_text, _canonical_field(alias_to_entity.get(g.get("alias")), g.get("prop")))
                    ]
                    new_select = [
                        s for s in new_select
                        if not _is_derived_dimension(s)
                        or _question_mentions_field(question_text, _canonical_field(alias_to_entity.get(s.get("alias")), s.get("prop")))
                    ]

                spec["group_by"] = group_by
                spec["aggregations"] = aggregations
                spec["select"] = new_select or [g for g in group_by if isinstance(g, dict)]

            q_lower = (question_text or "").lower()
            if any(t in q_lower for t in (" top ", "top ", " top", " vs ", " versus ", "compare")):
                if isinstance(spec.get("limit"), int) and spec["limit"] < 2000:
                    spec["limit"] = 2000
            return spec

        def _pred_has_value(node) -> bool:
            if isinstance(node, dict):
                if "value" in node:
                    return True
                return any(_pred_has_value(v) for v in node.values())
            if isinstance(node, list):
                return any(_pred_has_value(v) for v in node)
            return False

        def _pred_aliases(node) -> set[str]:
            out: set[str] = set()
            if isinstance(node, dict):
                if "alias" in node and isinstance(node.get("alias"), str):
                    out.add(node["alias"])
                for v in node.values():
                    out |= _pred_aliases(v)
            elif isinstance(node, list):
                for v in node:
                    out |= _pred_aliases(v)
            return out

        def _pred_is_join(node) -> bool:
            if not isinstance(node, dict):
                return False
            if node.get("op") != "==":
                return False
            left = node.get("left")
            right = node.get("right")
            if not isinstance(left, dict) or not isinstance(right, dict):
                return False
            if "value" in left or "value" in right:
                return False
            return bool(left.get("alias") and right.get("alias"))

        def _has_join_pred(where_list: list) -> bool:
            for pred in where_list or []:
                if _pred_is_join(pred):
                    return True
            return False

        def _ensure_select_has_key(selects: list, alias: str, fields: set[str]) -> None:
            if not fields:
                return
            preferred = ["rmid", "mandateid", "mandateid_str", "rm_name", "meet_year", "posyear", "meet_mon", "posmon"]
            key = next((k for k in preferred if k in fields), None)
            if not key:
                return
            if any(isinstance(s, dict) and s.get("alias") == alias and s.get("prop") == key for s in selects):
                return
            selects.append({"alias": alias, "prop": key, "as": key})

        def _split_rai_spec_if_unbounded(spec: dict, specs_list: list) -> list[dict]:
            if not isinstance(spec, dict):
                return [spec]
            binds = spec.get("bind") or []
            entity_binds = [b for b in binds if isinstance(b, dict) and b.get("alias") and b.get("entity")]
            if len(entity_binds) <= 1:
                return [spec]
            if _pred_has_value(spec.get("where") or []):
                return [spec]
            if (spec.get("aggregations") or spec.get("group_by")) and _has_join_pred(spec.get("where") or []):
                return [spec]

            fields_by_entity = {s.name: set(s.fields or []) for s in (specs_list or [])}
            out_specs: list[dict] = []
            for b in entity_binds:
                alias = b["alias"]
                ent = b["entity"]
                new_spec = {
                    "bind": [b],
                    "where": [],
                    "select": [],
                    "order_by": [],
                    "group_by": [],
                    "aggregations": [],
                }
                for pred in spec.get("where") or []:
                    aliases = _pred_aliases(pred)
                    if aliases and aliases.issubset({alias}):
                        new_spec["where"].append(pred)
                for sel in spec.get("select") or []:
                    if isinstance(sel, dict) and sel.get("alias") == alias:
                        new_spec["select"].append(sel)
                for g in spec.get("group_by") or []:
                    if isinstance(g, dict) and g.get("alias") == alias:
                        new_spec["group_by"].append(g)
                for a in spec.get("aggregations") or []:
                    term = a.get("term") if isinstance(a, dict) else None
                    if isinstance(term, dict) and term.get("alias") == alias:
                        new_spec["aggregations"].append(a)
                _ensure_select_has_key(new_spec["select"], alias, fields_by_entity.get(ent, set()))
                for ob in spec.get("order_by") or []:
                    term = (ob or {}).get("term") or {}
                    if isinstance(term, dict) and term.get("alias") == alias:
                        new_spec["order_by"].append(ob)
                for key in ("limit", "offset", "distinct"):
                    if key in spec:
                        new_spec[key] = spec[key]
                if new_spec["select"]:
                    out_specs.append(new_spec)

            return out_specs or [spec]

        def _coerce_rai_value_types(spec: dict, specs_list: list) -> dict:
            if not isinstance(spec, dict):
                return spec
            try:
                import copy
                out = copy.deepcopy(spec)
            except Exception:
                out = json.loads(json.dumps(spec, default=str))

            alias_to_entity = {}
            for b in out.get("bind") or []:
                if isinstance(b, dict) and b.get("alias") and b.get("entity"):
                    alias_to_entity[b["alias"]] = b["entity"]

            field_name_by_entity = {}
            expr_to_field_by_entity = {}
            types_by_entity = {}
            roles_by_entity = {}
            derived_by_entity = {}
            for s in specs_list or []:
                fields = list(getattr(s, "fields", []) or [])
                derived = list(getattr(s, "derived_fields", []) or [])
                field_name_by_entity[s.name] = {str(f).lower(): f for f in (fields + derived)}
                derived_by_entity[s.name] = set(derived)
                exprs = dict(getattr(s, "field_exprs", {}) or {})
                expr_to_field_by_entity[s.name] = {
                    str(expr).lower(): field for field, expr in (exprs or {}).items() if expr
                }
                types_by_entity[s.name] = dict(getattr(s, "field_types", {}) or {})
                roles_by_entity[s.name] = dict(getattr(s, "field_roles", {}) or {})

            def _dtype_for(term: dict) -> str:
                alias = term.get("alias")
                prop = term.get("prop")
                if not alias or not prop:
                    return ""
                ent = alias_to_entity.get(alias)
                prop_norm = str(prop).lower()
                canonical = (field_name_by_entity.get(ent, {}) or {}).get(prop_norm)
                if not canonical:
                    canonical = (expr_to_field_by_entity.get(ent, {}) or {}).get(prop_norm)
                if not canonical:
                    if prop_norm in (derived_by_entity.get(ent, set()) or set()):
                        canonical = prop_norm
                if canonical:
                    dtype = (types_by_entity.get(ent, {}) or {}).get(canonical)
                    if dtype:
                        return str(dtype or "").lower()
                    role = (roles_by_entity.get(ent, {}) or {}).get(canonical, "")
                    if str(role).lower() == "metric":
                        return "number"
                if prop_norm.startswith("signal_") or prop_norm.endswith("_score") or prop_norm.endswith("_trend"):
                    return "number"
                return ""

            ym_match = re.compile(r"^(\d{4})-(\d{2})$")

            def _parse_dt(value):
                if not isinstance(value, str):
                    return value
                raw = value.strip()
                m = ym_match.match(raw)
                if m:
                    raw = f"{m.group(1)}-{m.group(2)}-01"
                try:
                    return datetime.fromisoformat(raw)
                except Exception:
                    return value

            def _is_numeric_dtype(dt: str) -> bool:
                dt = (dt or "").lower()
                return any(
                    token in dt
                    for token in ("number", "numeric", "float", "double", "decimal", "int")
                )

            def _coerce_value(value, dtype: str):
                if _is_numeric_dtype(dtype):
                    def _to_num(v):
                        if isinstance(v, bool):
                            return 1 if v else 0
                        if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                            return 1 if v.strip().lower() == "true" else 0
                        return v
                    if isinstance(value, list):
                        return [_to_num(v) for v in value]
                    return _to_num(value)
                if dtype not in ("date", "datetime", "timestamp"):
                    return value
                if isinstance(value, list):
                    return [_parse_dt(v) for v in value]
                return _parse_dt(value)

            def _coerce_pred(node: dict):
                if not isinstance(node, dict):
                    return
                if "op" in node:
                    left = node.get("left")
                    right = node.get("right")
                    if isinstance(left, dict) and "value" not in left and isinstance(right, dict) and "value" in right:
                        dtype = _dtype_for(left)
                        right["value"] = _coerce_value(right.get("value"), dtype)
                    elif isinstance(right, dict) and "value" not in right and isinstance(left, dict) and "value" in left:
                        dtype = _dtype_for(right)
                        left["value"] = _coerce_value(left.get("value"), dtype)
                    return
                if "between" in node:
                    spec_node = node.get("between") or {}
                    left = spec_node.get("left")
                    if isinstance(left, dict):
                        dtype = _dtype_for(left)
                        low = spec_node.get("low")
                        high = spec_node.get("high")
                        if isinstance(low, dict) and "value" in low:
                            low["value"] = _coerce_value(low.get("value"), dtype)
                        if isinstance(high, dict) and "value" in high:
                            high["value"] = _coerce_value(high.get("value"), dtype)
                    return
                if "in" in node:
                    spec_node = node.get("in") or {}
                    left = spec_node.get("left")
                    right = spec_node.get("right")
                    if isinstance(left, dict) and isinstance(right, dict) and "value" in right:
                        dtype = _dtype_for(left)
                        right["value"] = _coerce_value(right.get("value"), dtype)
                    return

                # Recurse through nested boolean expressions (and/or/not).
                for v in node.values():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                _coerce_pred(item)
                    elif isinstance(v, dict):
                        _coerce_pred(v)
                    return
                if "not" in node:
                    _coerce_pred(node.get("not"))
                    return
                if "and" in node:
                    for child in node.get("and") or []:
                        _coerce_pred(child)
                    return
                if "or" in node:
                    for child in node.get("or") or []:
                        _coerce_pred(child)
                    return

            for pred in out.get("where") or []:
                _coerce_pred(pred)

            return out

        
        # Identifier detection & coercion
        def is_identifier_name(name: str) -> bool:
            if not isinstance(name, str): return False
            u = name.upper()
            return (
                u in ("RMID","RM_ID","MANDATEID","MANDATE_ID","RELATIONSHIP_MANAGER_ID")
                or u.endswith("_ID")
                or ("ID" in u and not any(k in u for k in ("COUNT","IDENTIFIER_COUNT")))
            )
        
        def force_category_axis(fig, x_is_id: bool):
            if x_is_id and fig is not None:
                fig.update_xaxes(type="category", categoryorder="category ascending")
        
        def smart_default_agg_for_metric(col_name: str) -> str:
            u = str(col_name).upper()
            if any(k in u for k in ("AUM","SNAPSHOT")): return "last"
            if any(k in u for k in ("PERFORMANCE","SCORE","SENTIMENT","YIELD","RATE","RATIO","AVG")): return "avg"
            if any(k in u for k in ("REVENUE","AMOUNT","AMT","PROFIT","TOPUP","INFLOW","OUTFLOW","FEE")): return "sum"
            if any(k in u for k in ("COUNT","NUM","FREQUENCY")): return "sum"
            return "sum"
        
        def aggregate_for_chart(df: pd.DataFrame, x: str, y: str, agg_mode: str, date_hint: Optional[str]=None) -> pd.DataFrame:
            if x not in df.columns or y not in df.columns: return df[[c for c in df.columns if c in (x,y)]].dropna()
            d = df[[x,y]].copy()
            if is_identifier_name(x):
                d[x] = d[x].astype(str)
                d[x] = pd.Categorical(d[x], ordered=False)
            if agg_mode == "sum": return d.groupby(x, as_index=False)[y].sum()
            if agg_mode == "avg": return d.groupby(x, as_index=False)[y].mean()
            if agg_mode == "median": return d.groupby(x, as_index=False)[y].median()
            if agg_mode == "count":
                out = d.groupby(x, as_index=False)[y].count(); out.rename(columns={y:"COUNT"}, inplace=True); out[y]=out["COUNT"]; out.drop(columns=["COUNT"], inplace=True); return out
            if agg_mode == "last":
                date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                chosen_date = date_hint if (date_hint in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_hint])) else (date_cols[0] if date_cols else None)
                if chosen_date:
                    d2 = df[[x,y,chosen_date]].dropna()
                    if is_identifier_name(x):
                        d2[x] = d2[x].astype(str); d2[x] = pd.Categorical(d2[x], ordered=False)
                    idx = d2.groupby(x)[chosen_date].idxmax()
                    return d2.loc[idx, [x,y]].reset_index(drop=True)
                return d.groupby(x, as_index=False).tail(1)[[x,y]]
            return d.groupby(x, as_index=False)[y].sum()

        def coerce_identifier_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if not isinstance(df, pd.DataFrame):
                return df
            id_cols = [c for c in df.columns if is_identifier_name(c)]
            if not id_cols:
                return df
            out = df.copy()
            for col in id_cols:
                try:
                    out[col] = out[col].astype(str)
                except Exception:
                    pass
            return out

        PLOTLY_SCHEMA_COL_CAP = 40
        PLOTLY_SCHEMA_HEAD = 5

        def _schema_for_frames(frames: dict) -> dict:
            out = {}
            for name, df in (frames or {}).items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    cols = list(df.columns)[:PLOTLY_SCHEMA_COL_CAP]
                    out[str(name)] = {
                        "columns": cols,
                        "dtypes": {str(c): str(df[c].dtype) for c in cols},
                        "preview": df[cols].head(PLOTLY_SCHEMA_HEAD).astype(str).to_dict("records"),
                        "rows": int(len(df)),
                    }
            return out

        _LLM_PREAMBLE = """
You will write pure Plotly Python code to visualize data that is already loaded into a dict called `frames`.
Each key in `frames` maps to a pandas DataFrame. Plotly and pandas are already imported as:
- pandas as `pd`
- plotly.express as `px`
- plotly.graph_objects as `go`

Hard constraints:
- Do NOT import anything; never use os/sys/subprocess/network.
- Use only the datasets and columns shown in the provided SCHEMA.
- MUST assign the final Plotly figure to a variable named `fig`.
- You may set an optional string variable `caption` (one concise sentence).
- No line-continuation backslashes across lines.
- Do not invent identifiers; the runtime provides only: frames, pd, px, go, and integer helpers: n, topn, top_k, topk, k.
- If you need Top-N, use `n` directly.
- Prefer names over IDs for axes/labels (use RM_NAME instead of RMID).
- If you plot identifier columns like RMID or MANDATEID, cast them to string first.
- Do not plot ID or date fields as metrics (Y values). Date/Year/Month may be used on X or as grouping.
- If a date/month column has a companion `<COL>_LABEL`, prefer the label column for the x-axis.
- Never cast date columns to integers; use them as dates or formatted strings.
- Be robust to missing columns or empty frames and fall back gracefully.

Chart selection rules:
- For Top/Bottom/Rank by entity: use a sorted bar chart, X=entity, Y=metric. If year comparison implied, use grouped bar by year.
- For trends over time: use a line chart with time on X.
- For relationships: use scatter (X=metric1, Y=metric2).

Metric intent:
- If question/summary mentions performance, prefer columns containing PERFORMANCE.
- If mentions revenue, prefer REVENUE.
- If mentions AUM, prefer AUM or TOTAL_AUM.
- If mentions sentiment, prefer SENTIMENT.
"""

        def _prepare_chart_frame(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df
            out = df.copy()
            for col in list(out.columns):
                try:
                    col_up = str(col).upper()
                    series = out[col]
                    # Try to coerce numeric date-like columns
                    if ("DATE" in col_up or "MONTH" in col_up):
                        ser_num = None
                        if pd.api.types.is_numeric_dtype(series):
                            ser_num = pd.to_numeric(series, errors="coerce")
                        elif series.dtype == object:
                            ser_num = pd.to_numeric(series, errors="coerce")
                        if ser_num is not None and ser_num.notna().any():
                            mx = ser_num.max(skipna=True)
                            if mx and mx > 1e12:
                                # likely epoch ns
                                out[col] = pd.to_datetime(ser_num, unit="ns", errors="coerce")
                            elif mx and mx > 1e10:
                                # likely epoch ms
                                out[col] = pd.to_datetime(ser_num, unit="ms", errors="coerce")
                            elif mx and mx > 1e9:
                                # likely epoch seconds
                                out[col] = pd.to_datetime(ser_num, unit="s", errors="coerce")
                            series = out[col]

                    if pd.api.types.is_datetime64_any_dtype(series):
                        label_col = f"{col}_LABEL"
                        if "MONTH" in col_up:
                            out[label_col] = series.dt.to_period("M").astype(str)
                        else:
                            out[label_col] = series.dt.strftime("%Y-%m-%d")
                except Exception:
                    continue
            return out

        def _build_llm_plotly_prompt(question: str, summary: str, schema: dict, overrides: Optional[dict] = None) -> str:
            few_shots = r"""
# EXAMPLE A - Top 5 RM 2025 vs 2024 (grouped bar by RM_NAME and MEET_YEAR)
# d = frames["combined"][["RM_NAME","MEET_YEAR","METRIC_VALUE"]].dropna()
# top = d[d["MEET_YEAR"] == 2025].sort_values("METRIC_VALUE", ascending=False).head(n)
# keep = set(top["RM_NAME"].astype(str))
# d = d[d["RM_NAME"].astype(str).isin(keep)]
# fig = px.bar(d, x="RM_NAME", y="METRIC_VALUE", color="MEET_YEAR", barmode="group")

# EXAMPLE B - Trend over time (line)
# d = frames["s1"][["MONTH_DATE","PERFORMANCE"]].dropna()
# fig = px.line(d.sort_values("MONTH_DATE"), x="MONTH_DATE", y="PERFORMANCE")
"""
            override_text = ""
            if overrides:
                override_lines = []
                ds = overrides.get("dataset")
                if ds:
                    override_lines.append(f"- Use frames['{ds}'] as the primary dataset.")
                x_field = overrides.get("x")
                if x_field:
                    override_lines.append(f"- Set the x-axis column to '{x_field}'.")
                series_field = overrides.get("series")
                if series_field:
                    override_lines.append(f"- Use '{series_field}' as the series/grouping column.")
                metrics = overrides.get("metrics") or []
                if metrics:
                    pretty_metrics = ", ".join(f"'{m}'" for m in metrics)
                    override_lines.append(f"- Plot the metric column(s): {pretty_metrics}.")
                chart_type = overrides.get("chart_type")
                if chart_type and chart_type != "auto":
                    override_lines.append(f"- The figure must be a {chart_type} chart.")
                notes = overrides.get("notes")
                if notes:
                    override_lines.append(f"- Additional constraint: {notes}")
                if override_lines:
                    override_text = "\nUSER OVERRIDES:\n" + "\n".join(override_lines) + "\n"

            return (
                _LLM_PREAMBLE
                + "\nQUESTION:\n" + (question or "")
                + "\n\nSUMMARY:\n" + (summary or "")
                + (override_text or "")
                + "\n\nANALYSIS GUIDANCE:\n"
                + "- Align the visualization with the QUESTION and SUMMARY.\n"
                + "- Combine data from multiple frames when helpful (join on RMID/RM_NAME, MEET_YEAR, MONTH_DATE).\n"
                + "- Prefer the most informative metrics/columns that support the stated conclusion.\n"
                + "\nRUNTIME HINTS:\n"
                + "- Integer helpers `n`, `topn`, `top_k`, `topk`, `k` are available. Use `n` for Top-N.\n"
                + "- Available dataframes are in `frames`. Choose the one(s) whose columns match the need.\n"
                + "- Prefer `RM_NAME` for entity labels; `MEET_YEAR` for year.\n"
                + "- Avoid ID/date columns as metrics.\n"
                + "\nSCHEMA (JSON):\n" + json.dumps(schema, ensure_ascii=False, indent=2)
                + "\n\nREFERENCE SHAPES:\n" + few_shots
                + "\n\nReturn ONLY Python code. You MUST assign the Plotly figure to a variable named `fig`."
            )

        _FORBIDDEN = ("import ", "__import__", "open(", "os.", "sys.", "subprocess", "eval(", "exec(", "input(")

        def _is_plotly_code_safe(code: str) -> bool:
            if not code or len(code) > 20000:
                return False
            low = code.lower()
            return not any(bad in low for bad in _FORBIDDEN)

        def _sanitize_llm_code(s: str) -> str:
            if not s:
                return s
            s = s.strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1]
            s = (s.replace("\\r\\n", "\n")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace('\\"', '"')
                .replace("\\'", "'"))
            fence = re.search(r"```(?:python)?\\s*([\\s\\S]*?)\\s*```", s, flags=re.IGNORECASE)
            if fence:
                s = fence.group(1)
            s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
            lines = []
            for ln in s.split("\n"):
                lstrip = ln.lstrip()
                if lstrip.startswith("%") or lstrip.startswith("!") or lstrip.startswith("%%"):
                    continue
                lines.append(ln)
            s = "\n".join(lines)
            raw_lines = s.split("\n")
            fixed = []
            i = 0
            while i < len(raw_lines):
                line = raw_lines[i]
                if line.rstrip().endswith("\\"):
                    nxt = raw_lines[i + 1] if i + 1 < len(raw_lines) else ""
                    if nxt.lstrip().startswith("#") or nxt.strip() == "":
                        line = line.rstrip()[:-1]
                        fixed.append(line + " ")
                        i += 2
                        continue
                    joined = line.rstrip()[:-1] + " " + nxt.lstrip()
                    fixed.append(joined)
                    i += 2
                    continue
                fixed.append(line)
                i += 1
            s = "\n".join(fixed)
            s = re.sub(r"\\s+\\n", "\n", s)
            s = re.sub(r"^\\s*\\\\\\s*", "", s)
            return s.strip()

        def _exec_plotly_code_safely(code: str, frames: dict):
            import pandas as _pd

            class _FramesProxy(dict):
                def __init__(self, data: Optional[dict]):
                    super().__init__()
                    if isinstance(data, dict):
                        for k, v in data.items():
                            self[k] = v

                def get(self, key, default=None):
                    if dict.__contains__(self, key):
                        return dict.get(self, key)
                    return default if default is not None else _pd.DataFrame()

            frames = _FramesProxy(frames)

            def _strip_fences(s: str) -> str:
                s = (s or "").strip()
                m = re.search(r"```(?:python)?\s*([\s\S]*?)```", s, re.IGNORECASE)
                return (m.group(1).strip() if m else s)

            def _coerce_fig_assignment(s: str) -> str:
                if re.search(r"(?m)^\s*fig\s*=", s):
                    return s
                px_call = None
                for m in re.finditer(r"(px\.\w+\s*\([^)]*\))", s, re.DOTALL):
                    px_call = m.group(1)
                if px_call:
                    return s.replace(px_call, f"fig = {px_call}")
                m = re.search(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*(px\.\w+\s*\([^)]*\)|go\.Figure\s*\([^)]*\))", s)
                if m:
                    varname = m.group(1)
                    return s + f"\nfig = {varname}"
                return s + "\nfig = go.Figure()"

            if not _is_plotly_code_safe(code):
                raise RuntimeError("Generated code failed safety checks.")

            code0 = _strip_fences(code)
            code1 = _sanitize_llm_code(code0)
            code2 = _coerce_fig_assignment(code1)

            import builtins
            safe_builtins = {
                "abs": builtins.abs, "min": builtins.min, "max": builtins.max, "sum": builtins.sum,
                "len": builtins.len, "range": builtins.range, "enumerate": builtins.enumerate,
                "zip": builtins.zip, "map": builtins.map, "filter": builtins.filter,
                "any": builtins.any, "all": builtins.all,
                "list": builtins.list, "tuple": builtins.tuple, "set": builtins.set, "dict": builtins.dict,
                "float": builtins.float, "int": builtins.int, "str": builtins.str, "round": builtins.round,
                "sorted": builtins.sorted
            }
            g = {"__builtins__": safe_builtins, "pd": pd, "px": px, "go": go, "frames": frames}

            n_default = 10
            l = {
                "frames": frames,
                "fig": None,
                "caption": "",
                "n": n_default,
                "topn": n_default,
                "top_k": n_default,
                "topk": n_default,
                "k": n_default,
            }

            def _pick_primary_df(frames_dict) -> _pd.DataFrame:
                for key in ("df_long", "long", "main", "data"):
                    v = frames_dict.get(key)
                    if isinstance(v, _pd.DataFrame) and not v.empty:
                        return v
                for v in frames_dict.values():
                    if isinstance(v, _pd.DataFrame) and not v.empty:
                        return v
                for v in frames_dict.values():
                    if isinstance(v, _pd.DataFrame):
                        return v
                return _pd.DataFrame()

            primary_df = _pick_primary_df(frames)
            l.setdefault("df", primary_df)
            l.setdefault("data", primary_df)
            l.setdefault("df_long", primary_df)
            g.setdefault("df", primary_df)
            g.setdefault("data", primary_df)
            g.setdefault("df_long", primary_df)

            alias_taken = set(list(l.keys()) + list(g.keys()))

            def _safe_alias(name: str) -> str:
                if not name:
                    return ""
                alias = re.sub(r"[^0-9A-Za-z_]+", "_", str(name))
                alias = alias.strip("_")
                if not alias or not alias[0].isalpha():
                    alias = f"f_{alias}" if alias else "frame"
                alias = alias[:24]
                if not alias[0].isalpha():
                    alias = "f_" + alias
                base = alias
                i = 2
                while alias in alias_taken:
                    alias = f"{base}_{i}"
                    i += 1
                alias_taken.add(alias)
                return alias

            merged_alias_given = False
            for _k, _v in (frames or {}).items():
                if not isinstance(_v, _pd.DataFrame):
                    continue
                key_str = str(_k)
                key_lower = key_str.lower()
                if re.match(r"^[A-Za-z_]\w{0,24}$", key_str):
                    if key_str not in l:
                        l.setdefault(key_str, _v)
                        g.setdefault(key_str, _v)
                        alias_taken.add(key_str)
                else:
                    alias = _safe_alias(key_str)
                    if alias and alias not in l:
                        l.setdefault(alias, _v)
                        g.setdefault(alias, _v)
                        alias_taken.add(alias)
                prefix = key_lower.split("_", 1)[0]
                if prefix and prefix.isidentifier() and prefix not in l:
                    l.setdefault(prefix, _v)
                    g.setdefault(prefix, _v)
                    alias_taken.add(prefix)
                if "merged" in key_lower and not merged_alias_given:
                    l.setdefault("merged", _v)
                    g.setdefault("merged", _v)
                    alias_taken.add("merged")
                    merged_alias_given = True

            def _resolve_missing_name(name: str):
                if not name:
                    return None
                name_l = name.lower()
                if name in frames and isinstance(frames[name], _pd.DataFrame):
                    return frames[name]
                for key, val in (frames or {}).items():
                    if not isinstance(val, _pd.DataFrame):
                        continue
                    k_str = str(key)
                    k_low = k_str.lower()
                    if k_str == name or k_low == name_l:
                        return val
                    if k_low.split("_", 1)[0] == name_l:
                        return val
                if name_l in ("df", "data", "dataset"):
                    return primary_df
                if re.fullmatch(r"(df|data|dataset)\d*", name_l):
                    return primary_df
                if re.fullmatch(r"p\d+", name_l) or re.fullmatch(r"px\d*", name_l):
                    return px
                if name_l.startswith("go") and name_l not in ("go",):
                    return go
                return None

            def _exec_with_missing_handling(code_text: str):
                base_locals = dict(l)
                resolved = {}
                last_exc = None
                for _attempt in range(3):
                    exec_locals = dict(base_locals)
                    exec_locals.update(resolved)
                    try:
                        exec(code_text, g, exec_locals)
                        return exec_locals
                    except NameError as exc:
                        last_exc = exc
                        missing = getattr(exc, "name", None)
                        if not missing:
                            m = re.search(r"name '([^']+)' is not defined", str(exc))
                            missing = m.group(1) if m else None
                        if not missing:
                            raise
                        resolved_value = _resolve_missing_name(missing)
                        resolved[missing] = resolved_value
                        if resolved_value is not None:
                            g[missing] = resolved_value
                        else:
                            raise
                if last_exc:
                    raise last_exc
                return exec_locals

            def _maybe_fix_unterminated_string(code_text: str, err: SyntaxError) -> Optional[str]:
                try:
                    lineno = err.lineno or 0
                    if lineno <= 0:
                        return None
                    lines = code_text.splitlines()
                    if lineno > len(lines):
                        return None
                    target_line = lines[lineno - 1]
                    def _needs_closing(line: str, quote: str) -> bool:
                        count = 0
                        escaped = False
                        for ch in line:
                            if escaped:
                                escaped = False
                                continue
                            if ch == "\\":
                                escaped = True
                                continue
                            if ch == quote:
                                count += 1
                        return (count % 2) == 1
                    quote_char = None
                    if _needs_closing(target_line, '"'):
                        quote_char = '"'
                    elif _needs_closing(target_line, "'"):
                        quote_char = "'"
                    if not quote_char:
                        return None
                    lines[lineno - 1] = target_line + quote_char
                    return "\n".join(lines)
                except Exception:
                    return None

            try:
                l_out = _exec_with_missing_handling(code2)
            except SyntaxError as first_err:
                code2b = _coerce_fig_assignment(_sanitize_llm_code(code2))
                try:
                    l_out = _exec_with_missing_handling(code2b)
                except SyntaxError as second_err:
                    fixed = None
                    msg = str(second_err).lower()
                    if any(token in msg for token in ("unterminated string", "eol while scanning string literal")):
                        fixed = _maybe_fix_unterminated_string(code2b, second_err)
                    if fixed:
                        l_out = _exec_with_missing_handling(fixed)
                    else:
                        raise

            fig = l_out.get("fig")
            if fig is None:
                for v in l_out.values():
                    try:
                        if isinstance(v, go.Figure):
                            fig = v
                            break
                    except Exception:
                        pass
            if fig is None:
                raise RuntimeError("Generated code did not set fig.")

            def _coerce_identifier_axes(fig_obj):
                if fig_obj is None or not getattr(fig_obj, "data", None):
                    return
                def _is_id_like(value: Any) -> bool:
                    if value is None:
                        return False
                    s = str(value).strip()
                    return s.isdigit() and len(s) >= 4
                axes_to_update = set()
                for trace in fig_obj.data:
                    x_data = getattr(trace, "x", None)
                    if x_data is None:
                        continue
                    try:
                        values = list(x_data) if not hasattr(x_data, "tolist") else list(x_data.tolist())
                    except Exception:
                        continue
                    sample = [v for v in values if v not in (None, "")]
                    if not sample:
                        continue
                    sample = sample[:20]
                    if not all(_is_id_like(v) for v in sample):
                        continue
                    coerced = [str(v).strip() if v is not None else None for v in values]
                    trace.update(x=coerced)
                    axis_ref = getattr(trace, "xaxis", "x")
                    axes_to_update.add(axis_ref)
                for axis_ref in axes_to_update:
                    layout_key = "xaxis" if axis_ref == "x" else f"xaxis{axis_ref[1:]}"
                    fig_obj.update_layout(**{layout_key: dict(type="category", categoryorder="category ascending")})

            _coerce_identifier_axes(fig)

            def _coerce_epoch_axes(fig_obj):
                if fig_obj is None or not getattr(fig_obj, "data", None):
                    return
                for trace in fig_obj.data:
                    x_data = getattr(trace, "x", None)
                    if x_data is None:
                        continue
                    try:
                        values = list(x_data) if not hasattr(x_data, "tolist") else list(x_data.tolist())
                    except Exception:
                        continue
                    if not values:
                        continue
                    try:
                        ser_num = pd.to_numeric(pd.Series(values), errors="coerce")
                    except Exception:
                        continue
                    if ser_num.isna().all():
                        continue
                    mx = ser_num.max(skipna=True)
                    if not mx or mx <= 1e9:
                        continue
                    if mx > 1e12:
                        dt = pd.to_datetime(ser_num, unit="ns", errors="coerce")
                    elif mx > 1e10:
                        dt = pd.to_datetime(ser_num, unit="ms", errors="coerce")
                    else:
                        dt = pd.to_datetime(ser_num, unit="s", errors="coerce")
                    if dt.isna().all():
                        continue
                    years = dt.dt.year
                    if not (((years >= 1970) & (years <= 2100)).mean() >= 0.8):
                        continue
                    trace.update(x=list(dt))
                    axis_ref = getattr(trace, "xaxis", "x")
                    layout_key = "xaxis" if axis_ref == "x" else f"xaxis{axis_ref[1:]}"
                    fig_obj.update_layout(**{layout_key: dict(type="date")})

            _coerce_epoch_axes(fig)

            cap = l_out.get("caption") if isinstance(l_out.get("caption"), str) else None
            return fig, cap

        def _invoke_llm_for_plotly(prompt: str) -> str:
            safe = str(prompt).replace("$$", "$ $")
            q = "select snowflake.cortex.complete('{model}', $${body}$$) as response".format(
                model=CORTEX_LLM_MODEL,
                body=safe,
            )
            df = session.sql(q).to_pandas()
            if df is None or df.empty:
                raise RuntimeError("Cortex returned empty response.")
            return df.iloc[0, 0]

        def _figure_has_visible_data(fig: Optional[go.Figure]) -> bool:
            if fig is None:
                return False
            data = getattr(fig, "data", None)
            if not data:
                return False
            for trace in data:
                if trace is None:
                    continue
                for attr in ("x", "y", "z", "values"):
                    seq = getattr(trace, attr, None)
                    if seq is None:
                        continue
                    try:
                        iterable = list(seq)
                    except Exception:
                        continue
                    non_null = [item for item in iterable if item is not None and not (isinstance(item, float) and pd.isna(item))]
                    if non_null:
                        return True
            return False

        def compute_llm_chart(frames_src: dict, question_or_state, summary_or_shared=None) -> dict:
            if isinstance(question_or_state, dict):
                question = (question_or_state.get("question") or st.session_state.get("last_query", ""))
                summary = ""
                if isinstance(st.session_state.get("last_result"), dict):
                    summary = st.session_state["last_result"].get("insights", "") or ""
            else:
                question = question_or_state or ""
                summary = summary_or_shared or ""

            result = {"fig": None, "caption": None, "status": "info", "message": None}
            if not isinstance(frames_src, dict) or not frames_src:
                result["message"] = "No frames yet to chart."
                return result
            frames_payload = {k: v for k, v in frames_src.items() if isinstance(v, pd.DataFrame) and not v.empty}
            combined_df = st.session_state.get("__single_chart_df__")
            if isinstance(combined_df, pd.DataFrame) and not combined_df.empty:
                frames_payload = dict(frames_payload)
                frames_payload.setdefault("combined", combined_df)
            if not frames_payload:
                result["message"] = "No data available for the proposed visualization."
                return result
            frames_payload = {k: _prepare_chart_frame(coerce_identifier_columns(v)) for k, v in frames_payload.items()}

            def _frame_fingerprint(df: pd.DataFrame) -> dict:
                cols = [str(c) for c in df.columns]
                return {"columns": cols, "rows": int(len(df))}

            fingerprint = {
                "question": question or "",
                "summary": summary or "",
                "frames": {str(k): _frame_fingerprint(v) for k, v in sorted(frames_payload.items(), key=lambda item: str(item[0]))},
            }
            cache = st.session_state.setdefault("llm_chart_cache", {})
            cached_fp = cache.get("fingerprint")
            cached_fig = cache.get("figure_dict")
            cached_cap = cache.get("caption")
            if cached_fp == fingerprint and cached_fig:
                try:
                    result["fig"] = go.Figure(cached_fig)
                    result["caption"] = cached_cap
                    result["status"] = "ok"
                    return result
                except Exception:
                    pass

            schema = _schema_for_frames(frames_payload)
            max_attempts = 4
            override_notes = None
            last_error = None

            for attempt in range(max_attempts):
                overrides = {"notes": override_notes} if override_notes else None
                try:
                    prompt = _build_llm_plotly_prompt(question, summary, schema, overrides)
                    llm_raw = _invoke_llm_for_plotly(prompt)
                    fig, cap = _exec_plotly_code_safely(llm_raw, frames_payload)
                except Exception as e:
                    last_error = str(e)
                    override_notes = (
                        f"Attempt {attempt + 1} failed: {last_error}. "
                        "Return minimal Plotly code only. Use only listed columns, "
                        "no unsafe ops, no external files, no extra imports. "
                        "Keep the chart simple and valid."
                    )
                    try:
                        trace("plotly_llm_exception", {"attempt": attempt + 1, "error": last_error[:MAX_TRACE_PAYLOAD]})
                    except Exception:
                        pass
                    continue

                if not _figure_has_visible_data(fig):
                    last_error = "empty_chart"
                    override_notes = (
                        "Previous chart had no visible data. "
                        "Remove filters, pick columns with non-null values, "
                        "and ensure at least one trace has data."
                    )
                    try:
                        trace("plotly_llm_empty", {"attempt": attempt + 1})
                    except Exception:
                        pass
                    continue

                cache.update({
                    "fingerprint": fingerprint,
                    "figure_dict": fig.to_dict(),
                    "caption": cap,
                    "code": llm_raw,
                })

                result["fig"] = fig
                result["caption"] = cap
                result["status"] = "ok"
                result["message"] = None
                return result

            if last_error == "empty_chart":
                result["status"] = "warning"
                result["message"] = "Tried twice but the generated chart stayed empty. Refine the question or adjust filters."
            else:
                cached_fp = cache.get("fingerprint")
                fallback_fig = cache.get("figure_dict")
                fallback_cap = cache.get("caption")
                if cached_fp == fingerprint and fallback_fig:
                    try:
                        result["fig"] = go.Figure(fallback_fig)
                        result["caption"] = fallback_cap
                        result["status"] = "warning"
                        result["message"] = (
                            f"Chart regeneration failed after retries ({last_error or 'unknown error'}). Showing last saved chart."
                        )
                        return result
                    except Exception:
                        pass
                result["status"] = "error"
                result["message"] = f"Chart generation failed after retries: {last_error or 'unknown error'}"
            return result

        # ──────────────────────────────────────────────────────────────────────────────
        # Deterministic metrics toolbox (LLM never does arithmetic)

        import numpy as _np
        

        def _pd_to_num(s):
            import pandas as pd
            try:
                return pd.to_numeric(s, errors="coerce")
            except Exception:
                return s

        def _coerce_list(s):
            try: return list(s)
            except Exception: return [s]

        def _pick_date_col(df: 'pd.DataFrame', date_hint: Optional[str]=None) -> Optional[str]:
            import pandas as pd
            if date_hint and date_hint in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_hint]):
                return date_hint
            for c in df.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[c]): return c
                except Exception:
                    pass
            return None

        def _last_value(df_col_with_date: Tuple['pd.Series','pd.Series']) -> Any:
            values, dates = df_col_with_date
            import pandas as pd
            if values is None or dates is None: return None
            try:
                idx = dates.idxmax()
                return values.loc[idx]
            except Exception:
                try: return values.iloc[-1]
                except Exception: return None

        def _apply_simple_filters(df, filters):
            """
            Deterministic filter evaluator.
            Supports:
              - {"COL": 5}                     -> equality
              - {"COL": {">": 5, "<=": 10}}    -> chained comparisons (AND)
              - {"COL": ["A","B"]}             -> IN list
              - {"COL": {"in": [...]} }        -> IN
              - {"COL": {"not in": [...]} }    -> NOT IN
              - {"COL": {"between": [lo, hi]}} -> BETWEEN inclusive
              - {"COL": {"contains": "foo"}}   -> substring contains (case-insensitive by default)
              - {"COL": {"startswith": "x"}}   -> startswith
              - {"COL": {"endswith": "y"}}     -> endswith
              - {"COL": {"isnull": true}}      -> is null
              - {"COL": {"notnull": true}}     -> not null
            NEVER evaluates a Series/DataFrame in boolean context.
            """
            import pandas as pd
            import numpy as np
            if not filters:
                return df

            d = df
            mask = pd.Series(True, index=d.index)

            def _ensure_series(col):
                if col in d.columns:
                    return d[col]
                # allow case-insensitive match
                up = {str(c).upper(): c for c in d.columns}
                cu = str(col).upper()
                if cu in up:
                    return d[up[cu]]
                # if not found, skip this filter
                return None

            # ---------- INSERT: dtype harmonizers ----------
            def _to_numeric_series(s: pd.Series) -> pd.Series:
                """Try numeric coercion; if it stays object, return original."""
                try:
                    sn = pd.to_numeric(s, errors="coerce")
                    # If we actually produced a numeric dtype (not object all-NaN), use it
                    return sn if str(sn.dtype) != "object" else s
                except Exception:
                    return s

            def _numify_value(v):
                """Try to numeric-coerce scalar or each element of a list/tuple/set."""
                try:
                    if isinstance(v, (list, tuple, set)):
                        return [pd.to_numeric(x, errors="coerce") for x in v]
                    return pd.to_numeric(v, errors="coerce")
                except Exception:
                    return v

            def _looks_like_MM_series(s: pd.Series) -> bool:
                """Heuristic: series values are 1–2 digit month codes (e.g., 8, '08')."""
                try:
                    ss = s.astype(str)
                    # allow some NAs; require majority to look like 1–2 digits
                    return (ss.str.fullmatch(r"\d{1,2}").mean() or 0) >= 0.8
                except Exception:
                    return False

            def _harmonize(s: pd.Series, v):
                """
                Return (series_for_compare, value_for_compare) so that:
                  - int vs '2025' vs Decimal('2025') compare equal
                  - month 8 vs '08' compare equal when series is MM-like strings
                For dict (operator) values we only prepare rhs; caller may override.
                """
                # Dict handled by caller (ops); here we just pass through
                if isinstance(v, dict):
                    return s, v

                # Try numeric on both sides
                sn = _to_numeric_series(s)
                vn = _numify_value(v)

                # If series is numeric and value successfully numeric, compare numerically
                if getattr(sn, "dtype", None) is not None and str(sn.dtype) != "object":
                    if isinstance(v, (list, tuple, set)):
                        return sn, [_numify_value(x) for x in v]
                    return sn, vn

                # If series looks like MM strings, zero-pad scalar month
                if _looks_like_MM_series(s):
                    try:
                        if isinstance(v, (list, tuple, set)):
                            vv = [str(int(x)).zfill(2) for x in v]
                        else:
                            vv = str(int(v)).zfill(2)
                        return s.astype(str).str.zfill(2), vv
                    except Exception:
                        pass

                # Fallback: string compare
                if isinstance(v, (list, tuple, set)):
                    return s.astype(str), [str(x) for x in v]
                return s.astype(str), str(v)
            # ---------- /INSERT ----------

            for k, v in (filters or {}).items():
                s = _ensure_series(k)
                if s is None:
                    # unknown column -> skip this filter safely
                    continue

                # guard against pathological values
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    continue

                cur = pd.Series(True, index=d.index)

                # --- literals / lists ---
                if not isinstance(v, dict):
                    s_cmp, v_cmp = _harmonize(s, v)
                    if isinstance(v, (list, tuple, set)):
                        cur = s_cmp.isin(list(v_cmp))
                    else:
                        cur = s_cmp == v_cmp
                    mask &= cur
                    continue

                # --- operator dict ---
                ops_mask_parts = []
                for op, rhs in v.items():
                    op_norm = str(op).strip().lower()

                    if isinstance(rhs, (pd.Series, pd.DataFrame)):
                        # unsafe rhs: skip this sub-op
                        continue

                    # For textual ops, force string path; for numeric/relational, harmonize
                    if op_norm in ("contains", "startswith", "starts_with", "endswith", "ends_with"):
                        s_str = s.astype(str)
                        if op_norm == "contains":
                            ops_mask_parts.append(s_str.str.contains(str(rhs), case=False, na=False))
                        elif op_norm in ("startswith", "starts_with"):
                            ops_mask_parts.append(s_str.str.startswith(str(rhs), na=False))
                        elif op_norm in ("endswith", "ends_with"):
                            ops_mask_parts.append(s_str.str.endswith(str(rhs), na=False))
                        continue

                    # Harmonize for numeric / set / range comparisons
                    s_cmp, rhs_cmp = _harmonize(s, rhs)

                    if op_norm in (">", "gt"):
                        ops_mask_parts.append(s_cmp > rhs_cmp)
                    elif op_norm in (">=", "ge"):
                        ops_mask_parts.append(s_cmp >= rhs_cmp)
                    elif op_norm in ("<", "lt"):
                        ops_mask_parts.append(s_cmp < rhs_cmp)
                    elif op_norm in ("<=", "le"):
                        ops_mask_parts.append(s_cmp <= rhs_cmp)
                    elif op_norm in ("=", "==", "eq"):
                        ops_mask_parts.append(s_cmp == rhs_cmp)
                    elif op_norm in ("!=", "<>", "ne"):
                        ops_mask_parts.append(s_cmp != rhs_cmp)
                    elif op_norm in ("in",):
                        rhs_list = list(rhs_cmp) if isinstance(rhs_cmp, (list, tuple, set)) else [rhs_cmp]
                        ops_mask_parts.append(s_cmp.isin(rhs_list))
                    elif op_norm in ("not in", "nin"):
                        rhs_list = list(rhs_cmp) if isinstance(rhs_cmp, (list, tuple, set)) else [rhs_cmp]
                        ops_mask_parts.append(~s_cmp.isin(rhs_list))
                    elif op_norm in ("between",):
                        lo, hi = (rhs_cmp or [None, None])[:2]
                        part = pd.Series(True, index=d.index)
                        if lo is not None:
                            part &= (s_cmp >= lo)
                        if hi is not None:
                            part &= (s_cmp <= hi)
                        ops_mask_parts.append(part)
                    elif op_norm in ("isnull", "is_null", "null"):
                        ops_mask_parts.append(s.isna() if rhs_cmp else ~s.isna())
                    elif op_norm in ("notnull", "not_null", "nonnull"):
                        ops_mask_parts.append(~s.isna() if rhs_cmp else s.isna())
                    else:
                        # unknown operator: skip
                        continue

                if ops_mask_parts:
                    # AND all sub-conditions
                    part_mask = ops_mask_parts[0]
                    for pm in ops_mask_parts[1:]:
                        part_mask = part_mask & pm
                    mask &= part_mask

            return d[mask]


        def _prep_last(df: 'pd.DataFrame', column: str, date_col: Optional[str]) -> Tuple['pd.Series','pd.Series']:
            dc = _pick_date_col(df, date_col)
            vals = df[column] if column in df.columns else None
            dates = df[dc] if (dc and dc in df.columns) else None
            return (vals, dates)

        _METRIC_FUNS = {
            "sum":     lambda s: _np.nansum(_pd_to_num(s)),
            "avg":     lambda s: _np.nanmean(_pd_to_num(s)),
            "mean":    lambda s: _np.nanmean(_pd_to_num(s)),
            "median":  lambda s: _np.nanmedian(_pd_to_num(s)),
            "min":     lambda s: _np.nanmin(_pd_to_num(s)),
            "max":     lambda s: _np.nanmax(_pd_to_num(s)),
            "count":   lambda s: int(s.notna().sum()) if hasattr(s, "notna") else int(len(_coerce_list(s))),
            "distinct_count": lambda s: int(s.nunique(dropna=True)) if hasattr(s, "nunique") else int(len(set(_coerce_list(s)))),
            "last":    lambda df_col_with_date: _last_value(df_col_with_date),
        }


        def _tool_pct_where(datasets, spec, extra_refs=None):
            """
            Percentage of a base set that satisfies a condition.

            Spec:
              tool: "pct_where"
              step: "s1"
              where: {...}                 # predicate (same shape as your filters)
              filters: {...}               # pre-filters applied to the base universe (optional)
              group_by: ["RMID"]           # output groups (optional)
              within: "group"|"partition"|"global"  # denominator scope (default "group")
              partition_by: ["MEET_YEAR"]  # needed if within="partition"
              base: { "distinct_on": "CLIENT_ID" }  # entity for the base set (rows if omitted)
              scale: 100                   # 100 for %, 1 for fraction (default 100)
              on_zero: "null"|"zero"      # denominator==0 behavior (default "null")
              alias: "PCT_WHERE"          # output column name (default)
            """
            import pandas as pd, numpy as np

            df = datasets[(spec.get("step") or "").strip()].copy()
            group_by     = list(spec.get("group_by") or [])
            within       = str(spec.get("within") or "group").lower()
            partition_by = list(spec.get("partition_by") or [])
            base         = spec.get("base") or {}
            distinct_on  = base.get("distinct_on")
            scale        = float(spec.get("scale", 100.0))
            on_zero      = (spec.get("on_zero") or "null").lower()
            alias        = spec.get("alias") or "PCT_WHERE"

            # 1) Pre-filter base universe
            if spec.get("filters"):
                df = _apply_simple_filters(df, spec.get("filters"))

            # 2) Numerator: rows/entities satisfying the predicate
            if spec.get("where"):
                num_df = _apply_simple_filters(df.copy(), spec.get("where"))
            else:
                num_df = df.iloc[0:0].copy()  # no predicate → 0%

            # Scope keys for denominator
            if within == "group":
                denom_keys = group_by[:]
            elif within == "partition":
                denom_keys = partition_by[:]
            else:  # global
                denom_keys = []

            # 3) Denominator counts
            if distinct_on:
                dn_keys = denom_keys + ([distinct_on] if isinstance(distinct_on, str) else list(distinct_on))
                denom = (
                    df[dn_keys]
                    .drop_duplicates(subset=distinct_on)
                    .groupby(denom_keys, observed=True)
                    .size()
                    .rename("DENOM")
                    .reset_index()
                )
            else:
                denom = (
                    df.groupby(denom_keys, observed=True)
                      .size()
                      .rename("DENOM")
                      .reset_index()
                )

            # 4) Numerator counts (respect the same base entity if provided)
            g_keys = group_by + (partition_by if within == "partition" else [])
            if distinct_on:
                nm_keys = g_keys + ([distinct_on] if isinstance(distinct_on, str) else list(distinct_on))
                num = (
                    num_df[nm_keys]
                    .drop_duplicates(subset=distinct_on)
                    .groupby(g_keys, observed=True)
                    .size()
                    .rename("NUM")
                    .reset_index()
                )
            else:
                num = (
                    num_df.groupby(g_keys, observed=True)
                          .size()
                          .rename("NUM")
                          .reset_index()
                )

            # 5) Join numerator with denominator per scope
            if within == "group":
                out = pd.merge(num, denom, on=group_by, how="right")
            elif within == "partition":
                out = pd.merge(num, denom, on=partition_by, how="right")
                # ensure group_by columns exist
                for col in group_by:
                    if col not in out.columns:
                        out[col] = pd.NA
            else:  # global
                total_denom = int(denom["DENOM"].sum()) if len(denom_keys) else (int(denom["DENOM"].iloc[0]) if not denom.empty else 0)
                if not group_by:
                    total_num = int(num["NUM"].sum()) if not num.empty else 0
                    val = (scale * total_num / total_denom) if total_denom else (None if on_zero == "null" else 0.0)
                    return {"ok": True, "value": val}
                out = num.copy()
                out["DENOM"] = total_denom

            # 6) Finalize %
            if "NUM" not in out.columns: out["NUM"] = 0
            if "DENOM" not in out.columns: out["DENOM"] = 0
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = np.where(out["DENOM"].to_numpy() == 0,
                               (np.nan if on_zero == "null" else 0.0),
                               scale * out["NUM"].to_numpy() / out["DENOM"].to_numpy())
            out[alias] = pct

            keep_cols = group_by + [alias]
            out = out[keep_cols].drop_duplicates().reset_index(drop=True)
            return {"ok": True, "grouped": out}



        def run_metric_tool(
            datasets: Dict[str, 'pd.DataFrame'],
            spec: Dict[str, Any],
            extra_refs: Optional[Dict[str,'pd.DataFrame']] = None
        ) -> Dict[str, Any]:
            """
            spec = {
              "tool": "sum|avg|median|min|max|count|distinct_count|last",
              "step": "s1",                 # dataset id
              "column": "CURRENT_AUM",      # metric column
              "filters": {"MONTH_DATE":"2025-08"},  # optional filters
              "group_by": ["RMID"],         # optional
              "date_col": "MONTH_DATE",     # for 'last'
              "join": {"using": "RMID", "ref": "rm_map"}  # optional join before computing
            }
            """
            import pandas as pd
            tool = (spec.get("tool") or "").lower()
            step = spec.get("step")
            if step not in datasets:
                return {"ok": False, "error": f"Unknown step '{step}'"}

            if tool == "pct_where":
                return _tool_pct_where(datasets, spec, extra_refs)


            left = datasets[step].copy()
            left_cols = list(left.columns)  # to detect/restore left cardinality

            # ── Ground-truth CALC DATAFRAME (never sees joins) ────────────────
            calc_df = _apply_simple_filters(left.copy(), spec.get("filters"))

            if tool in ("corr", "correlation"):
                x = spec.get("x") or spec.get("column")
                y = spec.get("y") or spec.get("column_y")
                if not x or not y:
                    return {"ok": False, "error": "corr requires 'x' and 'y' columns"}
                if x not in calc_df.columns or y not in calc_df.columns:
                    return {"ok": False, "error": f"corr columns not found: {x}, {y}"}
                d = calc_df[[x, y] + [c for c in (spec.get("group_by") or []) if c in calc_df.columns]].copy()
                d[x] = pd.to_numeric(d[x], errors="coerce")
                d[y] = pd.to_numeric(d[y], errors="coerce")
                d = d.dropna(subset=[x, y])
                gby = [g for g in (spec.get("group_by") or []) if g in d.columns]
                if gby:
                    out = []
                    for keys, g in d.groupby(gby, dropna=False):
                        val = None
                        if len(g) >= 2:
                            try:
                                val = g[x].corr(g[y])
                            except Exception:
                                val = None
                        out.append({"group": keys if isinstance(keys, tuple) else (keys,), "value": _try_float(val)})
                    return {"ok": True, "tool": "corr", "grouped": out}
                val = None
                if len(d) >= 2:
                    try:
                        val = d[x].corr(d[y])
                    except Exception:
                        val = None
                return {"ok": True, "tool": "corr", "value": _try_float(val)}

            # ── Preview DF (can include joins strictly for LLM context/frames) ─
            df = left  # keep the old name for the preview/join path below


            join = spec.get("join")
            if join and extra_refs:
                ref_name = join.get("ref")
                on = join.get("using")
                if ref_name in extra_refs and on:
                    right = extra_refs[ref_name].copy()

                    # Canonicalize common id columns so 'on' is guaranteed to exist
                    def _canon_ids(_df):
                        up = {c.upper(): c for c in _df.columns}
                        ren = {}
                        if "RM_ID" in up and "RMID" not in _df.columns: ren[up["RM_ID"]] = "RMID"
                        if "RELATIONSHIP_MANAGER_ID" in up and "RMID" not in _df.columns: ren[up["RELATIONSHIP_MANAGER_ID"]] = "RMID"
                        if "MANDATE_ID" in up and "MANDATEID" not in _df.columns: ren[up["MANDATE_ID"]] = "MANDATEID"
                        if ren: _df = _df.rename(columns=ren)
                        return _df

                    right = _canon_ids(right)

                    # If the tool plans to GROUP BY any column only present on the join side,
                    # expansion is intentional (e.g., group by MANDATEID). In that case, don't de-fan-out.
                    needs_group_by = spec.get("group_by") or []
                    group_uses_right = any(g not in left_cols for g in needs_group_by)

                    # If join key still isn't on the right side, skip dedup and just merge (or fail clearly)
                    if on not in right.columns:
                        # Be explicit so the error shows up in your tool panel
                        try:
                            return {"ok": False, "error": f"Join key '{on}' not found in right-side columns: {list(right.columns)}"}
                        except Exception:
                            return {"ok": False, "error": f"Join key '{on}' missing on right"}

                    # Collapse RIGHT to 1 row / key to avoid fan-out when we DON'T need right-side grouping.
                    if not group_uses_right:
                        if "CREATE_DATE" in right.columns:
                            right = right.sort_values("CREATE_DATE").drop_duplicates(subset=[on], keep="last")
                        else:
                            right = right.drop_duplicates(subset=[on])

                    df = left.merge(right, on=on, how="left")

                    # If we are not grouping by a right-only column, restore left cardinality to kill duplicates.
                    if not group_uses_right:
                        # Only keep the first occurrence per original left row signature
                        try:
                            df = df.drop_duplicates(subset=left_cols)
                        except Exception:
                            # defensive fallback
                            df = df.loc[:, ~df.columns.duplicated(keep="first")].drop_duplicates()



            df_preview = _apply_simple_filters(df, spec.get("filters"))

            # --- TopK / Rank ---
            if tool in ("topk", "rank"):
                tk = dict(spec.get("topk") or {})
                k = int(tk.get("k") or 5)
                by = str(tk.get("by") or "VALUE").upper()
                ascending = bool(tk.get("ascending", False))
                agg = str(tk.get("agg") or "avg").lower()

                gby = [g for g in (spec.get("group_by") or []) if g in calc_df.columns]

                col = spec.get("column")
                if not col:
                    for cand in ("METRIC_VALUE", "VALUE"):
                        if cand in calc_df.columns:
                            col = cand
                            break
                    if not col:
                        num_cols = [c for c in calc_df.columns if pd.api.types.is_numeric_dtype(calc_df[c])]
                        if num_cols:
                            col = num_cols[0]

                if not gby:
                    # If already aggregated, just sort rows by metric
                    if col and col in calc_df.columns:
                        out = calc_df.sort_values(col, ascending=ascending, kind="mergesort").head(k)
                        return {"ok": True, "tool": tool, "frame": out.to_dict(orient="records")}
                    return {"ok": False, "error": "topk/rank requires 'group_by' and 'column'"}
                if not col or col not in calc_df.columns:
                    return {"ok": False, "error": "topk/rank requires 'group_by' and 'column'"}

                # aggregate to one row per group
                if agg == "sum":
                    s = calc_df.groupby(gby, dropna=False)[col].sum()
                elif agg in ("avg", "mean"):
                    s = calc_df.groupby(gby, dropna=False)[col].mean()
                elif agg == "median":
                    s = calc_df.groupby(gby, dropna=False)[col].median()
                elif agg == "min":
                    s = calc_df.groupby(gby, dropna=False)[col].min()
                elif agg == "max":
                    s = calc_df.groupby(gby, dropna=False)[col].max()
                elif agg == "count":
                    s = calc_df.groupby(gby, dropna=False)[col].count()
                else:
                    s = calc_df.groupby(gby, dropna=False)[col].mean()  # safe default

                out = s.reset_index().rename(columns={col: "VALUE"})

                # choose the sort key: "VALUE" unless user asked otherwise
                sort_col = "VALUE" if by.upper() == "VALUE" else by
                if sort_col not in out.columns:
                    sort_col = "VALUE"

                out = out.sort_values(sort_col, ascending=ascending, kind="mergesort").head(k)

                # Return a frame payload (stable for downstream)
                return {
                    "ok": True,
                    "tool": tool,
                    "frame": out.to_dict(orient="records")
                }


            if tool == "last":
                v = _METRIC_FUNS["last"](_prep_last(calc_df, spec.get("column"), spec.get("date_col")))
                return {"ok": True, "tool": "last", "value": _try_float(v)}


            if spec.get("group_by"):
                gby = [g for g in (spec.get("group_by") or []) if g in calc_df.columns]
                out = []
                for keys, g in calc_df.groupby(gby, dropna=False):
                    ser = g[spec["column"]] if spec["column"] in g.columns else None

                    val = _METRIC_FUNS.get(tool, _METRIC_FUNS["sum"])(ser) if ser is not None else None
                    out.append({"group": keys if isinstance(keys, tuple) else (keys,), "value": _try_float(val)})
                return {"ok": True, "tool": tool, "grouped": out}

            ser = calc_df[spec["column"]] if spec["column"] in calc_df.columns else None
            val = _METRIC_FUNS.get(tool, _METRIC_FUNS["sum"])(ser) if ser is not None else None
            return {"ok": True, "tool": tool, "value": _try_float(val)}


        def _try_float(v):
            try:
                fv = float(v)
                if _np.isfinite(fv): return fv
            except Exception:
                pass
            return v
        # ──────────────────────────────────────────────────────────────────────────────

        @st.cache_data(show_spinner="Loading RM↔Mandate mapping…", ttl=900)
        def load_rm_to_mandate_map(session) -> 'pd.DataFrame':
            sql = """
            select *
            from tfo.tfo_schema.rmtomandate
            where create_date = (
              select max(create_date)
              from tfo.tfo_schema.rmtomandate
            )
            """
            df = session.sql(sql).to_pandas()
            df.columns = [str(c).strip().upper() for c in df.columns]
            return df


        # ──────────────────────────────────────────────────────────────────────────────
        # AUM guardrails — deterministic needs that adapt to the question intent
        # Place this right above summarize_with_tools().

        import re as _re

        _AUM_COL_PAT = _re.compile(r"(?:^|[_\W])(aum|current[_\W]*aum|snap[_\W]*current[_\W]*aum)(?:$|[_\W])", _re.I)

        def _guess_cols_for_aum(_df):
            """Return (aum_col, date_col, rmid_col) from a DataFrame."""
            import pandas as pd
            if _df is None or not isinstance(_df, pd.DataFrame) or _df.empty:
                return (None, None, None)

            # AUM column
            aum_col = None
            for c in _df.columns:
                if _AUM_COL_PAT.search(str(c)):
                    aum_col = c; break
                if str(c).upper() in ("AUM", "CURRENT_AUM", "SNAP_CURRENT_AUM"):
                    aum_col = c; break
            if not aum_col:
                for c in _df.columns:
                    if "AUM" in str(c).upper():
                        aum_col = c; break

            # Date column
            date_col = _pick_date_col(_df)

            # RMID column
            up = {str(c).upper(): c for c in _df.columns}
            rmid_col = up.get("RMID") or up.get("RM_ID") or up.get("RELATIONSHIP_MANAGER_ID")

            return (aum_col, date_col, rmid_col)

        def _trend_intent(q: str) -> bool:
            ql = (q or "").lower()
            return any(w in ql for w in ("trend", "over time", "by month", "monthly", "timeline"))

        def _all_rms_intent(q: str) -> bool:
            ql = (q or "").lower()
            return any(w in ql for w in ("all rms", "across all rms", "overall", "total", "portfolio", "consolidated"))

        def _single_month_hint(q: str):
            # Reuse your existing extractor if available
            try:
                return _extract_time_hints(q).get("single_month")
            except Exception:
                return None

        def _explicit_years(q: str):
            try:
                return _extract_time_hints(q).get("explicit_years") or []
            except Exception:
                return []

        def _latest_month_in_year(df, date_col: str, year: int):
            import pandas as pd
            if not date_col or date_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                return None
            d = df[[date_col]].dropna()
            d = d[d[date_col].dt.year == year]
            if d.empty: return None
            return d[date_col].max().to_period("M").to_timestamp()

        def _ym_str(ts):
            try:
                return f"{int(ts.year)}-{int(ts.month):02d}" if ts is not None else None
            except Exception:
                return None

        def aum_autoplan(question: str, frames: dict) -> list[dict]:
            """
            Returns a list of deterministic `spec` dicts (for run_metric_tool) so the LLM
            never computes AUM. Specs cover:
              • Trend in a year  → monthly total + per-RM monthly
              • Total AUM in year → sum at latest month in that year
              • AUM in a month → sum across RMs for that month (+ optional per-RM)
            """
            specs: list[dict] = []
            if not isinstance(frames, dict) or not frames:
                return specs

            # Pick the first step that actually contains AUM
            target = None
            for step_id, df in frames.items():
                aum_col, date_col, rmid_col = _guess_cols_for_aum(df)
                if aum_col and date_col:
                    target = (step_id, aum_col, date_col, rmid_col)
                    break
            if not target:
                return specs

            step_id, aum_col, date_col, rmid_col = target
            q = question or ""
            years = _explicit_years(q)
            single = _single_month_hint(q)

            # 1) Trend intent → totals by month (+ per-RM monthly)
            if _trend_intent(q):
                if len(years) == 1:
                    yr = years[0]
                    # Monthly total across all RMs
                    specs.append({
                        "tool": "sum", "step": step_id, "column": aum_col,
                        "filters": {"MEET_YEAR": yr} if "MEET_YEAR" in frames[step_id].columns else None,
                        "group_by": [date_col], "date_col": date_col
                    })
                    # Per-RM monthly (if RMID present)
                    if rmid_col:
                        specs.append({
                            "tool": "sum", "step": step_id, "column": aum_col,
                            "filters": {"MEET_YEAR": yr} if "MEET_YEAR" in frames[step_id].columns else None,
                            "group_by": [rmid_col, date_col], "date_col": date_col
                        })
                else:
                    specs.append({
                        "tool": "sum", "step": step_id, "column": aum_col,
                        "group_by": [date_col], "date_col": date_col
                    })
                    if rmid_col:
                        specs.append({
                            "tool": "sum", "step": step_id, "column": aum_col,
                            "group_by": [rmid_col, date_col], "date_col": date_col
                        })
                return specs

            # 2) Single explicit month → sum across RMs (+ per-RM)
            if single:
                ym = f"{single['year']}-{int(single['month']):02d}"
                specs.append({
                    "tool": "sum", "step": step_id, "column": aum_col,
                    "filters": {date_col: ym}
                })
                if rmid_col:
                    specs.append({
                        "tool": "sum", "step": step_id, "column": aum_col,
                        "filters": {date_col: ym}, "group_by": [rmid_col]
                    })
                return specs

            # 3) "Total AUM in <YEAR>" → sum at latest month in that year (snapshot, not annual sum)
            if _all_rms_intent(q) or ("total aum" in (q or "").lower()):
                if len(years) == 1:
                    yr = years[0]
                    df = frames[step_id]
                    last_ts = _latest_month_in_year(df, date_col, yr)
                    ym = _ym_str(last_ts)
                    if ym:
                        specs.append({
                            "tool": "sum",
                            "step": step_id,
                            "column": aum_col,
                            "filters": {date_col: ym},        # snapshot: only the latest month of that year
                            "date_col": date_col              # tell the tool what the month field is
                        })
                        if rmid_col:
                            specs.append({
                                "tool": "sum",
                                "step": step_id,
                                "column": aum_col,
                                "filters": {date_col: ym},
                                "group_by": [rmid_col],        # optional per-RM for narrative (still a snapshot)
                                "date_col": date_col
                            })
                        return specs


            # Fallback → portfolio total at latest month in the step
            df = frames[step_id]
            ts = None
            try:
                ts = df[date_col].dropna().max().to_period("M").to_timestamp()
            except Exception:
                pass
            ym = _ym_str(ts)
            if ym:
                specs.append({
                    "tool": "sum", "step": step_id, "column": aum_col,
                    "filters": {date_col: ym}
                })
            return specs


        def summarize_with_tools(question: str,
                                 frames: Dict[str, 'pd.DataFrame'],
                                 deterministic_insights: Optional[List[str]] = None,
                                 extra_refs: Optional[Dict[str,'pd.DataFrame']] = None,
                                 on_tool=None) -> Dict[str, Any]:
            """
            Pass 1: Ask LLM what deterministic metrics it needs (never do math in LLM).
            Pass 2: Compute with run_metric_tool(), convert to ground-truth lines.
            Pass 3: Call existing final_summary_llm() with those lines injected.
            """
            import json, pandas as pd

            # ---- UI helpers (no-op if no callback given) ----
            def _emit(msg: str):
                try:
                    if callable(on_tool):
                        on_tool(msg)
                except Exception:
                    pass

            def _schema_preview_for_llm(frames: dict, max_rows: int = 5) -> str:
                import pandas as pd
                blocks = []
                for sid, df in (frames or {}).items():
                    try:
                        dtypes = {str(c): str(t) for c, t in df.dtypes.items()}
                        prev = df.head(max_rows).to_dict(orient="records")
                    except Exception:
                        dtypes = {}
                        prev = []
                    blocks.append({
                        "step": sid,
                        "schema": dtypes,
                        "preview": prev
                    })
                import json
                return json.dumps(blocks, ensure_ascii=False, indent=2, default=str)


            def _fmt_val(v):
                try:
                    return f"{float(v):,.2f}"
                except Exception:
                    return str(v)

            def _pretty_need(spec: dict) -> str:
                tool = str((spec or {}).get("tool", "")).lower()
                col  = (spec or {}).get("column")
                step = (spec or {}).get("step")
                gb   = (spec or {}).get("group_by") or []
                fil  = (spec or {}).get("filters") or {}
                verb = {
                  "sum":"Summing","avg":"Averaging","mean":"Averaging","median":"Median of",
                  "min":"Minimum of","max":"Maximum of","count":"Counting",
                  "distinct_count":"Counting distinct","last":"Latest of",
                  "pct_where":"Percent of base where", # ← add this line
                  "corr":"Correlation between"
                }.get(tool, tool.title() or "Computing")
                if tool == "corr":
                    x = spec.get("x") or spec.get("column")
                    y = spec.get("y") or spec.get("column_y")
                    col = f"{x} vs {y}"

                parts = [f"{verb} **{col}**", f"in *{step}*"]
                if fil: parts.append(f"with filters {fil}")
                if gb:  parts.append(f"grouped by {', '.join(gb)}")
                return " ".join(parts)


            # --- Pass 1: get "needs" from LLM (tools it wants us to run) ---
            schema = {k: [str(c) for c in v.columns] for k, v in (frames or {}).items() if isinstance(v, pd.DataFrame)}
            preview = {k: v.head(30).to_dict(orient="records") for k, v in (frames or {}).items() if isinstance(v, pd.DataFrame)}

            # AUM guardrails: pre-plan deterministic specs so LLM never computes AUM
            auto_needs = []
            try:
                if "aum" in (question or "").lower():
                    auto_needs = aum_autoplan(question, frames)
                    if auto_needs:
                        for spec in auto_needs:
                            _emit(f"🔧 (guardrail) {_pretty_need(spec)} …")
            except Exception as _e:
                _emit(f"  ️ AUM guardrail planning skipped: {str(_e)[:120]}")

            NEEDS_PROMPT = r"""
You are a tool planner.

INPUTS YOU WILL RECEIVE:
- QUESTION: the user's request (free text)
- DATAFRAMES: a JSON list of dataframes, each with {"step": <id>, "schema": {col: dtype}, "preview": [rows...]}

YOUR TASK:
Return a SINGLE JSON object describing which tools to run PER STEP.

STRICT RULES:
- Output MUST be valid JSON. Do not include comments, prose, markdown, or trailing commas.
- Use the field **by_step** only. It MUST be a list where each item is:
  {"step": "<one of ALLOWED_STEPS>", "needs": [<tool-spec>, ...]}
- Each tool-spec MUST include the "step" field set to the SAME value as the parent item.
- Only use columns that appear in that step's schema.
- Use only these tool names: "topk","rank","percentile","yoy","mom","sum","avg","median","min","max","last","pct_of_total","pct_where","corr".
- If the QUESTION asks for correlation/relationship, use tool 'corr' with x and y columns.
- If the QUESTION mentions years/months and the step has MEET_YEAR/MEET_MON/POSYEAR/POSMON/YEAR/MONTH, use simple equality filters (e.g., {"MEET_YEAR": 2025}).
- If the step has a datetime column but no year/month columns, you MAY omit filters; we will still compute.
- Prefer "agg":"avg" for performance/score metrics; "agg":"sum" for flows/revenue.
- You MUST NOT invent steps. The only valid steps are in ALLOWED_STEPS.
- If no tools are relevant for a step, return an empty "needs": [] for that step.
- Do not copy any examples. Construct specs based on QUESTION + DATAFRAMES.
- Return JSON only; no extra text.

JSON SHAPE (schema, not an example):
{
  "by_step": [
    {
      "step": "<one of ALLOWED_STEPS>",
      "needs": [
        {
          "tool": "<one of the tools>",
          "step": "<same as parent step>",
          "column": "<column from this step's schema>",
          "x": "<column for corr>",
          "y": "<column for corr>",
          "group_by": ["<optional col>", "..."],
          "filters": {"<optional simple equality or between>": ...},
          "date_col": "<optional date column for yoy/mom>",
          "topk": {"k": <int>, "by": "VALUE", "ascending": <bool>, "agg": "<sum|avg|median|min|max>"},
          "rank": {"by": "VALUE", "ascending": <bool>},
          "percentile": {"p": <0-100>},
          "output": "frame"
        }
      ]
    }
  ]
}

ALLOWED_STEPS:
{{ALLOWED_STEPS_JSON}}

QUESTION:
{{QUESTION}}

DATAFRAMES:
{{DATAFRAMES_BLOCK}}
""".strip()

            _emit("🧭 planning metrics to compute…")

            # --- Pass 1: decide tools/needs (per-step) ---
            needs = []

            try:
                dataframes_block = _schema_preview_for_llm(frames, max_rows=5)

                import json as __json
                allowed_steps_json = __json.dumps(list((frames or {}).keys()), ensure_ascii=False)

                pbody = (
                    NEEDS_PROMPT
                    .replace("{{ALLOWED_STEPS_JSON}}", allowed_steps_json)
                    .replace("{{QUESTION}}", str(question))
                    .replace("{{DATAFRAMES_BLOCK}}", dataframes_block)
                )

                q = "select snowflake.cortex.complete('{model}', $${body}$$) as response".format(
                    model=CORTEX_LLM_MODEL,
                    body=pbody.replace("$$", "$ $"),
                )

                df_prompt = session.sql(q).to_pandas()
                resp_text = df_prompt.iloc[0, 0]

                # Parse robustly
                js = None
                try:
                    js = safe_json_loads(json_extract(resp_text) or resp_text)
                except Exception:
                    try:
                        js = json.loads(resp_text)
                    except Exception:
                        js = None

                by_step = []
                if isinstance(js, dict) and isinstance(js.get("by_step"), list):
                    by_step = js["by_step"]
                elif isinstance(js, dict) and isinstance(js.get("needs"), list):
                    # fallback to single flat list; we'll apply specs to their declared step or skip if missing
                    by_step = [{"step": s.get("step"), "needs": [s]} for s in (js["needs"] or []) if isinstance(s, dict)]
                else:
                    by_step = []

                # flatten but keep valid 'step'
                out_needs = []
                for item in by_step:
                    st = (item or {}).get("step")
                    arr = (item or {}).get("needs") or []
                    for spec in arr:
                        if not isinstance(spec, dict):
                            continue
                        if not spec.get("step"):
                            # if the LLM forgot, set step if present at wrapper-level
                            if st:
                                spec["step"] = st
                        if spec.get("step") in frames:
                            out_needs.append(spec)

                needs = out_needs

                # Merge in guardrail specs (dedupe by JSON signature)
                if auto_needs:
                    seen = set()
                    for s in needs:
                        try:
                            seen.add(json.dumps(s, sort_keys=True, default=str))
                        except Exception:
                            pass
                    for s in auto_needs:
                        try:
                            key = json.dumps(s, sort_keys=True, default=str)
                        except Exception:
                            key = None
                        if key and key in seen:
                            continue
                        needs.append(s)
                        if key:
                            seen.add(key)

                # stash for UI/debug
                try:
                    st.session_state.setdefault("summarizer_debug", {})["last_tool_calls"] = {
                        "question": question,
                        "needs": needs,
                        "resolved": []
                    }
                except Exception:
                    pass

            except Exception as _e:
                _emit(f"⚠️ Needs LLM failed: {_e}")
                needs = []



            # --- Baseline specs per step if planner returned nothing for that step ---
            def _baseline_for_step(step_id: str, df):
                # try to find a performance-like column and RMID for a simple Top-5
                up = {str(c).upper(): c for c in df.columns}
                perf = None
                for c in df.columns:
                    u = str(c).upper()
                    if ("PERFORMANCE" in u or "SCORE" in u) and ("AVG" in u or "_MEAN" in u or "AVERAGE" in u):
                        perf = c; break
                    if ("PERFORMANCE" in u or "SCORE" in u) and perf is None:
                        perf = c
                rmid = up.get("RMID") or up.get("RM_ID")
                if perf and rmid:
                    return [{
                        "tool":"topk","step":step_id,"column":perf,
                        "group_by":[rmid],
                        "topk":{"k":5,"by":"VALUE","ascending":False,"agg":"avg"},
                        "output":"frame"
                    }]
                return []

            # Merge: keep planner specs and add baseline only for steps with none
            steps_with_needs = {s.get("step") for s in needs if isinstance(s, dict)}
            for sid, df_step in (frames or {}).items():
                if sid not in steps_with_needs:
                    needs.extend(_baseline_for_step(sid, df_step))


            if needs:
                for spec in needs:
                    _emit(f"🔧 {_pretty_need(spec)} …")
            else:
                _emit("🔧 no extra metrics requested; composing answer…")

            def _normalize_tool_spec(spec: dict, df: pd.DataFrame, question_text: str) -> dict:
                if not isinstance(spec, dict) or df is None or not isinstance(df, pd.DataFrame):
                    return spec

                spec = dict(spec)
                col_map = {str(c).lower(): c for c in df.columns}

                def _resolve(name):
                    if name is None:
                        return name
                    if name in df.columns:
                        return name
                    low = str(name).lower()
                    if low in col_map:
                        return col_map[low]
                    norm = re.sub(r"[^a-z0-9]", "", low)
                    for key, col in col_map.items():
                        if re.sub(r"[^a-z0-9]", "", key) == norm:
                            return col
                    return name

                tool = str(spec.get("tool") or "").lower()

                for key in ("column", "x", "y", "column_y", "date_col"):
                    if key in spec:
                        spec[key] = _resolve(spec.get(key))

                if isinstance(spec.get("group_by"), list):
                    spec["group_by"] = [c for c in (_resolve(c) for c in spec["group_by"]) if c]

                if isinstance(spec.get("filters"), dict):
                    mapped = {}
                    for k, v in spec["filters"].items():
                        mapped[_resolve(k)] = v
                    spec["filters"] = mapped

                if isinstance(spec.get("topk"), dict):
                    by = spec["topk"].get("by")
                    if by and str(by).upper() != "VALUE":
                        spec["topk"]["by"] = _resolve(by)

                if isinstance(spec.get("rank"), dict):
                    by = spec["rank"].get("by")
                    if by and str(by).upper() != "VALUE":
                        spec["rank"]["by"] = _resolve(by)

                def _guess_metric_column():
                    ql = (question_text or "").lower()
                    patterns = []
                    if "performance" in ql or "score" in ql:
                        patterns.extend(["performance", "score"])
                    if "profit" in ql:
                        patterns.append("profit")
                    if "revenue" in ql:
                        patterns.append("revenue")
                    if "aum" in ql:
                        patterns.append("aum")
                    if "meeting" in ql:
                        patterns.append("meeting")
                    for pat in patterns:
                        for col in df.columns:
                            if pat in str(col).lower():
                                return col
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return col
                    return None

                def _guess_group_by():
                    for cand in (
                        "RM_NAME",
                        "RMID",
                        "RM_ID",
                        "RELATIONSHIP_MANAGER",
                        "RELATIONSHIP_MANAGER_NAME",
                        "MANDATEID",
                        "MANDATE_ID",
                    ):
                        resolved = _resolve(cand)
                        if resolved in df.columns:
                            return resolved
                    for col in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            return col
                    return None

                if tool in ("topk", "rank") and not spec.get("group_by"):
                    gb = _guess_group_by()
                    if gb:
                        spec["group_by"] = [gb]

                if not spec.get("column") and tool in ("topk", "rank", "sum", "avg", "median", "min", "max", "last"):
                    guessed = _guess_metric_column()
                    if guessed:
                        spec["column"] = guessed

                if tool == "topk":
                    topk_spec = dict(spec.get("topk") or {})
                    if "k" not in topk_spec:
                        m = re.search(r"\btop\s+(\d{1,3})\b", question_text or "", re.IGNORECASE)
                        if m:
                            try:
                                topk_spec["k"] = int(m.group(1))
                            except Exception:
                                pass
                    if "agg" not in topk_spec and spec.get("column"):
                        col_l = str(spec["column"]).lower()
                        if "performance" in col_l or "score" in col_l:
                            topk_spec["agg"] = "avg"
                        else:
                            topk_spec["agg"] = "sum"
                    spec["topk"] = topk_spec

                return spec

            # --- Pass 2: compute requested metrics deterministically ---
            computed = []
            for spec in (needs or []):
                if not isinstance(spec, dict):
                    continue
                # Guard: unknown step → skip cleanly
                step_id = spec.get("step")
                if step_id and step_id not in (frames or {}):
                    err = f"Unknown step '{step_id}'"
                    res = {"ok": False, "error": err}
                    computed.append({"spec": spec, "result": res})
                    try:
                        _emit(f"  ️ {_pretty_need(spec)} failed: {err}")
                    except Exception:
                        pass
                    continue

                try:
                    df_step = frames.get(step_id) if step_id else None
                    spec = _normalize_tool_spec(spec, df_step, question)
                    # Correct signature: spec first, datasets=frames
                    res = run_metric_tool(datasets=frames, spec=spec, extra_refs=extra_refs)

                except Exception as e:
                    res = {"ok": False, "error": str(e)}

                computed.append({"spec": spec, "result": res})

                # Debug emit (value / grouped / frame aware)
                try:
                    if res.get("ok"):
                        if "value" in res:
                            _emit(f"✅ {_pretty_need(spec)} → **{_fmt_val(res.get('value'))}**")
                        elif "grouped" in res:
                            _emit(f"✅ {_pretty_need(spec)} → **{len(res.get('grouped') or [])} groups**")
                        elif "frame" in res:
                            _emit(f"✅ {_pretty_need(spec)} → **{len(res.get('frame') or [])} rows**")
                        else:
                            _emit(f"✅ {_pretty_need(spec)}")
                    else:
                        _emit(f"  ️ {_pretty_need(spec)} failed: {res.get('error')}")
                except Exception:
                    pass

            driver_payload = None
            try:
                driver_payload = _run_driver_analysis(question, frames)
            except Exception as _e:
                try:
                    trace("driver_analysis_run_error", str(_e)[:MAX_TRACE_PAYLOAD])
                except Exception:
                    pass
                driver_payload = None

            if driver_payload:
                computed.append(driver_payload)
                try:
                    msg = (driver_payload.get("result") or {}).get("value")
                    if msg:
                        snippet = msg if len(msg) <= 160 else (msg[:157] + "...")
                        _emit(f"🔍 {snippet}")
                except Exception:
                    pass

            scenario_notes = []
            scenario_payload = None
            try:
                scenario_notes, scenario_payload = _scenario_projection_payload(question, frames)
            except Exception as _e:
                try:
                    trace("scenario_projection_error", str(_e)[:MAX_TRACE_PAYLOAD])
                except Exception:
                    pass
                scenario_notes = []
                scenario_payload = None
            if scenario_payload:
                computed.append(scenario_payload)
                try:
                    _emit("📊 Estimated scenario impact via regression.")
                except Exception:
                    pass

            # Update UI/debug stash with resolved tool results
            try:
                sd = st.session_state.get("summarizer_debug", {}).get("last_tool_calls")
                if isinstance(sd, dict):
                    sd["resolved"] = computed
            except Exception:
                pass
            try:
                trace("llm_tools_resolved", {"count": len(computed)})
            except Exception:
                pass



            # --- Pass 3: convert to ground-truth lines & call final summarizer ---
            import json as _json

            def _fmt_filters(f):
                try:
                    return _json.dumps(f or {}, ensure_ascii=False, default=str)
                except Exception:
                    return str(f or {})

            def _preview_frame(df, max_rows=5):
                try:
                    import pandas as pd
                    if not hasattr(df, "head"):
                        return []
                    sample = df.head(max_rows)
                    # Try to show entity + VALUE for readability
                    # Pick a display column:
                    colsU = {str(c).upper(): c for c in sample.columns}
                    name_col = None
                    for cand in ("RM_NAME","RMNAME","RELATIONSHIP_MANAGER","RELATIONSHIP_MANAGER_NAME","RMID","RM_ID"):
                        if cand in colsU:
                            name_col = colsU[cand]; break
                    if name_col is None:
                        # fallback: first non-VALUE
                        for c in sample.columns:
                            if str(c).upper() != "VALUE":
                                name_col = c; break
                    if name_col is None:
                        name_col = sample.columns[0]
                    out = []
                    for _, r in sample.iterrows():
                        nm = r.get(name_col, None)
                        v  = r.get("VALUE", None)
                        try:
                            v = None if v is None else float(v)
                        except Exception:
                            pass
                        out.append({"name": None if nm is None else str(nm), "VALUE": v})
                    return out
                except Exception:
                    return []

            deterministic_lines = []
            for item in (computed or []):
                spec = item.get("spec") or {}
                res  = item.get("result") or {}
                tool = spec.get("tool")
                step = spec.get("step")
                col  = spec.get("column")
                if tool == "corr":
                    x = spec.get("x") or spec.get("column")
                    y = spec.get("y") or spec.get("column_y")
                    col = f"{x} vs {y}"
                filt = spec.get("filters") or {}
                gby  = spec.get("group_by") or []
                filt_str = _fmt_filters(filt)

                if res.get("ok"):
                    if "frame" in res:
                        head = _preview_frame(res["frame"], max_rows=5)
                        deterministic_lines.append(
                            f"[RESOLVED] {tool} on {step}.{col} group_by={gby or []} filters={filt_str} → rows={len(res['frame'])}, preview={head}"
                        )
                    elif "grouped" in res:
                        head = (res.get("grouped") or [])[:5]
                        deterministic_lines.append(
                            f"[RESOLVED] {tool} (grouped) on {step}.{col} group_by={gby or []} filters={filt_str} → {head}"
                        )
                    elif "value" in res:
                        deterministic_lines.append(
                            f"[RESOLVED] {tool} on {step}.{col} filters={filt_str} → {res.get('value')}"
                        )
                        if tool == "driver_analysis":
                            for rec in (res.get("recommendations") or [])[:3]:
                                deterministic_lines.append(f"[RECOMMEND] {rec}")
                    else:
                        deterministic_lines.append(
                            f"[RESOLVED] {tool} on {step}.{col} filters={filt_str} → ok"
                        )
                else:
                    deterministic_lines.append(
                        f"[FAILED] {tool} on {step}.{col} filters={filt_str} → {res.get('error')}"
                    )

            if scenario_notes:
                deterministic_lines.extend(scenario_notes)

            _emit("   assembling insights…")

            # Call your summarizer; if it returns nothing, build a compact deterministic fallback
            summary_out = final_summary_llm(question, frames, deterministic_lines, resolved_tools=computed)
            if driver_payload and isinstance(summary_out, dict):
                try:
                    summary_out.setdefault("diagnostics", {})["driver_analysis"] = driver_payload.get("result")
                except Exception:
                    pass
            if scenario_payload and isinstance(summary_out, dict):
                try:
                    summary_out.setdefault("diagnostics", {})["scenario_projection"] = scenario_payload.get("result")
                except Exception:
                    pass
            if not summary_out or not isinstance(summary_out, dict) or not summary_out.get("narrative"):
                bullets = "\n".join(f"• {ln}" for ln in deterministic_lines[:12]) or "• (no metrics computed)"
                return {
                    "narrative": "Here’s what I computed deterministically:\n" + bullets,
                    "kpis": [],
                    "chart": {},
                    "needs_more": False,
                    "followup_prompt": "",
                    "required": {},
                }
            return summary_out






        def validate_visualization(vis: dict, df: pd.DataFrame, metric_candidates: list) -> tuple[bool, dict]:
            """
            Validate and normalize a visualization dict returned by LLMs.
            Returns (is_valid, normalized_vis)
            Normalized keys: chart,x,y,series,agg,allowed_chart_types,confidence,rationale,top_n
            """
            norm = {
                "chart": None,
                "x": None,
                "y": None,
                "series": None,
                "agg": "sum",
                "allowed_chart_types": ["table"],
                "confidence": 0.0,
                "rationale": "",
                "top_n": None
            }
            try:
                if not isinstance(vis, dict):
                    return False, norm
                norm["chart"] = vis.get("chart") or norm["chart"]
                norm["x"] = vis.get("x") or None
                norm["y"] = vis.get("y") or None
                norm["series"] = vis.get("series") or None
                if vis.get("agg") in ("sum","avg","median","last","count"):
                    norm["agg"] = vis.get("agg")
                allowed = [t for t in (vis.get("allowed_chart_types") or []) if t in ("line","bar","scatter","table")]
                norm["allowed_chart_types"] = allowed or ["table"]
                try:
                    c = float(vis.get("confidence")) if vis.get("confidence") is not None else 0.0
                    norm["confidence"] = max(0.0, min(1.0, c))
                except Exception:
                    norm["confidence"] = 0.0
                norm["rationale"] = str(vis.get("rationale") or "")
                if isinstance(vis.get("top_n"), int): 
                    norm["top_n"] = vis.get("top_n")
            except Exception:
                pass
            # quick field sanity
            return True, norm


        
              

        # --- New: convert DESCRIBE results to serializable records for caching ---
        def describe_rows_to_records(rows):
            """Convert Snowpark Row objects (or similar) into list-of-dicts safely."""
            if not rows:
                return []
            recs = []
            for r in rows:
                try:
                    recs.append(r.as_dict())
                except Exception:
                    try:
                        recs.append(r.asDict())
                    except Exception:
                        try:
                            recs.append(dict(r))
                        except Exception:
                            # last resort: convert via JSON roundtrip if possible
                            try:
                                recs.append(safe_json_loads(str(r)))
                            except Exception:
                                recs.append({"_repr": str(r)})
            return recs

        @st.cache_data(show_spinner="Loading semantic schemas…", ttl=600)
        def parse_describe_records(records):
            """Parse DESCRIBE-like records (list[dict]) into schema metadata.
            This function accepts only serializable input (list of dicts) so Streamlit can cache it safely.
            """
            out = {"ok": False, "fields": [], "types": {}, "raw": None}
            try:
                import pandas as _pd
            except Exception:
                _pd = None
            if not records:
                return out
            # Prefer pandas DataFrame for convenience if available
            try:
                if _pd:
                    df = _pd.DataFrame.from_records(records)
                else:
                    df = records
            except Exception:
                df = records
            # Keep only FACT/DIMENSION rows so planner never sees tables/extensions
            if "object_kind" in df.columns:
                df = df[df["object_kind"].astype(str).str.upper().isin(["FACT","DIMENSION"])].copy()

            # Normalize column names if pandas DF
            try:
                if _pd and isinstance(df, _pd.DataFrame):
                    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
                    out["raw"] = df.head(200)
                    # identify object/name column
                    obj_col = None
                    for cand in ("object_name","objectname","name","object","column_name","column"):
                        for c in df.columns:
                            if c.lower() == cand:
                                obj_col = c; break
                        if obj_col: break
                    if not obj_col:
                        obj_col = df.columns[0] if len(df.columns) else None
                    try:
                        fields = df[obj_col].dropna().astype(str).unique().tolist() if obj_col else []
                    except Exception:
                        fields = []
                else:
                    # records is list of dicts
                    out["raw"] = records[:200]
                    obj_col = None
                    if len(records) > 0:
                        keys = list(records[0].keys())
                        for cand in ("object_name","objectname","name","object","column_name","column"):
                            for k in keys:
                                if k.lower() == cand:
                                    obj_col = k; break
                            if obj_col: break
                        if not obj_col:
                            obj_col = keys[0] if keys else None
                    fields = []
                    for r in records:
                        v = r.get(obj_col) if isinstance(r, dict) else None
                        if v: fields.append(str(v))
                    # dedupe preserving order
                    seen = set(); fields = [x for x in fields if not (x in seen or seen.add(x))]
                types = {}
                for f in fields:
                    up = f.upper()

                    # Tokenize on non-alphanumerics; 'MANDATEID' should *not* yield a DATE token
                    _tokens = re.split(r'[^A-Z0-9]+', up)
                    def _has_date_token(u: str) -> bool:
                        toks = re.split(r'[^A-Z0-9]+', u)
                        # True only if 'DATE' is a standalone token or a canonical suffix like _DATE
                        return ("DATE" in toks) or u.endswith("_DATE") or u.endswith("_DT")

                    # Prefer schema-first typing if available; only then fall back to names
                    if _has_date_token(up) and not re.search(r'(^|_)[A-Z0-9]*ID($|_)', up):
                        types[f] = "DATE"
                    elif any(k in up for k in ("AUM","AMOUNT","TOTAL","REVEN","PROFIT","TOPUP","COUNT","SCORE","PERFORMANCE")):
                        types[f] = "NUMBER"
                    elif any(k in up for k in ("ID","RMID","MANDATEID","UID","CODE")):
                        types[f] = "STRING"
                    else:
                        types[f] = ""

                out.update({"ok": bool(fields), "fields": fields, "types": types})
            except Exception as e:
                out = {"ok": False, "fields": [], "types": {}, "raw": records, "error": str(e)}
            return out

        # --- end describe -> records helper ---

        @st.cache_data(show_spinner="Loading semantic schemas…", ttl=600)
        def parse_describe_df(df_raw: pd.DataFrame) -> Dict[str, Any]:
            out = {"ok": False, "fields": [], "types": {}, "raw": None}
            if df_raw is None or df_raw.empty: return out
            df = df_raw.copy(); df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
            out["raw"] = df.head(200)
            # Keep only semantic fields, not tables or extensions
            if "object_kind" in df.columns:
                df = df[df["object_kind"].astype(str).str.upper().isin(["FACT","DIMENSION"])].copy()

            obj_col = None
            for cand in ("object_name","objectname","name","object","column_name"):
                for c in df.columns:
                    if c.lower() == cand: obj_col = c; break
                if obj_col: break
            if not obj_col: obj_col = df.columns[0]
            try: fields = df[obj_col].dropna().astype(str).unique().tolist()
            except Exception: fields = []
            types = {}
            for f in fields:
                up = f.upper()
                if "DATE" in up or up.endswith("_DATE"): types[f] = "DATE"
                elif any(k in up for k in ("AUM","AMOUNT","TOTAL","SUM","PRICE","REVEN","PROFIT","TOPUP","COUNT","SCORE")): types[f] = "NUMBER"
                elif any(k in up for k in ("ID","RMID","MANDATEID","UID","CODE")): types[f] = "STRING"
                else: types[f] = ""
            return {"ok": bool(fields), "fields": fields, "types": types, "raw": df}
        
        @st.cache_data(show_spinner="Building catalog…", ttl=600)
        def load_catalog(views: List[str]) -> Dict[str, Dict[str, Any]]:
            catalog = {}
            for v in views:
                tried = [v]
                try:
                    try:
                        rows = _describe_semantic_rows(session, v); records = describe_rows_to_records(rows); parsed = parse_describe_records(records)
                    except Exception:
                        qv = '"' + v.replace('.', '"."') + '"'; tried.append(qv)
                        rows = _describe_semantic_rows(session, qv); records = describe_rows_to_records(rows); parsed = parse_describe_records(records)
                    parsed = parse_describe_records(records)
                    catalog[v] = {"describe_ok": parsed["ok"], "fields": parsed["fields"], "types": parsed["types"], "raw": parsed["raw"], "tried": tried}
                    trace("catalog", {v: {"fields": len(parsed["fields"]), "ok": parsed["ok"]}})
                except Exception as e:
                    trace("catalog_error", {v: str(e)})
                    catalog[v] = {"describe_ok": False, "fields": [], "types": {}, "raw": None, "tried": tried}
            return catalog
        
        def _normalize_token(s: str) -> str: return re.sub(r'[^a-z0-9]', '', s.lower())
        
        @st.cache_data(ttl=600)
        def build_schema_synonyms(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
            per_view: Dict[str, Dict[str, str]] = {}
            patterns = {
                "current_aum": ["current_aum", "snap_current_aum", "aum_current", "aum"],
                "performance": ["performance", "rm_performance", "rmperformance", "rm_perf", "rm_score", "rmscore", "performance_score"],
                "revenue": ["revenue","current_revenue","rev_amount","fee","fees"],
                "profit": ["profit","net_profit","margin"],
                "topup": ["topup","topup_amount","inflow","contribution","deposit"],
                "meetings": ["meeting_count","meetings","num_meetings","meeting"],
                "sentiment": ["sentiment","sentiment_score","tone","score"],
                "transactions": ["transaction_count","transactions","txn","txns","count"],
                "rmid": ["rmid","rm_id","relationship_manager_id"],
                "rm_name": ["rm_name","relationship_manager","rm"],
                "month_date": ["month_date","monthdate"],
                "meeting_date": ["meeting_date","meetingdate"],
                "any_date": ["month_date","meeting_date","date","*_date"]
            }
            for v, meta in catalog.items():
                fields = meta.get("fields", [])
                norm_map = {f: _normalize_token(f) for f in fields}
                m: Dict[str, str] = {}
                for target, pats in patterns.items():
                    best = None
                    for f, nf in norm_map.items():
                        if any(_normalize_token(p) == nf for p in pats): best = f; break
                    if not best:
                        for f, nf in norm_map.items():
                            if any(_normalize_token(p) in nf for p in pats): best = f; break
                    if not best and target in ("any_date","month_date","meeting_date"):
                        types = (meta or {}).get("types", {}) or {}
                        # prefer true date/timestamp typed columns
                        for f in fields:
                            t = str(types.get(f, "")).upper()
                            if t in {"DATE","DATETIME","TIMESTAMP","TIMESTAMP_NTZ","TIMESTAMP_LTZ","TIMESTAMP_TZ"}:
                                best = f
                                break
                        # otherwise allow only clean suffix/word matches (avoid picking MAN**DATE**)
                        if not best:
                            for f in fields:
                                U = f.upper()
                                if U.endswith("_DATE") or U in {"DATE","MONTH_DATE","MEETING_DATE","AS_OF_DATE"}:
                                    best = f
                                    break


                    m[target] = best or ""
                per_view[v] = m
            return per_view
        
        # ──────────────────────────────────────────────────────────────────────────────
        # Planner & Analyst
        PLANNER_PROMPT = """
        You are a **data planner** LLM.
        INPUTS:
        - User question
        - Compact schema JSON for available semantic views (fields & types)
        - A View Feature Map (families like: aum, finance, meetings, sentiment, mandates)
        - Time Hints (free-form; may include explicit months/years or relative ranges)

        OUTPUT:
        Return **ONLY** valid JSON (no prose, no backticks):

        {
          "steps": [
            {
              "id": "s1",
              "view": "<semantic_view_name>",
              "metric": {"field": "<field>", "mode": "sum|avg|median|percentile|last|count|snapshot", "p": "<0-100|nullable>"},
              "dim": {"field": "<field>|null"},
              "time": {"field": "<field>|null", "grain": "month|year|null"},
              "select_also": ["<schema_field_1>", "<schema_field_2>", "..."],
              "filters": [{"field":"<field>","op":">=", "..."],
              "order": "desc",
              "limit": 5000
            }
          ],
          "visualization": {
            "chart":"line|bar|scatter|table|auto",
            "x":"<col>|null",
            "y":"<col>|null",
            "agg":"sum|avg|median|last|count",
            "top_n": null,
            "allowed_chart_types":["line","bar","scatter","table"],
            "confidence": 0.0,
            "rationale": ""
          },
          "answer_from": "s1"
        }

        PLANNING RULES & HINTS (IMPORTANT):
        - Use **ONLY** fields that exist in the schema JSON for the chosen view. Do not invent fields.
        - Use the **View Feature Map** to choose relevant view(s) by family.
        - If the question spans multiple families (e.g., AUM with meetings/sentiment, or mandates with revenue),
          CREATE MULTIPLE STEPS (≤ 4), one per relevant view, and ALIGN THEM FOR MERGING.

        - **Split explicit time targets when present:**
          • When the question contains multiple explicit months/years (e.g., "Aug vs Jul 2025"), create **separate steps** (one per target)
            with precise filters that pin each step to exactly **one month or one year**.
          • Otherwise infer the timeframe from the question/Time Hints and set a sensible default (e.g., last 12 months).

        - **Ranking policy (only when the user asks for top/bottom/rank/best/worst for RMs):**
          • Rank strictly by **PERFORMANCE**.
          • If the timeframe covers **more than one month** (multiple months or a full year), set `metric.mode = "avg"` for PERFORMANCE.
            **Never sum PERFORMANCE.**
          • If the timeframe is a **single month**, use `metric.mode = "last"` or `"snapshot"` for PERFORMANCE.
          • Set `dim.field = RMID` if present; else `RM_NAME`.
          • Set `order = "desc"` for top/best and `order = "asc"` for bottom/worst.
          • Set `limit` to the requested N (default **5** if not specified).

          • (Optional context for a holistic answer) In addition to the PERFORMANCE step, you **may** add parallel step(s) for the **same timeframe**
            to fetch **AUM** (stock metric) and **COMMITMENTS/REVENUE** (flow metric) so the summarizer can enrich the narrative.
            – AUM is a **stock/snapshot** metric: it must **not** be summed. For a single month use `"snapshot"`.
              For a multi-month window or a year, by **default** return the **latest month in the window** (end-of-window snapshot).
              Only use `"avg"` for AUM **if the user explicitly requests an average** of AUM.
            – COMMITMENTS/REVENUE-like flows are **additive**: use `"sum"` over the requested timeframe.
            – Ensure these context steps align on the same `dim.field` and time filters so they can be merged on RMID/RM_NAME and month/year.
            – If step budget is tight (≤ 4), prioritize PERFORMANCE first; include AUM and then COMMITMENTS if capacity allows.

        - **General metric conventions (apply to non-ranking questions too):**
          • **PERFORMANCE**: average across multi-month periods/years (`"avg"`); single month → `"last"`/`"snapshot"`. Do **not** sum.
          • **AUM (stock)**: never sum. Single month → `"snapshot"`. Multi-month or year → **latest month only** by default (end-of-window snapshot).
            Only use `"avg"` for AUM if the user explicitly asks for an average; otherwise prefer the snapshot rule above.
          • **COMMITMENTS/REVENUE (flows)**: use `"sum"` across the requested timeframe.
          • “now/current/today” → prefer point-in-time metrics (`"snapshot"`).

        - **Joinability:** In each step, include **common join keys** inside `select_also` so downstream merging is possible.
          Prefer one or more of: RMID (or RM_NAME), MANDATEID, and a single **raw time column** when relevant (e.g., a date/month field from schema).
          • Use **actual schema column names** in `select_also` (not aliases).
          • If time bucketing is needed, set `time.grain` and the Analyst will alias (e.g., DATE_TRUNC('month', ...) AS MONTH_DATE).

        - **Time analysis:**
          • Set `time.field` (a schema date/month column) and `time.grain` ("month" or "year") when the question implies it
            or the user specifies a relative timeframe or explicit year(s).
          • When explicit **years** are present, include a YEAR/MEET_YEAR column (from schema) in `select_also` to enable comparisons.

        - **Step count:** keep ≤ 4 steps; choose the most relevant views.

        - **Visualization:**
          • Prefer RM_NAME over RMID as X axis if available; if you recommend RMID, lower `confidence` and explain in `rationale` that RM_NAME
            would be more readable if present.
          • ALWAYS return `visualization.allowed_chart_types` so the UI can restrict chart choices.
          • If you cannot recommend a visualization, set x=null, y=null, chart="table",
            allowed_chart_types:["table"], and provide a short rationale.

        - **Limits:** keep `limit` moderate (e.g., 500-5000) unless the question requires more to answer correctly.
         - Only apply DATE functions (DATEADD/DATE_TRUNC/TO_DATE) to columns typed as DATE/TIMESTAMP or the view’s {{any_date}} alias. Never to counts/IDs/amounts.
         - If the question says “P90 / 90th percentile / top decile / quantile 0.90”, use mode=percentile and p=90 (never median).


        Notes on `select_also`:
        - Include join-friendly keys (RMID / RM_NAME / MANDATEID) and any explicit YEAR/MEET_YEAR or **raw** time column
          needed for bucketing/comparisons.
        - **Do not** place derived alias names (e.g., MONTH_DATE) into `select_also` unless they already exist as **schema fields**.
          The Analyst will create aliases from the `time` directive when needed.

        Question:
        {question}

        View Feature Map:
        {feature_map_json}

        Time Hints:
        {time_hints_json}

        Schema:
        {schema_json}
        """

        
        def _compact_schema(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
            out = {}
            for v, meta in catalog.items():
                fields = meta.get("fields", [])[:300]
                types = meta.get("types", {})
                out[v] = {
                    "fields": fields,
                    "numeric": [f for f in fields if "NUMBER" in types.get(f, "") or any(k in f.upper() for k in ("AUM","AMOUNT","REVEN","PROFIT","TOPUP","COUNT","SCORE"))],
                    "dates": [f for f in fields if types.get(f, "").upper() in ("DATE","DATETIME","TIMESTAMP","TIMESTAMP_NTZ","TIMESTAMP_LTZ","TIMESTAMP_TZ")]

                }
            return out

        def _view_feature_map(catalog: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
            fam_keywords = {
                "aum": r"AUM|CURRENT_AUM|TOTAL_AUM",
                "finance": r"REVEN|REVENUE|PROFIT|AMOUNT|TOPUP",
                "meetings": r"MEET|MEETING|MEET_YEAR|MEET_MONTH",
                "sentiment": r"SENTIMENT|EMOTION|SCORE",
                "mandates": r"MANDATE|PIPE"
            }
            out = {}
            for v, meta in catalog.items():
                fields = meta.get("fields", []) or []
                tags = set()
                for fam, pat in fam_keywords.items():
                    if any(re.search(pat, f, re.I) for f in fields):
                        tags.add(fam)
                keys = [f for f in fields if re.search(r"^(RMID|RM_NAME|MANDATEID|MONTH_DATE|MONTH|MEET_YEAR)$", f, re.I)]
                out[v] = {"tags": sorted(tags), "keys": keys[:6]}
            return out

        
        def planner_llm(question: str, catalog: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            """
            LLM-first planner that can split explicit time references (e.g., "Aug vs Jul 2025")
            into SEPARATE STEPS. It sends the question, compact schema, feature map, time hints,
            and explicit directives (time_targets, top_n, dimension hints) to the LLM.
            """
            import json, re

            # Helpers from your codebase:
            # - _compact_schema(catalog)
            # - _view_feature_map(catalog)
            # - _extract_time_hints(question)
            # - trace(...), session, CORTEX_LLM_MODEL
            schema_small = _compact_schema(catalog)
            feature_map = _view_feature_map(catalog)
            time_hints = _extract_time_hints(question)

            # Derive explicit month targets like ["2025-08-01","2025-07-01"]
            time_targets = []
            for it in (time_hints.get("months_exact") or []):
                y = it.get("year"); m = it.get("month")
                if isinstance(y, int) and isinstance(m, int):
                    time_targets.append(f"{y:04d}-{m:02d}-01")

            # Parse top/bottom N
            top_n = None
            sort = "desc"
            try:
                m_top = re.search(r"\btop\s+(\d{1,3})\b", question or "", re.I)
                m_bot = re.search(r"\bbottom\s+(\d{1,3})\b", question or "", re.I)
                if m_top:
                    top_n = int(m_top.group(1)); sort = "desc"
                elif m_bot:
                    top_n = int(m_bot.group(1)); sort = "asc"
            except Exception:
                pass

            # Prefer a configurable dimension hint across common views
            dim_hint = _analysis_dimension_hint("RMID")
            # (we won't scan per-view here; the LLM has schema details to validate)
            join_keys_pref = _analysis_list(
                "join_keys_preference",
                ["RMID", "RM_NAME", "MANDATEID", "MONTH_DATE", "MEET_YEAR", "YEAR"],
            )

            # Compose prompt using your PLANNER_PROMPT plus explicit directives
            # (avoid f-strings for any sections that contain literal braces)
            try:
                prompt = (
                    PLANNER_PROMPT
                    .replace("{question}", str(question))
                    .replace("{schema_json}", json.dumps(schema_small, indent=2, default=_json_default))
                    .replace("{feature_map_json}", json.dumps(feature_map, indent=2, default=_json_default))
                    .replace("{time_hints_json}", json.dumps(time_hints, indent=2, default=_json_default))

                )
            except Exception:
                # Very defensive fallback
                prompt = str(PLANNER_PROMPT) + "\n\n" + str(question)

            directives = {
                "split_time_into_separate_steps": True,
                "time_targets": time_targets,   # if empty, LLM may still infer from Time Hints
                "top_n_hint": top_n,
                "sort_hint": sort,              # "desc" for top, "asc" for bottom
                "dimension_hint": dim_hint,     # prefer RMID (or RM_NAME)
                "join_keys_preferred": join_keys_pref,
                "rules": [
                    "Create one step PER EXPLICIT MONTH/YEAR target when multiple are present (e.g., 'Aug vs Jul 2025').",
                    "Each step must include filters that pin it to exactly one time target.",
                    "Always include join-friendly keys via select_also so the executor can merge (RMID/RM_NAME, MANDATEID, and a raw time column).",
                    "If user asked 'top N' or 'bottom N', set order and limit accordingly.",
                ]
            }
            prompt = (
                prompt
                + "\n\nAdditional Directives (JSON):\n"
                + json.dumps(directives, indent=2)
            )

            try:
                trace("planner_prompt", prompt[:MAX_TRACE_PAYLOAD])
            except Exception:
                pass

            # Call LLM (Cortex)
            try:
                q = "select snowflake.cortex.complete('{model}', $${body}$$) as response".format(
                    model=CORTEX_LLM_MODEL, body=prompt
                )
                df = session.sql(q).to_pandas()
                if df is None or df.empty:
                    return None
                resp_text = df.iloc[0, 0]
            except Exception as e:
                try:
                    trace("planner_exception", str(e)[:MAX_TRACE_PAYLOAD])
                except Exception:
                    pass
                return None

            # Parse JSON robustly
            plan = None
            try:
                if 'safe_json_loads' in globals():
                    plan = safe_json_loads(resp_text) if isinstance(resp_text, str) else None
                else:
                    plan = json.loads(resp_text) if isinstance(resp_text, str) else None
            except Exception:
                plan = json_extract(resp_text) if isinstance(resp_text, str) else None

            # Minimal validation: ensure view fields exist
            if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
                return None

            cleaned_steps = []
            for i, s in enumerate(plan["steps"], start=1):
                try:
                    view = (s or {}).get("view")
                    if not view or view not in catalog:
                        continue
                    fields = set(catalog[view].get("fields") or [])
                    m = (s.get("metric") or {})
                    d = (s.get("dim") or {})
                    t = (s.get("time") or {})
                    ok_metric = m.get("field") in fields
                    ok_dim = (d.get("field") in fields) if d.get("field") else True
                    ok_time = (t.get("field") in fields) if t.get("field") else True
                    if not ok_metric or not ok_dim or not ok_time:
                        continue

                    step = dict(s)
                    step["id"] = step.get("id") or f"s{i}"
                    # Keep only schema-valid select_also
                    if "select_also" in step and isinstance(step["select_also"], list):
                        step["select_also"] = [c for c in step["select_also"] if c in fields]
                    cleaned_steps.append(step)
                except Exception:
                    continue

            if not cleaned_steps:
                return None

            return {
                "steps": cleaned_steps[:4],  # safety cap
                "visualization": plan.get("visualization"),
                "answer_from": plan.get("answer_from", cleaned_steps[0]["id"])
            }

        
        def _resolve_metric_key(question: str) -> str:
            ql = question.lower()
            for key, arr in BASE_SYNONYMS.items():
                if any(w in ql for w in arr): return key
            return "aum"

        def resolve_time_field(
            step: dict,
            schema_synonyms: dict,
            catalog: dict,
            fields_upper: dict,
            prefer: tuple = ("month_date", "meeting_date", "any_date"),
        ) -> str | None:
            """
            Returns a verified DATE/TIMESTAMP column name for this step's view,
            or None if no safe time field exists.
            """
            import re

            this_view = (step or {}).get("view")
            if not this_view:
                return None

            view_syn = (schema_synonyms or {}).get(this_view, {}) or {}
            # 1) Preferred aliases from synonyms, in order
            candidates = [view_syn.get(k) for k in prefer if view_syn.get(k)]

            # 2) Fallback to typed columns from catalog (DATE/TIMESTAMP types)
            view_meta = (catalog or {}).get(this_view, {}) or {}
            types_map = {str(k).upper(): str(v).upper() for k, v in (view_meta.get("types") or {}).items()}
            for phys, typ in types_map.items():
                if typ in {"DATE", "DATETIME", "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ"}:
                    # Only add if present in this step's fields mapping
                    candidates.append(phys)

            # 3) Very last fallback: name heuristics (but only if type says it's a date)
            name_pref = [r"\bMONTH_DATE\b", r"\bMEETING_DATE\b", r"\b.*_DATE\b"]
            for rx in name_pref:
                for phys, typ in types_map.items():
                    if typ.startswith("TIMESTAMP") or typ == "DATE":
                        if re.search(rx, phys, re.I):
                            candidates.append(phys)

            # Deduplicate while preserving order
            seen = set()
            ordered = []
            for c in candidates:
                u = str(c).upper()
                if u and u not in seen:
                    seen.add(u)
                    ordered.append(c)

            # Validate against fields_upper and type map
            def _is_date_col(name: str) -> bool:
                phys = (fields_upper.get(str(name).upper(), "") if 'fields_upper' in locals() else _phys_name(name)) or str(name)
                up = phys.upper()
                # Never classify anything that looks like an ID as a date
                if up.endswith("ID") or up.endswith("_ID") or re.search(r'(^|_)[A-Z0-9]*ID($|_)', up):
                    return False
                t = types_map.get(up, "") if 'types_map' in locals() else types_map.get(phys.upper(), "")
                if t in {"DATE","DATETIME","TIMESTAMP","TIMESTAMP_NTZ","TIMESTAMP_LTZ","TIMESTAMP_TZ"}:
                    return True
                # As a *last resort*, allow name tokens but only strict tokens like '_DATE' or 'DATE' word
                toks = re.split(r'[^A-Z0-9]+', up)
                return ("DATE" in toks) or up.endswith("_DATE") or up.endswith("_DT")


            for cand in ordered:
                if _is_date_col(cand):
                    # Return the physical name that will be emitted in SQL
                    return fields_upper.get(str(cand).upper())

            return None

        
        def heuristic_plan(question: str, catalog: Dict[str, Any]) -> Dict[str, Any]:
            syn = build_schema_synonyms(catalog)

            key = _resolve_metric_key(question)
            best_view, best_metric = None, None

            # If user didn't name a metric, prefer PERFORMANCE when the schema provides it,
            # especially for RM-oriented questions (Top/best RMs).
            if key is None:
                ql = (question or "").lower()
                rm_intent = ("rm" in ql) or ("relationship manager" in ql) or ("relationship managers" in ql)
                rank_intent = any(w in ql for w in ("top", "best", "rank", "leader", "highest"))
                # Try to find a view that has a performance-like column and RM keys
                for v in catalog.keys():
                    mv = syn.get(v, {})
                    if mv.get("performance") and (mv.get("rmid") or mv.get("rm_name")):
                        best_view, best_metric = v, mv["performance"]
                        key = "performance"
                        break

            # If a metric keyword *was* found (or the above block set key), map it to a field via synonyms.
            if key:
                # include performance in lookup map
                lookup = {
                    "aum":"current_aum",
                    "performance":"performance",
                    "revenue":"revenue",
                    "profit":"profit",
                    "topup":"topup",
                    "meetings":"meetings",
                    "sentiment":"sentiment",
                    "transactions":"transactions"
                }.get(key)

                if not best_view:
                    for v in catalog.keys():
                        mv = syn.get(v, {})
                        if lookup and mv.get(lookup):
                            best_view, best_metric = v, mv[lookup]; break

            # Fallbacks if still nothing
            if not best_view:
                for v, meta in catalog.items():
                    nums = [f for f in meta.get("fields", []) if re.search(r"AUM|AMOUNT|REVEN|PROFIT|COUNT|MEETING|SCORE|TOPUP|PERF", f, re.I)]
                    if nums: best_view, best_metric = v, nums[0]; break
                # if we fell back to a numeric without a key, make it performance if it looks like PERF
                if best_metric and re.search(r"PERF|PERFORMANCE", best_metric, re.I):
                    key = "performance"

            # Determine mode from DEFAULT_MODE; default to sum, except snapshot for AUM "now"
            mode = DEFAULT_MODE.get(key or "", "sum")
            if key == "aum" and "now" in (question or "").lower():
                mode = "snapshot"
            if key == "performance":
                mode = "avg"

            rm_id = syn.get(best_view, {}).get("rmid") or None
            rm_name = syn.get(best_view, {}).get("rm_name") or None
            dim_field = rm_id or rm_name
            time_field = syn.get(best_view, {}).get("month_date") or syn.get(best_view, {}).get("meeting_date") or syn.get(best_view, {}).get("any_date") or None
            # --- determine order & limit from question intent (planner responsibility) ---
            order = "desc"
            limit = 500
            if isinstance(question, str):
                m_bottom = re.search(r'\bbottom\s+(\d{1,3})\b', question, re.I)
                m_top = re.search(r'\btop\s+(\d{1,3})\b', question, re.I)
                if m_bottom:
                    order = "asc"
                    try: limit = int(m_bottom.group(1))
                    except Exception: pass
                elif m_top:
                    order = "desc"
                    try: limit = int(m_top.group(1))
                    except Exception: pass
                else:
                    # fallback check for words bottom/lowest/top/highest (without explicit number)
                    if re.search(r'\bbottom\b|\blowest\b|\bsmallest\b', question, re.I):
                        order = "asc"
                    elif re.search(r'\btop\b|\bhighest\b|\blargest\b', question, re.I):
                        order = "desc"

            step = {
                "id": "s1",
                "view": best_view,
                "metric": {"field": best_metric, "mode": mode},
                "dim": {"field": dim_field} if dim_field else None,
                "time": {"field": time_field, "grain": "month"} if time_field else None,
                "order": order,
                "limit": limit
            }
     

            # Visualization hint stays as before
            return {
                "steps":[step],
                "visualization":{"chart":"bar","x": dim_field or (time_field or ""), "y": best_metric},
                "answer_from":"s1"
            }
        
        
        def _parse_analyst_payload(resp_content: str) -> Dict[str, Any]:
            out = {"sql": "", "suggestions": [], "explanation": "", "warnings": [], "raw": None}
            try:
                payload = safe_json_loads(resp_content or "{}"); out["raw"] = payload
                # If payload is a raw string, try to extract a top-level "sql" value
                if isinstance(payload, str):
                    m = _re.search(r'"sql"\s*:\s*"([^"]+)"', payload)
                    if m:
                        out["sql"] = m.group(1)
                        return out
                    # fallthrough: try to find a balanced JSON inside
                    candidate = _find_balanced_json(payload)
                    if candidate:
                        try:
                            payload = safe_json_loads(candidate)
                            out["raw"] = payload
                        except Exception:
                            pass
                # If top-level dict with 'sql' key (simple analyst responses)
                if isinstance(payload, dict):
                    if payload.get("sql"):
                        out["sql"] = payload.get("sql")
                    # Also accept 'statement' or 'query' top-level keys
                    if not out["sql"] and payload.get("statement"):
                        out["sql"] = payload.get("statement")
                    if not out["sql"] and payload.get("query"):
                        out["sql"] = payload.get("query")
                    # Now handle the 'message' wrapped format used by the fuller analyst responses
                    msg = payload.get("message") or {}
                    for block in (msg.get("content") or []):
                        t = (block.get("type") or "").lower()
                        if t == "sql":
                            stmt = block.get("statement") or block.get("sql") or ""
                            if stmt:
                                out["sql"] = stmt
                        elif t == "text":
                            tx = block.get("text") or block.get("content") or ""
                            if tx:
                                # append with a newline if explanation already present
                                if out["explanation"]:
                                    out["explanation"] += "\n" + tx
                                else:
                                    out["explanation"] = tx
                        elif t == "suggestions":
                            out["suggestions"].extend(block.get("suggestions") or [])
                # Collect warnings if present
                if isinstance(payload, dict):
                    for w in (payload.get("warnings") or []):
                        m = (w.get("message") if isinstance(w, dict) else None)
                        if m: out["warnings"].append(m)
            except Exception as e:
                out["explanation"] = f"parse failed: {e}"
            return out
        
        
        def call_analyst_rest(
            user_text: str,
            view: Optional[Union[str, List[str]]] = None,
            timeout_seconds: int = 60,
        ) -> dict:
            """
            Call the Cortex Analyst REST endpoint with PAT auth and return a structured dict:
              { ok: bool, status_code: int|None, json: dict|None, text: str|None, error: str|None }
        
            Requirements implemented here:
            - Authorization: Bearer <ANALYST_REST_TOKEN> (taken from env; never hardcoded)
            - X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN (default; override via ANALYST_REST_TOKEN_TYPE)
            - Accept: application/json
            - Content-Type: application/json
            - Exactly one of semantic_view OR semantic_model OR semantic_model_file must be set; we honor `view` param.
            - Rich diagnostics for JSON parse failures and common 4xx messages (e.g., network policy required).
            """
            url = os.environ.get(
                "ANALYST_REST_URL",
                "https://OBEIKAN-O3AI.snowflakecomputing.com/api/v2/cortex/analyst/message"
            )
            if not url:
                acct = os.environ.get("SNOWFLAKE_ACCOUNT", "").strip()
                if acct:
                    if acct.endswith(".snowflakecomputing.com"):
                        url = f"https://{acct}/api/v2/cortex/analyst/message"
                    else:
                        url = f"https://{acct}.snowflakecomputing.com/api/v2/cortex/analyst/message"
                else:
                    url = "https://<account>.snowflakecomputing.com/api/v2/cortex/analyst/message"

            token = os.environ.get("ANALYST_REST_TOKEN", "").strip()
            token_type_tag = os.environ.get("ANALYST_REST_TOKEN_TYPE", "PROGRAMMATIC_ACCESS_TOKEN")

            if not token:
                try:
                    trace("analyst_rest_error", "Missing ANALYST_REST_TOKEN env var")
                except Exception:
                    pass
                return {"ok": False, "status_code": None, "json": None, "text": None, "error": "Missing ANALYST_REST_TOKEN. Set env var ANALYST_REST_TOKEN with a valid PAT."}

            headers = {
                "Authorization": f"Bearer {token}",
                "X-Snowflake-Authorization-Token-Type": token_type_tag,
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            # Build request body per Snowflake docs: messages[] with last role user and content[] containing text
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text}
                        ]
                    }
                ],
                # explicit non-streaming by default
                "stream": False
            }

            # Support: if view is a single semantic_view string -> use semantic_view
            # If view is a list of semantic view strings, use semantic_models array form
            if view:
                if isinstance(view, (list, tuple)):
                    models = []
                    for v in view:
                        kind = _semantic_kind_for_name(v)
                        key = "semantic_model" if kind == "model" else "semantic_view"
                        models.append({key: v})
                    body["semantic_models"] = models
                else:
                    kind = _semantic_kind_for_name(view)
                    if kind == "model":
                        body["semantic_model"] = view
                    else:
                        body["semantic_view"] = view

            # Add request_id client-side to help correlate traces/logs (optional)
            body_request_id = str(uuid.uuid4())
            body["client_request_id"] = body_request_id

            # Keep a compact trace preview (redact host optionally)
            try:
                trace("analyst_rest_request", {
                    "url": ("(redacted host)" if "://" in url and os.environ.get("REDACT_ANALYST_HOST") else url),
                    "client_request_id": body_request_id,
                    "has_view": bool(view),
                    "view": (view if isinstance(view, str) else (list(view) if isinstance(view, (list,tuple)) else None)),
                    "body_preview": (user_text[:200] + ("…" if len(user_text) > 200 else ""))
                })
            except Exception:
                pass

            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout_seconds)
            except requests.exceptions.RequestException as e:
                # Network-level failures
                try:
                    trace("analyst_rest_request_exception", str(e)[:MAX_TRACE_PAYLOAD])
                except Exception:
                    pass
                return {"ok": False, "status_code": None, "json": None, "text": None, "error": f"Request failed: {e}"}

            status = resp.status_code

            # Prefer text first (for diagnostics)
            try:
                text = resp.text
            except Exception:
                text = None

            # Attempt JSON parse
            parsed_json = None
            parse_error = None
            try:
                parsed_json = resp.json()
            except ValueError as e:
                parse_error = str(e)
                # Try to extract JSON-looking substring
                if text:
                    m = re.search(r"(\\{[\\s\\S]*\\})", text)
                    if m:
                        candidate = m.group(1)
                        try:
                            parsed_json = json.loads(candidate)
                            parse_error = None
                        except Exception as e2:
                            parse_error = f"{parse_error}; fallback-json-extract-failed: {e2}"

            # Trace debug info for non-OK or parse failures
            if (not resp.ok) or parse_error:
                try:
                    trace("analyst_rest_response_debug", {
                        "status": status,
                        "ok": resp.ok,
                        "client_request_id": body_request_id,
                        "resp_headers": dict(resp.headers),
                        "json_preview": (parsed_json if parsed_json is not None else (text[:MAX_TRACE_PAYLOAD] if text else None))
                    })
                except Exception:
                    pass

            # Build friendly error message if needed
            ok_flag = resp.ok and (parsed_json is not None)
            error_msg = None
            if not ok_flag:
                if parse_error:
                    error_msg = f"JSON parse error: {parse_error}"
                else:
                    # Map common 4xx bodies to clearer messages
                    body_msg = ""
                    try:
                        if isinstance(parsed_json, dict) and parsed_json.get("message"):
                            body_msg = parsed_json.get("message")
                        elif isinstance(parsed_json, dict) and parsed_json.get("error"):
                            body_msg = parsed_json.get("error")
                        elif text:
                            body_msg = text[:400]
                    except Exception:
                        body_msg = text[:400] if text else ""
                    if status == 401 and body_msg and "Network policy" in body_msg:
                        error_msg = "401 Unauthorized: Network policy is required or caller IP not allowed."
                    elif status == 400 and body_msg and "semantic view" in body_msg.lower():
                        error_msg = "400 Bad Request: Exactly one of semantic_view / semantic_model / semantic_model_file must be set."
                    else:
                        error_msg = f"HTTP {status}: {body_msg or 'Request failed'}"

            # If response OK but message missing, still return JSON but add note
            if resp.ok and parsed_json is not None and "message" not in parsed_json:
                try:
                    trace("analyst_rest_unexpected_body", {
                        "status": status,
                        "client_request_id": body_request_id,
                        "body_keys": list(parsed_json.keys())[:20]
                    })
                except Exception:
                    pass
                error_msg = error_msg or "Response OK but 'message' key missing in body"

            return {
                "ok": ok_flag,
                "status_code": status,
                "json": parsed_json,
                "text": text,
                "error": None if ok_flag else error_msg
            }



        def analyst_message(view: Optional[str], user_text: str, timeout: int = 60000) -> Dict[str, Any]:
            """Use Analyst REST API (preferred) and fallback to snowflake.cortex.complete if necessary."""
            # Check if we have a cached response for this prompt (optional, graceful fallback)
            cached_response = None
            try:
                from backend.ai_insights_orchestrator import _get_cached_llm_result, _cache_llm_result
                cached_response = _get_cached_llm_result(user_text)
            except ImportError:
                pass  # Cache import failed, proceed with fresh request
            
            if cached_response is not None:
                try:
                    # Parse cached response (it should already be parsed JSON or valid SQL)
                    parsed = _parse_analyst_payload(cached_response)
                    trace("analyst_request_cached", {"view": view or "NONE"})
                    return parsed
                except Exception:
                    pass  # Fall through to fresh request if cache parse fails
            
            # Build a prompt asking the model to return structured SQL or JSON with keys similar to previous API.
            body_prompt = (
                "Return ONLY a structured response that contains the SQL statement to run. "
                "If possible, return a JSON object with a 'sql' key, otherwise return the SQL text. "
                f"Task: {user_text}"
            )
            trace("analyst_request", {"view": view or "NONE", "prompt": user_text[:1000]})

            # Ensure payload is defined for the fallback branch
            payload = None
            allow_fallback = os.environ.get("ANALYST_ALLOW_FALLBACK", "0").strip() == "1"

            # First, try REST Analyst
            try:
                # timeout param in function is in ms in older code: convert to seconds for requests
                timeout_seconds = max(1, int(timeout / 1000)) if timeout and timeout > 0 else 60
                rest_resp = call_analyst_rest(user_text, view=view, timeout_seconds=timeout_seconds)

                if isinstance(rest_resp, dict) and rest_resp.get("error"):
                    # REST call failed; optionally fall back to SQL COMPLETE
                    trace("analyst_rest_error", rest_resp.get("error")[:MAX_TRACE_PAYLOAD])
                    if not allow_fallback:
                        return {
                            "ok": False,
                            "sql": "",
                            "suggestions": [],
                            "explanation": rest_resp.get("error") or "Analyst REST call failed.",
                            "warnings": [],
                            "raw": rest_resp,
                        }

                else:
                    # parse REST response which may already be structured or may contain message.content blocks
                    parsed = {"ok": False, "sql": "", "suggestions": [], "explanation": "", "warnings": [], "raw": rest_resp}
                    try:
                        # If REST returned a top-level 'message' with 'content' blocks, inspect them
                        payload = (rest_resp.get("json") if isinstance(rest_resp, dict) else None) or rest_resp
                        msg = payload.get("message") if isinstance(payload, dict) else None
                        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                            for block in msg.get("content", []):
                                btype = (block.get("type") or "").lower()
                                if btype in ("sql","semantic_sql"):
                                    parsed["sql"] = block.get("statement") or block.get("sql") or block.get("query") or ""
                                elif btype == "text" and not parsed.get("sql"):
                                    txt = (block.get("text") or "")
                                    fenced = _strip_code_fences(txt).strip()
                                    if isinstance(fenced, str) and fenced.lower().startswith("select"):
                                        parsed["sql"] = fenced
                                    else:
                                        parsed["explanation"] = (parsed.get("explanation") or "") + " " + txt
                                elif btype == "suggestions":
                                    parsed["suggestions"].extend(block.get("suggestions") or [])
                        else:
                            # Maybe REST returned a dict payload directly with 'sql' key or a string
                            parsed_candidate = _parse_analyst_payload(payload if isinstance(payload, (dict, str)) else rest_resp)
                            if isinstance(parsed_candidate, dict):
                                parsed.update(parsed_candidate)

                        # Root-level fallbacks
                        if not parsed.get("sql") and isinstance(payload, dict):
                            root_sql = payload.get("sql") or payload.get("semantic_sql") or payload.get("statement") or payload.get("query")
                            if isinstance(root_sql, str):
                                parsed["sql"] = _strip_code_fences(root_sql).strip()

                        parsed["ok"] = bool(parsed.get("sql") or parsed.get("suggestions"))
                        if parsed.get("warnings"):
                            trace("analyst_warnings", parsed["warnings"])

                        # Cache the result for future requests with same prompt
                        try:
                            _cache_llm_result(user_text, json.dumps(parsed, default=str))
                        except Exception:
                            pass
                        return parsed

                    except Exception as e:
                        trace("analyst_rest_parse_error", str(e)[:MAX_TRACE_PAYLOAD])

                # FALLBACK: use Snowflake COMPLETE via SQL (optional)
                if not allow_fallback:
                    parsed["ok"] = bool(parsed.get("suggestions"))
                    return parsed
                try:
                    q = f"select snowflake.cortex.complete('{CORTEX_LLM_MODEL}', $${body_prompt}$$) as response"
                    df = session.sql(q).to_pandas()
                    if df is None or df.empty:
                        trace("analyst_exception", "empty LLM response")
                        return {"ok": False, "sql": "", "suggestions": [], "explanation": "no response", "warnings": [], "raw": None}

                    resp = df.iloc[0,0]
                    trace("analyst_raw", str(resp)[:MAX_TRACE_PAYLOAD])

                    parsed = _parse_analyst_payload(resp)

                    # Root-level fallbacks (only if payload was set earlier)
                    if not parsed.get("sql") and isinstance(payload, dict):
                        root_sql = payload.get("sql") or payload.get("semantic_sql") or payload.get("statement") or payload.get("query")
                        if isinstance(root_sql, str):
                            parsed["sql"] = _strip_code_fences(root_sql).strip()

                    parsed["ok"] = bool(parsed.get("sql") or parsed.get("suggestions"))
                    if parsed.get("warnings"):
                        trace("analyst_warnings", parsed["warnings"])

                    # Cache the result for future requests with same prompt
                    try:
                        _cache_llm_result(user_text, json.dumps(parsed, default=str))
                    except Exception:
                        pass
                    # ✅ Always return parsed from fallback
                    return parsed

                except Exception as e:
                    trace("analyst_exception", str(e)[:MAX_TRACE_PAYLOAD])
                    return {"ok": False, "sql": "", "suggestions": [], "explanation": str(e), "warnings": [], "raw": None}

            except Exception as e:
                trace("analyst_exception", str(e)[:MAX_TRACE_PAYLOAD])
                return {"ok": False, "sql": "", "suggestions": [], "explanation": str(e), "warnings": [], "raw": None}
        def analyst_query_execute(sql_stmt: str) -> Tuple[bool, pd.DataFrame, Optional[str]]:
            try:
                # Check cache first to avoid redundant queries (optional, graceful fallback)
                cached_result = None
                try:
                    from backend.ai_insights_orchestrator import _get_cached_query_result, _cache_query_result
                    cached_result = _get_cached_query_result(sql_stmt)
                except ImportError:
                    # If import fails, just execute query normally (no caching)
                    pass
                
                if cached_result is not None:
                    return True, cached_result, None
                
                df = session.sql(sql_stmt).to_pandas()
                
                # Cache the result for future use (optional, graceful fallback)
                try:
                    from backend.ai_insights_orchestrator import _cache_query_result
                    _cache_query_result(sql_stmt, df)
                except ImportError:
                    pass  # Caching failed, but query executed successfully
                
                return True, df, None
            except Exception as e:
                return False, pd.DataFrame(), str(e)
        
        # RM lookup & enrichment
        def _find_rmid_col_any(df: pd.DataFrame) -> Optional[str]:
            if df is None or df.empty: return None
            for c in df.columns:
                u = c.upper()
                if u in ("RMID","RM_ID","RELATIONSHIP_MANAGER_ID"): return c
                if "RMID" in u or ("RELATIONSHIP_MANAGER" in u and "ID" in u): return c
            return None
        
        @st.cache_data(ttl=1200, show_spinner=False)
        def fetch_rm_lookup(catalog: Dict[str, Any]) -> Optional[pd.DataFrame]:
            rm_view = next((v for v in catalog if "RM_PERFORMANCE" in v.upper()), None)
            if not rm_view: return None
            fields = catalog.get(rm_view, {}).get("fields", [])
            if not any(f.upper() == "RMID" for f in fields) or not any(f.upper() == "RM_NAME" for f in fields): return None
            prompt = f"Return RMID and RM_NAME for all RMs. Limit {RM_LOOKUP_LIMIT}. Use only semantic model fields."
            parsed = analyst_message(rm_view, prompt)
            if not parsed.get("sql"): return None
            ok, dfmap, err = analyst_query_execute(parsed["sql"])
            if not ok or dfmap is None or dfmap.empty: return None
            dfmap = clean_cols(dfmap)
            rmid_col = _find_rmid_col_any(dfmap)
            rname_col = next((c for c in dfmap.columns if c.upper() in ("RM_NAME","RELATIONSHIP_MANAGER","NAME")), None)
            if not rmid_col or not rname_col: return None
            dfmap = dfmap[[rmid_col, rname_col]].drop_duplicates(subset=[rmid_col])
            dfmap.columns = ["RMID", "RM_NAME"]
            return dfmap
        
        def enrich_rm_names(frames: Dict[str, pd.DataFrame], rm_lookup: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
            if rm_lookup is None or rm_lookup.empty: return frames
            lu = rm_lookup.copy(); lu["RMID"] = lu["RMID"].astype(str)
            for k, df in list(frames.items()):
                if df is None or df.empty: continue
                rcol = _find_rmid_col_any(df)
                if not rcol: continue
                df = df.copy()
                try: df[rcol] = df[rcol].astype(str)
                except Exception: pass
                merged = df.merge(lu, left_on=rcol, right_on="RMID", how="left")
                if "RM_NAME_y" in merged.columns and "RM_NAME_x" in merged.columns:
                    merged["RM_NAME"] = merged["RM_NAME_x"].fillna(merged["RM_NAME_y"])
                    merged.drop(columns=[c for c in ("RM_NAME_x","RM_NAME_y") if c in merged.columns], inplace=True, errors="ignore")
                frames[k] = merged; trace("rm_enrich", {"frame": k, "rows": len(merged)})
            return frames

        def _normalize_rm_text(value: Any) -> str:
            import re
            try:
                text = re.sub(r"[^a-z0-9]+", " ", str(value).lower())
                return " ".join(text.split())
            except Exception:
                return ""

        def _fuzzy_rm_from_question(question_text: str, rm_lookup: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
            """
            Cheap local fuzzy match to map a free-text RM name in the question to an exact RM_NAME/RMID.
            No extra queries or LLM calls.
            """
            import difflib
            import re

            if not question_text or rm_lookup is None or rm_lookup.empty:
                return None

            q_lower = str(question_text).lower()
            # Skip fuzzy matching for broad/plural RM queries
            if re.search(r"\b(rms|all rms|top\s+\d+\s+rms|bottom\s+\d+\s+rms)\b", q_lower):
                return None

            # If an explicit RMID is present, do not override
            if re.search(r"\brm\s*\d{2,}\b", q_lower) or "rmid" in q_lower:
                return None

            names = (
                rm_lookup.get("RM_NAME")
                .dropna()
                .astype(str)
                .unique()
                .tolist()
                if "RM_NAME" in rm_lookup.columns
                else []
            )
            if not names:
                return None

            norm_names = {_normalize_rm_text(n): n for n in names if _normalize_rm_text(n)}
            name_norm_list = list(norm_names.keys())
            q_norm = _normalize_rm_text(question_text)
            if not q_norm:
                return None

            # 1) Token match: allow single-token queries like "bukhari"
            if len(q_norm) >= 4:
                token_hits: list[tuple[str, str]] = []
                for n_norm, n_raw in norm_names.items():
                    tokens = n_norm.split()
                    if q_norm in tokens:
                        token_hits.append((n_raw, n_norm))
                if token_hits:
                    if len(token_hits) == 1:
                        return {"rm_name": token_hits[0][0], "method": "token", "confidence": 0.92}
                    best_name = None
                    best_score = -1.0
                    best_len = 1_000_000
                    for n_raw, n_norm in token_hits:
                        score = difflib.SequenceMatcher(None, q_norm, n_norm).ratio()
                        if score > best_score or (score == best_score and len(n_norm) < best_len):
                            best_name = n_raw
                            best_score = score
                            best_len = len(n_norm)
                    if best_name:
                        return {"rm_name": best_name, "method": "token", "confidence": 0.9}

            # 1) Direct substring match on normalized names (highest confidence)
            best_name = None
            best_len = 0
            for n_norm, n_raw in norm_names.items():
                if len(n_norm) < 4:
                    continue
                if n_norm in q_norm and len(n_norm) > best_len:
                    best_name = n_raw
                    best_len = len(n_norm)
            if best_name:
                return {"rm_name": best_name, "method": "substring", "confidence": 1.0}

            # 2) If question contains "RM <name>" or "relationship manager <name>", fuzzy match that tail
            stop_tokens = {
                "do", "does", "did", "to", "for", "in", "on", "over", "within",
                "across", "during", "next", "increase", "improve", "boost",
                "his", "her", "their", "the", "this", "that", "score", "scores",
                "help", "can", "could", "should", "would", "will", "show", "give",
                "tell", "explain", "trend", "trends", "compare", "versus", "vs"
            }
            candidate = None
            try:
                m = re.search(r"\b(?:rm|relationship manager)[\s\-:]+([A-Za-z][A-Za-z\s'.-]{1,60})", question_text, re.I)
            except Exception:
                m = None
            if m:
                tail = m.group(1).strip()
                tokens = tail.split()
                picked: list[str] = []
                for tok in tokens:
                    t = tok.strip(" ,.-")
                    if not t:
                        continue
                    if t.lower() in stop_tokens:
                        break
                    picked.append(t)
                candidate = " ".join(picked).strip() if picked else None

            def _best_close_match(text_in: str, cutoff: float = 0.86) -> Optional[str]:
                norm_in = _normalize_rm_text(text_in)
                if not norm_in:
                    return None
                close = difflib.get_close_matches(norm_in, name_norm_list, n=1, cutoff=cutoff)
                if not close:
                    return None
                return norm_names.get(close[0])

            if candidate:
                best = _best_close_match(candidate, cutoff=0.86)
                if best:
                    return {"rm_name": best, "method": "keyword_fuzzy", "confidence": 0.9}

            # 3) Fallback: n-gram fuzzy match on the question
            q_tokens = q_norm.split()
            if len(q_tokens) >= 2:
                ngrams = []
                for size in (4, 3, 2):
                    for i in range(len(q_tokens) - size + 1):
                        ngrams.append(" ".join(q_tokens[i : i + size]))
                for phrase in ngrams:
                    best = _best_close_match(phrase, cutoff=0.9)
                    if best:
                        return {"rm_name": best, "method": "ngram_fuzzy", "confidence": 0.88}

            return None

        def _apply_rm_hint_to_plan(plan: Dict[str, Any], rm_hint: Optional[Dict[str, Any]], catalog: Dict[str, Any]) -> Dict[str, Any]:
            if not rm_hint or not isinstance(plan, dict) or not plan.get("steps"):
                return plan
            rm_name = rm_hint.get("rm_name")
            if not rm_name:
                return plan
            for step in plan.get("steps") or []:
                try:
                    view = step.get("view")
                    if not view or view not in catalog:
                        continue
                    fields = set(catalog[view].get("fields") or [])
                    existing = step.get("filters") or []
                    has_rmid_filter = any(
                        isinstance(f, dict) and str(f.get("field", "")).upper() in {"RMID", "RM_ID"}
                        for f in existing
                    )
                    if has_rmid_filter:
                        continue
                    rm_name_filters = [
                        f for f in existing
                        if isinstance(f, dict) and str(f.get("field", "")).upper() in {"RM_NAME", "RELATIONSHIPMANAGER", "RELATIONSHIP_MANAGER"}
                    ]
                    if rm_name_filters:
                        for f in rm_name_filters:
                            f["value"] = rm_name
                        continue
                    target_field = None
                    if "RM_NAME" in fields:
                        target_field = "RM_NAME"
                    elif "RELATIONSHIPMANAGER" in fields:
                        target_field = "RELATIONSHIPMANAGER"
                    elif "RELATIONSHIP_MANAGER" in fields:
                        target_field = "RELATIONSHIP_MANAGER"
                    if not target_field:
                        continue
                    step.setdefault("filters", []).append({"field": target_field, "op": "=", "value": rm_name})
                except Exception:
                    continue
            return plan


        def _rbac_gate_frames(
            frames: dict,
            role: str,
            scope_rmids: list[str],
            allowed_mandates: list[str],
            allowed_names: list[str] | None = None
        ) -> dict:
            """
            Final gate: for each frame, keep rows where ANY of these match scope:
              - any column containing 'RMID' matches scope_rmids
              - any column containing 'MANDATEID' matches allowed_mandates
              - column 'RM_NAME' matches allowed_names (if provided)
            If role == OTHER or no scope, returns frames unchanged.
            """
            import pandas as pd  # ensure local in case top scope differs
            if not role or role == "OTHER":
                return frames

            rset = set(map(str, scope_rmids or []))
            mset = set(map(str, allowed_mandates or []))
            nset = set(map(str, (allowed_names or [])))

            out: dict = {}
            for k, df in list(frames.items()):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    out[k] = df
                    continue

                d = df.copy()
                keep = None

                for c in d.columns:
                    cu = str(c).upper()
                    try:
                        col = d[c].astype(str)
                    except Exception:
                        continue

                    hit = None
                    if "RMID" in cu:
                        hit = col.isin(rset)
                    elif "MANDATEID" in cu or ("MANDATE" in cu and "ID" in cu):
                        hit = col.isin(mset)
                    elif cu == "RM_NAME" and nset:
                        hit = col.isin(nset)

                    if hit is not None:
                        keep = hit if keep is None else (keep | hit)

                if keep is not None:
                    d = d[keep]
                else:
                    # No RBAC-identifying columns in this frame.
                    # For non-OTHER roles, drop the frame entirely so we don't leak unscoped data
                    # and so post_gate_rows falls to 0 (triggering the No-access stop).
                    d = d.iloc[0:0]

                out[k] = d

            return out

        def _total_rows(frames: dict) -> int:
            """Count total rows across all pandas DataFrames in a frames dict."""
            
            total = 0
            for _k, _df in (frames or {}).items():
                if isinstance(_df, pd.DataFrame) and not _df.empty:
                    try:
                        total += int(len(_df))
                    except Exception:
                        pass
            return total



        def _find_mandateid_col_any(df: pd.DataFrame) -> Optional[str]:
            if df is None or df.empty: return None
            for c in df.columns:
                u = c.upper()
                if u in ("MANDATEID","MANDATE_ID"): return c
                if "MANDATE" in u and "ID" in u: return c
            return None

        @st.cache_data(ttl=900, show_spinner=False)
        def fetch_rm_to_mandate_map() -> Optional[pd.DataFrame]:
            """
            Loads latest RM↔MANDATE snapshot:
            select * from tfo.tfo_schema.rmtomandate
            where create_date = (select max(create_date) from tfo.tfo_schema.rmtomandate)
            """
            try:
                sql = """
                select *
                from tfo.tfo_schema.rmtomandate
                where create_date = (
                  select max(create_date) from tfo.tfo_schema.rmtomandate
                )
                """
                df = session.sql(sql).to_pandas()
            except Exception:
                return None
            df = clean_cols(df)

            # Canonicalize id column names so joins like on="RMID" / "MANDATEID" always work
            cols_upper = {c.upper(): c for c in df.columns}
            if "RM_ID" in cols_upper:
                df.rename(columns={cols_upper["RM_ID"]: "RMID"}, inplace=True)
            if "RELATIONSHIP_MANAGER_ID" in cols_upper:
                df.rename(columns={cols_upper["RELATIONSHIP_MANAGER_ID"]: "RMID"}, inplace=True)
            if "MANDATE_ID" in cols_upper:
                df.rename(columns={cols_upper["MANDATE_ID"]: "MANDATEID"}, inplace=True)

            # Ensure id types are strings
            for k in ("RMID", "MANDATEID"):
                if k in df.columns:
                    try:
                        df[k] = df[k].astype(str)
                    except Exception:
                        pass

            # Normalize id types when present
            for k in df.columns:
                if k.upper() in ("RMID","RM_ID","RELATIONSHIP_MANAGER_ID","MANDATEID","MANDATE_ID"):
                    try: df[k] = df[k].astype(str)
                    except Exception: pass
            return df

        def enrich_rm_mandate(frames: Dict[str, pd.DataFrame],
                              rmmand_map: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
            """
            Safe enrichment:
            - If a frame already has MANDATEID but lacks RMID → add RMID by joining on MANDATEID (no row explosion)
            - If a frame already has RMID but lacks MANDATEID → **skip** by default to avoid 1→N expansion
              (use run_metric_tool(join={"using":"RMID","ref":"rm_map"}) for explicit expansions when needed)
            """
            if rmmand_map is None or rmmand_map.empty: return frames

            lu = rmmand_map.copy()
            # Find lookup-side columns, case-insensitively
            l_rmid = None
            l_mand = None
            for c in lu.columns:
                u = c.upper()
                if u in ("RMID","RM_ID","RELATIONSHIP_MANAGER_ID"): l_rmid = l_rmid or c
                if u in ("MANDATEID","MANDATE_ID"): l_mand = l_mand or c
            if not l_mand:  # need at least mandate id to enrich safely
                return frames

            # Make ids comparable
            try:
                if l_mand in lu.columns: lu[l_mand] = lu[l_mand].astype(str)
                if l_rmid and l_rmid in lu.columns: lu[l_rmid] = lu[l_rmid].astype(str)
            except Exception:
                pass

            # We only enrich frames that already carry a mandate id (safe join)
            for k, df in list(frames.items()):
                if df is None or df.empty: continue
                mcol = _find_mandateid_col_any(df)
                rcol = _find_rmid_col_any(df)
                if mcol and not rcol:
                    d = df.copy()
                    try: d[mcol] = d[mcol].astype(str)
                    except Exception: pass
                    use_cols = [l_mand] + ([l_rmid] if l_rmid else [])
                    merged = d.merge(
                        lu[use_cols].drop_duplicates(subset=[l_mand]),
                        left_on=mcol, right_on=l_mand, how="left"
                    )
                    # If RMID was introduced as a new column name, normalize to "RMID"
                    if l_rmid and l_rmid in merged.columns and "RMID" not in merged.columns:
                        merged.rename(columns={l_rmid: "RMID"}, inplace=True)
                    frames[k] = merged
                    trace("rm_mandate_enrich", {"frame": k, "rows": len(merged)})
            return frames

        
        # Sufficiency helpers
        def has_join_key(df: pd.DataFrame) -> bool:
            if df is None or df.empty: return False
            up = [c.upper() for c in df.columns]
            for k in ("RMID","MANDATEID","RM_NAME","MONTH","MONTH_DATE","MEET_YEAR","MEETINGDATE"): 
                if k in up: return True
            return False
        
        def looks_overly_aggregated(df: pd.DataFrame) -> bool:
            if df is None or df.empty: return True
            if len(df.columns) <= 3 and any("__" in c for c in df.columns) and not has_join_key(df): return True
            if len(df) < MIN_ROWS_THRESHOLD: return True
            return False
        
        def is_insufficient(df: pd.DataFrame, plan_step: Optional[Dict[str, Any]]=None) -> bool:
            if df is None or df.empty: return True
            if looks_overly_aggregated(df): return True
            if plan_step:
                dim = (plan_step.get("dim") or {}).get("field")
                if dim and dim in df.columns and df[dim].nunique() < MIN_UNIQUE_GROUPS: return True
            return False
        
        def build_followups(frames: Dict[str, pd.DataFrame], catalog: Dict[str, Any], plan: Dict[str, Any], *, widen_months: int = 36) -> List[Tuple[str, str]]:
            syn = build_schema_synonyms(catalog); prompts = []
            for step in (plan.get("steps") or []):
                sid = step.get("id"); df = frames.get(sid)
                if not is_insufficient(df, step): continue
                view = step.get("view")
                time_field = syn.get(view, {}).get("month_date") or syn.get(view, {}).get("meeting_date") or syn.get(view, {}).get("any_date")
                fields = catalog.get(view, {}).get("fields", [])
                prefer = ["MANDATEID","RMID","RM_NAME","MONTH_DATE","MEET_YEAR","CURRENT_AUM","MEETING_COUNT","CURRENT_REVENUE","TOPUP_AMOUNT","PROFIT_AMOUNT"]
                candidates = [p for p in prefer if p in fields] or fields[:6]
                cols = ", ".join(candidates) if candidates else "*"
                if time_field:
                    prompt = (f"Return raw rows with columns: {cols}. "
                              f"Filter WHERE {time_field} >= DATEADD(month, -{widen_months}, CURRENT_DATE). Aggregate by month. Limit 5000. Use only semantic model fields.")
                else:
                    prompt = (f"Return raw rows with columns: {cols}. "
                              f"Aggregate by Month. Limit 5000. Use only semantic model fields.")
                prompts.append((view, prompt))
            any_rmid = any((isinstance(df, pd.DataFrame) and not df.empty and _find_rmid_col_any(df)) for df in frames.values())
            if any_rmid and "rm_lookup" not in frames:
                rm_view = next((v for v in catalog if 'RM_PERFORMANCE' in v.upper()), None)
                if rm_view:
                    prompt = f"Return RMID and RM_NAME for all RMs. Limit {RM_LOOKUP_LIMIT}. Use only semantic model fields."
                    prompts.append((rm_view, prompt))
            seen=set(); uniq=[]
            for v,p in prompts:
                if (v,p) in seen: continue
                uniq.append((v,p)); seen.add((v,p))
            return uniq
        
        # Step execution
        def _pick_dim(view: str, catalog: Dict[str, Any]) -> Optional[Dict[str,str]]:
            fields = [f.upper() for f in catalog.get(view, {}).get("fields", [])]
            for cand in ("MONTH_DATE","MEETING_DATE","DATE","MONTH"):
                for f in fields:
                    if cand in f: return {"field": f, "kind": "month_or_date"}
            for cand in ("MEET_YEAR","YEAR"):
                for f in fields:
                    if cand in f: return {"field": f, "kind": "year"}
            for f in fields:
                if f in ("RMID","RM_ID","RELATIONSHIP_MANAGER_ID"): return {"field": f, "kind": "rmid"}
            for f in fields:
                if f == "MANDATEID": return {"field": f, "kind": "mandateid"}
            return None
        
        def _alias_for_step(step: Dict[str,Any]) -> str:
            sid = (step.get("id") or "STEP").upper()
            fld = (step.get("metric") or {}).get("field") or "METRIC"
            mode = (step.get("metric") or {}).get("mode") or "sum"
            safe = re.sub(r'[^A-Z0-9_]', '_', str(fld).upper())
            return f"{sid}__{safe}__{str(mode).upper()}"[:64]


        def _canonicalize_month_cols(df):
            import pandas as pd
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df

            # If MONTH_DATE already exists but is not datetime, try to rebuild it
            month_col = None
            year_col = None

            # Prefer explicit year columns
            for c in ["MEET_YEAR", "YEAR", "CAL_YEAR", "FISCAL_YEAR"]:
                if c in df.columns:
                    year_col = c
                    break

            # Prefer explicit month bucket columns
            for c in ["MEET_MON", "MONTH", "CAL_MONTH", "FISCAL_MONTH"]:
                if c in df.columns:
                    month_col = c
                    break

            # Case A: We already have a datetime-like MONTH_DATE → normalize
            if "MONTH_DATE" in df.columns and pd.api.types.is_datetime64_any_dtype(df["MONTH_DATE"]):
                # Make sure it’s month-start
                try:
                    df["MONTH_DATE"] = pd.to_datetime(df["MONTH_DATE"]).dt.to_period("M").dt.to_timestamp("MS")
                except Exception:
                    pass

            # Case B: We have MONTH_DATE but it’s numeric (e.g., 1..12). Rebuild if we also have a year
            elif "MONTH_DATE" in df.columns and year_col is not None and pd.api.types.is_numeric_dtype(df["MONTH_DATE"]):
                try:
                    y = pd.to_numeric(df[year_col], errors="coerce")
                    m = pd.to_numeric(df["MONTH_DATE"], errors="coerce")
                    dt = pd.to_datetime({"year": y, "month": m, "day": 1}, errors="coerce")
                    df["MONTH_DATE"] = dt.dt.to_period("M").dt.to_timestamp("MS")
                except Exception:
                    pass

            # Case C: We have year + month separate → create MONTH_DATE
            elif year_col is not None and month_col is not None:
                try:
                    y = pd.to_numeric(df[year_col], errors="coerce")
                    m = pd.to_numeric(df[month_col], errors="coerce")
                    dt = pd.to_datetime({"year": y, "month": m, "day": 1}, errors="coerce")
                    df["MONTH_DATE"] = dt.dt.to_period("M").dt.to_timestamp("MS")
                except Exception:
                    pass

            # Provide a JSON-friendly ISO key for previews/summarizer (without time component)
            if "MONTH_DATE" in df.columns:
                try:
                    df["MONTH_KEY"] = pd.to_datetime(df["MONTH_DATE"]).dt.strftime("%Y-%m")
                except Exception:
                    # if MONTH_DATE not convertible (edge cases), fall back to year+month if available
                    if year_col is not None and month_col is not None:
                        try:
                            y = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)
                            m = pd.to_numeric(df[month_col], errors="coerce").fillna(0).astype(int)
                            df["MONTH_KEY"] = (y.astype(str) + "-" + m.astype(str).str.zfill(2)).where((y > 0) & (m.between(1, 12)))
                        except Exception:
                            pass

            return df

        
        def run_step(step: Dict[str, Any], catalog: Dict[str, Any], *, months_window: int, question: str, year_filter: Optional[int] = None, rm_lookup: Optional[Any] = None) -> Dict[str, Any]:
            import pandas as pd
            import numpy as np
            import re
            import unicodedata
            import difflib
            import datetime

            # ---------- Validate inputs ----------
            view = (step or {}).get("view")
            alias = _alias_for_step(step)
            dim = ((step or {}).get("dim") or {}).get("field")
            metric = ((step or {}).get("metric") or {}).get("field")
            mode = ((step or {}).get("metric") or {}).get("mode", "sum")

            if not view or view not in (catalog or {}):
                return {"ok": False, "df": pd.DataFrame(), "sql": "", "explanation": f"unknown view {view!r}"}
            if metric is None:
                return {"ok": False, "df": pd.DataFrame(), "sql": "", "explanation": "no metric"}

            view_info = catalog.get(view, {}) or {}
            fields: List[str] = list(view_info.get("fields") or [])
            types: Dict[str, str] = view_info.get("types") or {}
            fields_upper = {f.upper(): f for f in fields}

            def _has_field(name: Optional[str]) -> bool:
                return isinstance(name, str) and name.upper() in fields_upper

            def _type_of(name: Optional[str]) -> Optional[str]:
                if not isinstance(name, str):
                    return None
                t = types.get(name)
                if t:
                    return str(t).upper()
                for k, v in types.items():
                    if k.upper() == name.upper():
                        return str(v).upper()
                return None

            def _is_numeric_field(name: Optional[str]) -> bool:
                t = (_type_of(name) or "").upper()
                if any(x in t for x in ["NUMBER", "DECIMAL", "NUMERIC", "INT", "FLOAT", "DOUBLE", "REAL", "BIGINT", "SMALLINT"]):
                    return True
                if isinstance(name, str):
                    nu = name.upper()
                    if any(tok in nu for tok in ["AMOUNT", "AMT", "VALUE", "BALANCE", "AUM", "REVENUE", "COMMIT", "SCORE", "SENTIMENT_SCORE", "POLARITY", "COMPOUND"]):
                        return True
                return False

            # ---------- Strict date handling ----------
            SAFE_DATE_NAMES = {
                "MONTH_DATE", "MEETING_DATE", "MEETINGDATE", "AS_OF_DATE",
                "TXN_DATE", "TRADE_DATE", "BOOKING_DATE", "POSTING_DATE",
                "VALUATION_DATE", "VALUE_DATE", "EFFECTIVE_DATE", "CREATED_DATE",
                "MODIFIED_DATE", "UPDATED_DATE", "DATE"
            }
            SAFE_MONTH_BUCKETS = {"MEET_MON", "MONTH_DATE", "FISCAL_MONTH", "CAL_MONTH", "MEET_MONTH"}

            def _is_date_typed(name: Optional[str]) -> bool:
                t = _type_of(name)
                return t in {"DATE", "DATETIME", "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ"}

            def _is_date_like(name: Optional[str]) -> bool:
                if not isinstance(name, str):
                    return False
                if _is_date_typed(name):
                    return True
                return name.upper() in SAFE_DATE_NAMES

            def _is_month_bucket(name: Optional[str]) -> bool:
                return isinstance(name, str) and name.upper() in SAFE_MONTH_BUCKETS

            # ---------- Resolve time field ----------
            step_time = (step or {}).get("time") or {}
            step_time_field = step_time.get("field") if isinstance(step_time.get("field"), str) else None
            step_time_grain = (step_time.get("grain") or "").lower().strip() if isinstance(step_time.get("grain"), str) else None

            syn = build_schema_synonyms(catalog) or {}
            syn_time = syn.get(view, {}) if isinstance(syn.get(view, {}), dict) else {}
            syn_candidates = [syn_time.get("month_date"), syn_time.get("meeting_date"), syn_time.get("any_date")]

            time_field = None
            if _has_field(step_time_field) and (_is_date_like(step_time_field) or _is_month_bucket(step_time_field)):
                time_field = fields_upper[step_time_field.upper()]
            if not time_field:
                for c in syn_candidates:
                    if _has_field(c) and (_is_date_like(c) or _is_month_bucket(c)):
                        time_field = fields_upper[str(c).upper()]
                        break
            if not time_field:
                for n in SAFE_DATE_NAMES:
                    if _has_field(n) and _is_date_like(n):
                        time_field = fields_upper[n]
                        break
            if not time_field:
                for n in SAFE_MONTH_BUCKETS:
                    if _has_field(n):
                        time_field = fields_upper[n]
                        break

            # ---------- Dimension ----------
            if isinstance(dim, str) and not _has_field(dim):
                dim = None
            elif isinstance(dim, str):
                dim = fields_upper[dim.upper()]

            # ---------- Fuzzy RM_NAME → RMID resolver ----------
            def _normalize_txt(s: str) -> str:
                s = unicodedata.normalize("NFKD", s)
                s = s.encode("ascii", "ignore").decode("ascii")
                s = s.lower()
                s = re.sub(r"[^a-z0-9]+", " ", s)
                return " ".join(s.split())

            def _collect_name_pairs_from_lookup(lookup) -> list[tuple[str, str]]:
                pairs = []
                try:
                    if isinstance(lookup, dict):
                        for k, v in lookup.items():
                            if isinstance(v, dict):
                                rid = None; rname = None
                                for kk, vv in v.items():
                                    ku = str(kk).upper()
                                    if ku in {"RMID", "RM_ID", "ID"}:
                                        rid = str(vv)
                                    if ku in {"RM_NAME", "NAME", "RMNAME"}:
                                        rname = str(vv)
                                if rid and rname:
                                    pairs.append((rid, rname))
                            elif isinstance(v, (str, int)) and isinstance(k, str):
                                if len(str(v)) <= 64 and not re.search(r"\s", str(v)) and not re.search(r"\d", k):
                                    pairs.append((str(v), k))
                                else:
                                    pairs.append((str(k), str(v)))
                    elif isinstance(lookup, pd.DataFrame):
                        cols = {c.upper(): c for c in lookup.columns}
                        name_col = next((cols[c] for c in ["RM_NAME", "NAME", "RMNAME"] if c in cols), None)
                        id_col = next((cols[c] for c in ["RMID", "RM_ID", "ID"] if c in cols), None)
                        if name_col and id_col:
                            for _, row in lookup[[id_col, name_col]].dropna().iterrows():
                                pairs.append((str(row[id_col]), str(row[name_col])))
                    elif isinstance(lookup, list):
                        for row in lookup:
                            if isinstance(row, dict):
                                rid = None; rname = None
                                for kk, vv in row.items():
                                    ku = str(kk).upper()
                                    if ku in {"RMID", "RM_ID", "ID"}:
                                        rid = str(vv)
                                    if ku in {"RM_NAME", "NAME", "RMNAME"}:
                                        rname = str(vv)
                                if rid and rname:
                                    pairs.append((rid, rname))
                except Exception:
                    pass
                seen = set(); out = []
                for rid, rname in pairs:
                    key = (rid, rname)
                    if key not in seen:
                        out.append((rid, rname))
                        seen.add(key)
                return out

            _NAME_INDEX = []
            _CANON_TO_IDS = {}
            _NAMES_LIST = []
            if rm_lookup is not None:
                for rid, rname in _collect_name_pairs_from_lookup(rm_lookup):
                    canon = _normalize_txt(rname)
                    _NAME_INDEX.append((rid, rname, canon))
                    _CANON_TO_IDS.setdefault(canon, set()).add(rid)
                    _NAMES_LIST.append(rname)

            def _resolve_rmids_from_value(val) -> list[str]:
                candidates: list[str] = []
                if not _NAME_INDEX:
                    return candidates
                names_in = []
                if isinstance(val, str):
                    names_in = [val]
                elif isinstance(val, (list, tuple)):
                    names_in = [str(x) for x in val if isinstance(x, (str, int))]
                else:
                    return candidates
                for raw in names_in:
                    q = _normalize_txt(str(raw))
                    if not q:
                        continue
                    if q in _CANON_TO_IDS:
                        candidates.extend(list(_CANON_TO_IDS[q]))
                        continue
                    for rid, _nm, canon in _NAME_INDEX:
                        if q in canon or canon in q:
                            candidates.append(rid)
                    try:
                        close = difflib.get_close_matches(str(raw), _NAMES_LIST, n=10, cutoff=0.86)
                        if close:
                            close_canons = {_normalize_txt(x) for x in close}
                            for c in close_canons:
                                if c in _CANON_TO_IDS:
                                    candidates.extend(list(_CANON_TO_IDS[c]))
                    except Exception:
                        pass
                dedup = []
                seen = set()
                for rid in candidates:
                    if rid not in seen:
                        dedup.append(rid)
                        seen.add(rid)
                return dedup[:50]

            def _is_numeric_like(s: Any) -> bool:
                try:
                    return bool(re.fullmatch(r"\d+", str(s).strip()))
                except Exception:
                    return False

            # ---------- select_also + soft join keys ----------
            planner_select_also = [c for c in ((step or {}).get("select_also") or []) if _has_field(c)]
            candidate_keys = ["MANDATEID", "MANDATE_ID", "RMID", "RM_ID", "RM_NAME",
                              "MONTH_DATE", "MEETING_DATE", "MEETINGDATE", "MEET_MON", "MEET_YEAR", "YEAR"]

            # If planner set a yearly grain, do not suggest MEET_MON (to avoid per-month duplication)
            step_time = (step or {}).get("time") or {}
            if str(step_time.get("grain", "")).lower() == "year" and "MEET_MON" in candidate_keys:
                candidate_keys = [k for k in candidate_keys if k != "MEET_MON"]

            join_keys_auto = [fields_upper[k] for k in candidate_keys if k in fields_upper]
            select_also_cols = []

            seen_cols = set()
            for c in (planner_select_also + join_keys_auto):
                cu = c.upper()
                if cu not in seen_cols:
                    select_also_cols.append(c)
                    seen_cols.add(cu)

            # ---------- Fix non-numeric metrics requesting numeric modes ----------
            mode_lc = str(mode).lower()

            sentiment_mapping_intent = (
                isinstance(view, str) and "MEETING_SENTIMENT" in view.upper()
                and str(metric).upper() in {"SENTIMENT", "MEETING_SENTIMENT"}
                and mode_lc in {"avg", "mean", "average"}
            )
            if (not sentiment_mapping_intent) and (not _is_numeric_field(metric) and mode_lc in {"avg", "sum", "median"}):

                replacement = None
                preferred_tokens = ["SENTIMENT_SCORE", "POLARITY", "COMPOUND", "SCORE", "SENTI", "VALENCE"]
                for f in fields:
                    fu = f.upper()
                    if any(tok in fu for tok in preferred_tokens) and _is_numeric_field(f):
                        replacement = f
                        break
                if replacement:
                    try: trace("metric_rewritten_for_numeric", {"from": metric, "to": replacement, "mode": mode_lc, "view": view})
                    except Exception: pass
                    metric = replacement
                else:
                    try: trace("metric_coerced_to_count", {"from": metric, "orig_mode": mode_lc, "view": view})
                    except Exception: pass
                    mode = "count"; mode_lc = "count"

            # ---------- Aggregation token ----------
            agg = "COUNT(*)" if mode_lc == "count" else f"{str(mode).upper()}({metric})"

            # ---------- Helpers for relative month rewrites ----------
            def _month_add(d: datetime.date, months: int) -> datetime.date:
                yy = d.year + (d.month - 1 + months) // 12
                mm = (d.month - 1 + months) % 12 + 1
                return datetime.date(yy, mm, 1)

            def _parse_dateadd_months(expr: str) -> Optional[int]:
                if not isinstance(expr, str): return None
                # DATEADD(month, -12, CURRENT_DATE)
                m = re.search(r"DATEADD\s*\(\s*month\s*,\s*([+-]?\d+)\s*,\s*CURRENT_DATE\s*\)", expr, re.I)
                if m:
                    try: return int(m.group(1))
                    except Exception: return None
                # DATE_TRUNC('month', DATEADD(month, -6, CURRENT_DATE))
                m2 = re.search(r"DATE_TRUNC\s*\(\s*'month'\s*,\s*DATEADD\s*\(\s*month\s*,\s*([+-]?\d+)\s*,\s*CURRENT_DATE\s*\)\s*\)", expr, re.I)
                if m2:
                    try: return int(m2.group(1))
                    except Exception: return None
                # DATE_TRUNC('month', CURRENT_DATE)
                if re.search(r"DATE_TRUNC\s*\(\s*'month'\s*,\s*CURRENT_DATE\s*\)", expr, re.I):
                    return 0
                return None

            def _parse_to_date_ym(expr: str) -> Optional[tuple[int, int]]:
                if not isinstance(expr, str): return None
                m = re.search(r"TO_DATE\s*\(\s*'(\d{4})-(\d{2})-01'\s*\)", expr, re.I)
                if not m: return None
                try:
                    return int(m.group(1)), int(m.group(2))
                except Exception:
                    return None

            def _rewrite_month_bucket_filter(month_field: str, op: str, val_expr: Any) -> Optional[str]:
                """Rewrite MEET_MON >= DATEADD(...)/TO_DATE(...) to YEAR+MEET_MON logic."""
                year_col = None
                if "MEET_YEAR" in fields_upper: year_col = fields_upper["MEET_YEAR"]
                elif "YEAR" in fields_upper: year_col = fields_upper["YEAR"]
                if not year_col:
                    return None  # can't safely rewrite without a year field

                # numeric literal month? (rare in these prompts) → require equality with an explicit year (not available)
                if isinstance(val_expr, (int, float)) and 1 <= int(val_expr) <= 12:
                    # Without a year anchor, it's ambiguous; skip rewrite
                    return None

                # parse relative/current expressions
                y0 = m0 = None
                if isinstance(val_expr, str):
                    delta = _parse_dateadd_months(val_expr)
                    if delta is not None:
                        base = datetime.date.today()
                        dt = _month_add(base, delta)
                        y0, m0 = dt.year, dt.month
                    else:
                        parsed = _parse_to_date_ym(val_expr)
                        if parsed:
                            y0, m0 = parsed

                if y0 is None or m0 is None:
                    return None

                mf = fields_upper[month_field.upper()]
                op = (op or "=").strip()
                if op in (">=", "=>"):
                    return f"(({year_col} > {y0}) OR ({year_col} = {y0} AND {mf} >= {m0}))"
                if op == ">":
                    return f"(({year_col} > {y0}) OR ({year_col} = {y0} AND {mf} > {m0}))"
                if op in ("<=", "=<"):
                    return f"(({year_col} < {y0}) OR ({year_col} = {y0} AND {mf} <= {m0}))"
                if op == "<":
                    return f"(({year_col} < {y0}) OR ({year_col} = {y0} AND {mf} < {m0}))"
                if op == "=":
                    return f"({year_col} = {y0} AND {mf} = {m0})"
                return None

            # ---------- Filters (RM_NAME ↔ RMID; month-bucket rewrites; guard subqueries/next-month) ----------
            explicit_filters_in_step = (step.get("filters") if isinstance(step, dict) else None)
            # --- ADAPT: accept both dict and list-of-objects for filters (match old behavior) ---
            exp = explicit_filters_in_step

            def _adapt_filters(exp_any):
                """
                Old builder expects: [{"field": "...", "op": "=", "value": ...}, ...]
                New planner sometimes emits: {"MEET_YEAR": 2025, "RMID": {"in":[1,2]}}
                This adapter normalizes to the old list shape.
                """
                if isinstance(exp_any, list):
                    # assume already old-shape (defensive sanitize)
                    out = []
                    for f in exp_any:
                        if isinstance(f, dict) and "field" in f and "op" in f and "value" in f:
                            out.append({"field": f["field"], "op": str(f["op"]), "value": f["value"]})
                    return out
                if isinstance(exp_any, dict):
                    out = []
                    for k, v in exp_any.items():
                        if isinstance(v, dict):
                            # operators dict, e.g. {"between":[2023,2025]}, {">=":2024}, {"in":[...]}
                            for op, val in v.items():
                                out.append({"field": k, "op": str(op), "value": val})
                        else:
                            out.append({"field": k, "op": "=", "value": v})
                    return out
                return []

            explicit_filters_in_step = _adapt_filters(exp)
            # --- /ADAPT ---

            filter_clauses = []

            def _fmt_val(v):
                if isinstance(v, (int, float)):
                    return str(v)
                if isinstance(v, str):
                    if re.search(r"\bSELECT\b", v, re.I):
                        return None
                    if re.search(r"\b(CURRENT_DATE|DATEADD|TO_DATE|EXTRACT|NOW|DATE_TRUNC)\b", v, re.I):
                        return v
                    return "'" + v.replace("'", "''") + "'"
                return str(v)

            def _is_next_month_expr(s: str) -> bool:
                if not isinstance(s, str): return False
                return re.search(r"DATEADD\s*\(\s*month\s*,\s*\+?1\s*,", s, re.I) is not None

            def _add_filter(field_name: str, op: str, value):
                """
                Adds a filter clause safely:
                  - If the value contains date expressions, forces the LHS to be a real date column.
                  - Hard-bans DATE math on count-ish fields (e.g., *_CNT, *_COUNT).
                  - Never emits a raw planner filter; only emits the rewritten/validated clause.
                """
                # View/type context
                this_view = step.get("view")
                view_meta = (catalog or {}).get(this_view, {}) if this_view else {}
                # types_map keys should be the *physical* column names for this view
                types_map = {str(k).upper(): str(v).upper() for k, v in (view_meta.get("types") or {}).items()}

                def _phys_name(name: str) -> str:
                    # Map any alias/synonym to the physical column name used in SQL
                    return fields_upper.get((name or "").upper(), "")

                def _is_date_col(name: str) -> bool:
                    phys = _phys_name(name)
                    return types_map.get(phys.upper(), "") in {
                        "DATE", "DATETIME", "TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ"
                    }

                def _is_countish(name: str) -> bool:
                    u = (name or "").upper()
                    return u.endswith("_CNT") or u.endswith("_COUNT") or u == "COUNT" or u.endswith("_NUM") or u.endswith("_N")

                # Bail if field doesn't exist at all
                # If field not present, try YEAR/MEET_YEAR synonyms before bailing
                if not _has_field(field_name):
                    fu = (field_name or "").upper()
                    if fu in {"MEET_YEAR", "FY", "FISCAL_YEAR"} and _has_field("YEAR"):
                        field_name = "YEAR"
                    elif fu == "YEAR" and _has_field("MEET_YEAR"):
                        field_name = "MEET_YEAR"
                    else:
                        return


                # Does RHS look like a date expression?
                is_date_expr = isinstance(value, str) and re.search(
                    r"\b(CURRENT_DATE|DATEADD|TO_DATE|EXTRACT|NOW|DATE_TRUNC)\b", value, re.I
                )

                # If RHS is DATE math, LHS must be a DATE/TIMESTAMP column
                needs_date_lhs = bool(is_date_expr)

                # Additionally, never allow DATE math on count-ish LHS even if types_map is wrong
                if _is_countish(field_name) and is_date_expr:
                    needs_date_lhs = True

                # If we need a date LHS and current field isn't date, rewrite to a real date column
                if needs_date_lhs and not _is_date_col(field_name):
                    syn = (schema_synonyms or {}).get(this_view, {}) if this_view else {}
                    # Preference order for date columns in this view
                    for candidate in (syn.get("month_date"), syn.get("meeting_date"), syn.get("any_date")):
                        if candidate and _has_field(candidate) and _is_date_col(candidate):
                            field_name = candidate
                            break
                    else:
                        # No valid date column available → drop the clause quietly
                        return

                # Format RHS
                fv = _fmt_val(value)
                if fv is None:
                    return

                # Compose final LHS using physical name (ensures consistency)
                phys_lhs = _phys_name(field_name)
                if not phys_lhs:
                    return

                # Special case: MEET_MON buckets—let your existing month/year rewrite handle it later
                clause = f"{phys_lhs} {op} {fv}"
                filter_clauses.append(clause)

                # (Optional) emit for debug so the Analyst text mirrors what we truly execute
                try:
                    _emit(f"⛳ FINAL FILTER → {clause}")
                except Exception:
                    pass

            # explicit_filters_in_step already set above
            exp = explicit_filters_in_step

            # ✅ Accept both shapes: dict and list-of-objects
            if isinstance(exp, dict):
                adapted = []
                for k, v in exp.items():
                    if isinstance(v, dict):
                        # operators dict, e.g. {"between":[2023,2025]}, {">=":2024}
                        for op, val in v.items():
                            adapted.append({"field": k, "op": str(op), "value": val})
                    else:
                        adapted.append({"field": k, "op": "=", "value": v})
                explicit_filters_in_step = adapted


            if isinstance(explicit_filters_in_step, list) and explicit_filters_in_step:
                for f in explicit_filters_in_step:
                    try:
                        f_field = f.get("field")
                        f_op = (f.get("op") or "=").strip()
                        f_val_raw = f.get("value")
                        if isinstance(f_val_raw, str) and _is_next_month_expr(f_val_raw):
                            continue

                        # (A) If the filter is on a month bucket with a relative date expression, rewrite to YEAR+MONTH
                        if isinstance(f_field, str) and _is_month_bucket(f_field) and isinstance(f_val_raw, str):
                            rewritten = _rewrite_month_bucket_filter(f_field, f_op, f_val_raw)
                            if rewritten:
                                filter_clauses.append(rewritten)
                                continue  # skip the original broken clause

                        # (B) RM_NAME-like filter → resolve to RMIDs or fallback to LIKE
                        if isinstance(f_field, str) and f_field.upper() in {"RM_NAME", "RMNAME", "NAME"}:
                            if "RMID" in fields_upper or "RM_ID" in fields_upper:
                                rids = _resolve_rmids_from_value(f_val_raw)
                                if rids:
                                    id_field = "RMID" if "RMID" in fields_upper else "RM_ID"
                                    if len(rids) == 1 and f_op in {"=", "=="}:
                                        _add_filter(id_field, "=", str(rids[0]))
                                    else:
                                        in_list = ", ".join(_fmt_val(str(x)) for x in rids if _fmt_val(str(x)))
                                        if in_list:
                                            filter_clauses.append(f"{fields_upper[id_field]} IN ({in_list})")
                                    continue
                            if _has_field(f_field) and isinstance(f_val_raw, str):
                                pattern = f"'%{_normalize_txt(f_val_raw).replace(' ', '%').upper()}%'"
                                filter_clauses.append(f"UPPER({fields_upper[f_field.upper()]}) LIKE {pattern}")
                            continue

                        # (C) RMID-like filter but value looks like a NAME → resolve via lookup
                        if isinstance(f_field, str) and f_field.upper() in {"RMID", "RM_ID"}:
                            id_field_key = "RMID" if "RMID" in fields_upper else ("RM_ID" if "RM_ID" in fields_upper else None)
                            if id_field_key is not None:
                                if isinstance(f_val_raw, (list, tuple)):
                                    id_values, name_values = [], []
                                    for vv in f_val_raw:
                                        (id_values if _is_numeric_like(vv) else name_values).append(str(vv))
                                    resolved = []
                                    for nv in name_values:
                                        resolved.extend(_resolve_rmids_from_value(nv))
                                    final_ids = list({*id_values, *resolved})
                                    if final_ids:
                                        if f_op.lower() in {"=", "=="} and len(final_ids) == 1:
                                            _add_filter(id_field_key, "=", final_ids[0])
                                        else:
                                            in_list = ", ".join(_fmt_val(x) for x in final_ids)
                                            filter_clauses.append(f"{fields_upper[id_field_key]} IN ({in_list})")
                                        continue
                                    else:
                                        if "RM_NAME" in fields_upper and name_values:
                                            pattern = f"'%{_normalize_txt(' '.join(name_values)).replace(' ', '%').upper()}%'"
                                            filter_clauses.append(f"UPPER({fields_upper['RM_NAME']}) LIKE {pattern}")
                                            continue
                                        continue
                                else:
                                    if isinstance(f_val_raw, str) and not _is_numeric_like(f_val_raw):
                                        rids = _resolve_rmids_from_value(f_val_raw)
                                        if rids:
                                            if len(rids) == 1 and f_op in {"=", "=="}:
                                                _add_filter(id_field_key, "=", str(rids[0]))
                                            else:
                                                in_list = ", ".join(_fmt_val(str(x)) for x in rids if _fmt_val(str(x)))
                                                if in_list:
                                                    filter_clauses.append(f"{fields_upper[id_field_key]} IN ({in_list})")
                                            continue
                                        else:
                                            if "RM_NAME" in fields_upper:
                                                pattern = f"'%{_normalize_txt(f_val_raw).replace(' ', '%').upper()}%'"
                                                filter_clauses.append(f"UPPER({fields_upper['RM_NAME']}) LIKE {pattern}")
                                                continue
                                            continue
                                    else:
                                        _add_filter(id_field_key, f_op, f_val_raw)
                                        continue

                        # (D) Default: pass-through if field exists
                        if _has_field(f_field):
                            _add_filter(f_field, f_op, f_val_raw)
                    except Exception:
                        continue
            elif time_field and _is_date_like(time_field):
                qtext = (st.session_state.get("last_query","") or "").lower()
                # Skip default window for yearly rollups or explicit "all years"
                if not (step_time_grain in {"year","yearly"} or "all years" in qtext):
                    filter_clauses.append(f"{time_field} >= DATEADD(month, -{months_window}, CURRENT_DATE)")


            # ---------- Order / Limit: honor planner only when NOT time-bucketed ----------
            order_from_step = (step.get("order") or None)
            limit_from_step = None
            try:
                if step.get("limit") is not None:
                    limit_from_step = int(step.get("limit"))
            except Exception:
                limit_from_step = None

            is_time_bucketed = isinstance(step_time_grain, str) and step_time_grain in {"month", "year", "monthly", "yearly"}

            # ---------- Minimal, neutral prompt (no question, no view mention) ----------
            parts = []
            shape_bits = []
            if dim:
                shape_bits.append(f"by {dim}")
            if step_time_grain in ("month", "monthly") and time_field:
                if _is_month_bucket(time_field) and not _is_date_typed(time_field):
                    shape_bits.append("one row per month using the month column")
                else:
                    shape_bits.append("one row per month")
            elif step_time_grain in ("year", "yearly"):
                shape_bits.append("summarized by year")

            if mode_lc == "count" and (not metric or not _is_numeric_field(metric)):
                metric_phrase = f"count of rows as {alias}"
            else:
                metric_phrase = f"{mode_lc} of {metric} as {alias}"

            # If averaging SENTIMENT in Meeting Sentiment view, tell Analyst what "average" means
            if (
                isinstance(view, str) and "MEETING_SENTIMENT" in view.upper()
                and str(metric).upper() in {"SENTIMENT", "MEETING_SENTIMENT"}
                and str(mode_lc) in {"avg", "mean", "average"}
            ):
                metric_phrase = f"average of {metric} (map Positive→+1, Neutral→0, Negative→-1 first) as {alias}"


            parts.append("Show " + ", ".join([metric_phrase] + shape_bits if shape_bits else [metric_phrase]) + ".")
            extras = [c for c in select_also_cols if c not in (dim, "YEAR", "MONTH_DATE")]
            if extras:
                parts.append("Include " + ", ".join(extras) + ".")
            if filter_clauses:
                parts.append("Filters: " + " AND ".join(filter_clauses) + ".")

            if not is_time_bucketed and isinstance(order_from_step, str) and order_from_step.strip():
                o = order_from_step.strip().lower()
                if o in ("asc", "ascending", "low", "lowest", "min", "smallest"):
                    parts.append(f"Sort by {alias} ascending.")
                elif o in ("desc", "descending", "high", "highest", "max", "largest"):
                    parts.append(f"Sort by {alias} descending.")

            if not is_time_bucketed and (limit_from_step is not None):
                parts.append(f"Limit {limit_from_step} rows.")

            prompt = " ".join(parts).strip()

            parsed = analyst_message(view, prompt) or {}
            if not (parsed.get("sql") or ""):
                retry = f"Show {metric_phrase}."
                if filter_clauses:
                    retry += " Filters: " + " AND ".join(filter_clauses) + "."
                parsed2 = analyst_message(view, retry) or {}
                if parsed2.get("sql"):
                    parsed = parsed2
                elif parsed.get("suggestions"):
                    sug = parsed.get("suggestions")
                    s0 = sug[0] if isinstance(sug, (list, tuple)) and len(sug) > 0 else sug
                    parsed3 = analyst_message(view, s0) or {}
                    if parsed3.get("sql"):
                        parsed = parsed3

            if not (parsed.get("sql") or ""):
                return {"ok": False, "df": pd.DataFrame(), "sql": "", "explanation": parsed.get("explanation", "no sql")}

            ok, df, err = analyst_query_execute(parsed["sql"])
            if not ok:
                return {"ok": False, "df": pd.DataFrame(), "sql": parsed["sql"], "explanation": err}

            # ---------- Post-processing ----------
            df = parse_dates(clean_cols(df))

            for c in ["MEET_YEAR","YEAR","POSYEAR","CAL_YEAR","FISCAL_YEAR"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

            for c in ["MEET_MON","MONTH","POSMON","CAL_MONTH","FISCAL_MONTH"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

            try:
                df = _canonicalize_month_cols(df)
            except Exception:
                pass

            if isinstance(df, pd.DataFrame) and not df.empty:
                orig_cols = list(df.columns)
                seen_upper_cols = {}
                new_cols = []
                for c in orig_cols:
                    key = str(c).upper()
                    if key not in seen_upper_cols:
                        seen_upper_cols[key] = 0
                        new_cols.append(str(c))
                    else:
                        seen_upper_cols[key] += 1
                        new_name = f"{c}__dup{seen_upper_cols[key]}"
                        while new_name in new_cols:
                            seen_upper_cols[key] += 1
                            new_name = f"{c}__dup{seen_upper_cols[key]}"
                        new_cols.append(new_name)
                if new_cols != orig_cols:
                    try: trace("dedupe_columns", {"orig": orig_cols, "new": new_cols})
                    except Exception: pass
                    df.columns = new_cols

            if isinstance(df, pd.DataFrame) and not df.empty and alias:
                if not any(str(c).upper() == str(alias).upper() for c in df.columns):
                    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    if len(numeric_cols) == 1:
                        df.rename(columns={numeric_cols[0]: alias}, inplace=True)
                    else:
                        for c in df.columns:
                            if str(c).upper() == "METRIC":
                                df.rename(columns={c: alias}, inplace=True)
                                break

            if isinstance(df, pd.DataFrame) and "___source" not in df.columns:
                df["___source"] = step.get("id")

            if year_filter is not None and isinstance(df, pd.DataFrame) and "YEAR" in df.columns:
                try: df["YEAR"] = df["YEAR"].fillna(year_filter).astype(int)
                except Exception: pass

            return {"ok": True, "df": df, "sql": parsed["sql"], "explanation": parsed.get("explanation", ""), "alias": alias}



        
        # ──────────────────────────────────────────────────────────────────────────────
        # NEW: Auto-merge across frames on shared keys
        def _columns_upper(df: pd.DataFrame) -> Dict[str, str]:
            # Map UPPER->original
            return {c.upper(): c for c in df.columns}
        
        def _common_join_keys(a: pd.DataFrame, b: pd.DataFrame) -> List[str]:
            if a is None or b is None or a.empty or b.empty: return []
            au = _columns_upper(a); bu = _columns_upper(b)
            common_upper = [k for k in JOIN_KEYS_PREF if k in au and k in bu]
            return [ (au[k], bu[k]) for k in common_upper ]  # return pairs preserving original case
        
        def auto_merge_all_frames(frames: Dict[str, pd.DataFrame], *, max_pairs: int = 6, how: str = "inner") -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
            """Merge every pair of frames with a shared preferred key. Returns new frames dict and a list of merges done."""
            keys = list(frames.keys())
            merges_done = []
            out_frames = dict(frames)
            pairs_tried = 0
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    if pairs_tried >= max_pairs: break
                    a_key, b_key = keys[i], keys[j]
                    a = out_frames.get(a_key); b = out_frames.get(b_key)
                    if not isinstance(a, pd.DataFrame) or a.empty: continue
                    if not isinstance(b, pd.DataFrame) or b.empty: continue
                    # detect common preferred keys
                    pairs = _common_join_keys(a, b)  # list of (a_col, b_col) preserving case
                    if not pairs: continue
                    join_a, join_b = pairs[0]  # prefer first best key
                    try:
                        merged = a.merge(b, left_on=join_a, right_on=join_b, how=how, suffixes=(f"__{a_key}", f"__{b_key}"))
                        if not merged.empty:
                            name = f"merged_{a_key}__{b_key}"
                            # Avoid overwriting
                            idx = 1; nm = name
                            while nm in out_frames: idx += 1; nm = f"{name}_{idx}"
                            out_frames[nm] = merged
                            merges_done.append({"name": nm, "left": a_key, "right": b_key, "on_left": join_a, "on_right": join_b, "rows": len(merged)})
                            pairs_tried += 1
                    except Exception as e:
                        trace("merge_error", {"left": a_key, "right": b_key, "err": str(e)})
                if pairs_tried >= max_pairs: break
            return out_frames, merges_done
        
        # ──────────────────────────────────────────────────────────────────────────────
        # Final LLM
        FINAL_PROMPT = """
        You are a senior BI analyst. Use ONLY the data previews provided. Output JSON ONLY:
        {
          "narrative": "markdown string",
          "kpis": [ {"title":"", "value":"", "sub":""} ],
          "chart": {
            "chart":"scatter|line|bar|table",
            "x":"<col>|null",
            "y":"<col>|null",
            "series":"<col>|null",
            "agg":"sum|avg|last|median|count",
            "allowed_chart_types":["line","bar","scatter","table"],
            "top_n": null
          }

        }
        Sections in narrative: Overall Summary, Key Findings, Drivers & Diagnostics, Recommendations.
        If insufficient data, say exactly what rows/fields are missing.
        
        FILTER/REASONER CONTEXT:
        - Treat the dataset as already filtered by the query/reasoners. Do NOT generalize to the full population.
          Use language like "within the returned/filtered set" instead of "all mandates" unless coverage is explicit.
        - Columns like signal_*, *_condition, *_score are derived outputs from rules/reasoners; describe them as flags/labels
          on the returned records, not as raw base measurements.

  ANALYTICS GUARDRAILS:
  - Do NOT invent correlations, regressions, or projections. Only mention those if GROUND_TRUTH_JSON or
    deterministic notes explicitly include them.
  - Never mention "GROUND_TRUTH_JSON", "tools", "tool outputs", or any internal JSON in the user-facing narrative.
    Translate verified numbers into plain language (e.g., "Based on the computed metrics...").

        CHART GENERATION GUIDANCE:
        - Prefer derived/computed metrics (profit_margin, cost_to_revenue, aum_trend, revenue_trend) when available for richer insights.
        - For bar charts (rankings): x=entity/dimension (RMID, RM_NAME, MANDATEID), y=metric (prefer derived), series=optional time/category, top_n=5-10.
        - For line charts (trends): x=time (DATE, MONTH_DATE, MEET_YEAR/MEET_MON), y=metric, series=entity or category dimension.
        - For scatter: x,y=two numeric metrics to show correlation.
        - Use top_n strategically: null for comprehensive views, 5-10 for rankings/comparisons, 3 for executive summaries.
        - Set agg: sum for flows (revenue, cost), last/snapshot for point-in-time (AUM), avg for ratios/scores (profit_margin).
        """

        FINAL_PROMPT_FULL = _get_prompt_template(
            "summary_full",
            """
        You are a senior BI analyst. Use ONLY the full datasets provided in DATA_JSON.
        Output JSON ONLY:
        {
          "narrative": "markdown string",
          "kpis": [ {"title":"", "value":"", "sub":""} ],
          "chart": {
            "chart":"scatter|line|bar|table",
            "x":"<col>|null",
            "y":"<col>|null",
            "series":"<col>|null",
            "agg":"sum|avg|last|median|count",
            "allowed_chart_types":["line","bar","scatter","table"],
            "top_n": null
          }

          "needs_more": false,
          "followup_prompt": "",
          "required": { "views": [], "columns": {} }
        }

        Sections in narrative: Overall Summary, Key Findings, Drivers & Diagnostics, Recommendations.
        If insufficient data, say exactly what rows/fields are missing.

          IMPORTANT:
        - If the question or dataset includes FILTER_* columns (e.g., FILTER_CATEGORY, FILTER_MONTH) or clearly states a filter (e.g., "LOW category"),
          assume the metric is ALREADY computed on that filtered subset—even if the data does not show a category column.
          Do NOT recommend adding a category field in that case.
        - Treat the dataset as already filtered by the query/reasoners. Do NOT generalize to the full population.
          Use language like "within the returned/filtered set" instead of "all mandates" unless coverage is explicit.
        - Columns like signal_*, *_condition, *_score are derived outputs from rules/reasoners; describe them as flags/labels
          on the returned records, not as raw base measurements.
          - Do NOT invent correlations, regressions, or projections. Only mention those if GROUND_TRUTH_JSON or
            deterministic notes explicitly include them.
          - Never mention "GROUND_TRUTH_JSON", "tools", "tool outputs", or any internal JSON in the user-facing narrative.
            Translate verified numbers into plain language (e.g., "Based on the computed metrics...").
        - If X is time-like (DATE/DATETIME, YEARMONTH string, or integer months 1–12, optionally with YEAR), set "chart":"line" (not scatter) and sort X ascending. If YEAR is present with MEET_MON, plot a multi-series line with series = YEAR.
        - If the question contains multiple explicit years (e.g. "2024 vs 2025") or "vs", return a single comparison chart:
          • For rankings ("top/best/most/highest"): chart="bar", x="<RM or Mandate dimension>", y="<metric>", series="YEAR", top_n=the requested N (default 5).
          • For time-like X (months), prefer chart="line" with series="YEAR".
        - Only use fields that exist in the frames; prefer YEAR/MEET_YEAR for series if present.
        - Set "agg" consistent with metric conventions (AUM=snapshot/last, Scores=avg, flows=sum).

        CHART GENERATION (Enhanced):
        - Prefer derived/computed metrics (profit_margin, cost_to_revenue, aum_trend, revenue_trend) for visualizations; they provide richer insights.
        - Classify columns: ENTITIES (RMID, RM_NAME, MANDATEID) for x/series, METRICS (aum, revenue, cost, profit_margin, sentiment) for y, TIME (DATE, MONTH_DATE, MEET_YEAR, MEET_MON) for x in time-series.
        - For bar charts (rankings/comparison): x=entity or category, y=metric (prefer derived), series=optional time/category, top_n=5-10.
        - For line charts (trends): x=time, y=metric, series=entity or category dimension; sort x ascending.
        - For scatter: x,y=two numeric metrics to show correlation.
        - If reasoning context identifies flagged entities (at-risk mandates, etc.), highlight or filter them in top_n.
        - Set top_n: null for comprehensive views, 5-10 for rankings, 3 for executive summaries.

            """,
        )

        CHUNK_PROMPT = _get_prompt_template(
            "summary_chunk",
            """
        You are a BI analyst. INPUTS: QUESTION and ONE_FRAME rows.
        Summarize this CHUNK ONLY and return JSON ONLY:
        {
          "intermediate": true,
          "frame_id": "<id>",
          "coverage": {"rows": 0, "months_observed": 0, "first_month": null, "last_month": null},
          "signals": [
            {"type":"top_rm","by":"<metric_col>","items":[{"RMID":"", "RM_NAME":"", "value":0.0}]},
            {"type":"trend","metric":"<metric_col>","direction":"up|down|flat","strength":"weak|moderate|strong"}
          ],
          "kpi_candidates": [ {"title":"", "value":"", "sub":""} ],
          "chart_hints": [{"chart":"line|bar|table","x":"<col>","y":"<col>"}]
        }
        Keep it compact and numeric; no prose. Do not repeat rows.
            """,
        )

        REDUCE_PROMPT = _get_prompt_template(
            "summary_reduce",
            """
        You are a senior BI analyst. Combine ALL chunk-level JSONs (INTERMEDIATE_JSON) to answer the QUESTION.
        Output JSON ONLY with the same schema as FINAL_PROMPT_FULL:
        {
          "narrative": "markdown string",
          "kpis": [ {"title":"", "value":"", "sub":""} ],
          "chart": {"chart":"scatter|line|bar|table","x":"<col>|null","y":"<col>|null"},
          "needs_more": false,
          "followup_prompt": "",
          "required": { "views": [], "columns": {} }
        }
        Be definitive if coverage across chunks is sufficient; if not, set needs_more with a concise follow-up request.
            """,
        )


        
        def final_summary_llm(
            question: str,
            frames: Dict[str, Any],
            deterministic_insights: Optional[List[str]] = None,
            resolved_tools: Optional[list] = None   # <— add this
        ) -> Dict[str, Any]:
            """
            Multi-step reliability mode:
              - If multiple frames (≥2): revert to the proven PREVIEW summarizer (like your original),
                with explicit multi-frame guidance so the LLM writes a full, aligned summary.
              - If single frame: keep the full-data path for rich answers.
            Returns: narrative, kpis, chart, needs_more, followup_prompt, required
            """
            import json
            import pandas as pd
            import numpy as np
            import datetime as _dt
            import math
            import re

            deterministic_insights = deterministic_insights or []

            # ---------------- JSON helpers ----------------
            def _json_default(obj):
                try:
                    if isinstance(obj, (pd.Timestamp,)):
                        return obj.isoformat()
                    if isinstance(obj, (np.datetime64,)):
                        return pd.Timestamp(obj).isoformat()
                    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
                        return obj.isoformat()
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        try:
                            if np.isnan(obj):
                                return None
                        except Exception:
                            pass
                        return float(obj)
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                    return str(obj)
                except Exception:
                    return str(obj)

            def _coerce_json(resp_text: str) -> dict:
                """Be tolerant to prose/fences; extract the first valid {...}."""
                if isinstance(resp_text, dict):
                    return resp_text
                s = str(resp_text or "").strip()
                if not s:
                    return {}
                m = re.search(r"```(?:json)?(.*?)```", s, flags=re.S | re.I)
                if m:
                    s = m.group(1).strip()
                try:
                    return json.loads(s)
                except Exception:
                    pass
                m = re.search(r"\{.*\}", s, flags=re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        return {"narrative": s}
                return {"narrative": s}

            # ---------------- Small helpers ----------------
            def _pick_numeric_cols(df: pd.DataFrame, k: int = 6):
                return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:k]

            def _pick_time_cols(df: pd.DataFrame):
                cu = {c.upper(): c for c in df.columns}
                year = cu.get("MEET_YEAR", cu.get("YEAR"))
                month = cu.get("MEET_MON", cu.get("MONTH"))
                date = None
                for cand in ["MONTH_DATE", "MEETING_DATE", "MEETINGDATE", "AS_OF_DATE", "DATE"]:
                    if cand in cu:
                        date = cu[cand]
                        break
                return year, month, date

            def _join_keys(df: pd.DataFrame):
                cu = {c.upper(): c for c in df.columns}
                keys = []
                for k in ["RMID", "RM_NAME", "MANDATEID", "MEET_YEAR", "YEAR", "MEET_MON", "MONTH_DATE"]:
                    if k in cu:
                        keys.append(cu[k])
                return keys

            def _sample_df(df: pd.DataFrame) -> dict:
                """Keep your original preview style, but prefer join/time+top numerics."""
                try:
                    cols = []
                    # join-ish keys first
                    cols.extend([c for c in _join_keys(df) if c not in cols])
                    # add time cols if not already present
                    y, m, d = _pick_time_cols(df)
                    for c in [y, m, d]:
                        if c and c not in cols:
                            cols.append(c)
                    # add up to 6 numeric cols
                    for c in _pick_numeric_cols(df, k=6):
                        if c not in cols:
                            cols.append(c)
                    # cap total columns to ~12 for clarity
                    cols = cols[:12] if cols else list(df.columns)[:12]
                    rows = df[cols].head(24).replace({np.nan: None}).to_dict(orient="records")
                    # time coverage (best-effort)
                    y, m, d = _pick_time_cols(df)
                    cov = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
                    try:
                        if y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
                            cov["year_min"] = int(pd.to_numeric(df[y], errors="coerce").min())
                            cov["year_max"] = int(pd.to_numeric(df[y], errors="coerce").max())
                    except Exception:
                        pass
                    return {
                        "columns": cols,
                        "rows": rows,
                        "shape": [int(df.shape[0]), int(df.shape[1])],
                        "join_keys": _join_keys(df),
                        "time_cols": [x for x in [y, m, d] if x],
                        "coverage": cov
                    }
                except Exception:
                    return {
                        "columns": list(df.columns)[:12],
                        "rows": df.head(24).replace({np.nan: None}).to_dict(orient="records"),
                        "shape": [int(getattr(df, "shape", [0, 0])[0]), int(getattr(df, "shape", [0, 0])[1])],
                        "join_keys": _join_keys(df),
                        "time_cols": [x for x in _pick_time_cols(df) if x],
                        "coverage": {"rows": int(getattr(df, "shape", [0, 0])[0]), "cols": int(getattr(df, "shape", [0, 0])[1])}
                    }

            def _build_metric_glossary(frames: Dict[str, Any], max_items: int = 24) -> str:
                cols_seen: dict[str, str] = {}
                cols_order: list[str] = []
                for df in (frames or {}).values():
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    for col in df.columns:
                        key = str(col).lower()
                        if key not in cols_seen:
                            cols_seen[key] = str(col)
                            cols_order.append(key)
                if not cols_order:
                    return ""

                meta_map: dict[str, dict[str, Any]] = {}
                try:
                    from rai_semantic_registry import load_registry
                    for entity in load_registry():
                        for f in getattr(entity, "fields", []) or []:
                            name = (getattr(f, "name", "") or "").lower()
                            if not name:
                                continue
                            entry = meta_map.get(name, {})
                            if not entry or (not entry.get("description") and getattr(f, "description", None)):
                                meta_map[name] = {
                                    "description": getattr(f, "description", "") or "",
                                    "derived": bool(getattr(f, "derived", False)),
                                    "dtype": getattr(f, "dtype", "") or "",
                                    "expr": getattr(f, "expr", "") or "",
                                    "role": getattr(f, "role", "") or "",
                                }
                except Exception:
                    meta_map = {}

                def _heuristic_desc(key: str) -> str:
                    if key.startswith("signal_"):
                        return "derived flag (1/0) indicating condition met."
                    if key.endswith("_condition"):
                        return "derived categorical label from rules."
                    if key.endswith("_score"):
                        return "derived score from rules; higher indicates greater intensity."
                    if key.endswith("_trend"):
                        return "derived period-over-period change rate."
                    if key.startswith("prev_"):
                        return "prior-period value used in trend calculations."
                    return ""

                lines: list[str] = []
                for key in cols_order:
                    meta = meta_map.get(key, {})
                    derived = bool(meta.get("derived"))
                    desc = (meta.get("description") or "").strip()
                    if not desc:
                        desc = _heuristic_desc(key)
                    include = (
                        derived
                        or key.startswith("signal_")
                        or key.endswith("_score")
                        or key.endswith("_condition")
                        or key.endswith("_trend")
                        or key.startswith("prev_")
                    )
                    if not include:
                        continue
                    tags: list[str] = []
                    if derived:
                        tags.append("derived")
                    if key.startswith("signal_"):
                        tags.append("flag")
                    if key.endswith("_condition"):
                        tags.append("label")
                    if key.endswith("_score"):
                        tags.append("score")
                    if key.endswith("_trend"):
                        tags.append("trend")
                    if key.startswith("prev_"):
                        tags.append("prior-period")
                    dtype = (meta.get("dtype") or "").strip()
                    if dtype:
                        tags.append(f"type={dtype}")
                    line = f"- {cols_seen[key]}: {desc}" if desc else f"- {cols_seen[key]}"
                    if tags:
                        line += f" ({', '.join(tags)})"
                    if len(line) > 220:
                        line = line[:217] + "..."
                    lines.append(line)
                    if len(lines) >= max_items:
                        break
                return "\n".join(lines)

            # Decide path: multi-step (preview) vs single-step (full-data)
            multi_step = sum(1 for _, df in (frames or {}).items() if isinstance(df, pd.DataFrame) and not df.empty) >= 2
            glossary_text = _build_metric_glossary(frames)
            glossary_block = (
                "\n\nMETRICS GLOSSARY (columns in this data):\n" + glossary_text
                if glossary_text
                else ""
            )

            # ---------------- SINGLE-STEP (full data) ----------------
            if not multi_step:
                # Keep your richer full-data path for 1 dataset (same shape you already expect)
                try:
                    # Serialize full frame safely
                    data_json = []
                    for fid, df in (frames or {}).items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            safe = df.copy()
                            for c in safe.columns:
                                if pd.api.types.is_datetime64_any_dtype(safe[c]):
                                    safe[c] = pd.to_datetime(safe[c], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S").replace("NaT", None)
                            safe = safe.replace({np.nan: None})
                            data_json.append({"id": fid, "rows": json.loads(safe.to_json(orient="records"))})
                    FINAL_PROMPT_FULL = _get_prompt_template(
                        "summary_full_single",
                        _get_prompt_template(
                            "summary_full",
                            """
You are a senior BI analyst. Use ONLY the full dataset(s) provided in DATA_JSON.
Output JSON ONLY:
{
  "narrative": "markdown string",
  "kpis": [ {"title":"", "value":"", "sub":""} ],
  "chart": {
    "chart":"scatter|line|bar|table",
    "x":"<col>|null",
    "y":"<col>|null",
    "series":"<col>|null",
    "agg":"sum|avg|last|median|count",
    "allowed_chart_types":["line","bar","scatter","table"],
    "top_n": null
  }

  "needs_more": false,
  "followup_prompt": "",
  "required": {"views": [], "columns": {}}
}
Sections: Overall Summary, Key Findings, Drivers & Diagnostics, Recommendations.
If insufficient data, say exactly what rows/fields are missing.

          IMPORTANT:
- If the question or dataset includes FILTER_* columns (e.g., FILTER_CATEGORY, FILTER_MONTH) or clearly states a filter,
  assume the metric is ALREADY computed on that filtered subset. Do NOT ask for category fields in that case.
- Treat the dataset as already filtered by the query/reasoners. Do NOT generalize to the full population.
  Use language like "within the returned/filtered set" instead of "all mandates" unless coverage is explicit.
- Columns like signal_*, *_condition, *_score are derived outputs from rules/reasoners; describe them as flags/labels
  on the returned records, not as raw base measurements.
  - Do NOT invent correlations, regressions, or projections. Only mention those if GROUND_TRUTH_JSON or
    deterministic notes explicitly include them.
  - Never mention "GROUND_TRUTH_JSON", "tools", "tool outputs", or any internal JSON in the user-facing narrative.
    Translate verified numbers into plain language (e.g., "Based on the computed metrics...").
""".strip(),
                        ),
                    )
                    gt_json = json.dumps({"resolved_tools": resolved_tools or []},
                                         ensure_ascii=False, default=_json_default)

                    prompt = (
                        FINAL_PROMPT_FULL
                        + "\n\nQuestion:\n" + str(question)
                        + glossary_block
                        + "\n\nDATA_JSON:\n" + json.dumps(data_json, ensure_ascii=False, default=_json_default)
                        + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                        + (
                            "\n\nDeterministic notes:\n"
                            + json.dumps(deterministic_insights, ensure_ascii=False, default=_json_default)
                            if deterministic_insights else ""
                          )
                        + "\n\nSTRICT RULES:\n"
                          "- Prefer numbers from GROUND_TRUTH_JSON when present; treat them as source of truth.\n"
                          "- If GROUND_TRUTH_JSON is empty or tools failed, you may compute from DATA_JSON and label as estimate.\n"
                          "- If GROUND_TRUTH_JSON conflicts with DATA_JSON, follow GROUND_TRUTH_JSON and note the mismatch briefly.\n"
                          "- When a tool result has a scalar 'value', copy it verbatim (format to 2 decimals if needed).\n"
                          "- When a tool result is 'grouped', you may list top items by their given 'value' only.\n"

                    )

                    # --- HARD BUDGET GATE: Final (full-data) prompt ----------
                    if _approx_tokens(prompt) > FINAL_INPUT_LIMIT_TOKENS and glossary_block:
                        glossary_block = ""
                        prompt = (
                            FINAL_PROMPT_FULL
                            + "\n\nQuestion:\n" + str(question)
                            + "\n\nDATA_JSON:\n" + json.dumps(data_json, ensure_ascii=False, default=_json_default)
                            + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                            + (
                                "\n\nDeterministic notes:\n"
                                + json.dumps(deterministic_insights, ensure_ascii=False, default=_json_default)
                                if deterministic_insights else ""
                              )
                            + "\n\nSTRICT RULES:\n"
                              "- Prefer numbers from GROUND_TRUTH_JSON when present; treat them as source of truth.\n"
                              "- If GROUND_TRUTH_JSON is empty or tools failed, you may compute from DATA_JSON and label as estimate.\n"
                              "- If GROUND_TRUTH_JSON conflicts with DATA_JSON, follow GROUND_TRUTH_JSON and note the mismatch briefly.\n"
                              "- When a tool result has a scalar 'value', copy it verbatim (format to 2 decimals if needed).\n"
                              "- When a tool result is 'grouped', you may list top items by their given 'value' only.\n"
                        )

                    if _approx_tokens(prompt) > FINAL_INPUT_LIMIT_TOKENS:
                        # Drop deterministic notes first (GT stays!)
                        prompt = (
                            FINAL_PROMPT_FULL
                            + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                            + "\n\nQuestion:\n" + str(question)
                            + glossary_block
                            + "\n\nDATA_JSON:\n" + json.dumps(data_json, ensure_ascii=False, default=_json_default)
                            + "\n\nSTRICT RULES:\n"
                              "- Prefer numbers from GROUND_TRUTH_JSON when present; treat them as source of truth.\n"
                              "- If GROUND_TRUTH_JSON is empty or tools failed, you may compute from DATA_JSON and label as estimate.\n"
                              "- If GROUND_TRUTH_JSON conflicts with DATA_JSON, follow GROUND_TRUTH_JSON and note the mismatch briefly.\n"
                              "- When a tool result has a scalar 'value', copy it verbatim (format to 2 decimals if needed).\n"
                              "- When a tool result is 'grouped', you may list top items by their given 'value' only.\n"
                        )

                    if (_approx_tokens(prompt) > FINAL_INPUT_LIMIT_TOKENS or len(prompt) > FINAL_INPUT_LIMIT_CHARS) and glossary_block:
                        glossary_block = ""
                        prompt = (
                            FINAL_PROMPT_FULL
                            + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                            + "\n\nQuestion:\n" + str(question)
                            + "\n\nDATA_JSON:\n" + json.dumps(data_json, ensure_ascii=False, default=_json_default)
                            + "\n\nSTRICT RULES:\n"
                              "- Prefer numbers from GROUND_TRUTH_JSON when present; treat them as source of truth.\n"
                              "- If GROUND_TRUTH_JSON is empty or tools failed, you may compute from DATA_JSON and label as estimate.\n"
                              "- If GROUND_TRUTH_JSON conflicts with DATA_JSON, follow GROUND_TRUTH_JSON and note the mismatch briefly.\n"
                              "- When a tool result has a scalar 'value', copy it verbatim (format to 2 decimals if needed).\n"
                              "- When a tool result is 'grouped', you may list top items by their given 'value' only.\n"
                        )

                    if _approx_tokens(prompt) > FINAL_INPUT_LIMIT_TOKENS or len(prompt) > FINAL_INPUT_LIMIT_CHARS:
                        # Minimal form: GT + question (+ tiny DATA excerpt), but NEVER truncate GT.
                        tiny_data = json.dumps(data_json, ensure_ascii=False, default=_json_default)
                        if len(tiny_data) > 8000:  # ~2k tokens
                            tiny_data = tiny_data[:8000] + "\n...[DATA_JSON truncated]"
                        prompt = (
                            FINAL_PROMPT_FULL
                            + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                            + "\n\nQuestion:\n" + str(question)
                            + glossary_block
                            + "\n\nDATA_JSON (trimmed for size only; use only if GT missing or tools failed):\n" + tiny_data
                            + "\n\nSTRICT RULES:\n"
                              "- Prefer numbers from GROUND_TRUTH_JSON when present; treat them as source of truth.\n"
                              "- If GROUND_TRUTH_JSON is empty or tools failed, you may use DATA_JSON and label as estimate.\n"
                              "- If GROUND_TRUTH_JSON conflicts with DATA_JSON, follow GROUND_TRUTH_JSON and note the mismatch briefly.\n"
                        )

                    if _approx_tokens(prompt) > FINAL_INPUT_LIMIT_TOKENS or len(prompt) > FINAL_INPUT_LIMIT_CHARS:
                        # Last resort: char slice TAIL; GT is at the top so it survives
                        prompt = _hard_slice_to_limit(prompt, FINAL_INPUT_LIMIT_TOKENS)
                        if len(prompt) > FINAL_INPUT_LIMIT_CHARS:
                            prompt = prompt[:FINAL_INPUT_LIMIT_CHARS] + "\n...[prompt hard-truncated at boundary]"

                    # --- PARANOID CLAMP (final guard before calling COMPLETE$V6) ---
                    if _approx_tokens(prompt) > FINAL_INPUT_LIMIT_TOKENS or len(prompt) > FINAL_INPUT_LIMIT_CHARS:
                        prompt = _hard_slice_to_limit(prompt, FINAL_INPUT_LIMIT_TOKENS)
                        if len(prompt) > FINAL_INPUT_LIMIT_CHARS:
                            prompt = prompt[:FINAL_INPUT_LIMIT_CHARS] + "\n...[prompt hard-truncated at boundary]"



                    q = "select snowflake.cortex.complete('" + str(CORTEX_LLM_MODEL) + "', $$" + prompt.replace("$$", "$ $") + "$$) as response"
                    df_resp = session.sql(q).to_pandas()
                    if df_resp is None or df_resp.empty:
                        return {"narrative": "No LLM response", "kpis": [], "chart": {}, "needs_more": False, "followup_prompt": "", "required": {}}
                    out = _coerce_json(df_resp.iloc[0, 0]) or {}
                    return {
                        "narrative": str(out.get("narrative") or "").strip(),
                        "kpis": out.get("kpis") or [],
                        "chart": out.get("chart") or {},
                        "needs_more": bool(out.get("needs_more")),
                        "followup_prompt": str(out.get("followup_prompt") or "").strip(),
                        "required": out.get("required") or {}
                    }
                except Exception as e:
                    try:
                        trace("final_single_exception", str(e)[:MAX_TRACE_PAYLOAD])
                    except Exception:
                        pass
                    return {
                        "narrative": f"Final LLM call failed: {e}",
                        "kpis": [],
                        "chart": {},
                        "needs_more": False,
                        "followup_prompt": "",
                        "required": {}
                    }

            # ---------------- MULTI-STEP (preview mode like original) ----------------
            # Build compact, guided previews to make the LLM write a full cross-view summary.
            preview: dict = {}
            frames_meta: dict = {}
            for name, df in (frames or {}).items():
                try:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        pv = _sample_df(df)
                        preview[name] = {
                            "columns": pv["columns"],
                            "rows": pv["rows"],
                            "shape": pv["shape"]
                        }
                        frames_meta[name] = {
                            "join_keys": pv.get("join_keys", []),
                            "time_cols": pv.get("time_cols", []),
                            "coverage": pv.get("coverage", {})
                        }
                except Exception:
                    continue

            # Trim preview JSON to a safe window (bigger than before to keep it “fuller”)
            try:
                preview_json = json.dumps(preview, default=_json_default)
                if len(preview_json) > 24000:
                    preview_json = preview_json[:24000]
            except Exception:
                preview_json = str(preview)[:24000]

            # Meta guide to help LLM align frames (join/time)
            try:
                meta_json = json.dumps(frames_meta, default=_json_default)
            except Exception:
                meta_json = str(frames_meta)

            PREVIEW_PROMPT_MULTI = """
You are a senior BI analyst. Use ONLY the data previews provided (multiple datasets).
Output JSON ONLY:
{
  "narrative": "markdown string",
  "kpis": [ {"title":"", "value":"", "sub":""} ],
  "chart": {
    "chart":"scatter|line|bar|table",
    "x":"<col>|null",
    "y":"<col>|null",
    "series":"<col>|null",
    "agg":"sum|avg|last|median|count",
    "allowed_chart_types":["line","bar","scatter","table"],
    "top_n": null
  }

  "needs_more": false,
  "followup_prompt": "",
  "required": {"views": [], "columns": {}}
}
Sections in narrative: Overall Summary, Key Findings, Drivers & Diagnostics, Recommendations.

MULTI-DATASET GUIDANCE:
- Treat each frame as a dataset; align insights by common join keys (e.g., RMID/RM_NAME/MANDATEID) and by time (MEET_YEAR/MEET_MON or date).
- Synthesize a single coherent story across views (e.g., how revenue and sentiment move together by RM over time).
- If X is time-like (DATE/DATETIME, YEARMONTH string, or integer months 1–12, optionally with YEAR), set "chart":"line" (not scatter) and sort X ascending. For comparisons like 2025 vs 2024 with MEET_MON, plot a multi-series line (series = YEAR).
- If the question contains multiple explicit years (e.g. "2024 vs 2025") or "vs", return a single comparison chart:
  • For rankings ("top/best/most/highest"): chart="bar", x="<RM or Mandate dimension>", y="<metric>", series="YEAR", top_n=the requested N (default 5).
  • For time-like X (months), prefer chart="line" with series="YEAR".
- Only use fields that exist in the frames; prefer YEAR/MEET_YEAR for series if present.
- Set "agg" consistent with metric conventions (AUM=snapshot/last, Scores=avg, flows=sum).

CHART GENERATION (Enhanced):
- Prefer derived/computed metrics (profit_margin, cost_to_revenue, aum_trend, revenue_trend) when available; they provide richer business insights.
- Classify available columns: ENTITIES (RMID, RM_NAME, MANDATEID) → use for x-axis or series, METRICS (aum, revenue, cost, sentiment, profit_margin) → use for y-axis, TIME (DATE, MONTH_DATE, MEET_YEAR, MEET_MON) → use for x-axis in trends.
- For bar charts (rankings/comparison): x=entity or category, y=metric (prefer derived metrics), series=optional time/category, top_n=5-10 to highlight top/bottom.
- For line charts (trends over time): x=time, y=metric, series=entity or category dimension; sort x ascending for temporal clarity.
- For scatter plots: x,y = two numeric metrics to show correlation or relationship.
- If reasoning context identifies flagged/at-risk entities, highlight them in top_n selection or narrative callout.
- Set top_n strategically: null for comprehensive views, 5-10 for focused rankings/comparisons, 3 for executive summaries.

- If question or data includes FILTER_* columns or explicit filters, assume they are already applied. Do NOT ask to add category fields.
FILTER/REASONER CONTEXT:
- Treat the dataset as already filtered by the query/reasoners. Do NOT generalize to the full population.
  Use language like "within the returned/filtered set" instead of "all mandates" unless coverage is explicit.
- Columns like signal_*, *_condition, *_score are derived outputs from rules/reasoners; describe them as flags/labels
  on the returned records, not as raw base measurements.
ANALYTICS GUARDRAILS:
- Do NOT invent correlations, regressions, or projections. Only mention those if GROUND_TRUTH_JSON or
  deterministic notes explicitly include them.

LOOP-CONTROL:
- If data is insufficient to answer confidently, set "needs_more": true and provide a concise "followup_prompt"
  telling exactly what to fetch (views/columns/time grain). Also set "required": {"views":[...], "columns":{"VIEW":[...]}}.
- Always request join-friendly keys in required columns when you need more (RMID/RM_NAME, MANDATEID, raw time column).
""".strip()

            gt_json = json.dumps({"resolved_tools": resolved_tools or []}, default=_json_default)

            reasoning_context_text = ""
            reasoner_evidence_text = ""

            today_str = datetime.date.today().isoformat()
            FINAL_PROMPT = (
                PREVIEW_PROMPT_MULTI
                + f"\n\nToday: {today_str}"
                + "\n\nQuestion:\n" + str(question)
                + "\n\nDataset preview JSON (trimmed):\n" + preview_json
                + "\n\nAlignment hints per frame:\n" + meta_json
                + glossary_block
                + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                + (
                    "\n\nDeterministic insights to consider (optional):\n"
                    + json.dumps(deterministic_insights, default=_json_default)
                    if deterministic_insights else ""
                  )
                + reasoning_context_text
                + reasoner_evidence_text
                + "\n\nSTRICT RULES:\n"
                  "- Prefer numbers from GROUND_TRUTH_JSON when present; treat them as source of truth.\n"
                  "- If GROUND_TRUTH_JSON is empty or tools failed, you may compute from preview data and label as estimate.\n"
                  "- If GROUND_TRUTH_JSON conflicts with preview data, follow GROUND_TRUTH_JSON and note the mismatch briefly.\n"
                  "- When a tool result has a scalar 'value', copy it verbatim (format to 2 decimals if needed).\n"
                  "- When a tool result is 'grouped', you may list top items by their given 'value' only.\n"
            )

            # --- HARD BUDGET GATE: Final (preview) prompt -------------------
            if _approx_tokens(FINAL_PROMPT) > FINAL_INPUT_LIMIT_TOKENS and glossary_block:
                glossary_block = ""
                FINAL_PROMPT = (
                    PREVIEW_PROMPT_MULTI
                    + "\n\nQuestion:\n" + str(question)
                    + "\n\nDataset preview JSON (trimmed):\n" + preview_json
                    + "\n\nAlignment hints per frame:\n" + meta_json
                    + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                    + (
                        "\n\nDeterministic insights to consider (optional):\n"
                        + json.dumps(deterministic_insights, default=_json_default)
                        if deterministic_insights else ""
                      )
                    + reasoning_context_text
                    + "\n\nSTRICT RULES:\n"
                      "- Prefer numbers from GROUND_TRUTH_JSON when present; treat them as source of truth.\n"
                      "- If GROUND_TRUTH_JSON is empty or tools failed, you may compute from preview data and label as estimate.\n"
                      "- If GROUND_TRUTH_JSON conflicts with preview data, follow GROUND_TRUTH_JSON and note the mismatch briefly.\n"
                      "- When a tool result has a scalar 'value', copy it verbatim (format to 2 decimals if needed).\n"
                      "- When a tool result is 'grouped', you may list top items by their given 'value' only.\n"
                )

            if _approx_tokens(FINAL_PROMPT) > FINAL_INPUT_LIMIT_TOKENS:
                # Drop deterministic notes first (GT stays!)
                FINAL_PROMPT = (
                    PREVIEW_PROMPT_MULTI
                    + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                    + "\n\nQuestion:\n" + str(question)
                    + "\n\nDataset preview JSON (trimmed):\n" + preview_json
                    + "\n\nAlignment hints per frame:\n" + meta_json
                    + glossary_block
                )

            if _approx_tokens(FINAL_PROMPT) > FINAL_INPUT_LIMIT_TOKENS:
                # Trim preview/meta ONLY (GT stays!)
                if len(preview_json) > 12_000:
                    preview_json = preview_json[:12_000] + "\n...[preview truncated]"
                if len(meta_json) > 6_000:
                    meta_json = meta_json[:6_000] + "\n...[meta truncated]"
                FINAL_PROMPT = (
                    PREVIEW_PROMPT_MULTI
                    + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                    + "\n\nQuestion:\n" + str(question)
                    + "\n\nDataset preview JSON (trimmed):\n" + preview_json
                    + "\n\nAlignment hints per frame:\n" + meta_json
                )

            if _approx_tokens(FINAL_PROMPT) > FINAL_INPUT_LIMIT_TOKENS:
                # Minimal form: GT + question (NO preview/meta/notes)
                FINAL_PROMPT = (
                    PREVIEW_PROMPT_MULTI
                    + "\n\nGROUND_TRUTH_JSON (do not ignore):\n" + gt_json
                    + "\n\nQuestion:\n" + str(question)
                )

            if _approx_tokens(FINAL_PROMPT) > FINAL_INPUT_LIMIT_TOKENS or len(FINAL_PROMPT) > FINAL_INPUT_LIMIT_CHARS:
                # Last resort: hard slice TAIL, GT is at the top so it survives
                FINAL_PROMPT = _hard_slice_to_limit(FINAL_PROMPT, FINAL_INPUT_LIMIT_TOKENS)
                if len(FINAL_PROMPT) > FINAL_INPUT_LIMIT_CHARS:
                    FINAL_PROMPT = FINAL_PROMPT[:FINAL_INPUT_LIMIT_CHARS] + "\n...[prompt hard-truncated at boundary]"

            # --- PARANOID CLAMP (final guard before calling COMPLETE$V6) ---
            if _approx_tokens(FINAL_PROMPT) > FINAL_INPUT_LIMIT_TOKENS or len(FINAL_PROMPT) > FINAL_INPUT_LIMIT_CHARS:
                FINAL_PROMPT = _hard_slice_to_limit(FINAL_PROMPT, FINAL_INPUT_LIMIT_TOKENS)
                if len(FINAL_PROMPT) > FINAL_INPUT_LIMIT_CHARS:
                    FINAL_PROMPT = FINAL_PROMPT[:FINAL_INPUT_LIMIT_CHARS] + "\n...[prompt hard-truncated at boundary]"


            try:
                trace("final_prompt_multi", FINAL_PROMPT[:MAX_TRACE_PAYLOAD])
            except Exception:
                pass

            # Call Cortex, with robust JSON coercion
            try:
                q = "select snowflake.cortex.complete('" + str(CORTEX_LLM_MODEL) + "', $$" + FINAL_PROMPT.replace("$$", "$ $") + "$$) as response"
                df_resp = session.sql(q).to_pandas()
                if df_resp is None or df_resp.empty:
                    return {"narrative": "No LLM response", "kpis": [], "chart": {}, "needs_more": False, "followup_prompt": "", "required": {}}
                raw = df_resp.iloc[0, 0]
                plan = _coerce_json(raw)

                out = {
                    "narrative": str(plan.get("narrative") or "").strip(),
                    "kpis": plan.get("kpis") or [],
                    "chart": plan.get("chart") or {},
                    "needs_more": bool(plan.get("needs_more")),
                    "followup_prompt": str(plan.get("followup_prompt") or "").strip(),
                    "required": plan.get("required") or {}
                }

                # Never blank: if narrative is empty, add a safe, informative synthesis from previews
                if not out["narrative"]:
                    lines = ["### Overall Summary"]
                    # Surface per-frame top metric glimpses to avoid bare response
                    for name, df in (frames or {}).items():
                        if not isinstance(df, pd.DataFrame) or df.empty:
                            continue
                        # pick a numeric column to highlight
                        nums = _pick_numeric_cols(df, k=1)
                        label = "RM_NAME" if "RM_NAME" in df.columns else ("RMID" if "RMID" in df.columns else None)
                        if nums and label:
                            try:
                                top = df[[label, nums[0]]].dropna().sort_values(nums[0], ascending=False).head(3)
                                items = ", ".join(f"{r[label]} ({float(r[nums[0]]):.3f})" for _, r in top.iterrows())
                                lines.append(f"- **{name}:** top by **{nums[0]}** → {items}")
                            except Exception:
                                lines.append(f"- **{name}:** {len(df):,} rows, {len(df.columns):,} cols")
                        else:
                            lines.append(f"- **{name}:** {len(df):,} rows, {len(df.columns):,} cols")
                    out["narrative"] = "\n".join(lines)
                    if not out["chart"]:
                        out["chart"] = {"chart": "table", "x": None, "y": None}
                return out

            except Exception as e:
                try:
                    trace("final_exception_multi", str(e)[:MAX_TRACE_PAYLOAD])
                except Exception:
                    pass
                return {
                    "narrative": f"Final LLM call failed: {e}",
                    "kpis": [],
                    "chart": {},
                    "needs_more": False,
                    "followup_prompt": "",
                    "required": {}
                }

        def _human_time(step: Dict[str, Any]) -> str:
            t = (step or {}).get("time") or {}
            tf = t.get("field"); tg = (t.get("grain") or "").lower().strip() if isinstance(t.get("grain"), str) else ""
            filters = (step or {}).get("filters") or []
            # Try to extract simple month/year hints from filters (best effort)
            filtxt = []
            for f in filters:
                try:
                    fld = f.get("field"); op = f.get("op") or ""; val = f.get("value")
                    if isinstance(val, str):
                        v = val.replace("DATEADD", "DATEADD(...)").replace("CURRENT_DATE", "today")
                    else:
                        v = val
                    if fld and val is not None:
                        filtxt.append(f"{fld} {op} {v}")
                except Exception:
                    continue
            if tf and tg:
                base = f"{tf}@{tg}"
            elif tf:
                base = f"{tf}"
            else:
                base = ""
            if filtxt:
                return (base + " | " if base else "") + "; ".join(filtxt)
            return base

        def _step_to_human(step: Dict[str, Any]) -> str:
            vid = step.get("id") or "step"
            view = step.get("view") or "view"
            m = (step.get("metric") or {})
            metric = m.get("field") or "metric"
            mode = (m.get("mode") or "").lower() or "sum"
            d = (step.get("dim") or {})
            dim = d.get("field")
            order = (step.get("order") or "desc").lower()
            limit = step.get("limit") or 500
            also = step.get("select_also") or []
            time_desc = _human_time(step)
            rank_txt = ""
            if "top" in order or (order == "desc" and isinstance(limit, int) and limit <= 10):
                rank_txt = f" (top {limit})"
            elif order == "asc" and isinstance(limit, int) and limit <= 10:
                rank_txt = f" (bottom {limit})"
            parts = []
            parts.append(f"{vid}: {metric} ({mode})")
            if dim:
                parts.append(f"by {dim}")
            parts.append(f"in {view}")
            if time_desc:
                parts.append(f"[{time_desc}]")
            if rank_txt:
                parts.append(rank_txt)
            if also:
                parts.append(f"+ keys: {', '.join(also[:6])}{'…' if len(also) > 6 else ''}")
            return " ".join(parts)

        def _friendly_label(name: str) -> str:
            import re
            if not name:
                return ""
            n = str(name)
            u = n.upper()

            # Common metric field friendly names
            metric_map = {
                "REVENUE": "revenue",
                "CURRENT_REVENUE": "revenue",
                "SENTIMENT": "meeting sentiment",
                "MEETING_SENTIMENT": "meeting sentiment",
                "PERFORMANCE": "performance score",
                "AUM": "assets under management",
                "CURRENT_AUM": "assets under management",
                "COMMITMENTS": "commitments",
                "TOPUP": "top-ups",
                "PROFIT": "profit",
                "MEETING_COUNT": "meeting count",
            }
            for k, v in metric_map.items():
                if k in u:
                    return v

            # Common dimensions / keys
            dim_map = {
                "RMID": "relationship manager",
                "RM_ID": "relationship manager",
                "RM_NAME": "relationship manager",
                "MANDATEID": "mandate",
                "MANDATE_ID": "mandate",
                "MONTH_DATE": "month",
                "MEETINGDATE": "meeting month",
                "MEET_YEAR": "year",
                "YEAR": "year",
            }
            for k, v in dim_map.items():
                if u == k:
                    return v

            # Otherwise prettify identifier
            n = re.sub(r"[_\W]+", " ", n).strip()
            return n.lower()

        def _friendly_view(view_name: str) -> str:
            if not view_name:
                return "data"
            # take last identifier, prettify
            last = str(view_name).split(".")[-1]
            pretty = last.replace("_", " ").strip()
            if pretty:
                return pretty.title()
            return "data"

        def _friendly_time_phrase(step: Dict[str, Any]) -> str:
            """Turn step['time'] + filters into human wording like 'for Aug 2025', 'last 12 months', 'by year in 2024', etc."""
            import re
            from calendar import month_abbr

            def mmmyyyy(y, m):
                try:
                    return f"{month_abbr[int(m)]} {int(y)}"
                except Exception:
                    return None

            t = (step or {}).get("time") or {}
            grain = (t.get("grain") or "").lower().strip() if isinstance(t.get("grain"), str) else ""
            filters = (step or {}).get("filters") or []

            # Look for explicit month IN (...)
            joined = " ".join([f"{f.get('field','')} {f.get('op','')} {f.get('value','')}" for f in filters if isinstance(f, dict)])
            # TO_DATE('YYYY-MM-01')
            dates = re.findall(r"TO_DATE\('(\d{4})-(\d{2})-01'\)", joined)
            if dates:
                if len(dates) == 1:
                    y, m = dates[0]
                    label = mmmyyyy(y, m)
                    return f"for {label}" if label else "for the selected month"
                # try compact range if same year and consecutive-ish
                months = [(int(y), int(m)) for (y, m) in dates]
                years = {y for (y, _) in months}
                if len(years) == 1:
                    months_only = [m for (_, m) in months]
                    mn, mx = min(months_only), max(months_only)
                    if mx >= mn:
                        s = mmmyyyy(list(years)[0], mn)
                        e = mmmyyyy(list(years)[0], mx)
                        if s and e:
                            return f"from {s} to {e}"
                # fallback list
                labels = [mmmyyyy(y, m) for (y, m) in dates if mmmyyyy(y, m)]
                if labels:
                    if len(labels) == 2:
                        return f"for {labels[0]} and {labels[1]}"
                    return "for " + ", ".join(labels[:4]) + ("…" if len(labels) > 4 else "")

            # Look for YEAR = 2025 or EXTRACT(YEAR FROM …) = 2025
            year_m = re.findall(r"\b(?:MEET_YEAR|YEAR)\s*=\s*(20\d{2})\b", joined)
            if year_m:
                y = year_m[0]
                return f"in {y}"

            # Look for last N months
            if re.search(r"DATEADD\(\s*month\s*,\s*-\d+\s*,\s*CURRENT_DATE\s*\)", joined, re.I):
                m = re.search(r"DATEADD\(\s*month\s*,\s*-(\d+)\s*,", joined, re.I)
                n = m.group(1) if m else "12"
                return f"for the last {n} months"

            # Grain-only hints
            if grain == "year":
                return "by year"
            if grain == "month":
                return "by month"

            return ""

        def _friendly_rank_phrase(order: Optional[str], limit: Optional[int]) -> str:
            o = (order or "").lower()
            n = int(limit) if isinstance(limit, int) else None
            if o in ("asc", "ascending"):
                return f"(bottom {n})" if n else "(bottom)"
            if o in ("desc", "descending", ""):
                return f"(top {n})" if n else "(top)"
            return ""

        def _step_to_friendly_sentence(step: Dict[str, Any]) -> str:
            """e.g., 'Revenue by mandate for the last 12 months (top 5)'"""
            view = _friendly_view(step.get("view"))
            m = (step.get("metric") or {})
            metric = _friendly_label(m.get("field") or "metric")
            mode = (m.get("mode") or "").lower().strip() or "sum"
            d = (step.get("dim") or {})
            dim = _friendly_label(d.get("field") or "")
            time_phrase = _friendly_time_phrase(step)
            rank = _friendly_rank_phrase(step.get("order"), step.get("limit"))

            # Mode hints in parentheses only when helpful
            mode_hint = ""
            if mode in ("avg", "median", "count"):
                mode_hint = f" ({mode})"

            pieces = []
            # Metric + mode
            pieces.append(f"{metric}{mode_hint}".strip())
            # Dimension
            if dim:
                pieces.append(f"by {dim}")
            # Time phrase
            if time_phrase:
                pieces.append(time_phrase)
            # Rank
            if rank:
                pieces.append(rank)
            # View context (light)
            if view and view.lower() not in ("data",):
                pieces.append(f"from {view}")

            # Build sentence
            s = " ".join(pieces).strip()
            # Uppercase first letter
            return s[0].upper() + s[1:] if s else ""
 
        def _json_default(o):
            import datetime as dt, numpy as np
            try:
                import pandas as pd
            except Exception:
                class pd: Timestamp = ()  # no-op if pandas missing

            if isinstance(o, (dt.datetime, dt.date)):
                return o.isoformat()
            if hasattr(pd, "Timestamp") and isinstance(o, pd.Timestamp):
                return o.isoformat()
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)):    return bool(o)
            if isinstance(o, (set, tuple)):   return list(o)
            return str(o)


        def orchestrate(
            question: str,
            views: List[str],
            *,
            months_window: int = DEFAULT_MONTHS_WINDOW,
            widen_time_on_insufficient: bool = True,
        ) -> Dict[str, Any]:
            """
            Planner → execute → link → summarize (replan up to 2x).
            Clean UI:
              • Single card, single stage row (updated in-place)
              • Natural “What I’ll fetch” list (no bracketed timeframe)
              • Minimal calm copy: Planning → Fetching → Linking → Summarizing
              • Keeps existing data logic (RM lookup, merging, follow-up loops)
            """
            import re
            import pandas as pd

            # ---------- helpers (wording & formatting) ----------
            def _friendly_view_name(v: str) -> str:
                base = (v or "").split(".")[-1]
                base = base.replace("_", " ").strip()
                titled = base.title()
                titled = re.sub(r"\bRm\b", "RM", titled)
                titled = re.sub(r"\bAum\b", "AUM", titled)
                return titled or v

            def _nice_metric_name(tok: str) -> str:
                if not tok:
                    return "metric"
                t = tok.replace("_", " ").strip()
                up = t.upper()
                if up in {"AUM", "NNM", "NAV", "P&L"}:
                    return up
                return t.lower()

            def _mode_label(mode: str) -> str:
                m = (mode or "").strip().lower()
                return m if m in {"snapshot", "last", "avg", "sum", "median", "count"} else ""

            def _dim_phrase(step: Dict[str, Any]) -> str:
                d = ((step or {}).get("dim") or {}).get("field")
                if not d:
                    return ""
                du = str(d).upper()
                if du in {"RMID", "RM_ID", "RM_NAME"}:
                    return " by RM"
                if du in {"MANDATEID", "MANDATE_ID"}:
                    return " by mandate"
                return f" by {d.replace('_', ' ').lower()}"

            def _step_to_friendly_sentence(step: Dict[str, Any]) -> str:
                metric = ((step or {}).get("metric") or {}).get("field")
                mode = ((step or {}).get("metric") or {}).get("mode", "")
                view = (step or {}).get("view", "")
                mname = _nice_metric_name(str(metric))
                mlabel = _mode_label(str(mode))
                dimtxt = _dim_phrase(step)
                left = f"{mname}" + (f" ({mlabel})" if mlabel else "")
                return f"• {left}{dimtxt} — {_friendly_view_name(view)}"

            def _qualify_table(name: str) -> str:
                if not name:
                    return name
                if "." in name:
                    return name
                db = os.getenv("SNOWFLAKE_DATABASE")
                schema = os.getenv("SNOWFLAKE_SCHEMA")
                if db and schema:
                    return f"{db}.{schema}.{name}"
                return name

            def _infer_time_source(views_in: List[str]) -> Optional[str]:
                env_tbl = os.getenv("AI_INSIGHTS_TIME_TABLE")
                if env_tbl:
                    return _qualify_table(env_tbl)
                if views_in:
                    v0 = str(views_in[0])
                    parts = v0.split(".")
                    base = parts[-1]
                    if base.upper().endswith("_MODEL"):
                        parts[-1] = base[:-6]
                        return ".".join(parts)
                return None

            def _latest_year_from_source(source_table: str, fields_upper: set) -> Optional[int]:
                if not source_table:
                    return None
                cache = st.session_state.setdefault("latest_year_cache", {})
                cache_key = source_table.upper()
                if cache_key in cache:
                    return cache[cache_key]
                sql = None
                if "MEET_YEAR" in fields_upper:
                    sql = f"select max(MEET_YEAR) as Y from {source_table}"
                elif "YEAR" in fields_upper:
                    sql = f"select max(YEAR) as Y from {source_table}"
                elif "MONTH_DATE" in fields_upper:
                    sql = f"select year(max(MONTH_DATE)) as Y from {source_table}"
                if not sql:
                    return None
                try:
                    df = session.sql(sql).to_pandas()
                    if df is None or df.empty:
                        return None
                    yval = df.iloc[0, 0]
                    if yval is None or (hasattr(pd, "isna") and pd.isna(yval)):
                        return None
                    yr = int(yval)
                    cache[cache_key] = yr
                    return yr
                except Exception:
                    return None

            def _maybe_append_default_year(q: str, views_in: List[str], cat_in: Dict[str, Any]) -> str:
                if not q or re.search(r"assume\s+year", q, re.I):
                    return q
                hints = _extract_time_hints(q)
                if hints.get("explicit_years") or not hints.get("month_only"):
                    return q
                source_table = _infer_time_source(views_in)
                fields = (cat_in.get(views_in[0], {}).get("fields") or []) if views_in else []
                fields_upper = {str(f).upper() for f in fields}
                yr = _latest_year_from_source(source_table, fields_upper)
                if yr:
                    new_q = f"{q} (assume year {yr})"
                    trace("question_augmented", {"from": q, "to": new_q})
                    return new_q
                return q

            use_rai_dynamic = os.environ.get("AI_INSIGHTS_RAI_DYNAMIC", "1").strip().lower() not in ("0", "false", "no", "off")
            skip_describe_env = os.environ.get("AI_INSIGHTS_SKIP_DESCRIBE", "").strip().lower()
            skip_describe = skip_describe_env in ("1", "true", "yes", "on") or (not skip_describe_env and use_rai_dynamic)

            # ---------- catalog ----------
            if skip_describe:
                cat = {v: {"describe_ok": False, "fields": [], "types": {}, "raw": None, "tried": []} for v in (views or [])}
                try:
                    trace("catalog_skipped", {"reason": "AI_INSIGHTS_SKIP_DESCRIBE or RAI dynamic mode"})
                except Exception:
                    pass
            else:
                cat = load_catalog(views)
            try:
                question = _maybe_append_default_year(question, views, cat)
            except Exception:
                pass
            any_fields = any(cat.get(v, {}).get("fields") for v in views)
            if not any_fields and not use_rai_dynamic:
                return {
                    "question": question,
                    "frames": {},
                    "plan": {},
                    "insights": "Schema discovery returned no fields. Check privileges/view names.",
                    "kpis": [],
                    "chart": {},
                    "log": ["No fields discovered"],
                    "sqls": [],
                    "followups": [],
                    "catalog": cat
                }
            if not any_fields and use_rai_dynamic:
                try:
                    trace("catalog_warning", {"reason": "no fields in semantic catalog; continuing with registry-driven RAI"})
                except Exception:
                    pass

            use_analyst_only = os.environ.get("AI_INSIGHTS_ANALYST_ONLY", "0").strip() == "1"
            if use_analyst_only:
                log = []
                sqls = []
                merges = []

                parsed = analyst_message(views, question)
                if not parsed or not parsed.get("sql"):
                    msg = (parsed or {}).get("explanation") or "No SQL returned from Analyst."
                    return {
                        "question": question,
                        "frames": {},
                        "plan": {},
                        "insights": msg,
                        "kpis": [],
                        "chart": {},
                        "log": log,
                        "sqls": sqls,
                        "merges": merges,
                        "followups": [],
                        "catalog": cat
                    }

                ok, df, err = analyst_query_execute(parsed["sql"])
                if not ok:
                    return {
                        "question": question,
                        "frames": {},
                        "plan": {},
                        "insights": f"Query failed: {err}",
                        "kpis": [],
                        "chart": {},
                        "log": log,
                        "sqls": [("s1", parsed["sql"])],
                        "merges": merges,
                        "followups": [],
                        "catalog": cat
                    }

                frames = {"s1": df}
                def _ui_progress(msg: str):
                    try:
                        _body([msg])
                    except Exception:
                        pass
                summary = summarize_with_tools(question, frames, deterministic_insights=[], extra_refs=None, on_tool=_ui_progress)
                sqls.append(("s1", parsed["sql"]))
                
                # Generate chart using LLM - OVERRIDE the spec from summarizer with actual Plotly figure
                chart_result = {}
                try:
                    frames_src = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame) and not v.empty}
                    if frames_src:
                        # Call compute_llm_chart to generate actual Plotly visualization
                        llm_chart_data = compute_llm_chart(frames_src, question, summary.get("narrative", ""))
                        if llm_chart_data and isinstance(llm_chart_data, dict):
                            fig = llm_chart_data.get("fig")
                            if fig is not None:
                                try:
                                    # Convert figure to dict with data and layout
                                    chart_fig_dict = fig.to_dict()
                                    chart_result = {
                                        "data": chart_fig_dict.get("data", []),
                                        "layout": chart_fig_dict.get("layout", {}),
                                    }
                                except Exception:
                                    pass
                except Exception:
                    pass
                
                return {
                    "question": question,
                    "frames": frames,
                    "plan": {},
                    "insights": summary.get("narrative", "") if isinstance(summary, dict) else "",
                    "kpis": summary.get("kpis", []) if isinstance(summary, dict) else [],
                    "chart": chart_result if chart_result else (summary.get("chart", {}) if isinstance(summary, dict) else {}),
                    "needs_more": summary.get("needs_more", False) if isinstance(summary, dict) else False,
                    "followup_prompt": summary.get("followup_prompt", "") if isinstance(summary, dict) else "",
                    "required": summary.get("required", {}) if isinstance(summary, dict) else {},
                    "summary_obj": summary if isinstance(summary, dict) else {},
                    "log": log,
                    "sqls": sqls,
                    "merges": merges,
                    "followups": [],
                    "catalog": cat
                }

            # ---------- single, modern card UI with fixed placeholders ----------
            try:
                # CSS once
                st.markdown(
                    """
                    <style>
                      .ai-card{background:#fff;border:1px solid #e6e9ef;border-radius:14px;padding:14px 16px;
                               box-shadow:0 4px 14px rgba(17,24,39,0.04);margin-top:6px;}
                      .ai-title{font-size:16px;font-weight:600;color:#0f172a;margin:0 0 8px 0;display:flex;gap:.5rem;align-items:center;}
                      .ai-stage{display:flex;gap:8px;margin:6px 0 10px 0;flex-wrap:wrap}
                      .ai-pill{padding:4px 10px;border:1px solid #e6e9ef;border-radius:999px;font-size:12px;color:#475569;background:#f8fafc;}
                      .ai-pill.on{border-color:#2563eb33;background:linear-gradient(180deg,#eef6ff,#e8f0ff);color:#1e3a8a;}
                      .ai-body{font-size:13px;color:#334155;line-height:1.55;}
                      .ai-muted{color:#64748b;}
                      .ai-list{margin:6px 0 0 0}
                      .ai-list li{margin:4px 0}
                      .ai-loader{width:16px;height:16px;border-radius:50%;border:3px solid #c7d2fe;border-top-color:#4f46e5;animation:ai-spin 1s linear infinite}
                      @keyframes ai-spin{to{transform:rotate(360deg)}}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # One card with three placeholders we update in-place
                card = st.container()
                ph_header = card.empty()
                ph_stage = card.empty()
                ph_body = card.empty()

                def _render_header():
                    ph_header.markdown(
                        f"<div class='ai-card'><div class='ai-title'>"
                        f"<div class='ai-loader'></div><div>🧭 Looking into:&nbsp;<b>{question}</b></div>"
                        f"</div></div>",
                        unsafe_allow_html=True
                    )

                def _render_stage(active: str):
                    items = [("planning", "Planning"), ("fetching", "Fetching"), ("linking", "Linking"), ("summarizing", "Summarizing")]
                    pills = []
                    for key, label in items:
                        cls = "ai-pill on" if key == active else "ai-pill"
                        pills.append(f"<div class='{cls}'>{label}</div>")
                    ph_stage.markdown(
                        "<div class='ai-card' style='padding-top:0'><div class='ai-stage'>" + "".join(pills) + "</div></div>",
                        unsafe_allow_html=True
                    )

                def _body(lines: list[str]):
                    html = "<div class='ai-card' style='padding-top:0'><div class='ai-body'>" + "<br/>".join(lines) + "</div></div>"
                    ph_body.markdown(html, unsafe_allow_html=True)

                _render_header()
                _render_stage("planning")
                _body(["<span class='ai-muted'>Figuring out which data to use…</span>"])
            except Exception:
                # headless fallback
                def _render_stage(_a: str): pass
                def _body(_l: list[str]): pass

            if use_rai_dynamic:
                def _cortex_complete_text(prompt: str) -> str:
                    q = "select snowflake.cortex.complete('{model}', $${body}$$) as response".format(
                        model=CORTEX_LLM_MODEL,
                        body=prompt.replace("$$", "$ $"),
                    )
                    try:
                        df_resp = session.sql(q).to_pandas()
                        if df_resp is None or df_resp.empty:
                            return ""
                        return str(df_resp.iloc[0, 0] or "")
                    except Exception:
                        return ""

                _render_stage("planning")
                try:
                    rebuild_token = os.environ.get("AI_INSIGHTS_FORCE_KG_REBUILD", "").strip()
                    builder, specs = _rai_builder_and_specs(rebuild_token=rebuild_token)
                except Exception as exc:
                    detail = getattr(exc, "raw_content", None) or getattr(exc, "content", None)
                    detail_text = f"\n{detail}" if detail else ""
                    return {
                        "question": question,
                        "frames": {},
                        "plan": {},
                        "insights": f"RAI initialization failed: {exc}{detail_text}",
                        "kpis": [],
                        "chart": {},
                        "log": [f"RAI initialization failed: {exc}"] + ([detail] if detail else []),
                        "sqls": [],
                        "merges": [],
                        "followups": [],
                        "catalog": cat,
                    }
                if not specs:
                    return {
                        "question": question,
                        "frames": {},
                        "plan": {},
                        "insights": "RAI ontology not available.",
                        "kpis": [],
                        "chart": {},
                        "log": ["RAI ontology not available"],
                        "sqls": [],
                        "merges": [],
                        "followups": [],
                        "catalog": cat,
                    }

                ontology_text = render_ai_insights_ontology_text(specs)
                join_keys = infer_join_keys(specs)

                max_query_attempts = int(os.environ.get("AI_INSIGHTS_RAI_DYNAMIC_QUERY_RETRIES", "3"))
                lqp_fallback_done = False
                failure_context = None
                spec = None
                specs_to_run = []
                reasoner_ids = []
                frames = {}
                reasoner_contexts = {}
                reasoner_results = {}
                last_exc = None
                last_err_detail = None
                last_err_trace = None
                log_items = []
                graph_enabled = os.environ.get("AI_INSIGHTS_ENABLE_GRAPH_REASONERS", "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
                graph_spec = None
                graph_df = None
                graph_label = None
                graph_primary = False
                graph_from_reasoner = False
                graphs = None
                reasoners_enabled = os.environ.get("AI_INSIGHTS_ENABLE_REASONERS", "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
                inline_reasoners = os.environ.get("AI_INSIGHTS_INLINE_REASONERS", "1").strip().lower() not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
                if graph_enabled:
                    try:
                        kg_spec = load_kg_spec() or {}
                        graphs = kg_spec.get("graphs") if isinstance(kg_spec, dict) else None
                        try:
                            graph_count = len(graphs) if isinstance(graphs, list) else 0
                            log_items.append(f"[GRAPH_ROUTER] registry graphs={graph_count}")
                        except Exception:
                            log_items.append("[GRAPH_ROUTER] registry graphs=<unknown>")
                        if isinstance(graphs, list):
                            graph_spec, graph_reasoning = _select_graph_for_question_llm(
                                question, graphs, _cortex_complete_text
                            )
                            if graph_spec is None:
                                fallback_enabled = os.environ.get(
                                    "AI_INSIGHTS_GRAPH_ROUTER_FALLBACK", "1"
                                ).strip().lower() not in ("0", "false", "no", "off")
                                if fallback_enabled:
                                    graph_spec = _select_graph_for_question(question, graphs)
                                    if graph_spec is not None:
                                        log_items.append(
                                            f"[GRAPH_ROUTER] fallback selected graph={graph_spec.get('id')}"
                                        )
                                else:
                                    log_items.append("[GRAPH_ROUTER] fallback disabled; no graph selected")
                            else:
                                log_items.append(
                                    f"[GRAPH_ROUTER] llm selected graph={graph_spec.get('id')}; reasoning={graph_reasoning}"
                                )
                            if graph_spec and _is_graph_intent(question):
                                graph_primary = True
                    except Exception:
                        graph_spec = None
                else:
                    log_items.append("[GRAPH_ROUTER] graph routing disabled (AI_INSIGHTS_ENABLE_GRAPH_REASONERS=0)")
                if graph_enabled and graph_spec is None:
                    log_items.append("[GRAPH_ROUTER] no graph selected")
                if graph_enabled and graph_spec is not None:
                    log_items.append(f"[GRAPH_ROUTER] graph_primary={graph_primary}")
                if graph_primary:
                    exclusive = os.environ.get("AI_INSIGHTS_GRAPH_PRIMARY_EXCLUSIVE", "0").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                    )
                    if exclusive:
                        reasoners_enabled = False
                        inline_reasoners = False
                if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_ROUTER", "").strip().lower() in ("1", "true", "yes"):
                    import sys
                    for item in log_items:
                        if isinstance(item, str) and item.startswith("[GRAPH_ROUTER]"):
                            print(item, file=sys.stderr)

                def _shorten(text: str, limit: int = 1200) -> str:
                    if not text:
                        return ""
                    if len(text) <= limit:
                        return text
                    return text[:limit] + " ...[truncated]"

                def _spec_dump(spec_obj: object) -> str:
                    try:
                        return json.dumps(spec_obj, indent=2, sort_keys=True, default=str)
                    except Exception:
                        return str(spec_obj)

                def _spec_diff(label: str, before: object, after: object) -> None:
                    if os.environ.get("RAI_DEBUG_SPEC_DIFF", "").lower() not in ("1", "true", "yes"):
                        return
                    import sys
                    import difflib
                    before_s = _spec_dump(before)
                    after_s = _spec_dump(after)
                    diff = difflib.unified_diff(
                        before_s.splitlines(),
                        after_s.splitlines(),
                        fromfile=f"{label}:before",
                        tofile=f"{label}:after",
                        lineterm="",
                    )
                    print("\n".join(diff), file=sys.stderr)

                def _summarize_reasoner_results(results: dict, max_sources: int = 2, max_records: int = 3) -> str:
                    if not isinstance(results, dict) or not results:
                        return ""
                    summary = {}
                    for step_id, rr in results.items():
                        if not isinstance(rr, dict):
                            continue
                        step_out = {}
                        for rid, entry in rr.items():
                            drill = (entry or {}).get("drilldowns") or {}
                            if "steps" in drill:
                                steps_out = {}
                                for sid, payload in (drill.get("steps") or {}).items():
                                    rows = (payload or {}).get("rows") or []
                                    samples = []
                                    for row_block in rows[:max_sources]:
                                        recs = (row_block or {}).get("records") or []
                                        if recs:
                                            samples.extend(recs[:max_records])
                                    steps_out[sid] = {
                                        "sources": len(rows),
                                        "sample_records": samples[:max_records],
                                    }
                                step_out[rid] = {"steps": steps_out}
                            elif "machines" in drill:
                                step_out[rid] = {"machines": len(drill.get("machines") or [])}
                            elif "audits" in drill:
                                step_out[rid] = {"audits": len(drill.get("audits") or [])}
                        if step_out:
                            summary[step_id] = step_out
                    if not summary:
                        return ""
                    try:
                        return json.dumps(summary, default=_json_default, indent=2)
                    except Exception:
                        return str(summary)

                def _summarize_graph_frame(df: object, label: str | None, max_rows: int = 10) -> str:
                    try:
                        import pandas as pd
                        if not isinstance(df, pd.DataFrame) or df.empty:
                            return ""
                        graph_id = label or "graph"
                        sample = df.head(max_rows).to_dict(orient="records")
                        payload = {"graph_id": graph_id, "rows": sample}
                        return f"GRAPH_RESULTS: {json.dumps(payload, default=_json_default)}"
                    except Exception:
                        return ""

                def _collect_unresolved_types(spec_in: dict, specs_list: list) -> list[str]:
                    if not isinstance(spec_in, dict):
                        return ["spec is not a dict"]
                    binds = spec_in.get("bind") or []
                    alias_to_entity = {}
                    for b in binds:
                        if isinstance(b, dict) and b.get("alias") and b.get("entity"):
                            alias_to_entity[b["alias"]] = b["entity"]
                    field_name_by_entity = {}
                    derived_by_entity = {}
                    expr_to_field_by_entity = {}
                    types_by_entity = {}
                    for s in specs_list or []:
                        try:
                            fields = list(s.fields or [])
                            derived = list(getattr(s, "derived_fields", []) or [])
                            field_name_by_entity[s.name] = {str(f).lower(): f for f in (fields + derived)}
                            derived_by_entity[s.name] = set(derived)
                            exprs = dict(getattr(s, "field_exprs", {}) or {})
                            expr_to_field_by_entity[s.name] = {
                                str(expr).lower(): field for field, expr in (exprs or {}).items() if expr
                            }
                            types_by_entity[s.name] = dict(getattr(s, "field_types", {}) or {})
                        except Exception:
                            continue

                    def _canonical_field(ent: str | None, prop: str | None) -> str:
                        if not ent or not prop:
                            return ""
                        prop_norm = str(prop).lower()
                        canonical = (field_name_by_entity.get(ent, {}) or {}).get(prop_norm)
                        if canonical:
                            return canonical
                        if prop_norm in (derived_by_entity.get(ent, set()) or set()):
                            return prop_norm
                        return (expr_to_field_by_entity.get(ent, {}) or {}).get(prop_norm, "")

                    def _dtype_for(ent: str | None, prop: str | None) -> str:
                        if not ent or not prop:
                            return ""
                        dtype = (types_by_entity.get(ent, {}) or {}).get(prop)
                        return str(dtype or "")

                    unresolved = []

                    def _check_term(term: dict, context: str):
                        if not isinstance(term, dict):
                            return
                        alias = term.get("alias")
                        prop = term.get("prop")
                        if not alias or not prop:
                            return
                        ent = alias_to_entity.get(alias)
                        if not ent:
                            unresolved.append(f"{context}: {alias}.{prop} (unknown alias)")
                            return
                        canonical = _canonical_field(ent, prop)
                        if not canonical:
                            unresolved.append(f"{context}: {alias}.{prop} (unknown field)")
                            return
                        dtype = _dtype_for(ent, canonical)
                        if not dtype or str(dtype).lower() in ("any", "unknown", "unresolved"):
                            unresolved.append(
                                f"{context}: {alias}.{prop} (unresolved type: {dtype or 'Any'})"
                            )

                    def _walk_pred(node: dict, context: str):
                        if not isinstance(node, dict):
                            return
                        if "op" in node:
                            _check_term(node.get("left"), context)
                            _check_term(node.get("right"), context)
                            return
                        if "between" in node:
                            spec = node.get("between") or {}
                            _check_term(spec.get("left"), context)
                            _check_term(spec.get("low"), context)
                            _check_term(spec.get("high"), context)
                            return
                        if "in" in node:
                            spec = node.get("in") or {}
                            _check_term(spec.get("left"), context)
                            _check_term(spec.get("right"), context)
                            return
                        if "not" in node:
                            _walk_pred(node.get("not"), context)
                            return
                        if "and" in node:
                            for child in node.get("and") or []:
                                _walk_pred(child, context)
                            return
                        if "or" in node:
                            for child in node.get("or") or []:
                                _walk_pred(child, context)
                            return

                    for item in spec_in.get("select") or []:
                        _check_term(item, "select")
                    for item in spec_in.get("group_by") or []:
                        _check_term(item, "group_by")
                    for item in spec_in.get("aggregations") or []:
                        term = item.get("term")
                        if isinstance(term, dict):
                            _check_term(term, "aggregation")
                    for pred in spec_in.get("where") or []:
                        _walk_pred(pred, "where")
                    for item in spec_in.get("order_by") or []:
                        term = item.get("term")
                        if isinstance(term, dict):
                            _check_term(term, "order_by")

                    return unresolved

                def _run_query_with_lqp_fallback(exec_spec: dict):
                    nonlocal builder, lqp_fallback_done
                    try:
                        return run_dynamic_query(builder, exec_spec)
                    except Exception as retry_exc:
                        if (not lqp_fallback_done) and _is_lqp_internal_error(retry_exc):
                            builder, switched = _switch_builder_to_legacy(builder, log_items)
                            if switched:
                                lqp_fallback_done = True
                                return run_dynamic_query(builder, exec_spec)
                        raise

                for attempt in range(max_query_attempts):
                    if attempt > 0:
                        _render_stage("planning")
                        _body([f"Retrying RAI planning after query failure (attempt {attempt + 1}/{max_query_attempts})..."])

                    spec, reasoner_ids = generate_dynamic_query_spec(
                        _cortex_complete_text,
                        question,
                        ontology_text,
                        join_keys,
                        failure_context=failure_context,
                    )
                    graph_reasoners = _graph_reasoners_from_ids(reasoner_ids or [])
                    graph_reasoner_ids = [r.id for r in graph_reasoners if getattr(r, "id", None)]
                    inline_reasoner_ids = [r for r in (reasoner_ids or []) if r not in graph_reasoner_ids]
                    if graph_reasoners:
                        graph_from_reasoner = True
                        selected_graph_id = str(getattr(graph_reasoners[0], "graph_id", "") or "").strip()
                        if selected_graph_id and isinstance(graphs, list):
                            graph_spec = None
                            for g in graphs:
                                if isinstance(g, dict) and str(g.get("id") or "").strip() == selected_graph_id:
                                    graph_spec = g
                                    break
                        if graph_spec is not None:
                            graph_primary = True
                            log_items.append(
                                f"[GRAPH_ROUTER] graph selected from reasoner={graph_reasoners[0].id} graph={graph_spec.get('id')}"
                            )
                        else:
                            log_items.append(
                                f"[GRAPH_ROUTER] reasoner graph not found id={selected_graph_id}"
                            )
                        if graph_primary:
                            exclusive = os.environ.get("AI_INSIGHTS_GRAPH_PRIMARY_EXCLUSIVE", "0").strip().lower() in (
                                "1",
                                "true",
                                "yes",
                            )
                            if exclusive:
                                reasoners_enabled = False
                                inline_reasoners = False
                    if os.environ.get("AI_INSIGHTS_DEBUG_REASONERS", "").strip().lower() in (
                        "1",
                        "true",
                        "yes",
                    ):
                        import sys
                        print(f"[DEBUG] reasoner_ids selected: {reasoner_ids}", file=sys.stderr)
                    raw_spec = json.loads(json.dumps(spec, default=str)) if isinstance(spec, dict) else spec
                    # Normalizer disabled by request: keep the raw LLM spec as-is.
                    normalize_enabled = False
                    spec = raw_spec
                    if reasoners_enabled and inline_reasoners and inline_reasoner_ids:
                        before_inline = json.loads(json.dumps(spec, default=str)) if isinstance(spec, dict) else spec
                        spec = inject_reasoner_relations_into_spec(spec, inline_reasoner_ids)
                        _spec_diff("normalized_to_inline_reasoners", before_inline, spec)
                    _spec_diff("llm_to_normalized", raw_spec, spec)
                    specs_to_run = _split_rai_spec_if_unbounded(spec, specs)

                    unresolved = []
                    for run_spec in (specs_to_run or []):
                        unresolved.extend(_collect_unresolved_types(run_spec, specs))
                    if unresolved:
                        failure_context = _shorten(
                            "Unresolved types detected:\n"
                            + "\n".join(unresolved)
                            + f"\nSpec: {json.dumps(spec, default=str)}"
                        )
                        log_items.append(
                            f"RAI spec unresolved types on attempt {attempt + 1}/{max_query_attempts}: {len(unresolved)} issue(s)"
                        )
                        if attempt < max_query_attempts - 1:
                            continue
                        return {
                            "question": question,
                            "frames": {},
                            "plan": {"spec": spec} if isinstance(spec, dict) else {},
                            "insights": "RAI query plan contains unresolved field types.",
                            "kpis": [],
                            "chart": {},
                            "log": log_items + unresolved,
                            "sqls": [],
                            "merges": [],
                            "followups": [],
                            "catalog": cat,
                        }
                    if os.environ.get("RAI_DEBUG_SPEC", "").lower() in ("1", "true", "yes"):
                        if len(specs_to_run) > 1:
                            for idx, s in enumerate(specs_to_run, 1):
                                print(f"RAI spec {idx}:", json.dumps(s, indent=2, default=str))
                        else:
                            print("RAI spec:", json.dumps(spec, indent=2, default=str))
                    if not (isinstance(spec, dict) and spec.get("bind") and spec.get("select")):
                        last_exc = RuntimeError("RAI plan invalid or empty")
                        failure_context = _shorten(
                            f"Spec invalid or empty (missing bind/select). Last spec: {json.dumps(spec, default=str)}"
                        )
                        log_items.append(f"RAI plan invalid on attempt {attempt + 1}/{max_query_attempts}")
                        if attempt < max_query_attempts - 1:
                            continue
                        return {
                            "question": question,
                            "frames": {},
                            "plan": {},
                            "insights": "RAI query plan could not be generated for this question.",
                            "kpis": [],
                            "chart": {},
                            "log": log_items or ["RAI plan invalid or empty"],
                            "sqls": [],
                            "merges": [],
                            "followups": [],
                            "catalog": cat,
                        }

                    _render_stage("fetching")
                    frames = {}
                    reasoner_contexts = {}
                    try:
                        log_items.append(f"RAI dynamic query attempt {attempt + 1}/{max_query_attempts}")
                        for idx, run_spec in enumerate(specs_to_run, 1):
                            exec_spec = _coerce_rai_value_types(run_spec, specs)

                            if graph_primary and graph_enabled and graph_spec and graph_df is None:
                                try:
                                    log_items.append(
                                        f"[GRAPH_ROUTER] running graph query graph={graph_spec.get('id')}"
                                    )
                                    try:
                                        graph_keys = list(getattr(builder, "_graphs", {}).keys())
                                        if graph_keys:
                                            log_items.append(f"[GRAPH_ROUTER] available graphs: {graph_keys}")
                                        else:
                                            log_items.append("[GRAPH_ROUTER] available graphs: []")
                                    except Exception:
                                        log_items.append("[GRAPH_ROUTER] available graphs: <unavailable>")

                                    window_start, window_end = _extract_time_window_from_spec(exec_spec)
                                    if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
                                        try:
                                            log_items.append(
                                                f"[GRAPH_BUILD] extracted window_start={window_start} window_end={window_end}"
                                            )
                                            if not window_start and not window_end:
                                                where_raw = exec_spec.get("where") if isinstance(exec_spec, dict) else None
                                                log_items.append(
                                                    f"[GRAPH_BUILD] no time window found in exec_spec.where={_shorten(json.dumps(where_raw, default=str))}"
                                                )
                                        except Exception:
                                            pass
                                    filters = _extract_filter_values_from_spec(exec_spec, ["pu_id", "prod_id", "pl_id"])
                                    graph_df = _run_graph_reasoner_query(
                                        builder,
                                        graph_spec,
                                        graph_reasoners[0] if graph_reasoners else None,
                                        question,
                                        limit=10,
                                        window_start=window_start,
                                        window_end=window_end,
                                        filters=filters,
                                    )
                                    if graph_df is not None:
                                        graph_label = str(graph_spec.get("id") or "").strip() or "graph"
                                        frames[f"graph:{graph_label}"] = graph_df
                                        if graph_df.empty:
                                            log_items.append(f"[GRAPH_ROUTER] graph {graph_label} returned 0 rows")
                                        else:
                                            log_items.append(f"[GRAPH_ROUTER] graph {graph_label} rows={len(graph_df)}")
                                    else:
                                        log_items.append(
                                            f"[GRAPH_ROUTER] graph query returned None graph={graph_spec.get('id')}"
                                        )
                                except Exception:
                                    log_items.append(f"[GRAPH_ROUTER] graph query failed graph={graph_spec.get('id')}")
                                    graph_df = None

                            df_base = _run_query_with_lqp_fallback(exec_spec)
                            df_base = parse_dates(clean_cols(df_base))
                            df_base = _coerce_int128_cols(df_base)

                            if (not graph_primary) and graph_enabled and graph_spec and graph_df is None:
                                try:
                                    window_start, window_end = _extract_time_window_from_spec(exec_spec)
                                    if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILD", "").strip().lower() in ("1", "true", "yes"):
                                        try:
                                            log_items.append(
                                                f"[GRAPH_BUILD] extracted window_start={window_start} window_end={window_end}"
                                            )
                                            if not window_start and not window_end:
                                                where_raw = exec_spec.get("where") if isinstance(exec_spec, dict) else None
                                                log_items.append(
                                                    f"[GRAPH_BUILD] no time window found in exec_spec.where={_shorten(json.dumps(where_raw, default=str))}"
                                                )
                                        except Exception:
                                            pass
                                    filters = _extract_filter_values_from_spec(exec_spec, ["pu_id", "prod_id", "pl_id"])
                                    graph_df = _run_graph_reasoner_query(
                                        builder,
                                        graph_spec,
                                        graph_reasoners[0] if graph_reasoners else None,
                                        question,
                                        limit=10,
                                        window_start=window_start,
                                        window_end=window_end,
                                        filters=filters,
                                    )
                                    if graph_df is not None:
                                        graph_label = str(graph_spec.get("id") or "").strip() or "graph"
                                except Exception:
                                    graph_df = None

                            df_final = df_base

                            frames[f"s{idx}"] = df_final
                        # Success - exit retry loop
                        break
                    except Exception as exc:
                        import traceback
                        last_exc = exc
                        last_err_trace = traceback.format_exc()
                        err_detail = getattr(exc, "raw_content", None) or getattr(exc, "content", None)
                        if not err_detail:
                            problem = getattr(exc, "problem", None)
                            if isinstance(problem, dict):
                                report = problem.get("report")
                                message = problem.get("message")
                                err_detail = "\n".join([part for part in [report, message] if part])
                        last_err_detail = err_detail
                        failure_context = _shorten(
                            "\n".join(
                                [
                                    f"RAI query failed: {exc}",
                                    f"Detail: {err_detail}" if err_detail else "",
                                    f"Spec: {json.dumps(spec, default=str)}" if spec else "",
                                ]
                            ).strip()
                        )
                        log_items.append(f"RAI query failed on attempt {attempt + 1}/{max_query_attempts}: {exc}")
                        if attempt < max_query_attempts - 1:
                            continue
                        return {
                            "question": question,
                            "frames": {},
                            "plan": {"spec": spec} if isinstance(spec, dict) else {},
                            "insights": f"RAI query failed: {exc}",
                            "kpis": [],
                            "chart": {},
                            "log": log_items + ([str(last_err_detail)] if last_err_detail else []) + ([last_err_trace] if last_err_trace else []),
                            "sqls": [],
                            "merges": [],
                            "followups": [],
                            "catalog": cat,
                        }

                rm_mand_map = fetch_rm_to_mandate_map()
                frames = enrich_rm_mandate(frames, rm_mand_map)
                frames = {k: _coerce_int128_cols(v) for k, v in frames.items()}

                if graph_df is not None and graph_label:
                    frames[f"graph:{graph_label}"] = graph_df

                pre_gate_rows = _total_rows(frames)

                role = st.session_state.get("role")
                scope_rmids = st.session_state.get("scope_rmids") or []
                allowed_mandates = st.session_state.get("allowed_mandates") or []
                if role is None:
                    sess = init_snowflake_session(shared)
                    user_rmid = _current_rmid_from_session()
                    role, scope_rmids, _ = _role_and_scope_from_mapping(sess, user_rmid) if user_rmid else ("OTHER", [], [])
                    allowed_mandates = _mandates_from_scope(sess, scope_rmids) if role != "OTHER" else []
                    st.session_state["role"] = role
                    st.session_state["scope_rmids"] = scope_rmids
                    st.session_state["allowed_mandates"] = allowed_mandates

                allowed_names = []

                frames = _rbac_gate_frames(
                    frames,
                    role=role,
                    scope_rmids=list(map(str, scope_rmids or [])),
                    allowed_mandates=list(map(str, allowed_mandates or [])),
                    allowed_names=allowed_names,
                )
                frames = {k: _coerce_int128_cols(v) for k, v in frames.items()}

                post_gate_rows = _total_rows(frames)
                if pre_gate_rows > 0 and post_gate_rows == 0:
                    _render_stage("security")
                    _body(["**No access**: Your role-based access does not permit viewing this data."])
                    st.stop()

                if pre_gate_rows == 0 and post_gate_rows == 0:
                    _render_stage("no-data")
                    _body(["**No data available** for this request. Try changing filters or timeframe."])
                    st.stop()

                _render_stage("summarizing")
                def _ui_progress(msg: str):
                    try:
                        _body([msg])
                    except Exception:
                        pass

                det_ins: List[str] = []
                graph_note = _summarize_graph_frame(graph_df, graph_label)
                if graph_note:
                    det_ins.append(graph_note)

                summary = summarize_with_tools(
                    question,
                    frames,
                    deterministic_insights=det_ins,
                    extra_refs=None,
                    on_tool=_ui_progress,
                )

                # Generate chart using LLM - OVERRIDE the spec from summarizer with actual Plotly figure
                chart_result = {}
                try:
                    frames_src = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame) and not v.empty}
                    if frames_src:
                        # Call compute_llm_chart to generate actual Plotly visualization
                        llm_chart_data = compute_llm_chart(frames_src, question, summary.get("narrative", ""))
                        if llm_chart_data and isinstance(llm_chart_data, dict):
                            fig = llm_chart_data.get("fig")
                            if fig is not None:
                                try:
                                    # Convert figure to dict with data and layout
                                    chart_fig_dict = fig.to_dict()
                                    chart_result = {
                                        "data": chart_fig_dict.get("data", []),
                                        "layout": chart_fig_dict.get("layout", {}),
                                    }
                                except Exception:
                                    pass
                except Exception:
                    pass

                return {
                    "question": question,
                    "frames": frames,
                    "plan": {"spec": spec},
                    "insights": summary.get("narrative", "") if isinstance(summary, dict) else "",
                    "kpis": summary.get("kpis", []) if isinstance(summary, dict) else [],
                    "chart": chart_result if chart_result else (summary.get("chart", {}) if isinstance(summary, dict) else {}),
                    "needs_more": summary.get("needs_more", False) if isinstance(summary, dict) else False,
                    "followup_prompt": summary.get("followup_prompt", "") if isinstance(summary, dict) else "",
                    "required": summary.get("required", {}) if isinstance(summary, dict) else {},
                    "summary_obj": summary if isinstance(summary, dict) else {},
                    "reasoner_results": reasoner_results,
                    "log": log_items or ["RAI dynamic query"],
                    "sqls": [],
                    "merges": [],
                    "followups": [],
                    "catalog": cat,
                }

            def _select_view_llm(q: str, catalog: Dict[str, Dict[str, Any]], candidates: list[str]) -> Optional[str]:
                try:
                    schema_small = _compact_schema(catalog)
                    feature_map = _view_feature_map(catalog)
                    prompt = (
                        "You are a router. Choose the single best semantic view for the question.\n"
                        "Return ONLY JSON: {\"view\": \"<exact view name>\", \"reason\": \"<short>\"}\n"
                        f"Question: {q}\n"
                        f"Views: {candidates}\n"
                        f"Feature map: {json.dumps(feature_map, indent=2, default=_json_default)}\n"
                        f"Schema: {json.dumps(schema_small, indent=2, default=_json_default)}\n"
                    )
                    qsql = "select snowflake.cortex.complete('{model}', $${body}$$) as response".format(
                        model=CORTEX_LLM_MODEL, body=prompt
                    )
                    df = session.sql(qsql).to_pandas()
                    if df is None or df.empty:
                        return None
                    resp_text = df.iloc[0, 0]
                    parsed = None
                    try:
                        parsed = safe_json_loads(resp_text) if isinstance(resp_text, str) else None
                    except Exception:
                        parsed = json_extract(resp_text) if isinstance(resp_text, str) else None
                    if isinstance(parsed, dict):
                        view = parsed.get("view")
                        if isinstance(view, str) and view in candidates:
                            return view
                except Exception:
                    pass
                return None

            def _find_view_by_keywords(candidates: list[str], keywords: list[str]) -> Optional[str]:
                for v in candidates:
                    up = v.upper()
                    if all(k in up for k in keywords):
                        return v
                for v in candidates:
                    up = v.upper()
                    if any(k in up for k in keywords):
                        return v
                return None

            def _select_view_fallback(q: str, candidates: list[str]) -> Optional[str]:
                ql = (q or "").lower()
                if any(t in ql for t in ("meeting", "sentiment", "call", "conversation")):
                    v = _find_view_by_keywords(candidates, ["MEETING", "DETAIL"])
                    if v:
                        return v
                    v = _find_view_by_keywords(candidates, ["MEETING"])
                    if v:
                        return v
                if any(t in ql for t in ("mandate", "profit", "revenue", "topup", "deal")):
                    v = _find_view_by_keywords(candidates, ["MANDATE", "MONTHLY", "SUMMARY"])
                    if v:
                        return v
                    v = _find_view_by_keywords(candidates, ["MANDATE"])
                    if v:
                        return v
                v = _find_view_by_keywords(candidates, ["RM", "MONTHLY", "SUMMARY"])
                return v or (candidates[0] if candidates else None)

            def _normalize_suggestions(sug: Any) -> list[str]:
                if not sug:
                    return []
                if isinstance(sug, str):
                    return [sug]
                if isinstance(sug, (list, tuple)):
                    out = []
                    for s in sug:
                        if isinstance(s, dict):
                            s = s.get("text") or s.get("title") or s.get("suggestion") or str(s)
                        out.append(str(s))
                    return out
                return [str(sug)]

            def _extract_options_from_text(text: str) -> list[str]:
                if not text:
                    return []
                opts = []
                try:
                    for m in re.findall(r"(Top\s+\d+[^\\n\\?]*\\?)", text, re.I):
                        opts.append(m.strip())
                except Exception:
                    pass
                for line in (text or "").splitlines():
                    l = line.strip().lstrip("-•*").strip()
                    if len(l) < 10:
                        continue
                    if l.endswith("?") or l.lower().startswith("top "):
                        opts.append(l)
                # dedupe, preserve order
                seen = set()
                out = []
                for o in opts:
                    if o and o not in seen:
                        seen.add(o)
                        out.append(o)
                return out

            def _pick_suggestion_llm(q: str, options: list[str]) -> Optional[str]:
                if not options:
                    return None
                try:
                    prompt = (
                        "Pick the best option that answers the question. Return ONLY JSON:\n"
                        "{\"choice_index\": <1-based index>}\n"
                        f"Question: {q}\n"
                        "Options:\n"
                        + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
                    )
                    qsql = "select snowflake.cortex.complete('{model}', $${body}$$) as response".format(
                        model=CORTEX_LLM_MODEL, body=prompt
                    )
                    df = session.sql(qsql).to_pandas()
                    if df is None or df.empty:
                        return None
                    resp_text = df.iloc[0, 0]
                    parsed = None
                    try:
                        parsed = safe_json_loads(resp_text) if isinstance(resp_text, str) else None
                    except Exception:
                        parsed = json_extract(resp_text) if isinstance(resp_text, str) else None
                    if isinstance(parsed, dict):
                        idx = parsed.get("choice_index")
                        if isinstance(idx, int) and 1 <= idx <= len(options):
                            return options[idx - 1]
                except Exception:
                    pass
                return options[0]

            def _data_request_plan_llm(q: str, view_name: str) -> dict:
                view_meta = cat.get(view_name, {}) if isinstance(cat, dict) else {}
                fields = view_meta.get("fields", []) or []
                types = view_meta.get("types", {}) or {}
                time_hints = _extract_time_hints(q)
                prompt = (
                    "You are a data planner. Return raw data requirements only.\n"
                    "Return ONLY JSON: {\"columns\": [...], \"filters\": {...}, \"limit\": <int>, \"order_by\": <col|null>, \"order\": \"asc|desc\"}\n"
                    "Rules: use only schema columns; do NOT aggregate, rank, or compute correlations.\n"
                    "Always include RMID or RM_NAME if present, and a time column if present.\n"
                    "If filtering by RM_NAME/RELATIONSHIPMANAGER, prefer fuzzy match (like/ilike) over exact equality.\n"
                    "If the question mentions client(s), interpret that as mandate-level and include MANDATEID or MANDATEID_STR when available.\n"
                    f"Question: {q}\n"
                    f"Time hints: {json.dumps(time_hints, indent=2, default=_json_default)}\n"
                    f"Schema fields: {json.dumps(fields, indent=2, default=_json_default)}\n"
                    f"Schema types: {json.dumps(types, indent=2, default=_json_default)}\n"
                )
                try:
                    qsql = "select snowflake.cortex.complete('{model}', $${body}$$) as response".format(
                        model=CORTEX_LLM_MODEL, body=prompt
                    )
                    df = session.sql(qsql).to_pandas()
                    if df is None or df.empty:
                        return {}
                    resp_text = df.iloc[0, 0]
                    parsed = None
                    try:
                        parsed = safe_json_loads(resp_text) if isinstance(resp_text, str) else None
                    except Exception:
                        parsed = json_extract(resp_text) if isinstance(resp_text, str) else None
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}

            def _explicit_metric_key(question_text: str) -> Optional[str]:
                ql = (question_text or "").lower()
                for key, arr in BASE_SYNONYMS.items():
                    if any(w in ql for w in arr):
                        return key
                return None

            def _extract_rank_intent(question_text: str) -> tuple[Optional[int], str]:
                q = question_text or ""
                order = "desc"
                limit = None
                m_bottom = re.search(r"\bbottom\s+(\d{1,3})\b", q, re.I)
                m_top = re.search(r"\btop\s+(\d{1,3})\b", q, re.I)
                if m_bottom:
                    order = "asc"
                    try:
                        limit = int(m_bottom.group(1))
                    except Exception:
                        limit = None
                elif m_top:
                    order = "desc"
                    try:
                        limit = int(m_top.group(1))
                    except Exception:
                        limit = None
                else:
                    if re.search(r"\b(bottom|lowest|smallest|worst)\b", q, re.I):
                        order = "asc"
                        limit = 5
                    elif re.search(r"\b(top|highest|largest|best|rank)\b", q, re.I):
                        order = "desc"
                        limit = 5
                return limit, order

            def _is_ranking_question(question_text: str) -> bool:
                return bool(re.search(r"\b(top|bottom|highest|lowest|best|worst|rank)\b", question_text or "", re.I))

            def _ranking_request_prompt(q: str, view_name: str) -> str:
                view_meta = cat.get(view_name, {}) if isinstance(cat, dict) else {}
                fields = view_meta.get("fields", []) or []
                types = view_meta.get("types", {}) or {}
                up = {str(f).upper(): f for f in fields}
                syn = build_schema_synonyms(cat) if isinstance(cat, dict) else {}
                view_syn = syn.get(view_name, {}) or {}
                ql = (q or "").lower()

                explicit_key = _explicit_metric_key(q)
                key_priority = ["performance", "revenue", "profit", "aum", "sentiment", "topup", "meetings", "transactions"]
                chosen_key = explicit_key
                if not chosen_key and re.search(r"\b(up[- ]?sell|upsell|cross[- ]?sell|new product|new products)\b", ql):
                    for cand in ("topup", "revenue", "profit", "performance"):
                        syn_key = "current_aum" if cand == "aum" else cand
                        if view_syn.get(syn_key):
                            chosen_key = cand
                            break
                if not chosen_key:
                    for cand in key_priority:
                        syn_key = "current_aum" if cand == "aum" else cand
                        if view_syn.get(syn_key):
                            chosen_key = cand
                            break
                chosen_key = chosen_key or "performance"

                syn_key = "current_aum" if chosen_key == "aum" else chosen_key
                metric_field = view_syn.get(syn_key)
                if not metric_field:
                    keywords = {
                        "performance": ["PERF", "SCORE"],
                        "revenue": ["REVEN", "FEE", "INCOME"],
                        "profit": ["PROFIT", "MARGIN", "PNL"],
                        "aum": ["AUM", "ASSET"],
                        "sentiment": ["SENTIMENT", "TONE", "POSITIVE", "NEGATIVE"],
                        "topup": ["TOPUP", "CONTRIBUTION", "INFLOW", "DEPOSIT"],
                        "meetings": ["MEETING", "APPOINTMENT", "CALL"],
                        "transactions": ["TRANSACTION", "TXN", "COUNT"]
                    }
                    for f in fields:
                        fu = str(f).upper()
                        if any(tok in fu for tok in keywords.get(chosen_key, [])):
                            metric_field = f
                            break
                if not metric_field:
                    for f in fields:
                        t = str(types.get(f, "")).upper()
                        if any(tok in t for tok in ("NUMBER", "DECIMAL", "NUMERIC", "INT", "FLOAT", "DOUBLE", "REAL")):
                            metric_field = f
                            break
                if not metric_field and fields:
                    metric_field = fields[0]

                agg_mode = DEFAULT_MODE.get(chosen_key, "sum")
                mandate_intent = bool(re.search(r"\bmandate|deal|client|clients\b", ql))
                rmid_field = view_syn.get("rmid") or up.get("RMID")
                rm_name_field = view_syn.get("rm_name") or up.get("RM_NAME") or up.get("RELATIONSHIPMANAGER") or up.get("RELATIONSHIP_MANAGER")
                mandate_id_field = None
                mandate_name_field = None
                for cand in ("MANDATE_NAME", "MANDATEID_STR", "MANDATEID", "MANDATE_ID", "DEALNAME"):
                    if cand in up:
                        if cand in ("MANDATE_NAME", "MANDATEID_STR", "DEALNAME") and not mandate_name_field:
                            mandate_name_field = up[cand]
                        if cand in ("MANDATEID", "MANDATE_ID") and not mandate_id_field:
                            mandate_id_field = up[cand]
                entity_fields = []
                if mandate_intent and (mandate_name_field or mandate_id_field):
                    if mandate_name_field:
                        entity_fields.append(mandate_name_field)
                    if mandate_id_field and mandate_id_field not in entity_fields:
                        entity_fields.append(mandate_id_field)
                else:
                    if rm_name_field:
                        entity_fields.append(rm_name_field)
                    if rmid_field and rmid_field not in entity_fields:
                        entity_fields.append(rmid_field)
                year_field = up.get("MEET_YEAR") or up.get("POSYEAR") or up.get("YEAR")
                time_field = view_syn.get("month_date") or view_syn.get("meeting_date") or view_syn.get("any_date")

                years = (_extract_time_hints(q) or {}).get("explicit_years") or []
                limit, order = _extract_rank_intent(q)
                limit = limit or 5
                include_year = bool(years)

                metric_alias = re.sub(r"[^A-Z0-9_]+", "_", str(metric_field).upper())
                metric_alias = f"{metric_alias}_{agg_mode.upper()}"

                cols = []
                for c in entity_fields:
                    if c and c not in cols:
                        cols.append(c)
                if mandate_intent and rm_name_field and rm_name_field not in cols:
                    cols.append(rm_name_field)
                year_label = None
                if include_year:
                    year_label = year_field or "YEAR"
                    cols.append(year_label)
                cols.append(metric_alias)

                parts = [
                    "Return aggregated rows for ranking (do not return raw rows).",
                    f"Metric: {metric_field} using {agg_mode} aggregation, alias as {metric_alias}.",
                ]
                if include_year:
                    parts.append("Include a year column (use MEET_YEAR/POSYEAR/YEAR if present; otherwise derive YEAR from a date column).")
                parts.append(f"Columns: {', '.join(cols)}.")
                if years:
                    if year_field:
                        parts.append(f"Filters: {year_field} IN ({', '.join(map(str, years))}).")
                    elif time_field:
                        parts.append(f"Filters: YEAR({time_field}) IN ({', '.join(map(str, years))}).")
                if include_year:
                    parts.append(f"Order by year ascending, {metric_alias} {order}.")
                    parts.append(
                        f"Return top {limit} per year if possible (e.g., ROW_NUMBER over year <= {limit}); "
                        f"otherwise return {limit} rows per year ordered by {metric_alias}."
                    )
                else:
                    parts.append(f"Order by {metric_alias} {order}.")
                    parts.append(f"Return top {limit} rows overall ordered by {metric_alias}.")
                return " ".join(parts)

            def _data_request_fallback(q: str, view_name: str) -> dict:
                view_meta = cat.get(view_name, {}) if isinstance(cat, dict) else {}
                fields = view_meta.get("fields", []) or []
                up = {str(f).upper(): f for f in fields}
                ql = (q or "").lower()
                cols = []
                if "RM_NAME" in up: cols.append(up["RM_NAME"])
                if "RMID" in up and up["RMID"] not in cols: cols.append(up["RMID"])
                if ("mandate" in ql or "client" in ql) and "MANDATEID" in up:
                    cols.append(up["MANDATEID"])
                if ("mandate" in ql or "client" in ql) and "MANDATEID_STR" in up and up["MANDATEID_STR"] not in cols:
                    cols.append(up["MANDATEID_STR"])
                if "MONTH_DATE" in up: cols.append(up["MONTH_DATE"])
                for cand in ("MEET_YEAR","POSYEAR","YEAR"):
                    if cand in up and up[cand] not in cols: cols.append(up[cand])
                for cand in ("MEET_MON","POSMON","MONTH"):
                    if cand in up and up[cand] not in cols: cols.append(up[cand])

                def _pick_metric(match_terms):
                    for f in fields:
                        fu = str(f).upper()
                        if any(t in fu for t in match_terms):
                            return f
                    return None

                if "revenue" in ql: cols.append(_pick_metric(["REVENUE"]) or "REVENUE_AMOUNT")
                if "profit" in ql: cols.append(_pick_metric(["PROFIT"]) or "PROFIT_AMOUNT")
                if "sentiment" in ql: cols.append(_pick_metric(["SENTIMENT"]) or "SENTIMENT_SCORE")
                if "meeting" in ql: cols.append(_pick_metric(["MEETING_COUNT","DURATION"]) or "MEETING_COUNT")
                if "performance" in ql: cols.append(_pick_metric(["PERFORMANCE"]) or "PERFORMANCE")
                if "aum" in ql: cols.append(_pick_metric(["AUM"]) or "AUM")

                norm_cols = []
                for c in cols:
                    if not c: continue
                    if isinstance(c, str) and c.upper() in up:
                        norm_cols.append(up[c.upper()])
                seen=set(); norm_cols=[c for c in norm_cols if not (c in seen or seen.add(c))]

                filters = {}
                hints = _extract_time_hints(q)
                years = hints.get("explicit_years") or []
                year_col = up.get("MEET_YEAR") or up.get("POSYEAR") or up.get("YEAR")
                if years and year_col:
                    if len(years) == 1:
                        filters[year_col] = years[0]
                    else:
                        filters[year_col] = {"in": years}
                sm = hints.get("single_month")
                mon_col = up.get("MEET_MON") or up.get("POSMON") or up.get("MONTH")
                if sm and mon_col and year_col:
                    filters[year_col] = sm.get("year")
                    filters[mon_col] = sm.get("month")

                return {"columns": norm_cols, "filters": filters, "limit": 2000, "order_by": None, "order": "desc"}

            def _filters_to_prompt(filters: dict) -> str:
                def _is_name_field(field: str) -> bool:
                    return str(field or "").upper() in {
                        "RM_NAME",
                        "RELATIONSHIPMANAGER",
                        "RELATIONSHIP_MANAGER",
                        "NAME",
                    }
                parts = []
                for k, v in (filters or {}).items():
                    if isinstance(v, dict):
                        for op, val in v.items():
                            op_l = str(op).lower()
                            if op_l == "between" and isinstance(val, (list, tuple)) and len(val) == 2:
                                parts.append(f"{k} BETWEEN {val[0]} AND {val[1]}")
                            elif op_l == "in" and isinstance(val, (list, tuple)):
                                items = ", ".join([f"'{x}'" if isinstance(x,str) else str(x) for x in val])
                                parts.append(f"{k} IN ({items})")
                            else:
                                rhs = f"'{val}'" if isinstance(val,str) else str(val)
                                if op_l in {"=", "=="} and isinstance(val, str) and _is_name_field(k):
                                    parts.append(f"{k} is like {rhs}")
                                else:
                                    parts.append(f"{k} {op} {rhs}")
                    elif isinstance(v, (list, tuple, set)):
                        items = ", ".join([f"'{x}'" if isinstance(x,str) else str(x) for x in v])
                        parts.append(f"{k} IN ({items})")
                    else:
                        rhs = f"'{v}'" if isinstance(v,str) else str(v)
                        if isinstance(v, str) and _is_name_field(k):
                            parts.append(f"{k} is like {rhs}")
                        else:
                            parts.append(f"{k} = {rhs}")
                return " AND ".join(parts)

            def _data_request_prompt(q: str, view_name: str) -> tuple[str, dict]:
                view_meta = cat.get(view_name, {}) if isinstance(cat, dict) else {}
                fields = view_meta.get("fields", []) or []
                if _is_ranking_question(q):
                    prompt = _ranking_request_prompt(q, view_name)
                    return (prompt, {"ranking": True})
                plan = _data_request_plan_llm(q, view_name)
                if not plan:
                    plan = _data_request_fallback(q, view_name)
                cols = plan.get("columns") or []
                up = {str(f).upper(): f for f in fields}
                norm_cols = []
                for c in cols:
                    if isinstance(c, str) and c.upper() in up:
                        norm_cols.append(up[c.upper()])
                if not norm_cols:
                    norm_cols = fields[:10]
                if "RMID" in up and up["RMID"] not in norm_cols:
                    norm_cols.append(up["RMID"])
                if "RM_NAME" in up and up["RM_NAME"] not in norm_cols:
                    norm_cols.append(up["RM_NAME"])
                if "MONTH_DATE" in up and up["MONTH_DATE"] not in norm_cols:
                    norm_cols.append(up["MONTH_DATE"])
                order_by = plan.get("order_by")
                if isinstance(order_by, str) and order_by.upper() in up:
                    order_by = up[order_by.upper()]
                else:
                    order_by = None
                order = str(plan.get("order") or "desc").lower()
                filt_txt = _filters_to_prompt(plan.get("filters") or {})
                parts = ["Return raw rows (no aggregation or ranking).", f"Columns: {', '.join(norm_cols)}."]
                if filt_txt:
                    parts.append(f"Filters: {filt_txt}.")
                if order_by:
                    parts.append(f"Order by {order_by} {order}.")
                return (" ".join(parts), plan)


            def _agentic_analyst(question_text: str, view_name: str) -> Dict[str, Any]:
                prompt, _plan = _data_request_prompt(question_text, view_name)
                parsed = analyst_message(view_name, prompt) or {}
                sql = parsed.get("sql")
                if sql:
                    return {"sql": sql, "parsed": parsed, "suggestion": None, "prompt": prompt}
                options = _normalize_suggestions(parsed.get("suggestions"))
                if not options:
                    options = _extract_options_from_text(parsed.get("explanation", "") or "")
                if options:
                    chosen = _pick_suggestion_llm(question_text, options)
                    if chosen:
                        prompt2, _plan2 = _data_request_prompt(chosen, view_name)
                        parsed2 = analyst_message(view_name, prompt2) or {}
                        if parsed2.get("sql"):
                            return {"sql": parsed2.get("sql"), "parsed": parsed2, "suggestion": chosen, "prompt": prompt2}
                retry_prompt, _plan3 = _data_request_prompt(question_text + " (raw data only)", view_name)
                parsed3 = analyst_message(view_name, retry_prompt) or {}
                if parsed3.get("sql"):
                    return {"sql": parsed3.get("sql"), "parsed": parsed3, "suggestion": None, "prompt": retry_prompt}
                return {"sql": "", "parsed": parsed, "suggestion": None, "prompt": prompt}
            strict_routing = os.environ.get("AI_INSIGHTS_STRICT_ROUTING", "1").strip() == "1"
            if strict_routing:
                _render_stage("planning")
                candidates = [v for v in (views or []) if cat.get(v, {}).get("fields")] or list(views or [])
                chosen_view = _select_view_llm(question, cat, candidates) or _select_view_fallback(question, candidates)
                if not chosen_view:
                    return {
                        "question": question,
                        "frames": {},
                        "plan": {},
                        "insights": "No suitable semantic view found for this question.",
                        "kpis": [],
                        "chart": {},
                        "log": ["No view selected"],
                        "sqls": [],
                        "merges": [],
                        "followups": [],
                        "catalog": cat,
                    }
                _body([f"Selected view: **{chosen_view}**"])

                _render_stage("fetching")
                rm_lookup_cache = st.session_state.get("rm_lookup_cache")
                rm_hint = _fuzzy_rm_from_question(question, rm_lookup_cache)
                if rm_hint and rm_hint.get("rm_name"):
                    question_for_agent = question + f"\n\nFilter RM_NAME is like {rm_hint['rm_name']}."
                else:
                    question_for_agent = question
                agent = _agentic_analyst(question_for_agent, chosen_view)
                sql = agent.get("sql") or ""
                if not sql:
                    msg = (agent.get("parsed") or {}).get("explanation") or "Analyst did not return SQL."
                    return {
                        "question": question,
                        "frames": {},
                        "plan": {"view": chosen_view},
                        "insights": msg,
                        "kpis": [],
                        "chart": {},
                        "log": ["Analyst returned no SQL"],
                        "sqls": [],
                        "merges": [],
                        "followups": [],
                        "catalog": cat,
                    }

                ok, df, err = analyst_query_execute(sql)
                if not ok:
                    return {
                        "question": question,
                        "frames": {},
                        "plan": {"view": chosen_view},
                        "insights": f"Query failed: {err}",
                        "kpis": [],
                        "chart": {},
                        "log": [f"Query failed: {err}"],
                        "sqls": [("s1", sql)],
                        "merges": [],
                        "followups": [],
                        "catalog": cat,
                    }

                df = parse_dates(clean_cols(df))
                if isinstance(df, pd.DataFrame) and not df.empty:
                    up = {str(c).upper(): c for c in df.columns}
                    if "RM_NAME" not in up and ("RMID" in up or "RM_ID" in up):
                        rm_lookup = fetch_rm_lookup(cat)
                        if rm_lookup is not None and not rm_lookup.empty:
                            rmid_col = up.get("RMID") or up.get("RM_ID")
                            try:
                                df[rmid_col] = df[rmid_col].astype(str)
                            except Exception:
                                pass
                            lu = rm_lookup.copy()
                            try:
                                lu["RMID"] = lu["RMID"].astype(str)
                            except Exception:
                                pass
                            df = df.merge(lu, left_on=rmid_col, right_on="RMID", how="left")
                            if "RM_NAME_y" in df.columns and "RM_NAME_x" in df.columns:
                                df["RM_NAME"] = df["RM_NAME_x"].fillna(df["RM_NAME_y"])
                                df.drop(columns=[c for c in ("RM_NAME_x", "RM_NAME_y") if c in df.columns], inplace=True, errors="ignore")

                frames = {"s1": df}
                _render_stage("summarizing")
                def _ui_progress(msg: str):
                    try:
                        _body([msg])
                    except Exception:
                        pass
                summary = summarize_with_tools(
                    question,
                    frames,
                    deterministic_insights=[],
                    extra_refs=None,
                    on_tool=_ui_progress,
                )
                
                # Generate chart using LLM - OVERRIDE the spec from summarizer with actual Plotly figure
                chart_result = {}
                try:
                    frames_src = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame) and not v.empty}
                    if frames_src:
                        # Call compute_llm_chart to generate actual Plotly visualization
                        llm_chart_data = compute_llm_chart(frames_src, question, summary.get("narrative", ""))
                        if llm_chart_data and isinstance(llm_chart_data, dict):
                            fig = llm_chart_data.get("fig")
                            if fig is not None:
                                try:
                                    # Convert figure to dict with data and layout
                                    chart_fig_dict = fig.to_dict()
                                    chart_result = {
                                        "data": chart_fig_dict.get("data", []),
                                        "layout": chart_fig_dict.get("layout", {}),
                                    }
                                except Exception:
                                    pass
                except Exception:
                    pass
                
                return {
                    "question": question,
                    "frames": frames,
                    "plan": {"view": chosen_view},
                    "insights": summary.get("narrative", "") if isinstance(summary, dict) else "",
                    "kpis": summary.get("kpis", []) if isinstance(summary, dict) else [],
                    "chart": chart_result if chart_result else (summary.get("chart", {}) if isinstance(summary, dict) else {}),
                    "needs_more": summary.get("needs_more", False) if isinstance(summary, dict) else False,
                    "followup_prompt": summary.get("followup_prompt", "") if isinstance(summary, dict) else "",
                    "required": summary.get("required", {}) if isinstance(summary, dict) else {},
                    "summary_obj": summary if isinstance(summary, dict) else {},
                    "log": ["Strict routing"],
                    "sqls": [("s1", sql)],
                    "merges": [],
                    "followups": [],
                    "catalog": cat,
                }

            # ---------- RM lookup for name→ID resolution ----------
            rm_lookup = fetch_rm_lookup(cat)
            st.session_state["rm_lookup_cache"] = rm_lookup
            rm_mand_map = fetch_rm_to_mandate_map()


            # ---------- execution ----------
            def _execute_plan(plan: Dict[str, Any], *, loop_tag: str = "L1") -> tuple[Dict[str, pd.DataFrame], list, list, dict]:
                frames: Dict[str, pd.DataFrame] = {}
                log, sqls = [], []
                alias_by_id = {}
                for s in (plan.get("steps") or [])[:4]:
                    v = s.get("view"); m = (s.get("metric") or {}).get("field")
                    sid = s.get("id") or f"s_{loop_tag}"
                    if not v or not m or v not in cat:
                        continue
                    if m not in cat[v].get("fields", []) and not any(str(m).upper() == str(f).upper() for f in cat[v].get("fields", [])):
                        continue
                    out = run_step(s, cat, months_window=months_window, question=question, rm_lookup=rm_lookup)
                    if out.get("ok"):
                        df_out = out.get("df")
                        try:
                            cleaned = parse_dates(clean_cols(df_out))
                        except Exception:
                            cleaned = df_out
                        sid_full = f"{sid}_{loop_tag}" if not (sid or "").endswith(f"_{loop_tag}") else sid
                        frames[sid_full] = cleaned
                        log.append(f"✓ {sid_full}")
                        sqls.append((sid_full, out.get("sql", "")))
                        if out.get("alias"):
                            alias_by_id[sid_full] = out["alias"]
                    else:
                        log.append(f"✗ {(s.get('id') or 'step')} failed: {out.get('explanation')}")
                return frames, log, sqls, alias_by_id

            # ---------- main loop ----------
            overall_frames: Dict[str, pd.DataFrame] = {}
            all_logs, all_sqls, all_followups, merges_done = [], [], [], []
            loops_max = 2
            loop = 0
            current_question = question
            final = {"narrative": "", "kpis": [], "chart": {}, "needs_more": False, "followup_prompt": "", "required": {}}
            last_plan = {}

            while loop <= loops_max:
                _render_stage("planning")
                rm_hint = _fuzzy_rm_from_question(current_question, rm_lookup)
                plan = planner_llm(current_question, cat) or heuristic_plan(current_question, cat)
                try:
                    if _looks_like_driver_question(current_question):
                        plan = _augment_plan_for_driver(plan, cat)
                except Exception:
                    pass
                plan = _apply_rm_hint_to_plan(plan, rm_hint, cat)
                last_plan = plan if isinstance(plan, dict) else {}
                if not (isinstance(plan, dict) and plan.get("steps")):
                    all_logs.append("Planner returned no steps")
                    _body(["<span class='ai-muted'>I couldn’t find the right datasets to answer that.</span>"])
                    break

                # One concise “What I’ll fetch” list
                fetch_lines = ["📋 <b>What I’ll fetch</b>:"]
                for s in (plan.get("steps") or [])[:4]:
                    fetch_lines.append(_step_to_friendly_sentence(s))
                _body(fetch_lines)

                # Fetch
                _render_stage("fetching")
                frames, log, sqls, alias_map = _execute_plan(plan, loop_tag=f"L{loop + 1}")
                st.session_state["frame_aliases"] = {
                    **(st.session_state.get("frame_aliases") or {}),
                    **(alias_map or {})
                }
                all_logs.extend(log); all_sqls.extend(sqls)
                _body(["📥 Pulling data…"])

                # Link
                _render_stage("linking")
                overall_frames.update(frames)
                overall_frames = enrich_rm_mandate(overall_frames, rm_mand_map)
                overall_frames = enrich_rm_names(overall_frames, rm_lookup)

                # Pre-gate row count (after linking/enrichment, before RBAC)
                pre_gate_rows = _total_rows(overall_frames)

                # ── RBAC bootstrap (reuse session_state if set by other pages) ──
                role = st.session_state.get("role")
                scope_rmids = st.session_state.get("scope_rmids") or []
                allowed_mandates = st.session_state.get("allowed_mandates") or []
                if role is None:
                    sess = init_snowflake_session(shared)

                    user_rmid = _current_rmid_from_session()
                    role, scope_rmids, _ = _role_and_scope_from_mapping(sess, user_rmid) if user_rmid else ("OTHER", [], [])
                    allowed_mandates = _mandates_from_scope(sess, scope_rmids) if role != "OTHER" else []
                    st.session_state["role"] = role
                    st.session_state["scope_rmids"] = scope_rmids
                    st.session_state["allowed_mandates"] = allowed_mandates

                # Optional: allowed RM_NAMEs from lookup
                allowed_names = []
                try:
                    if rm_lookup is not None and not rm_lookup.empty and scope_rmids:
                        lu = rm_lookup.copy()
                        lu["RMID"] = lu["RMID"].astype(str)
                        allowed_names = (
                            lu[lu["RMID"].isin(list(map(str, scope_rmids)))]["RM_NAME"]
                            .dropna().astype(str).unique().tolist()
                        )
                except Exception:
                    pass

                # ── FINAL RBAC GATE: affects s1_L1, s2_L1, any future frames ──
                overall_frames = _rbac_gate_frames(
                    overall_frames,
                    role=role,
                    scope_rmids=list(map(str, scope_rmids or [])),
                    allowed_mandates=list(map(str, allowed_mandates or [])),
                    allowed_names=allowed_names
                )

                # Post-gate row count
                post_gate_rows = _total_rows(overall_frames)

                # If we had rows before gating, but none after → user has no access to this data slice
                if pre_gate_rows > 0 and post_gate_rows == 0:
                    _render_stage("security")
                    _body(["🔒 **No access**: Your role-based access does not permit viewing this data."])
                    st.stop()

                # If we had no rows to begin with (even before gate) → empty inputs, nothing to summarize
                if pre_gate_rows == 0 and post_gate_rows == 0:
                    _render_stage("no-data")
                    _body(["  ️ **No data available** for this request. Try changing filters or timeframe."])
                    st.stop()



                merged_frames, merges = auto_merge_all_frames(overall_frames, max_pairs=8, how="inner")
                overall_frames = merged_frames

                # ----- auto-build comparison dataset for any N and any dimension (multi-frame aware) -----
                try:
                    q_text = (res.get("question") or "") if isinstance(res, dict) else (question or "")
                except Exception:
                    q_text = (question or "")

                terms = _extract_compare_terms(q_text)

                def _candidate_frames_with_cmp_col(frames: dict):
                    out = []
                    for sid, df in (frames or {}).items():
                        if not isinstance(df, pd.DataFrame) or df.empty:
                            continue
                        cmp_col, cmp_vals = _find_compare_column(df, terms)
                        if cmp_col:
                            out.append((sid, df, cmp_col, cmp_vals))
                    return out

                # Need at least 2 explicit compare values (years/months/tokens)
                if any([len(terms.get("years", [])) >= 2,
                        len(terms.get("months", [])) >= 2,
                        len(terms.get("tokens", [])) >= 2]):

                    cands = _candidate_frames_with_cmp_col(overall_frames)
                    if cands:
                        # Prefer the first column choice from the first candidate to keep SERIES consistent
                        _, sample_df, sample_cmp_col, sample_vals = cands[0]

                        # if the first candidate returned no explicit values (rare), reuse term lists
                        cmp_values = sample_vals or terms.get("years") or terms.get("months") or terms.get("tokens") or []

                        # metric preference
                        metric = _pick_metric_col(sample_df, prefer=["S1__PERFORMANCE__AVG","PERFORMANCE","SCORE","AUM","REVENUE","PROFIT"]) or _pick_metric_col(sample_df)
                        if metric:
                            rm_lu = _rm_lookup_from_frames(overall_frames)

                            # Build from MULTIPLE frames: for each requested SERIES value, pick a frame that has rows for it
                            parts_frames = []
                            for i, val in enumerate(cmp_values):
                                # round-robin search for a frame that actually contains this SERIES value
                                chosen = None
                                for sid, df, cmp_col, _ in cands:
                                    try:
                                        ser = _derive_compare_series(df, cmp_col)
                                        # string compare to be dtype-agnostic
                                        if str(val) in set(ser.dropna().astype(str).unique()):
                                            chosen = (df, cmp_col, val)
                                            break
                                    except Exception:
                                        continue
                                # if none explicitly has that value, still fall back to first candidate (it will filter to empty safely)
                                if chosen is None:
                                    chosen = (cands[i % len(cands)][1], cands[i % len(cands)][2], val)
                                parts_frames.append(chosen)

                            # Aggregate each (df, cmp_col, value) and union → one long frame (robust per-frame metric)
                            long_parts = []
                            for df_part, cmp_col, val in parts_frames:
                                d = df_part.copy()
                                if not isinstance(d, pd.DataFrame) or d.empty:
                                    continue

                                # normalize SERIES and filter to this value (dtype-agnostic compare)
                                ser = _derive_compare_series(d, cmp_col)
                                d["__SERIES__"] = ser
                                d = d[d["__SERIES__"].astype(str) == str(val)]
                                if d.empty:
                                    continue

                                # choose metric for THIS frame if the global 'metric' is missing
                                up = {c.upper(): c for c in d.columns}
                                metric_col = metric if metric in d.columns else None
                                if metric_col is None:
                                    # try case-insensitive match first
                                    if metric and metric.upper() in up:
                                        metric_col = up[metric.upper()]
                                    else:
                                        # fallback: pick a sensible numeric column for this frame
                                        metric_col = _pick_metric_col(d, prefer=["S1__PERFORMANCE__AVG","PERFORMANCE","SCORE","AUM","REVENUE","PROFIT"])
                                if not metric_col or metric_col not in d.columns:
                                    # nothing to aggregate in this frame for this series value
                                    continue

                                # choose entity id/display columns
                                x_disp, x_id = _choose_entity_column(d)

                                # guard required columns exist
                                needed = [x_id, "__SERIES__", metric_col]
                                needed = [c for c in needed if c in d.columns]
                                if len(needed) < 3:
                                    continue

                                g = d[needed].dropna(subset=[x_id, metric_col]).copy()
                                if is_identifier_name(x_id):
                                    g[x_id] = g[x_id].astype(str)

                                # pick aggregation per metric (frame-aware)
                                agg_here = smart_default_agg_for_metric(metric_col)
                                if agg_here == "sum":
                                    gg = g.groupby([x_id, "__SERIES__"], dropna=False)[metric_col].sum().reset_index()
                                elif agg_here in ("avg","mean"):
                                    gg = g.groupby([x_id, "__SERIES__"], dropna=False)[metric_col].mean().reset_index()
                                elif agg_here == "median":
                                    gg = g.groupby([x_id, "__SERIES__"], dropna=False)[metric_col].median().reset_index()
                                elif agg_here == "count":
                                    gg = g.groupby([x_id, "__SERIES__"], dropna=False)[metric_col].count().reset_index()
                                elif agg_here == "last":
                                    # use last by time if a time column exists
                                    date_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
                                    if date_cols:
                                        dc = date_cols[0]
                                        # keep only the latest row per X_ID for this series
                                        d2 = d[[x_id, "__SERIES__", metric_col, dc]].sort_values(dc)
                                        idx = d2.groupby(x_id)[dc].idxmax()
                                        gg = d2.loc[idx, [x_id, "__SERIES__", metric_col]].reset_index(drop=True)
                                    else:
                                        gg = g.groupby([x_id, "__SERIES__"], dropna=False)[metric_col].last().reset_index()
                                else:
                                    gg = g.groupby([x_id, "__SERIES__"], dropna=False)[metric_col].sum().reset_index()

                                gg.rename(columns={x_id: "X_ID", metric_col: "VALUE"}, inplace=True)
                                long_parts.append(gg)


                            if long_parts:
                                cmp_long = pd.concat(long_parts, ignore_index=True)

                                # CONSISTENT COHORT OPTION:
                                # Keep only entities that appear in *all* series (intersection) to avoid “no data in 2024” issues
                                all_series = set(cmp_long["__SERIES__"].dropna().unique().tolist())
                                if len(all_series) >= 2:
                                    counts = cmp_long.groupby("X_ID")["__SERIES__"].nunique()
                                    keep_ids = counts[counts == len(all_series)].index.astype(str).tolist()
                                    cmp_long = cmp_long[cmp_long["X_ID"].astype(str).isin(keep_ids)]

                                if not cmp_long.empty:
                                    # Top-K: by the latest SERIES if sortable, else by first SERIES
                                    try:
                                        latest = sorted(cmp_long["__SERIES__"].dropna().unique().tolist())[-1]
                                    except Exception:
                                        latest = cmp_long["__SERIES__"].dropna().unique().tolist()[0]
                                    top_ids = (cmp_long[cmp_long["__SERIES__"] == latest]
                                               .sort_values("VALUE", ascending=False)
                                               .head(5)["X_ID"].astype(str).tolist())
                                    cmp_long = cmp_long[cmp_long["X_ID"].astype(str).isin(top_ids)]

                                    # Attach RM_NAME if available
                                    if rm_lu is not None and not rm_lu.empty:
                                        # harmonize rm_lu columns
                                        up = {c.upper(): c for c in rm_lu.columns}
                                        rmid = up.get("RMID")
                                        rmname = up.get("RM_NAME")
                                        if rmid and rmname:
                                            cmp_long = cmp_long.merge(
                                                rm_lu[[rmid, rmname]].rename(columns={rmid: "RMID", rmname: "RM_NAME"}),
                                                left_on="X_ID", right_on="RMID", how="left"
                                            )

                                    # finalize column names
                                    # finalize column names
                                    cmp_long.rename(columns={"__SERIES__": "SERIES"}, inplace=True)

                                    # ---------- friendly labels for charts ----------
                                    # Infer legend (series) label
                                    try:
                                        _s = cmp_long["SERIES"].dropna().astype(str)
                                        if (len(_s) > 0) and _s.str.fullmatch(r"20\d{2}").all():
                                            series_label = "Year"
                                        else:
                                            # check if values look like months 1..12
                                            _nums = pd.to_numeric(_s, errors="coerce").dropna().astype(int)
                                            if len(_nums) > 0 and set(_nums.unique()).issubset(set(range(1,13))):
                                                series_label = "Month"
                                            else:
                                                # fall back to the original compare column name if we have it
                                                series_label = sample_cmp_col if isinstance(sample_cmp_col, str) else "Series"
                                    except Exception:
                                        series_label = "Series"

                                    # Infer y-axis label from the metric we aggregated
                                    value_label = metric if isinstance(metric, str) else "Value"

                                    # Infer X label
                                    if "RM_NAME" in cmp_long.columns:
                                        x_label = "RM Name"
                                    elif "RMID" in cmp_long.columns:
                                        x_label = "RMID"
                                    elif "X_ID" in cmp_long.columns:
                                        x_label = "Entity"
                                    else:
                                        x_label = "Category"

                                    # Store labels on the DataFrame so the Charts block can read them
                                    cmp_long.attrs["series_label"] = series_label
                                    cmp_long.attrs["value_label"]  = value_label
                                    cmp_long.attrs["x_label"]      = x_label
                                    cmp_long.attrs["label"] = f"{value_label} by {x_label} — {series_label} comparison"
                                    # ---------- /friendly labels ----------

                                    # register & prefer this dataset in Charts
                                    overall_frames["compare_auto_any"] = cmp_long
                                    st.session_state["__auto_compare_dataset__"] = "compare_auto_any"





                merged_rows = _total_rows(merged_frames)
                if merged_rows == 0:
                    _render_stage("no-data")
                    _body(["  ️ **Insufficient data** after access restrictions and merges. Try broadening filters or timeframe."])
                    st.stop()
                merges_done.extend(merges)
                _body(["🔗 Connecting datasets…"])

                # Summarize
                _render_stage("summarizing")
                _body(["   Writing insights…"])

                def _ui_progress(msg: str):
                    # print into the same “Summarizing” stage
                    try:
                        _body([msg])
                    except Exception:
                        pass

                det_ins = []
                for k, df in overall_frames.items():
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    name_col = "RM_NAME" if "RM_NAME" in df.columns else None
                    metric_cols = [c for c in df.columns if "__" in c and pd.api.types.is_numeric_dtype(df[c])]
                    if name_col and metric_cols:
                        mcol = metric_cols[0]
                        top = df[[name_col, mcol]].dropna().sort_values(mcol, ascending=False).head(3)
                        if not top.empty:
                            det_ins.append("First look: " + ", ".join(f"{r[name_col]} ({float(r[mcol]):.2f})" for _, r in top.iterrows()))
                            break


                final = summarize_with_tools(
                    question,
                    overall_frames,
                    deterministic_insights=det_ins,
                    extra_refs={"rm_map": rm_mand_map},
                    on_tool=_ui_progress,   # ← stream progress here
                )

                



                if final.get("needs_more") and loop < loops_max:
                    fp = (final.get("followup_prompt") or "").strip()
                    req = final.get("required") or {}
                    pieces = [fp] if fp else []
                    views_hint = req.get("views") or []
                    cols_by_view = req.get("columns") or {}
                    if views_hint:
                        pieces.append(f"Focus on views: {', '.join(views_hint)}.")
                    if cols_by_view:
                        req_bits = []
                        for v, cols in cols_by_view.items():
                            if cols:
                                req_bits.append(f"{v}: {', '.join(cols)}")
                        if req_bits:
                            pieces.append("Ensure these columns → " + " | ".join(req_bits))
                    current_question = (question + "\n\n" + " ".join(pieces)).strip()
                    all_followups.append(current_question)
                    _body(["🔁 Adding a bit more context…"])
                    loop += 1
                    continue

                break

            try:
                st.markdown("")  # noop to ensure last render flushes
            except Exception:
                pass

        
        # Sidebar controls
        if "last_result" not in st.session_state: st.session_state["last_result"] = None
        if "user_prefs" not in st.session_state:
            st.session_state["user_prefs"] = {
                "widen_time": True,
                "dev_mode": False,
                "months_window": DEFAULT_MONTHS_WINDOW,
                "fit_on_load": True,
            }
        # Gate: only render data/insights/charts columns after first question is asked
        asked = bool(st.session_state.get("last_query")) or bool(st.session_state.get("last_result"))


        if not asked:
            st.markdown("### Ask a question to get started")
            st.caption("Examples: *Top 5 RMs in 2025 vs 2024 – what changed?* · *Compare Revenue & Sentiment across views*")
            st.markdown('<div class="sticky-spacer"></div>', unsafe_allow_html=True)
        else:
            # Layout

            col_data, col_insights, col_charts = st.columns(ratios, gap="small")
            res = st.session_state.get("last_result"); frames = res.get("frames", {}) if isinstance(res, dict) else {}
        
            # Data panel
            with col_data:
                st.markdown("### Data")
                # KPI cards stacked vertically
                if res and isinstance(res.get("kpis"), list) and res["kpis"]:
                    for i, card in enumerate(res["kpis"]):
                        st.markdown(
                            f"""
                            <div style="padding:12px;border-radius:10px;background:linear-gradient(90deg,#ffffff,#f7f8fb);box-shadow:0 1px 2px rgba(0,0,0,.05);margin-bottom:8px">
                              <div style="font-size:12px;color:#666">{card.get('title','')}</div>
                              <div style="font-size:20px;font-weight:800;margin-top:4px">{card.get('value','')}</div>
                              <div style="font-size:11px;color:#888;margin-top:4px">{card.get('sub','')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                if frames:
                    aliases  = st.session_state.get("frame_aliases", {}) or {}
                    ds_keys  = list(frames.keys())
                    labels   = [frame_label_for(k, frames.get(k), aliases) for k in ds_keys]
                    sel_lbl  = st.selectbox("Dataset to inspect", labels, index=0, key="inspect_ds")
                    sel_key  = ds_keys[labels.index(sel_lbl)]
                    df_show  = frames.get(sel_key)

                    if not isinstance(df_show, pd.DataFrame) or df_show.empty:
                        st.info("Selected dataset has no rows.")
                    else:
                        df_show = clean_cols(df_show)
                        # Replaced st_aggrid display with native Streamlit dataframe display per request
                        st.dataframe(df_show, height=360)
                else:
                    st.info("No frames yet. Ask a question below.")
                if st.session_state.get("user_prefs", {}).get("dev_mode", False) and res:
                    if res.get("log"):
                        with st.expander("Execution log"):
                            for l in res["log"]: st.write("-", l)
                    if res.get("sqls"):
                        with st.expander("SQL statements"):
                            for sid, sql in res["sqls"]: st.code(f"-- {sid}\n{sql}", language="sql")
                    if res.get("merges"):
                        with st.expander("Auto-joins performed"):
                            for m in res["merges"]:
                                st.write(f"• {m['name']} ← {m['left']} ⨝ {m['right']} on {m['on_left']}={m['on_right']} (rows={m['rows']})")
        
            # Insights
            with col_insights:
                st.markdown("### Insights")
                st.markdown(res.get("insights") if isinstance(res, dict) else "_No insights._")
                if st.session_state.get("user_prefs", {}).get("dev_mode", False) and res:
                    if res.get("followups"):
                        with st.expander("Follow-ups executed"):
                            for fid, view, sql, rows in res["followups"]: st.markdown(f"- {fid} from {view} — rows={rows}")
                    if "catalog" in res:
                        with st.expander("Catalog summary"):
                            for v, m in res["catalog"].items(): st.markdown(f"- {v}: fields={len(m.get('fields',[]))} ok={m.get('describe_ok',False)}")
        
            # Charts
            with col_charts:
                st.markdown("### Visualizations")
                if frames:
                    llm_flag = os.getenv("AI_INSIGHTS_LLM_CHART", "1")
                    use_llm_chart = str(llm_flag).strip().lower() not in ("0", "false", "no", "off")
                    if use_llm_chart:
                        frames_src = {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame) and not v.empty}
                        llm_question = (res.get("question") if isinstance(res, dict) else None) or ""
                        llm_summary = (res.get("insights") if isinstance(res, dict) else None) or ""
                        chart_result = compute_llm_chart(frames_src, llm_question, llm_summary)
                        if chart_result.get("fig") is not None:
                            st.plotly_chart(chart_result["fig"], use_container_width=True)
                            if chart_result.get("caption"):
                                st.caption(chart_result["caption"])
                            st.markdown("---")
                        elif st.session_state.get("user_prefs", {}).get("dev_mode", False):
                            msg = chart_result.get("message") or "LLM chart not available."
                            st.info(msg)

                else:
                    st.info("No frames yet to chart.")

        # Export nested helpers for headless reuse (FastAPI)
        # Only do this once to avoid repeated module-level assignments
        try:
            if globals().get('_orchestrate_will_export') and not globals().get('_orchestrate_exported'):
                import sys
                this_module = sys.modules[__name__]
                # Only export orchestrate (not all callables) for performance
                if 'orchestrate' in locals():
                    setattr(this_module, 'orchestrate', orchestrate)
                    globals()['_orchestrate_exported'] = True
        except Exception:
            pass

        # Chat input
        user_input = st.chat_input("Ask a question (e.g., Compare Revenue and Sentiment by RM across views)")
        if user_input:
            st.session_state["last_query"] = user_input
            try:
                out = orchestrate(
                    user_input,
                    DEFAULT_VIEWS,
                    months_window=int(
                        st.session_state.get("user_prefs", {}).get(
                            "months_window",
                            DEFAULT_MONTHS_WINDOW,
                        )
                    ),
                    widen_time_on_insufficient=st.session_state.get("user_prefs", {}).get("widen_time", True),
                )
                st.session_state["last_result"] = out; trace("user_query", {"q": user_input}); st.rerun()
            except Exception as e:
                trace("orchestrate_error", str(e) + " " + traceback.format_exc()[:MAX_TRACE_PAYLOAD]); st.error("Execution failed: " + str(e))
        
        # Trace (dev mode)
        if st.session_state.get("user_prefs",{}).get("dev_mode", False):
            with st.expander("Developer Trace (recent)"):
                tr = st.session_state.get("trace", [])[-200:]
                for t in tr:
                    st.markdown(f"**{t['when']} — {t['kind']}**")
                    try: st.json(t["payload"])
                    except Exception: st.text(str(t["payload"]))
    except Exception as e:
        st.error('App render failed: ' + str(e))

__all__ = ['render_main', 'render_sidebar', 'orchestrate']

if __name__ == "__main__":
    render_main()
