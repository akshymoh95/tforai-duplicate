from __future__ import annotations

import datetime
import json
import os
import re
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from rai_dynamic_query import DYNAMIC_QUERY_SCHEMA
from rai_semantic_registry import load_reasoners, load_registry_config


def _allow_multi_fact_aggregations() -> bool:
    try:
        cfg = load_registry_config()
        return bool(getattr(cfg, "allow_multi_fact_aggregations", False))
    except Exception:
        return False

# ============================================================
# JSON parsing helpers
# ============================================================


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _find_balanced_json(text: str) -> Optional[str]:
    if not text or "{" not in text:
        return None
    start = text.find("{")
    stack = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            stack += 1
        elif c == "}":
            stack -= 1
            if stack == 0:
                return text[start : i + 1]
    return None


def _parse_json_response(text: str, log_failure: bool = False) -> Dict[str, object]:
    """
    Parse JSON from LLM response with validation.
    Raises ValueError if response is invalid, logs details for debugging.
    """
    if not text or not isinstance(text, str):
        if log_failure:
            import sys
            print("[ERROR] Empty or invalid LLM response (not string)", file=sys.stderr)
        raise ValueError("LLM returned empty or non-string response")

    s = _strip_code_fences(text)

    # First attempt: direct JSON parse
    try:
        result = json.loads(s)
        if not isinstance(result, dict):
            raise ValueError(f"JSON root is not a dict: {type(result).__name__}")
        if not result:
            raise ValueError("JSON is empty dict (no spec content)")
        return result
    except json.JSONDecodeError as e:
        if log_failure:
            import sys
            print(f"[ERROR] Direct JSON parse failed: {e}", file=sys.stderr)
    except ValueError as e:
        if log_failure:
            import sys
            print(f"[ERROR] JSON validation failed: {e}", file=sys.stderr)
        raise

    # Second attempt: find balanced JSON
    candidate = _find_balanced_json(s)
    if candidate:
        try:
            result = json.loads(candidate)
            if not isinstance(result, dict):
                raise ValueError(f"Balanced JSON root is not a dict: {type(result).__name__}")
            if not result:
                raise ValueError("Balanced JSON is empty dict")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            if log_failure:
                import sys
                print(f"[ERROR] Balanced JSON parse failed: {e}", file=sys.stderr)

    # All attempts failed
    if log_failure:
        import sys
        print(f"[ERROR] Could not extract valid JSON from LLM response", file=sys.stderr)
        print(f"[ERROR] Raw response (first 1000 chars):\n{text[:1000]}", file=sys.stderr)
        print(f"[ERROR] Stripped response (first 1000 chars):\n{s[:1000]}", file=sys.stderr)
    
    raise ValueError(
        "LLM returned invalid JSON. "
        "Response does not contain a valid dict structure. "
        "Check logs for full response."
    )


def _load_semantic_registry() -> Optional[Dict[str, Any]]:
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
                return json.load(f)
        except Exception:
            continue
    return None


# ============================================================
# Datetime literal rewriting (prevents DateTime vs String IR errors)
# ============================================================

_REL_DATE_RE = re.compile(
    r"^\s*CURRENT_DATE\s*(?:([+-])\s*INTERVAL\s*'(\d+)\s*DAY'\s*)?$",
    re.IGNORECASE,
)
_REL_NOW_RE = re.compile(
    r"^\s*now\(\)\s*(?:([+-])\s*interval\s*'(\d+)\s*days?'\s*)?$",
    re.IGNORECASE,
)


def _format_dt_for_op(d: datetime.date, op: str) -> str:
    # >=/> => start of day, <=/< => end of day
    if op in (">=", ">"):
        return f"{d.isoformat()} 00:00:00"
    if op in ("<=", "<"):
        return f"{d.isoformat()} 23:59:59"
    return f"{d.isoformat()} 00:00:00"


def _rewrite_sqlish_datetime_value(val: str, op: str) -> Optional[str]:
    if not isinstance(val, str):
        return None

    m = _REL_DATE_RE.match(val)
    if m:
        sign = m.group(1)
        days = m.group(2)
        base = datetime.date.today()
        if sign and days:
            delta = timedelta(days=int(days))
            base = base + delta if sign == "+" else base - delta
        return _format_dt_for_op(base, op)

    m = _REL_NOW_RE.match(val)
    if m:
        sign = m.group(1)
        days = m.group(2)
        dt = datetime.datetime.now().replace(microsecond=0)
        if sign and days:
            delta = timedelta(days=int(days))
            dt = dt + delta if sign == "+" else dt - delta
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    return None


def _rewrite_where_tree(node: Any) -> Any:
    """
    Walk spec['where'] and rewrite {"value": "CURRENT_DATE - INTERVAL '7 DAY'"} etc. into ISO timestamps.
    Assumes 'between' has already been normalized away (see _normalize_where_predicates).
    """
    if isinstance(node, list):
        return [_rewrite_where_tree(x) for x in node]

    if not isinstance(node, dict):
        return node

    # logical nodes
    if "and" in node and isinstance(node["and"], list):
        node["and"] = [_rewrite_where_tree(x) for x in node["and"]]
        return node
    if "or" in node and isinstance(node["or"], list):
        node["or"] = [_rewrite_where_tree(x) for x in node["or"]]
        return node

    op = node.get("op")
    if op and "right" in node and isinstance(node["right"], dict):
        rv = node["right"].get("value")
        if isinstance(rv, str):
            rewritten = _rewrite_sqlish_datetime_value(rv, op)
            if rewritten is not None:
                node["right"]["value"] = rewritten

    if "left" in node:
        node["left"] = _rewrite_where_tree(node["left"])
    if "right" in node:
        node["right"] = _rewrite_where_tree(node["right"])

    return node


def _rewrite_datetime_literals_in_spec(spec: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(spec, dict):
        return spec
    if "where" in spec:
        spec["where"] = _rewrite_where_tree(spec["where"])
    return spec


# ============================================================
# Where normalization: convert non-schema "between" into >= and <=
# ============================================================


def _normalize_where_node(node: Any) -> List[Any]:
    """
    Returns a list of predicate nodes (because 'between' expands into two predicates).
    Supports recursion through 'and'/'or'.
    """
    if node is None:
        return []
    if isinstance(node, list):
        out: List[Any] = []
        for x in node:
            out.extend(_normalize_where_node(x))
        return out
    if not isinstance(node, dict):
        return [node]

    # Expand {"between": {...}} into two predicates
    if "between" in node and isinstance(node["between"], dict):
        b = node["between"]
        left = b.get("left")
        low = b.get("low")
        high = b.get("high")
        preds: List[dict] = []
        if left is not None and low is not None:
            preds.append({"op": ">=", "left": left, "right": low})
        if left is not None and high is not None:
            preds.append({"op": "<=", "left": left, "right": high})
        return preds

    # Preserve logical blocks but normalize inside them
    if "and" in node and isinstance(node["and"], list):
        node["and"] = [_normalize_where_node(x) for x in node["and"]]
        # flatten nested lists inside 'and'
        node["and"] = [y for xs in node["and"] for y in (xs if isinstance(xs, list) else [xs])]
        return [node]

    if "or" in node and isinstance(node["or"], list):
        node["or"] = [_normalize_where_node(x) for x in node["or"]]
        node["or"] = [y for xs in node["or"] for y in (xs if isinstance(xs, list) else [xs])]
        return [node]

    return [node]


def _normalize_where_predicates(spec: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(spec, dict):
        return spec
    wh = spec.get("where")
    if wh is None:
        return spec
    normalized = _normalize_where_node(wh)
    # Always store as a list for consistency
    spec["where"] = normalized if isinstance(normalized, list) else [normalized]
    return spec


# ============================================================
# Prompt builder
# ============================================================


def build_dynamic_query_prompt(
    question: str,
    ontology_text: str,
    join_keys: List[str],
    default_limit: int,
    failure_context: Optional[str] = None,
    reasoner_base_fields: Optional[List[str]] = None,
) -> str:
    join_hint = ", ".join(join_keys) if join_keys else "id, entity_id, date, month, year"
    failure_block = ""
    if failure_context:
        failure_block = (
            "PREVIOUS SPEC WAS REJECTED BY VALIDATOR (FIX THIS):\n"
            f"{failure_context}\n\n"
            "Fix by adjusting entities/fields/joins/aggregations so the query executes.\n"
            "Return ONLY the corrected JSON spec.\n\n"
        )
    reasoner_block = ""
    if reasoner_base_fields:
        reasoner_block = (
            "REASONER INPUT FIELDS (MUST INCLUDE IN BASE RESULT):\n"
            "====================================================\n"
            "The following fields are required as inputs for selected reasoners. Ensure they appear in the base result.\n"
            "They may be aliased, but must be present as columns in the base dataframe.\n"
            f"Required fields (can be aliased): {', '.join(reasoner_base_fields)}\n\n"
        )

    return (
        "You are a data planner for RelationalAI.\n"
        "Convert the user question into ONE JSON object that conforms exactly to the JSON Schema below.\n"
        "Return ONLY the JSON object. No code fences, no markdown, no extra text.\n\n"
        f"Today is: {datetime.date.today().isoformat()}\n\n"
        f"{failure_block}"
        "STRICT ONTOLOGY RULES (MUST FOLLOW):\n"
        "====================================\n"
        "1) Use only entities and fields that appear in the ontology.\n"
        "2) FIELD OWNERSHIP: Only use alias.prop if that prop is listed under that entity in the ontology.\n"
        "3) TYPE SAFETY: Do NOT use fields whose dtype is missing/Any/Unresolved in the ontology.\n"
        "   If a needed field is untyped, omit it and use the closest typed alternative.\n"
        "4) Prefer DERIVED fields listed in the ontology (e.g., availability_pct, oee_pct) instead of recomputing.\n\n"
        "MINIMAL BINDS (MUST FOLLOW):\n"
        "============================\n"
        "Bind the smallest set of entities needed to answer the question.\n"
        "Prefer one source of truth per measure.\n\n"
        "JOIN SAFETY RULE (MUST FOLLOW):\n"
        "===============================\n"
        "Goal: the final query must be ONE connected join graph.\n"
        "\n"
        "Allowed ways to connect entities:\n"
        "  A) BindPath edges in `bind` (PREFERRED)\n"
        "  B) Explicit join predicates (==) in `where` (only if no relationship exists)\n"
        "\n"
        "WHERE-EMPTY RULE:\n"
        "- It is OK for `where` to be empty IF all joins are expressed via BindPath in `bind`.\n"
        "- If you bind >1 entity AND you did NOT use BindPath to connect them, then `where` MUST include join predicates.\n"
        "\n"
        "CONNECTEDNESS RULE:\n"
        "- Every additional bound alias must be connected to an earlier alias by either:\n"
        "    - a BindPath from an earlier alias, OR\n"
        "    - an explicit equality join predicate.\n"
        "- If you cannot connect an alias, REMOVE that alias from bind.\n"
        "\n"
        "NO DUPLICATION:\n"
        "- Do NOT duplicate join logic: if a join is expressed via BindPath, do NOT repeat the same join in `where`.\n\n"
        "USE THE KG (PREFERRED):\n"
        "=======================\n"
        "BindPath (KG traversal):\n"
        "- Use BindPath when joining entities via a known registry/ontology relationship.\n"
        "- BindPath is for traversing ENTITY-TO-ENTITY relationships. It is not for selecting columns.\n"
        "- Each BindPath step MUST be a relationship name from the registry/ontology (not a column name).\n"
        "- When you use BindPath, do NOT also add explicit join predicates (==) for the same relationship in where.\n"
        "- Use explicit where equality joins only when no suitable relationship exists in the registry.\n\n"
        "Preference order:\n"
        "1) Single-entity queries (filter/aggregate one table)\n"
        "2) BindPath joins via relationships\n"
        "3) Explicit joins (==) only if no relationship exists\n\n"
        "FANOUT / MULTI-FACT JOIN RULE (MUST FOLLOW):\n"
        "===========================================\n"
        "Do NOT join two high-cardinality event/fact entities together in one aggregated query.\n"
        "For aggregated summaries: pick ONE primary fact entity and join only lookup/dimension entities for labels.\n"
        "For evidence/drilldown: create a separate drilldown query using the event entity as primary.\n\n"
        "TIME WINDOW RULE (MUST FOLLOW):\n"
        "===============================\n"
        "If the user says last week/last month/this week/this month/yesterday/past N days, ALWAYS add a concrete time window.\n"
        "Also set meta.time_window with {start,end,mode}.\n"
        " - mode='overlap' for interval events with start/end timestamps (interval semantics)\n"
        " - mode='point' for point-in-time facts (e.g., entry_on)\n"
        "Do NOT use SQL date expressions in values (no CURRENT_DATE, no INTERVAL, no now()).\n"
        "Use ISO timestamps like '2026-01-20 00:00:00'.\n\n"
        "INTERVAL WINDOW SEMANTICS (MUST FOLLOW):\n"
        "=======================================\n"
        "For interval/timed events, a window means INTERVAL OVERLAP:\n"
        "  start_time <= window_end AND end_time >= window_start\n"
        "When applying overlap, emit it as an explicit AND clause (two predicates) in `where`.\n"
        "Do NOT filter only by start_time when answering interval-in-window questions.\n\n"
        "RECURRENCE ACROSS ASSETS (MUST FOLLOW):\n"
        "======================================\n"
        "If the question asks 'recurring across assets/entities', do NOT group by the asset identifier field.\n"
        "Instead, group by fault/cause and include entities_affected = count_distinct(<asset_id_field>).\n\n"
        "GROUP BY + AGGREGATIONS RULES (MUST FOLLOW):\n"
        "============================================\n"
        "If `aggregations` is present:\n"
        "  - `select` MUST contain ONLY the group_by dimensions.\n"
        "  - Put all metrics in `aggregations`.\n"
        "  - NEVER reuse the same `as` name in both select and aggregations.\n\n"
        "SHADOWED VARIABLE RULE (MUST FOLLOW):\n"
        "=====================================\n"
        "Do NOT group_by and aggregate the same field (or same alias) in one query.\n"
        "Avoid using the same alias for both a group_by dimension and an aggregation output.\n\n"
        f"{reasoner_block}"
        "RAI REASONER RELATIONS (NATIVE RAI):\n"
        "===================================\n"
        "Some pipelines materialize reasoner drilldown outputs as entities named:\n"
        "  Reasoner_<reasoner_id>_<step_id>\n"
        "\n"
        "IMPORTANT:\n"
        "- Treat Reasoner_* entities as PARAMETERIZED outputs.\n"
        "- Do NOT bind Reasoner_* entities by default.\n"
        "- Only bind a Reasoner_* entity if the user explicitly asks for the underlying evidence rows\n"
        "  AND the required inputs are available in the spec (e.g., window_start/window_end and required keys).\n"
        "\n"
        "DEFAULT BEHAVIOR:\n"
        "- Prefer binding base entities directly (e.g., primary event/fact entities).\n"
        "- Use reasoners as selection/routing hints, not as required query sources.\n\n"
        "TEXT FILTER RULE:\n"
        "=================\n"
        "For fuzzy text matching, you MAY use:\n"
        "  - op='contains' (case-insensitive substring)\n"
        "  - op='ilike' with % wildcards (case-insensitive)\n"
        "  - op='like' with % wildcards (case-sensitive)\n\n"
        "ORDER BY RULE (MUST FOLLOW):\n"
        "============================\n"
        "If `aggregations` is present, order_by should reference either:\n"
        "  (a) a group_by dimension term, OR\n"
        "  (b) an aggregation alias using: {'term': {'value': '<agg_as_name>'}, 'dir': 'asc|desc'}\n\n"
        "ANALYTICAL QUESTIONS RULE:\n"
        "==========================\n"
        "For totals/averages/top/bottom/trends: use group_by + aggregations.\n"
        f"Join hint (use only if no BindPath relationship exists): {join_hint}.\n\n"
        "LIMIT RULE - CRITICAL (DOMAIN-AGNOSTIC):\n"
        "==========================================\n"
        "RULE 1: Questions WITHOUT comparison keywords (e.g., 'Top 5 RMs in 2025'):\n"
        "  - Use limit = N as requested\n\n"
        "RULE 2: Questions WITH comparison keywords (e.g., 'Top 5 RMs in 2025 vs 2024', 'Top 5 departments by revenue and headcount'):\n"
        "  - ALWAYS use a HIGH limit (e.g., 100-200) when there is group_by\n"
        "  - Let order_by sort all groups by the aggregation metric descending\n"
        "  - RAI will naturally return all groups ranked, achieving 'top N per group' semantics\n"
        "  - Example: 'Top 5 RMs vs by year' → limit=100, group_by=[rm, year], order_by=performance DESC → returns 50 rows (5 RMs × 2 years, ranked)\n"
        "  - Comparison keywords to detect: ' vs ', ' versus ', ' and ', ' by ', ' per ', ' in each ', ' for each '\n"
        "  - NEVER use limit=N for comparison queries; use limit=N*10 or higher\n\n"
        f"RULE 3: For all other cases: use limit = {default_limit} if not explicitly requested.\n\n"

        "JSON Schema:\n"
        f"{DYNAMIC_QUERY_SCHEMA}\n\n"
        "Ontology:\n"
        f"{ontology_text}\n\n"
        f"Question: {question}\n"
    )



# ============================================================
# Reasoner detection
# ============================================================


def build_reasoner_detection_prompt(question: str) -> str:
    reasoners = load_reasoners()
    reasoner_list_items = []
    for r in reasoners:
        rid = getattr(r, "id", None)
        if not rid:
            continue
        
        rtype = getattr(r, "type", None) or "unknown"
        graph_id = getattr(r, "graph_id", None)
        graph_suffix = f", graph_id={graph_id}" if graph_id else ""
        
        desc = getattr(r, "description", None) or getattr(r, "name", None) or "No description"
        desc = str(desc).strip() if desc else "No description"
        
        reasoner_list_items.append(f"- {rid} ({rtype}{graph_suffix}): {desc}")
    
    reasoner_list = "\n".join(reasoner_list_items) if reasoner_list_items else "(No reasoners available)"

    return f"""
You are selecting reasoners to run inside RAI.
Graph reasoners execute as graph queries within RAI (not post-processing).

AVAILABLE REASONERS:
{reasoner_list}

SELECTION RULES (MUST FOLLOW):
- If the user asks to explain/why/root-cause/diagnose/drilldown/evidence, you MUST return at least one reasoner_id.
- Choose the most relevant reasoners based on keyword overlap between the question and reasoner descriptions.
- Return 1 to 4 reasoner_ids (not more than 4).

Return ONLY JSON with keys: reasoning, reasoner_ids.
Question: {question}
""".strip()


def _extract_reasoner_base_fields(reasoner_ids: List[str]) -> List[str]:
    fields: set[str] = set()
    if not reasoner_ids:
        return []
    reasoner_map = {r.id: r for r in load_reasoners() if getattr(r, "id", None)}
    for rid in reasoner_ids:
        reasoner = reasoner_map.get(rid)
        if reasoner and str(getattr(reasoner, "type", "")).strip().lower() == "graph_reasoner":
            continue
        plan = getattr(reasoner, "drilldown_plan", None) or {}
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
            for _, src in inputs.items():
                if isinstance(src, str) and src.startswith("row."):
                    fields.add(src[4:])
    return sorted(fields)


def _fallback_reasoner_ids(question: str) -> List[str]:
    """
    Deterministic fallback: if user asks to explain/diagnose/root-cause, run relevant reasoners.
    Generic across domains using keyword overlap against reasoner metadata.
    """
    q = (question or "").lower()
    reasoners = load_reasoners()
    available = [r.id for r in reasoners if getattr(r, "id", None)]
    if not available:
        return []

    explain_intent = any(
        k in q
        for k in (
            "explain",
            "why",
            "root cause",
            "diagnose",
            "drill",
            "drilldown",
            "evidence",
            "break down",
            "breakdown",
        )
    )
    if not explain_intent:
        return []

    tokens = set(re.findall(r"[a-z0-9_]+", q))
    scored: List[Tuple[int, str]] = []
    for r in reasoners:
        rid = getattr(r, "id", "") or ""
        text = " ".join(
            [
                rid,
                (getattr(r, "name", "") or ""),
                (getattr(r, "description", "") or ""),
            ]
        ).lower()
        rtokens = set(re.findall(r"[a-z0-9_]+", text))
        score = len(tokens & rtokens)

        rtype = (getattr(r, "type", "") or "").lower()
        if "explain" in rtype:
            score += 5
        if "audit" in rtype:
            score += 3

        scored.append((score, rid))

    scored.sort(reverse=True)
    picked = [rid for score, rid in scored if score > 0][:4]
    if not picked:
        picked = [rid for _, rid in scored][:2]
    return picked


# ============================================================
# Relative window inference + injection (ontology-driven time field pick)
# ============================================================

_REL_WINDOW_RE = re.compile(r"(?:last|past)\s+(\d+)\s*(days?|weeks?)", re.IGNORECASE)
_RECENT_KEYWORDS = (
    "this week",
    "last week",
    "past week",
    "this month",
    "last month",
    "past month",
    "last seven",
    "past seven",
)

_SQLY_DATE_TOKENS = (
    "CURRENT_",
    "INTERVAL",
    "DATEADD",
    "DATEDIFF",
    "TIMESTAMP",
    "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
)


def _infer_recent_days(question: str) -> Optional[int]:
    q = (question or "").lower()
    m = _REL_WINDOW_RE.search(q)
    if m:
        count = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("week"):
            return max(1, count * 7)
        return max(1, count)
    if any(k in q for k in _RECENT_KEYWORDS):
        if "month" in q:
            return 30
        return 7
    return None


_TIME_FIELD_PRIORITY = [
    "event_time",
    "event_ts",
    "timestamp",
    "ts",
    "created_at",
    "updated_at",
    "start_at",
    "end_at",
    "start_time",
    "end_time",
    "operation_start",
    "entry_on",
    "createdon",
    "created_on",
    "result_on",
    "order_date",
    "invoice_date",
    "signup_date",
    "transaction_time",
    "date",
]

def _extract_time_fields_for_entity(ontology_text: str, entity: str) -> List[str]:
    """
    More robust: scans the entity block and extracts field names whose dtype contains DATE/TIMESTAMP.
    Falls back to known time field names if parsing is imperfect.
    """
    if not ontology_text or not entity:
        return []

    lines = ontology_text.splitlines()
    block: List[str] = []
    in_entity = False

    for line in lines:
        if line.startswith(f"- {entity} "):
            in_entity = True
            continue
        if in_entity:
            if line.startswith("- "):  # next entity
                break
            block.append(line)

    text = "\n".join(block)
    if not text:
        return []

    # First try: parse "fields:" line
    time_fields: List[str] = []
    m = re.search(r"fields:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        fields_line = m.group(1)
        # token pattern: name:TYPE(...)
        for match in re.finditer(r"([A-Za-z0-9_]+)\s*:\s*([A-Za-z_]+(?:\([0-9,]+\))?)", fields_line):
            name, ftype = match.group(1), match.group(2)
            if re.search(r"(DATE|TIMESTAMP)", ftype, re.IGNORECASE):
                time_fields.append(name)

    # Fallback: known fields present in block
    if not time_fields:
        lower_block = text.lower()
        for cand in _TIME_FIELD_PRIORITY:
            if re.search(rf"\b{re.escape(cand.lower())}\b", lower_block):
                time_fields.append(cand)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for f in time_fields:
        fl = f.lower()
        if fl not in seen:
            seen.add(fl)
            out.append(f)
    return out

def _score_time_field(name: str) -> int:
    n = (name or "").lower()
    score = 0
    if "start" in n:
        score += 6
    if "created" in n or "entry" in n:
        score += 5
    if "end" in n:
        score += 3
    if "time" in n:
        score += 2
    if "date" in n:
        score += 2
    return score


def _pick_time_fields_for_window(
    ontology_text: str, binds: List[Dict[str, object]]
) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Returns (alias, start_field, end_field_or_none).
    If a bind has both start_time and end_time, returns those for overlap windows.
    Otherwise chooses best single time field by priority.
    """
    for b in binds:
        if not isinstance(b, dict):
            continue
        alias = b.get("alias")
        entity = b.get("entity")
        if not alias or not entity:
            continue
        fields = _extract_time_fields_for_entity(ontology_text, entity)
        fset = {f.lower() for f in fields}

        if "start_time" in fset and "end_time" in fset:
            return (alias, "start_time", "end_time")

        for cand in _TIME_FIELD_PRIORITY:
            if cand.lower() in fset:
                return (alias, cand, None)

    return None


def _where_has_alias_prop(where: Any, alias: str, prop: str) -> bool:
    found = False

    def walk(node: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        for side in ("left", "right"):
            part = node.get(side)
            if isinstance(part, dict) and part.get("alias") == alias and part.get("prop") == prop:
                found = True
                return
        if "and" in node:
            walk(node.get("and"))
        if "or" in node:
            walk(node.get("or"))

    walk(where)
    return found


def _inject_last_n_days_filter_if_missing(
    spec: Dict[str, object],
    question: str,
    ontology_text: str,
    days: Optional[int] = None,
) -> Dict[str, object]:
    if not isinstance(spec, dict) or not days or days <= 0:
        return spec

    q = (question or "").lower()
    wants_recent = any(k in q for k in _RECENT_KEYWORDS) or _REL_WINDOW_RE.search(q)
    if not wants_recent:
        return spec

    binds = spec.get("bind") or []
    if not isinstance(binds, list) or not binds:
        return spec

    chosen = _pick_time_fields_for_window(ontology_text, binds)
    if not chosen:
        return spec

    alias, start_field, end_field = chosen

    wh = spec.get("where") or []
    if not isinstance(wh, list):
        wh = [wh]

    # If already has any filter referencing the start_field, do not inject (avoid double windows)
    if _where_has_alias_prop(wh, alias, start_field):
        spec["where"] = wh
        return spec

    # Use naive UTC timestamps (safe for TIMESTAMP_NTZ)
    end = datetime.datetime.utcnow().replace(microsecond=0)
    start = end - datetime.timedelta(days=int(days))
    start_s = start.strftime("%Y-%m-%d %H:%M:%S")
    end_s = end.strftime("%Y-%m-%d %H:%M:%S")

    if end_field:
        # overlap semantics for interval tables
        wh.extend(
            [
                {"op": "<=", "left": {"alias": alias, "prop": start_field}, "right": {"value": end_s}},
                {"op": ">=", "left": {"alias": alias, "prop": end_field}, "right": {"value": start_s}},
            ]
        )
        mode = "overlap"
    else:
        # point semantics
        wh.extend(
            [
                {"op": ">=", "left": {"alias": alias, "prop": start_field}, "right": {"value": start_s}},
                {"op": "<=", "left": {"alias": alias, "prop": start_field}, "right": {"value": end_s}},
            ]
        )
        mode = "point"

    spec["where"] = wh

    # Attach meta.time_window so downstream (executor / reasoners) know intent
    meta = spec.get("meta") if isinstance(spec.get("meta"), dict) else {}
    meta["time_window"] = {"start": start_s, "end": end_s, "mode": mode}
    spec["meta"] = meta

    return spec


# ============================================================
# Validation helpers
# ============================================================


def _aliases_in_aggregations(spec: Dict[str, object]) -> set:
    out = set()
    for a in (spec.get("aggregations") or []):
        if not isinstance(a, dict):
            continue
        term = a.get("term")
        if isinstance(term, dict):
            alias = term.get("alias")
            if alias:
                out.add(alias)
    return out


def _aliases_in_binds(spec: Dict[str, object]) -> set:
    out = set()
    for b in (spec.get("bind") or []):
        if isinstance(b, dict) and b.get("alias"):
            out.add(b["alias"])
    return out


def _aliases_in_where(spec: Dict[str, object]) -> set:
    out = set()

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        left = node.get("left")
        right = node.get("right")
        if isinstance(left, dict) and left.get("alias"):
            out.add(left["alias"])
        if isinstance(right, dict) and right.get("alias"):
            out.add(right["alias"])
        if "and" in node:
            walk(node.get("and"))
        if "or" in node:
            walk(node.get("or"))

    walk(spec.get("where") or [])
    return out


def _group_by_as_names(spec: Dict[str, object]) -> set:
    out = set()
    for g in (spec.get("group_by") or []):
        if isinstance(g, dict):
            a = g.get("as")
            if a:
                out.add(a)
    return out


def _agg_alias_term_counts(spec: Dict[str, object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for a in (spec.get("aggregations") or []):
        if not isinstance(a, dict):
            continue
        term = a.get("term")
        if isinstance(term, dict):
            alias = term.get("alias")
            if alias:
                counts[alias] = counts.get(alias, 0) + 1
    return counts


def _join_edges_from_where(where: Any) -> List[tuple]:
    edges: List[tuple] = []

    def walk(node: Any) -> None:
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
        left = node.get("left")
        right = node.get("right")
        if op == "==" and isinstance(left, dict) and isinstance(right, dict):
            la = left.get("alias")
            ra = right.get("alias")
            if la and ra and la != ra:
                edges.append((la, ra))

    walk(where)
    return edges


def _is_connected_join_graph(bind_aliases: set, edges: List[tuple]) -> bool:
    if len(bind_aliases) <= 1:
        return True
    adj = {a: set() for a in bind_aliases}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)

    start = next(iter(bind_aliases))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == bind_aliases


def _has_sqlish_datetime_literals(where: Any) -> bool:
    found = False

    def walk(node: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        if not isinstance(node, dict):
            return
        right = node.get("right")
        if isinstance(right, dict):
            v = right.get("value")
            if isinstance(v, str):
                up = v.upper()
                if any(tok in up for tok in _SQLY_DATE_TOKENS):
                    found = True
                    return
        if "and" in node:
            walk(node.get("and"))
        if "or" in node:
            walk(node.get("or"))
        if "left" in node:
            walk(node.get("left"))
        if "right" in node:
            walk(node.get("right"))

    walk(where)
    return found


def _join_adj_from_where(where: Any) -> Dict[str, set]:
    adj: Dict[str, set] = {}
    for u, v in _join_edges_from_where(where):
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _bindpath_edges(spec: Dict[str, object]) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    for b in (spec.get("bind") or []):
        if not isinstance(b, dict):
            continue
        if "from" in b and "path" in b and b.get("alias") and b.get("from"):
            edges.append((b["from"], b["alias"]))
    return edges


def _aliases_in_bindpaths(spec: Dict[str, object]) -> set:
    out: set = set()
    for b in (spec.get("bind") or []):
        if isinstance(b, dict) and "from" in b and "path" in b:
            if b.get("from"):
                out.add(b["from"])
            if b.get("alias"):
                out.add(b["alias"])
    return out


def _aliases_in_select_like(spec: Dict[str, object]) -> set:
    out: set = set()
    for sec in ("select", "group_by"):
        for t in (spec.get(sec) or []):
            if isinstance(t, dict) and t.get("alias"):
                out.add(t["alias"])
    for a in (spec.get("aggregations") or []):
        term = a.get("term") if isinstance(a, dict) else None
        if isinstance(term, dict) and term.get("alias"):
            out.add(term["alias"])
    for ob in (spec.get("order_by") or []):
        term = (ob or {}).get("term") or {}
        if isinstance(term, dict) and term.get("alias"):
            out.add(term["alias"])
    return out


def _registry_maps() -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    reg = _load_semantic_registry()
    rel_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    rel_name_index: Dict[str, List[Dict[str, Any]]] = {}
    entities_by_name: Dict[str, Dict[str, Any]] = {}
    if not reg:
        return rel_index, rel_name_index, entities_by_name

    for ent in reg.get("entities", []) or []:
        if isinstance(ent, dict) and ent.get("name"):
            entities_by_name[ent["name"]] = ent

    for rel in reg.get("relationships", []) or []:
        if not isinstance(rel, dict):
            continue
        from_ent = rel.get("from_entity")
        rel_name = rel.get("name")
        if not from_ent or not rel_name:
            continue
        rel_index[(from_ent, rel_name)] = rel
        rel_name_index.setdefault(rel_name, []).append(rel)

    return rel_index, rel_name_index, entities_by_name


def _invalid_bindpath_chains(spec: Dict[str, Any]) -> Optional[str]:
    if not isinstance(spec, dict):
        return None
    binds = spec.get("bind") or []
    if not isinstance(binds, list):
        return None

    rel_index, _, _ = _registry_maps()
    if not rel_index:
        return None

    alias_to_entity: Dict[str, str] = {}
    for b in binds:
        if isinstance(b, dict) and b.get("alias") and b.get("entity"):
            alias_to_entity[b["alias"]] = b["entity"]

    for b in binds:
        if not isinstance(b, dict):
            continue
        if "from" in b and "path" in b:
            src_alias = b.get("from")
            steps = b.get("path") or []
            if not src_alias or src_alias not in alias_to_entity:
                return f"BindPath from alias '{src_alias}' cannot be resolved to an entity."
            cur_entity = alias_to_entity[src_alias]
            for step in steps:
                meta = rel_index.get((cur_entity, step))
                if not meta:
                    return f"BindPath invalid at entity '{cur_entity}' step '{step}'."
                cur_entity = meta.get("to_entity") or cur_entity
    return None


def _topo_sort_binds(binds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    entity_binds = [b for b in binds if isinstance(b, dict) and "entity" in b]
    path_binds = [b for b in binds if isinstance(b, dict) and "from" in b and "path" in b]
    ordered = list(entity_binds)
    bound = {b.get("alias") for b in entity_binds if b.get("alias")}

    remaining = list(path_binds)
    progress = True
    while remaining and progress:
        progress = False
        for b in remaining[:]:
            if b.get("from") in bound:
                ordered.append(b)
                if b.get("alias"):
                    bound.add(b.get("alias"))
                remaining.remove(b)
                progress = True

    ordered.extend(remaining)
    return ordered


def _normalize_bindpaths(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        return spec
    binds = spec.get("bind") or []
    if not isinstance(binds, list) or not binds:
        return spec

    rel_index, rel_name_index, entities_by_name = _registry_maps()
    if not rel_index:
        spec["bind"] = _topo_sort_binds(binds)
        return spec

    alias_to_entity: Dict[str, str] = {}
    new_binds: List[Dict[str, Any]] = []
    extra_where: List[Dict[str, Any]] = []

    for b in binds:
        if not isinstance(b, dict):
            new_binds.append(b)
            continue
        if "entity" in b:
            new_binds.append(b)
            if b.get("alias") and b.get("entity"):
                alias_to_entity[b["alias"]] = b["entity"]
            continue

        if "from" in b and "path" in b:
            src_alias = b.get("from")
            steps = b.get("path") or []
            if not src_alias or src_alias not in alias_to_entity:
                new_binds.append(b)
                continue

            cur_entity = alias_to_entity[src_alias]
            ok = True
            for step in steps:
                meta = rel_index.get((cur_entity, step))
                if not meta:
                    ok = False
                    break
                cur_entity = meta.get("to_entity") or cur_entity

            if ok:
                new_binds.append(b)
                if b.get("alias"):
                    alias_to_entity[b["alias"]] = cur_entity
                continue

            # salvage reverse single-hop BindPath (relationship defined opposite direction)
            if len(steps) == 1:
                step = steps[0]
                candidates = rel_name_index.get(step, [])
                chosen = None
                for cand in candidates:
                    if cand.get("to_entity") == cur_entity:
                        chosen = cand
                        break
                if chosen and b.get("alias"):
                    fact_entity = chosen.get("from_entity")
                    fact_alias = b.get("alias")
                    if fact_entity and fact_alias:
                        new_binds.append({"alias": fact_alias, "entity": fact_entity})
                        alias_to_entity[fact_alias] = fact_entity
                        join_on = chosen.get("join_on") or []
                        for l, r in join_on:
                            if l and r:
                                extra_where.append(
                                    {
                                        "op": "==",
                                        "left": {"alias": fact_alias, "prop": l},
                                        "right": {"alias": src_alias, "prop": r},
                                    }
                                )
                        continue

            # rewrite invalid multi-hop BindPath (best-effort)
            if steps:
                step1 = steps[0]
                meta1 = rel_index.get((alias_to_entity[src_alias], step1))
                if meta1:
                    inter_alias = f"{b.get('alias')}__inter"
                    new_binds.append({"alias": inter_alias, "from": src_alias, "path": [step1]})
                    inter_entity = meta1.get("to_entity")
                    if inter_entity:
                        alias_to_entity[inter_alias] = inter_entity

                    # Try to salvage second hop by binding its from_entity and joining on shared keys
                    if len(steps) >= 2 and inter_entity:
                        candidates = rel_name_index.get(steps[1], [])
                        chosen = None
                        for cand in candidates:
                            if cand.get("to_entity") == inter_entity:
                                chosen = cand
                                break
                        if chosen:
                            fact_entity = chosen.get("from_entity")
                            fact_alias = b.get("alias")
                            if fact_entity and fact_alias:
                                new_binds.append({"alias": fact_alias, "entity": fact_entity})
                                alias_to_entity[fact_alias] = fact_entity
                                join_on = chosen.get("join_on") or []
                                for l, r in join_on:
                                    if l and r:
                                        extra_where.append(
                                            {
                                                "op": "==",
                                                "left": {"alias": fact_alias, "prop": l},
                                                "right": {"alias": inter_alias, "prop": r},
                                            }
                                        )
                                continue
            new_binds.append(b)
            continue

        new_binds.append(b)

    if extra_where:
        spec["where"] = (spec.get("where") or []) + extra_where
    spec["bind"] = _topo_sort_binds(new_binds)
    return spec


def _join_adj_from_edges(edges: List[tuple]) -> Dict[str, set]:
    adj: Dict[str, set] = {}
    for u, v in edges or []:
        if not u or not v:
            continue
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _is_star_fanout(bind_aliases: set, agg_aliases: set, edges: List[tuple]) -> bool:
    """
    Detect star-join fanout pattern:
    - multiple aggregation-producing aliases connected only via a single hub alias,
      with no direct joins between the aggregation aliases.
    """
    if len(bind_aliases) < 3 or len(agg_aliases) < 2:
        return False

    adj = _join_adj_from_edges(edges)
    hubs = []
    for a in bind_aliases:
        if len((adj.get(a, set()) & bind_aliases)) >= 2:
            hubs.append(a)
    if not hubs:
        return False

    for hub in hubs:
        others = {x for x in agg_aliases if x != hub}
        if len(others) < 2:
            continue

        # if any direct edge exists between two agg aliases, not a pure star
        for x in others:
            for y in others:
                if x != y and y in adj.get(x, set()):
                    return False
        return True

    return False


def _entity_kind_from_registry(entity: str, registry: Optional[Dict[str, Any]]) -> Optional[str]:
    if not entity or not registry:
        return None
    entities = {e.get("name"): e for e in (registry.get("entities") or []) if isinstance(e, dict)}
    meta = entities.get(entity)
    if not meta:
        return None
    etype = str(meta.get("entity_type") or "").lower()
    desc = str(meta.get("description") or "").lower()
    name = str(entity).lower()
    text = " ".join([name, etype, desc])

    fact_kws = ("fact", "event", "metric", "metrics", "transaction", "log", "snapshot")
    dim_kws = ("dimension", "lookup", "reference", "master", "config", "dictionary")
    fact_hit = any(k in text for k in fact_kws)
    dim_hit = any(k in text for k in dim_kws)
    if fact_hit and not dim_hit:
        return "fact"
    if dim_hit and not fact_hit:
        return "dimension"
    return None


def _alias_kind_map(spec: Dict[str, object]) -> Dict[str, Optional[str]]:
    reg = _load_semantic_registry()
    out: Dict[str, Optional[str]] = {}
    for b in (spec.get("bind") or []):
        if not isinstance(b, dict):
            continue
        alias = b.get("alias")
        ent = b.get("entity")
        if alias and ent:
            out[alias] = _entity_kind_from_registry(str(ent), reg)
    return out


# ============================================================
# Spec normalization: grouping shape
# ============================================================


def _normalize_grouping_shape(spec: Dict[str, object]) -> Dict[str, object]:
    """
    Enforce consistent grouping semantics:
    - If aggregations exist and group_by missing: set group_by = select dims.
    - If aggregations exist and group_by exists: force select to exactly group_by dims.
    """
    aggs = spec.get("aggregations") or []
    if not aggs:
        return spec

    sel = spec.get("select") or []
    gb = spec.get("group_by") or []

    if not gb and isinstance(sel, list) and len(sel) > 0:
        new_gb = []
        for s in sel:
            if isinstance(s, dict) and s.get("alias") and s.get("prop"):
                new_gb.append(
                    {"alias": s["alias"], "prop": s["prop"], "as": s.get("as", s["prop"])}
                )
        if new_gb:
            spec["group_by"] = new_gb

    gb = spec.get("group_by") or []
    if isinstance(gb, list) and len(gb) > 0:
        spec["select"] = [
            {"alias": g["alias"], "prop": g["prop"], "as": g.get("as", g["prop"])}
            for g in gb
            if isinstance(g, dict) and g.get("alias") and g.get("prop")
        ]

    return spec


def _enhance_analytical_spec(spec: Dict[str, object], question: str, join_keys: List[str]) -> Dict[str, object]:
    # Temporarily disable auto group_by normalization to avoid accidental
    # group_by creation on measures (can trigger shadowed-variable errors).
    return spec


# ============================================================
# Alias de-duplication (avoid shadowed variable errors)
# ============================================================


def _ensure_unique_output_aliases(spec: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(spec, dict):
        return spec

    select = list(spec.get("select") or [])
    group_by = list(spec.get("group_by") or [])
    aggregations = list(spec.get("aggregations") or [])
    order_by = list(spec.get("order_by") or [])

    used = set()
    for t in group_by + select:
        if isinstance(t, dict) and t.get("as"):
            used.add(t["as"])

    renamed = {}
    agg_used = set()
    for a in aggregations:
        if not isinstance(a, dict):
            continue
        a_as = a.get("as")
        if not a_as:
            continue

        base = a_as
        candidate = a_as
        i = 2
        while candidate in used or candidate in agg_used:
            candidate = f"{base}_{i}"
            i += 1

        if candidate != a_as:
            a["as"] = candidate
            renamed[a_as] = candidate
        agg_used.add(candidate)

    if renamed and order_by:
        for ob in order_by:
            term = (ob or {}).get("term") or {}
            if isinstance(term, dict) and "value" in term:
                v = term.get("value")
                if v in renamed:
                    term["value"] = renamed[v]
                    ob["term"] = term

    spec["aggregations"] = aggregations
    if order_by:
        spec["order_by"] = order_by
    return spec


# ============================================================
# Validation
# ============================================================


def _validate_spec_or_raise(spec: Dict[str, object], mode: str = "base") -> None:
    binds = spec.get("bind") or []
    where = spec.get("where", None)

    # Cartesian product check
    bindpath_edges = _bindpath_edges(spec)
    if isinstance(binds, list) and len(binds) > 1:
        if (where is None or (isinstance(where, list) and len(where) == 0)) and not bindpath_edges:
            raise ValueError("bind has >1 entity but where is empty and no BindPath edges -> cartesian product.")

    # SQL-ish datetime string check (after normalization + rewrite, should be gone)
    if _has_sqlish_datetime_literals(spec.get("where") or []):
        raise ValueError("where contains SQL-like datetime literals. Use concrete ISO timestamps only.")

    # BindPath chain validity check
    invalid_chain = _invalid_bindpath_chains(spec)
    if invalid_chain:
        raise ValueError(f"Invalid BindPath chain: {invalid_chain}")

    # Ensure all bound aliases are referenced somewhere (where/bindpath/select/group_by/agg/order_by)
    bind_aliases = _aliases_in_binds(spec)
    where_aliases = _aliases_in_where(spec)
    if len(bind_aliases) > 1:
        bindpath_aliases = _aliases_in_bindpaths(spec)
        select_aliases = _aliases_in_select_like(spec)
        referenced = where_aliases | bindpath_aliases | select_aliases
        missing_aliases = bind_aliases - referenced
        if missing_aliases:
            raise ValueError(
                "Some bound aliases are not referenced in where/bindpath/select/group_by/aggregations/order_by: "
                + ", ".join(sorted(missing_aliases))
                + ". Remove them or add proper joins."
            )

        edges = _join_edges_from_where(spec.get("where") or []) + _bindpath_edges(spec)
        if not _is_connected_join_graph(bind_aliases, edges):
            raise ValueError(
                "Bound entities are not connected by join predicates (== between aliases). "
                "Add join conditions to connect all aliases into one graph, or remove extra binds."
            )

    # Drilldown mode constraints
    if mode == "drilldown":
        if spec.get("aggregations"):
            raise ValueError("drilldown specs must not include aggregations.")
        limit = spec.get("limit")
        if isinstance(limit, int) and limit > 200:
            raise ValueError("drilldown specs must use a small limit (<= 200).")
        return

    # Base mode: fanout / multi-fact prevention
    allow_multi_fact = _allow_multi_fact_aggregations()
    if spec.get("aggregations"):
        agg_aliases = set(_agg_alias_term_counts(spec).keys())
        alias_kinds = _alias_kind_map(spec)
        can_classify_agg = all(alias_kinds.get(a) in ("fact", "dimension") for a in agg_aliases)
        fact_agg_aliases = {a for a in agg_aliases if alias_kinds.get(a) == "fact"}

        if not allow_multi_fact:
            # star-join fanout: multiple aggregation sources connected only via a hub alias
            if len(bind_aliases) >= 3 and len(agg_aliases) >= 2:
                if _is_star_fanout(bind_aliases, agg_aliases, edges):
                    raise ValueError(
                        "Star-join fanout: multiple aggregation sources joined only via a shared hub alias. "
                        "Split into multiple queries (seed + drilldowns) or use reasoners."
                    )

            # heuristic: 3+ binds and at least 2 aliases contribute aggregation terms -> high risk
            if isinstance(binds, list) and len(binds) >= 3 and len(agg_aliases) >= 2:
                if not (can_classify_agg and len(fact_agg_aliases) <= 1):
                    # If the model tried to "launder" by putting event-table fields in group_by,
                    # agg_aliases still catches it because the aggregations reference those aliases.
                    # We keep this as a hard reject to avoid slow/wrong results.
                    raise ValueError(
                        "Aggregations reference multiple bound aliases in a 3+ bind query -> likely fanout/multi-fact join. "
                        "Use one primary fact entity for aggregates; use reasoners/drilldowns for event evidence."
                    )

            # two-table strong fanout signal: multiple terms from both aliases
            counts = _agg_alias_term_counts(spec)
            if isinstance(binds, list) and len(binds) == 2:
                if sum(1 for c in counts.values() if c >= 2) >= 2:
                    if not can_classify_agg or len(fact_agg_aliases) >= 2:
                        raise ValueError(
                            "Aggregations include multiple terms from both bound entities -> likely multi-fact join fanout. "
                            "Prefer one primary fact entity for aggregates; use separate drilldown for the other entity."
                        )

    # Aggregation shape check: if aggregations and group_by exist, select must be subset of group_by dims
    if spec.get("aggregations") and spec.get("group_by"):
        gb = {(g.get("alias"), g.get("prop")) for g in (spec.get("group_by") or []) if isinstance(g, dict)}
        for s in (spec.get("select") or []):
            if isinstance(s, dict) and "alias" in s and "prop" in s:
                if (s["alias"], s["prop"]) not in gb:
                    raise ValueError("aggregations present but select contains non-group_by field -> invalid shape.")

    # order_by validation for aggregated queries
    if spec.get("aggregations") and spec.get("order_by"):
        gb_pairs = {
            (g.get("alias"), g.get("prop"))
            for g in (spec.get("group_by") or [])
            if isinstance(g, dict)
        }
        gb_as = _group_by_as_names(spec)
        agg_names = {
            a.get("as")
            for a in (spec.get("aggregations") or [])
            if isinstance(a, dict) and a.get("as")
        }
        for ob in (spec.get("order_by") or []):
            term = (ob or {}).get("term") or {}
            if "value" in term:
                v = term["value"]
                if v not in agg_names and v not in gb_as:
                    raise ValueError(
                        f"order_by uses value '{v}' but it is neither an aggregation alias nor a group_by alias."
                    )
            elif "alias" in term and "prop" in term:
                if (term.get("alias"), term.get("prop")) not in gb_pairs:
                    raise ValueError(
                        "order_by references a raw field that is not in group_by while aggregations are present."
                    )


# ============================================================
# Main entry: generate spec + reasoners
# ============================================================

def _find_entity_alias(spec: Dict[str, object], entity: str) -> Optional[str]:
    for b in (spec.get("bind") or []):
        if isinstance(b, dict) and b.get("entity") == entity and b.get("alias"):
            return b["alias"]
    return None


def _bound_entities(spec: Dict[str, object]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for b in (spec.get("bind") or []):
        if isinstance(b, dict) and b.get("entity") and b.get("alias"):
            out.append((b["entity"], b["alias"]))
    return out


def _extract_fields_for_entity(ontology_text: str, entity: str) -> List[str]:
    if not ontology_text or not entity:
        return []

    lines = ontology_text.splitlines()
    block: List[str] = []
    in_entity = False

    for line in lines:
        if line.startswith(f"- {entity} "):
            in_entity = True
            continue
        if in_entity:
            if line.startswith("- "):
                break
            block.append(line)

    text = "\n".join(block)
    if not text:
        return []

    fields: List[str] = []
    m = re.search(r"fields:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        fields_line = m.group(1)
        field_pat = r"(?:^|,\s*)([A-Za-z0-9_]+)\s*:\s*[A-Za-z_]+(?:\([0-9,]+\))?\s*:"
        for match in re.finditer(field_pat, fields_line):
            fields.append(match.group(1))

    seen = set()
    out: List[str] = []
    for f in fields:
        fl = f.lower()
        if fl not in seen:
            seen.add(fl)
            out.append(f)
    return out


def _pick_interval_entity(
    spec: Dict[str, object], ontology_text: str
) -> Optional[Tuple[str, str, str, str]]:
    pairs = [
        ("start_time", "end_time"),
        ("start_ts", "end_ts"),
        ("start", "end"),
        ("begin", "finish"),
    ]

    for entity, alias in _bound_entities(spec):
        fields = _extract_fields_for_entity(ontology_text, entity)
        if not fields:
            continue
        lower_map = {f.lower(): f for f in fields}
        for start_name, end_name in pairs:
            if start_name in lower_map and end_name in lower_map:
                return (alias, entity, lower_map[start_name], lower_map[end_name])

        time_fields = _extract_time_fields_for_entity(ontology_text, entity)
        if time_fields:
            starts = [f for f in time_fields if re.search(r"(start|begin)", f, re.IGNORECASE)]
            ends = [f for f in time_fields if re.search(r"(end|finish)", f, re.IGNORECASE)]
            if starts and ends:
                return (alias, entity, starts[0], ends[0])

    return None


def _pick_asset_id_field(fields: List[str]) -> Optional[str]:
    if not fields:
        return None
    lower_map = {f.lower(): f for f in fields}
    priority = [
        "machine_id",
        "asset_id",
        "device_id",
        "user_id",
        "account_id",
        "customer_id",
        "employee_id",
        "pu_id",
    ]
    for cand in priority:
        if cand in lower_map:
            return lower_map[cand]
    for f in fields:
        lf = f.lower()
        if lf.endswith("_id") or (lf.endswith("id") and len(lf) > 2):
            return f
    return None


def _pick_row_id_field(fields: List[str]) -> Optional[str]:
    if not fields:
        return None
    lower_map = {f.lower(): f for f in fields}
    priority = [
        "event_id",
        "row_id",
        "record_id",
        "incident_id",
        "ticket_id",
        "id",
        "tedet_id",
    ]
    for cand in priority:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _pick_time_field_for_entity(ontology_text: str, entity: str) -> Optional[str]:
    time_fields = _extract_time_fields_for_entity(ontology_text, entity)
    if not time_fields:
        return None
    lower_map = {f.lower(): f for f in time_fields}
    for cand in _TIME_FIELD_PRIORITY:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return sorted(time_fields, key=_score_time_field, reverse=True)[0]


def _pick_availability_field(fields: List[str]) -> Optional[str]:
    if not fields:
        return None
    patterns = ["availability", "uptime", "reliability", "on_time", "ontime", "sla"]
    for f in fields:
        lf = f.lower()
        if any(p in lf for p in patterns):
            return f
    return None


def _mentions_recurring_across_assets(q: str) -> bool:
    if "recurring across" in q:
        return True
    if "recurring" in q and "across" in q:
        return True
    if "across machines" in q or "across assets" in q or "across entities" in q:
        return True
    return False


def _pick_metrics_entity_for_trend(
    spec: Dict[str, object], ontology_text: str, interval_alias: Optional[str]
) -> Optional[Tuple[str, str, str]]:
    candidates = _bound_entities(spec)
    if interval_alias:
        candidates = [b for b in candidates if b[1] != interval_alias] + [
            b for b in candidates if b[1] == interval_alias
        ]
    for entity, alias in candidates:
        time_field = _pick_time_field_for_entity(ontology_text, entity)
        if time_field:
            return (alias, entity, time_field)
    return None


def _semantic_rewrite_for_question(
    spec: Dict[str, object], question: str, ontology_text: str
) -> Dict[str, object]:
    """
    Deterministic semantic fixes AFTER the LLM, to ensure the spec answers the question.
    - Recurring across assets => do not group by asset id; add entities_affected=count_distinct(asset_id)
    - Downtime-in-window for timed events => ensure meta.time_window.mode=overlap if interval table detected
    - Trends => force time series output (entity + time) when asked
    """
    if not isinstance(spec, dict):
        return spec

    q = (question or "").lower()

    interval_info = _pick_interval_entity(spec, ontology_text)
    interval_alias = interval_info[0] if interval_info else None
    interval_entity = interval_info[1] if interval_info else None

    # 1) Recurring across assets
    if _mentions_recurring_across_assets(q):
        rec_alias = interval_alias
        rec_entity = interval_entity
        if not rec_alias or not rec_entity:
            for entity, alias in _bound_entities(spec):
                fields = _extract_fields_for_entity(ontology_text, entity)
                if _pick_asset_id_field(fields):
                    rec_alias = alias
                    rec_entity = entity
                    break

        if rec_alias and rec_entity:
            fields = _extract_fields_for_entity(ontology_text, rec_entity)
            asset_id_field = _pick_asset_id_field(fields)
            if asset_id_field:
                # remove asset id from select/group_by
                spec["select"] = [
                    s for s in (spec.get("select") or [])
                    if not (isinstance(s, dict) and s.get("alias") == rec_alias and s.get("prop") == asset_id_field)
                ]
                spec["group_by"] = [
                    g for g in (spec.get("group_by") or [])
                    if not (isinstance(g, dict) and g.get("alias") == rec_alias and g.get("prop") == asset_id_field)
                ]

                aggs = list(spec.get("aggregations") or [])
                aggs = [
                    a
                    for a in aggs
                    if not (isinstance(a, dict) and a.get("as") in ("machines_affected", "entities_affected"))
                ]
                aggs.append(
                    {
                        "op": "count_distinct",
                        "term": {"alias": rec_alias, "prop": asset_id_field},
                        "as": "entities_affected",
                    }
                )

                row_id_field = _pick_row_id_field(fields) or asset_id_field
                if not any(isinstance(a, dict) and a.get("as") == "event_count" for a in aggs):
                    aggs.append(
                        {
                            "op": "count",
                            "term": {"alias": rec_alias, "prop": row_id_field},
                            "as": "event_count",
                        }
                    )

                spec["aggregations"] = aggs
                spec["order_by"] = [{"term": {"value": "entities_affected"}, "dir": "desc"}]
                if "limit" not in spec:
                    spec["limit"] = 50

    # 2) Trend/deteriorating availability: force time series output
    # Disabled by default (opt-in) to avoid unintended group_by injection.
    enable_trend_rewrite = os.environ.get("AI_INSIGHTS_ENABLE_TREND_REWRITE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if enable_trend_rewrite and any(k in q for k in ("trend", "deteriorat", "declin", "getting worse")):
        trend_info = _pick_metrics_entity_for_trend(spec, ontology_text, interval_alias)
        if trend_info:
            om_alias, om_entity, time_field = trend_info
            gb = list(spec.get("group_by") or [])
            sel = list(spec.get("select") or [])
            has_time = any(
                isinstance(g, dict) and g.get("alias") == om_alias and g.get("prop") == time_field for g in gb
            )
            if not has_time:
                gb.append({"alias": om_alias, "prop": time_field, "as": time_field})
                sel.append({"alias": om_alias, "prop": time_field, "as": time_field})
            spec["group_by"] = gb
            spec["select"] = sel

            aggs = list(spec.get("aggregations") or [])
            fields = _extract_fields_for_entity(ontology_text, om_entity)
            availability_field = _pick_availability_field(fields)
            if availability_field:
                has_avail = any(
                    isinstance(a, dict)
                    and a.get("op") == "avg"
                    and isinstance(a.get("term"), dict)
                    and a["term"].get("alias") == om_alias
                    and a["term"].get("prop") == availability_field
                    for a in aggs
                )
                if not has_avail:
                    aggs.append(
                        {
                            "op": "avg",
                            "term": {"alias": om_alias, "prop": availability_field},
                            "as": f"avg_{availability_field}",
                        }
                    )
            spec["aggregations"] = aggs

            spec["order_by"] = [
                {"term": {"alias": om_alias, "prop": time_field, "as": time_field}, "dir": "asc"}
            ]
            if "limit" not in spec:
                spec["limit"] = 2000

    return spec


def generate_dynamic_query_spec(
    cortex_complete: Callable[[str], str],
    question: str,
    ontology_text: str,
    join_keys: List[str],
    *,
    default_limit: int = 2000,
    max_retries: int = 6,
    failure_context: Optional[str] = None,
) -> Tuple[Dict[str, object], List[str]]:
    """
    Returns: (spec dict, list of reasoner_ids)
    """
    # Reasoner selection (used to include required inputs in base spec)
    reasoner_ids: List[str] = []
    reasoner_base_fields: List[str] = []
    try:
        reasoner_prompt = build_reasoner_detection_prompt(question)
        reasoner_response = cortex_complete(reasoner_prompt)
        reasoner_data = _parse_json_response(
            reasoner_response if isinstance(reasoner_response, str) else "", log_failure=False
        )
        if isinstance(reasoner_data, dict):
            reasoner_ids = reasoner_data.get("reasoner_ids", []) or []
    except Exception:
        reasoner_ids = []

    if not reasoner_ids:
        reasoner_ids = _fallback_reasoner_ids(question)

    reasoner_base_fields = _extract_reasoner_base_fields(reasoner_ids)

    required_fields = {"bind", "select"}
    local_failure = failure_context

    for _attempt in range(max_retries):
        prompt = build_dynamic_query_prompt(
            question,
            ontology_text,
            join_keys,
            default_limit,
            local_failure,
            reasoner_base_fields=reasoner_base_fields,
        )
        raw = cortex_complete(prompt)
        spec = _parse_json_response(raw if isinstance(raw, str) else "", log_failure=False)

        if not isinstance(spec, dict):
            local_failure = "LLM returned non-JSON or non-object response."
            continue

        if "limit" not in spec:
            spec["limit"] = default_limit

        missing = required_fields - set(spec.keys())
        if missing:
            local_failure = f"Missing required keys: {sorted(list(missing))}"
            continue

        # Normalize where (convert 'between' to >= and <=) BEFORE datetime rewrite/validation
        spec = _normalize_where_predicates(spec)

        # Normalize grouping shape for aggregates
        spec = _enhance_analytical_spec(spec, question, join_keys)

        # Rewrite SQL-ish datetime strings if any
        spec = _rewrite_datetime_literals_in_spec(spec)

        # Inject recent window if implied by question (adds meta.time_window too)
        spec = _inject_last_n_days_filter_if_missing(spec, question, ontology_text, days=_infer_recent_days(question))

        # Deterministic semantic corrections to ensure spec answers the question
        spec = _semantic_rewrite_for_question(spec, question, ontology_text)

        # De-duplicate output aliases to avoid shadowed-variable errors
        spec = _ensure_unique_output_aliases(spec)

        # Normalize/repair BindPath chains and order binds
        spec = _normalize_bindpaths(spec)

        # Validate final spec
        try:
            _validate_spec_or_raise(spec, mode="base")
        except Exception as e:
            local_failure = f"{type(e).__name__}: {str(e)}"
            continue

        return spec, reasoner_ids

    import sys
    if local_failure:
        print(
            f"[ERROR] Failed to generate valid spec after {max_retries} attempts. Last failure: {local_failure}",
            file=sys.stderr,
        )
    else:
        print(f"[ERROR] Failed to generate valid spec after {max_retries} attempts.", file=sys.stderr)

    return {"limit": default_limit}, reasoner_ids
