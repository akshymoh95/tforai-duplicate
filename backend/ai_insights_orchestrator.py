from __future__ import annotations

import contextvars
import time
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ENV_PATH = Path(os.environ.get("AI_INSIGHTS_ENV", Path(__file__).with_name(".env")))
load_dotenv(ENV_PATH, override=False)

# KG Enhancement control
_KG_ENHANCEMENT_ENABLED = os.environ.get("KG_ENHANCEMENT_ENABLED", "true").lower() == "true"
_KG_ENHANCEMENT_TIMEOUT = float(os.environ.get("KG_ENHANCEMENT_TIMEOUT", "5"))  # seconds

# Global stage tracking
_ORCHESTRATION_STATE = {
    "current_stage": "planning",
    "stage_callback": None,
    "requests": {},
}

# Query and LLM result caching to reduce redundant calls
_QUERY_RESULT_CACHE = {}  # Hash(SQL) -> DataFrame
_LLM_RESULT_CACHE = {}    # Hash(prompt) -> LLM response
_CACHE_MAX_SIZE = 100     # Prevent unbounded memory growth
_CACHE_TTL = 300          # 5 minutes cache lifetime

# Per-request context for concurrent orchestration
_REQUEST_ID = contextvars.ContextVar("rai_request_id", default=None)


def set_request_id(request_id: Optional[str]) -> None:
    _REQUEST_ID.set(request_id)


def _get_request_id() -> Optional[str]:
    return _REQUEST_ID.get()


def _hash_string(s: str) -> str:
    """Create a short hash of a string for caching"""
    import hashlib
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _get_cached_query_result(sql: str) -> Optional[pd.DataFrame]:
    """Retrieve cached query result if available and fresh"""
    h = _hash_string(sql)
    if h in _QUERY_RESULT_CACHE:
        cached_entry = _QUERY_RESULT_CACHE[h]
        import time
        if time.time() - cached_entry.get("timestamp", 0) < _CACHE_TTL:
            return cached_entry.get("result")
        else:
            del _QUERY_RESULT_CACHE[h]
    return None


def _cache_query_result(sql: str, df: pd.DataFrame) -> None:
    """Cache a query result with timestamp"""
    import time
    h = _hash_string(sql)
    # Prevent unbounded cache growth
    if len(_QUERY_RESULT_CACHE) >= _CACHE_MAX_SIZE:
        oldest_key = min(_QUERY_RESULT_CACHE.keys(), key=lambda k: _QUERY_RESULT_CACHE[k].get("timestamp", 0))
        del _QUERY_RESULT_CACHE[oldest_key]
    _QUERY_RESULT_CACHE[h] = {"result": df, "timestamp": time.time()}


def _get_cached_llm_result(prompt: str) -> Optional[str]:
    """Retrieve cached LLM result if available and fresh"""
    h = _hash_string(prompt)
    if h in _LLM_RESULT_CACHE:
        cached_entry = _LLM_RESULT_CACHE[h]
        import time
        if time.time() - cached_entry.get("timestamp", 0) < _CACHE_TTL:
            return cached_entry.get("result")
        else:
            del _LLM_RESULT_CACHE[h]
    return None


def _cache_llm_result(prompt: str, result: str) -> None:
    """Cache an LLM result with timestamp"""
    import time
    h = _hash_string(prompt)
    # Prevent unbounded cache growth
    if len(_LLM_RESULT_CACHE) >= _CACHE_MAX_SIZE:
        oldest_key = min(_LLM_RESULT_CACHE.keys(), key=lambda k: _LLM_RESULT_CACHE[k].get("timestamp", 0))
        del _LLM_RESULT_CACHE[oldest_key]
    _LLM_RESULT_CACHE[h] = {"result": result, "timestamp": time.time()}


def _cortex_complete_with_timeout(
    session,
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout_s: Optional[int] = None,
    response_key: str = "RESPONSE",
) -> str:
    """Call Snowflake Cortex LLM with timeout and parameterized SQL."""
    if session is None:
        raise RuntimeError("Snowflake session is required for Cortex calls")

    llm_model = model or os.environ.get("CORTEX_LLM_MODEL", "openai-gpt-5-chat")
    cortex_timeout = int(timeout_s or os.environ.get("CORTEX_TIMEOUT_SECONDS", "120"))

    result_holder = [None]
    exception_holder = [None]

    def cortex_thread():
        try:
            result_holder[0] = session.sql(
                "SELECT snowflake.cortex.complete(?, ?) as response",
                params=[llm_model, prompt],
            ).collect()
        except Exception as e:
            exception_holder[0] = e

    t = threading.Thread(target=cortex_thread, daemon=True)
    t.start()
    t.join(timeout=cortex_timeout)

    if t.is_alive():
        raise TimeoutError(f"Cortex LLM call timed out after {cortex_timeout}s")
    if exception_holder[0]:
        if _is_auth_expired_error(exception_holder[0]):
            raise AuthExpiredError(str(exception_holder[0]))
        raise exception_holder[0]

    result = result_holder[0]
    if result and len(result) > 0:
        # Snowpark Row can be accessed by key or index
        try:
            return str(result[0][response_key]).strip()
        except Exception:
            return str(result[0][0]).strip()
    return ""


class AuthExpiredError(RuntimeError):
    pass


def _is_auth_expired_error(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        "authentication token has expired" in msg
        or "token has expired" in msg
        or "must authenticate again" in msg
        or "390114" in msg
    )


def _coerce_json_response(resp_text: str) -> Dict[str, Any]:
    """Be tolerant to prose/fences; extract the first valid JSON object."""
    if isinstance(resp_text, dict):
        return resp_text
    s = str(resp_text or "").strip()
    if not s:
        return {}
    m = re.search(r"```(OK:json)OK(.*OK)```", s, flags=re.S | re.I)
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


_PLOTLY_FORBIDDEN = ("import ", "__import__", "open(", "os.", "sys.", "subprocess", "eval(", "exec(", "input(")


def _is_plotly_code_safe(code: str) -> bool:
    if not code or len(code) > 20000:
        return False
    low = code.lower()
    return not any(bad in low for bad in _PLOTLY_FORBIDDEN)


def _sanitize_llm_code(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = (
        s.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
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
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"^\s*\\\s*", "", s)
    return s.strip()


def _figure_has_visible_data(fig) -> bool:
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


def _exec_plotly_code_safely(code: str, df: pd.DataFrame):
    import builtins
    import plotly.express as px
    import plotly.graph_objects as go

    if not _is_plotly_code_safe(code):
        raise RuntimeError("Generated code failed safety checks.")

    def _strip_fences(s: str) -> str:
        s = (s or "").strip()
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
        return m.group(1).strip() if m else s

    code0 = _sanitize_llm_code(_strip_fences(code))
    if not re.search(r"(?m)^\s*fig\s*=", code0):
        px_call = None
        for m in re.finditer(r"(px\.\w+\s*\([^)]*\))", code0, re.DOTALL):
            px_call = m.group(1)
        if px_call:
            code0 = code0.replace(px_call, f"fig = {px_call}")
        else:
            m = re.search(r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*(px\.\w+\s*\([^)]*\)|go\.Figure\s*\([^)]*\))", code0)
            if m:
                varname = m.group(1)
                code0 = code0 + f"\nfig = {varname}"
            else:
                code0 = code0 + "\nfig = go.Figure()"

    safe_builtins = {
        "abs": builtins.abs,
        "min": builtins.min,
        "max": builtins.max,
        "sum": builtins.sum,
        "len": builtins.len,
        "range": builtins.range,
        "enumerate": builtins.enumerate,
        "zip": builtins.zip,
        "map": builtins.map,
        "filter": builtins.filter,
        "any": builtins.any,
        "all": builtins.all,
        "list": builtins.list,
        "tuple": builtins.tuple,
        "set": builtins.set,
        "dict": builtins.dict,
        "float": builtins.float,
        "int": builtins.int,
        "str": builtins.str,
        "round": builtins.round,
        "sorted": builtins.sorted,
    }
    g = {"__builtins__": safe_builtins, "pd": pd, "px": px, "go": go}
    l = {"df": df, "fig": None, "caption": ""}

    exec(code0, g, l)
    fig = l.get("fig")
    if fig is None:
        for v in l.values():
            try:
                if isinstance(v, go.Figure):
                    fig = v
                    break
            except Exception:
                pass
    if fig is None:
        raise RuntimeError("Generated code did not set fig.")
    return fig, l.get("caption")


def _generate_narrative_summary(
    session,
    question: str,
    results_df: Optional[pd.DataFrame] = None,
    insights: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate rich narrative + KPIs using Cortex LLM (like ai_insights.py).
    Returns: {"narrative": str, "kpis": List[Dict]}
    """
    try:
        from snowflake.snowpark.context import get_active_session

        if session is None:
            session = get_active_session()

        data_json = []
        if results_df is not None and not results_df.empty:
            try:
                safe_df = results_df.copy()
                for col in safe_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
                        safe_df[col] = (
                            pd.to_datetime(safe_df[col], errors="coerce")
                            .dt.strftime("%Y-%m-%dT%H:%M:%S")
                            .replace("NaT", None)
                        )
                safe_df = safe_df.replace({np.nan: None})
                data_json = [{
                    "id": "results",
                    "rows": json.loads(safe_df.to_json(orient="records"))
                }]
            except Exception as e:
                print(f"[WARNING] Failed to serialize DataFrame: {e}", file=sys.stderr)
                sys.stderr.flush()

        prompt = f"""
You are a senior BI analyst. Use ONLY the full dataset(s) provided in DATA_JSON.
Output JSON ONLY:
{{
  "narrative": "markdown string",
  "kpis": [ {{"title":"", "value":"", "sub":""}} ],
  "chart": {{
    "chart":"scatter|line|bar|table",
    "x":"<col>|null",
    "y":"<col>|null",
    "series":"<col>|null",
    "agg":"sum|avg|last|median|count",
    "allowed_chart_types":["line","bar","scatter","table"],
    "top_n": null
  }},
  "needs_more": false,
  "followup_prompt": "",
  "required": {{ "views": [], "columns": {{}} }}
}}
Sections in narrative: Overall Summary, Key Findings, Drivers & Diagnostics, Recommendations.
If insufficient data, say exactly what rows/fields are missing.

IMPORTANT:
- Treat the dataset as already filtered by the query/reasoners. Do NOT generalize to the full population.
- Columns like signal_*, *_condition, *_score are derived outputs; describe them as flags/labels.
- Do NOT invent correlations, regressions, or projections.
- Never mention internal JSON or tools in the narrative.
- Format numbers clearly: percentages as "X%", currency as "$X", counts as "X".

Question: {question}

DATA_JSON (the ONLY source of truth):
{json.dumps(data_json, default=str, indent=2)}

Deterministic insights from query:
{json.dumps(insights or [], default=str, indent=2)}

Output ONLY the JSON object above - no other text:
""".strip()

        cache_hit = _get_cached_llm_result(prompt)
        if cache_hit:
            response_text = cache_hit
        else:
            print(f"[DEBUG] Calling Cortex LLM for narrative (prompt len={len(prompt)})", file=sys.stderr)
            sys.stderr.flush()
            response_text = _cortex_complete_with_timeout(session, prompt)
            _cache_llm_result(prompt, response_text)

        if response_text:
            result_json = _coerce_json_response(response_text)
            narrative = result_json.get("narrative", response_text)
            kpis = result_json.get("kpis", [])
            reasoning = result_json.get("reasoning")

            print(f"[DEBUG] OK Narrative generated: {len(str(narrative))} chars, {len(kpis) if isinstance(kpis, list) else 0} KPIs", file=sys.stderr)
            sys.stderr.flush()

            return {
                "narrative": str(narrative).strip(),
                "kpis": kpis if isinstance(kpis, list) else [],
                "reasoning": str(reasoning).strip() if reasoning else ""
            }

        print("[WARNING] Cortex returned empty response", file=sys.stderr)
        sys.stderr.flush()
        return {
            "narrative": "\n".join(insights or []),
            "kpis": [],
            "reasoning": ""
        }

    except Exception as e:
        if isinstance(e, AuthExpiredError):
            raise
        print(f"[ERROR] Failed to generate narrative: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return {
            "narrative": "\n".join(insights or []),
            "kpis": [],
            "reasoning": ""
        }


def _generate_plotly_chart(
    session,
    question: str,
    results_df: Optional[pd.DataFrame] = None,
    narrative_summary: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate a Plotly chart spec from query results using Cortex LLM.
    Returns chart dict with 'data' and 'layout' keys (Plotly JSON format).
    """
    print(f"[DEBUG] _generate_plotly_chart called: results_df={results_df is not None}, df_empty={results_df.empty if results_df is not None else 'N/A'}", file=sys.stderr)
    sys.stderr.flush()

    if results_df is None or results_df.empty:
        print("[DEBUG] Skipping chart generation: results_df is None/empty", file=sys.stderr)
        sys.stderr.flush()
        return None

    try:
        from snowflake.snowpark.context import get_active_session

        if session is None:
            session = get_active_session()

        def _coerce_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame):
                return df
            id_cols = [c for c in df.columns if str(c).upper().endswith("ID") or str(c).upper() in ("RMID", "MANDATEID")]
            if not id_cols:
                return df
            out = df.copy()
            for col in id_cols:
                try:
                    out[col] = out[col].astype(str)
                except Exception:
                    pass
            return out

        def _prepare_chart_frame(df: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return df
            out = df.copy()
            for col in list(out.columns):
                try:
                    col_up = str(col).upper()
                    series = out[col]
                    if ("DATE" in col_up or "MONTH" in col_up):
                        ser_num = None
                        if pd.api.types.is_numeric_dtype(series):
                            ser_num = pd.to_numeric(series, errors="coerce")
                        elif series.dtype == object:
                            ser_num = pd.to_numeric(series, errors="coerce")
                        if ser_num is not None and ser_num.notna().any():
                            mx = ser_num.max(skipna=True)
                            if mx and mx > 1e12:
                                out[col] = pd.to_datetime(ser_num, unit="ns", errors="coerce")
                            elif mx and mx > 1e10:
                                out[col] = pd.to_datetime(ser_num, unit="ms", errors="coerce")
                            elif mx and mx > 1e9:
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

        df = _prepare_chart_frame(_coerce_identifier_columns(results_df))

        PLOTLY_SCHEMA_COL_CAP = 40
        PLOTLY_SCHEMA_HEAD = 5

        cols = list(df.columns)[:PLOTLY_SCHEMA_COL_CAP]
        schema = {
            "columns": cols,
            "dtypes": {str(c): str(df[c].dtype) for c in cols},
            "preview": df[cols].head(PLOTLY_SCHEMA_HEAD).astype(str).to_dict("records"),
            "rows": int(len(df)),
        }

        def _build_prompt(question_text: str, summary_text: str, schema_obj: dict, override_notes: Optional[str]) -> str:
            override_block = f"\nOVERRIDE NOTES:\n{override_notes}\n" if override_notes else ""
            return (
                "You will write pure Plotly Python code to visualize data that is already loaded into a pandas DataFrame called `df`.\n"
                "Plotly and pandas are already imported as:\n"
                "- pandas as `pd`\n"
                "- plotly.express as `px`\n"
                "- plotly.graph_objects as `go`\n\n"
                "Hard constraints:\n"
                "- Do NOT import anything; never use os/sys/subprocess/network.\n"
                "- Use only the columns shown in SCHEMA.\n"
                "- MUST assign the final Plotly figure to a variable named `fig`.\n"
                "- You may set an optional string variable `caption` (one concise sentence).\n"
                "- No line-continuation backslashes across lines.\n"
                "- Prefer names over IDs for axes/labels.\n"
                "- If you plot identifier columns (RMID/MANDATEID), cast them to string first.\n"
                "- Do not plot ID or date fields as metrics (Y values). Date/Year/Month may be used on X or as grouping.\n\n"
                "Chart selection rules:\n"
                "- For Top/Bottom/Rank by entity: use a sorted bar chart, X=entity, Y=metric. If year comparison implied, use grouped bar by year.\n"
                "- For trends over time: use a line chart with time on X.\n"
                "- For relationships: use scatter (X=metric1, Y=metric2).\n\n"
                f"QUESTION:\n{question_text or ''}\n\n"
                f"SUMMARY:\n{summary_text or ''}\n"
                f"{override_block}"
                "\nSCHEMA (JSON):\n" + json.dumps(schema_obj, ensure_ascii=False, indent=2)
                + "\n\nReturn ONLY Python code. You MUST assign the Plotly figure to a variable named `fig`."
            )

        max_attempts = 4
        override_notes = None
        last_error = None

        for attempt in range(max_attempts):
            try:
                prompt = _build_prompt(question, narrative_summary or "", schema, override_notes)
                llm_raw = _cortex_complete_with_timeout(session, prompt)

                print(f"[DEBUG] Generated Plotly code ({len(llm_raw or '')} chars)", file=sys.stderr)
                print("[DEBUG] ===== CODE START =====", file=sys.stderr)
                print(llm_raw, file=sys.stderr)
                print("[DEBUG] ===== CODE END =====", file=sys.stderr)
                sys.stderr.flush()

                if os.environ.get("AI_INSIGHTS_DEBUG_PLOTLY", "").strip().lower() in ("1", "true", "yes"):
                    try:
                        import tempfile
                        debug_dir = Path(tempfile.gettempdir()) / "rai_plotly_debug"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_file = debug_dir / f"chart_code_debug_attempt{attempt + 1}.py"
                        debug_file.write_text(llm_raw or "", encoding="utf-8")
                    except Exception:
                        pass

                fig, _cap = _exec_plotly_code_safely(llm_raw or "", df)

                if not _figure_has_visible_data(fig):
                    last_error = "empty_chart"
                    override_notes = (
                        "Previous chart had no visible data. "
                        "Remove filters, pick columns with non-null values, "
                        "and ensure at least one trace has data."
                    )
                    continue

                chart_dict = fig.to_dict()

                def convert_to_serializable(obj):
                    try:
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        if isinstance(obj, (np.integer, np.floating)):
                            return obj.item()
                        if isinstance(obj, dict):
                            return {k: convert_to_serializable(v) for k, v in obj.items()}
                        if isinstance(obj, (list, tuple)):
                            return [convert_to_serializable(item) for item in obj]
                        return obj
                    except Exception:
                        return obj

                chart_dict = convert_to_serializable(chart_dict)

                print(f"[DEBUG] OK Chart generated successfully on attempt {attempt + 1}", file=sys.stderr)
                sys.stderr.flush()
                return chart_dict

            except Exception as e:
                last_error = str(e)
                override_notes = (
                    f"Attempt {attempt + 1} failed: {last_error}. "
                    "Return minimal Plotly code only. Use only listed columns, "
                    "no unsafe ops, no external files, no extra imports. "
                    "Keep the chart simple and valid."
                )
                print(f"[WARNING] Attempt {attempt + 1} - Plotly generation failed: {last_error}", file=sys.stderr)
                sys.stderr.flush()
                continue

        print(f"[WARNING] Chart generation failed after {max_attempts} attempts: {last_error}", file=sys.stderr)
        sys.stderr.flush()
        return None

    except Exception as e:
        if isinstance(e, AuthExpiredError):
            raise
        print(f"[WARNING] Failed to generate chart: {e}", file=sys.stderr)
        sys.stderr.flush()
        return None


def _minimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types"""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    
    try:
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['object']).columns:
            # Convert string columns to category if they have few unique values
            if df[col].nunique() < len(df) * 0.05:  # Less than 5% unique values
                df[col] = df[col].astype('category')
    except Exception:
        pass  # If downcast fails, return original
    
    return df


def set_stage_callback(callback):
    """Set a callback function to be called when stage changes"""
    _ORCHESTRATION_STATE["stage_callback"] = callback


def _update_stage(stage_name: str):
    """Update the current orchestration stage and call callback if set"""
    request_id = _get_request_id()
    # If suppress_updates is set, do not propagate stage changes (used during init)
    if _ORCHESTRATION_STATE.get("suppress_updates"):
        return
    _ORCHESTRATION_STATE["current_stage"] = stage_name
    if request_id:
        req_state = _ORCHESTRATION_STATE.setdefault("requests", {}).setdefault(request_id, {})
        req_state["stage"] = stage_name
        req_state["updated_at"] = time.time()
    if _ORCHESTRATION_STATE.get("stage_callback"):
        try:
            _ORCHESTRATION_STATE["stage_callback"](stage_name)
        except Exception:
            pass


class _HeadlessContainer:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def empty(self):
        return _HeadlessContainer(self._st)

    def container(self):
        return _HeadlessContainer(self._st)

    def columns(self, spec, gap=None):
        return self._st.columns(spec, gap=gap)

    def expander(self, *args, **kwargs):
        return _HeadlessContainer(self._st)

    def slider(self, label, min_value, max_value, value=None, *args, **kwargs):
        return value if value is not None else min_value

    def number_input(self, label, min_value=None, max_value=None, value=None, *args, **kwargs):
        return value

    def selectbox(self, *args, **kwargs):
        options = kwargs.get("options") or (args[1] if len(args) > 1 else [])
        if options:
            return options[0]
        return None

    def button(self, *args, **kwargs):
        return False

    def toggle(self, *args, **kwargs):
        return bool(kwargs.get("value", False))

    def chat_input(self, *args, **kwargs):
        return None

    def markdown(self, text, unsafe_allow_html=False):
        """Intercept markdown to detect stage changes - delegate to parent st"""
        return self._st.markdown(text, unsafe_allow_html=unsafe_allow_html)

    def __getattr__(self, _name):
        return self._noop

    def _noop(self, *args, **kwargs):
        return None


class _HeadlessStreamlit:
    def __init__(self):
        self.session_state: Dict[str, Any] = {}
        self.sidebar = _HeadlessContainer(self)

    def container(self):
        return _HeadlessContainer(self)

    def empty(self):
        return _HeadlessContainer(self)

    def columns(self, spec, gap=None):
        if isinstance(spec, (list, tuple)):
            count = len(spec)
        else:
            try:
                count = int(spec)
            except Exception:
                count = 1
        return [_HeadlessContainer(self) for _ in range(count)]

    def expander(self, *args, **kwargs):
        return _HeadlessContainer(self)

    def chat_input(self, *args, **kwargs):
        return None

    def set_page_config(self, *args, **kwargs):
        return None

    def markdown(self, text, unsafe_allow_html=False):
        """Intercept markdown to detect stage changes"""
        import re
        if text and isinstance(text, str):
            # Check for long-running provisioning/initializing stages
            if "Processing background tasks" in text or "Provisioning engine" in text:
                _update_stage("provisioning")
            elif "Initializing data index" in text:
                _update_stage("initializing")
            # Check for regular stage transitions: <div class='ai-pill on'>StageName</div>
            elif "ai-pill" in text:
                # The stage name will be capitalized (Planning, Fetching, Linking, Summarizing)
                match = re.search(r"<div class='ai-pill on'>(\w+)</div>", text)
                if match:
                    stage_label = match.group(1).strip().lower()
                    # Verify it's a valid stage
                    if stage_label in ("planning", "fetching", "linking", "summarizing"):
                        _update_stage(stage_label)
        return None

    def stop(self):
        return None

    def rerun(self):
        return None

    def cache_data(self, *args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def decorator(fn):
            return fn
        return decorator

    def cache_resource(self, *args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        def decorator(fn):
            return fn
        return decorator

    def __getattr__(self, _name):
        return self._noop

    def _noop(self, *args, **kwargs):
        return None


class _SharedSession:
    def __init__(self, session):
        self._session = session

    def get_session(self):
        return self._session


_ORCHESTRATOR = None
_ORCHESTRATOR_LOCK = threading.Lock()
_HEADLESS_ST = _HeadlessStreamlit()
_AI_MODULE = None
_INIT_TIMESTAMP = 0
_INIT_TTL = 3600  # Cache orchestrator for 1 hour


def _ensure_root_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_ai_insights():
    global _AI_MODULE
    if _AI_MODULE is not None:
        return _AI_MODULE
    _ensure_root_on_path()
    sys.modules["streamlit"] = _HEADLESS_ST
    import ai_insights
    ai_insights.st = _HEADLESS_ST
    _AI_MODULE = ai_insights
    return ai_insights


def _init_orchestrator(session):
    import time
    global _ORCHESTRATOR, _INIT_TIMESTAMP, _INIT_TTL
    try:
        env_ttl = int(os.environ.get("AI_INSIGHTS_ORCHESTRATOR_TTL", str(_INIT_TTL)))
        _INIT_TTL = max(0, env_ttl)
    except Exception:
        pass
    force_reload = os.environ.get("AI_INSIGHTS_ORCHESTRATOR_FORCE_RELOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if force_reload:
        _ORCHESTRATOR = None
        globals()["_AI_MODULE"] = None

    current_time = time.time()
    # Check if cached orchestrator is still valid (not expired)
    if _ORCHESTRATOR is not None and _INIT_TTL > 0 and (current_time - _INIT_TIMESTAMP) < _INIT_TTL:
        return _ORCHESTRATOR
    
    with _ORCHESTRATOR_LOCK:
        # Double-check inside lock
        if _ORCHESTRATOR is not None and _INIT_TTL > 0 and (current_time - _INIT_TIMESTAMP) < _INIT_TTL:
            return _ORCHESTRATOR
        
        ai = _load_ai_insights()
        shared = _SharedSession(session)
        # Suppress stage update events while rendering the Streamlit UI (initialization)
        _ORCHESTRATION_STATE["suppress_updates"] = True
        try:
            ai.render_main(shared=shared)
        finally:
            _ORCHESTRATION_STATE.pop("suppress_updates", None)
        if not hasattr(ai, "orchestrate"):
            raise RuntimeError("orchestrate was not exported from ai_insights.render_main")
        _ORCHESTRATOR = ai.orchestrate
        _INIT_TIMESTAMP = current_time
        return _ORCHESTRATOR


def _serialize_frame(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None:
        return []
    if not isinstance(df, pd.DataFrame):
        return df
    if df.empty:
        return []
    try:
        # Limit serialization to first 1000 rows for performance
        # (UI typically doesn't display more than that anyway)
        df_limited = df.head(1000) if len(df) > 1000 else df
        result = json.loads(df_limited.to_json(orient="records", date_format="iso"))
        # Return list directly (Pydantic expects list, not dict with metadata)
        return result if isinstance(result, list) else [result]
    except Exception:
        df_limited = df.head(1000) if len(df) > 1000 else df
        result = df_limited.astype(str).to_dict(orient="records")
        return result if isinstance(result, list) else [result]


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return _serialize_frame(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        return value.isoformat()
    except Exception:
        return value


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(result or {})
    frames = payload.get("frames") or {}
    if isinstance(frames, dict):
        # Optimize frames before serialization
        optimized_frames = {}
        for k, v in frames.items():
            if isinstance(v, pd.DataFrame):
                v = _minimize_dataframe_memory(v)
            optimized_frames[k] = _serialize_frame(v)
        payload["frames"] = optimized_frames
    payload = _jsonable(payload)
    return payload


def _add_legacy_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    frames = payload.get("frames") or {}
    first_rows: Optional[List[Dict[str, Any]]] = None
    first_key = None
    if isinstance(frames, dict) and frames:
        first_key = next(iter(frames.keys()))
        first_rows = frames.get(first_key)
    payload.setdefault("narrative", payload.get("insights"))
    payload.setdefault("rows", first_rows or [])
    payload.setdefault("view", payload.get("view") or first_key or "")
    if payload.get("sql") is None:
        sqls = payload.get("sqls") or []
        if sqls and isinstance(sqls, list) and len(sqls[0]) >= 2:
            payload["sql"] = sqls[0][1]
    kpis = payload.get("kpis")
    if isinstance(kpis, list):
        for item in kpis:
            if isinstance(item, dict):
                if "label" not in item and "title" in item:
                    item["label"] = item.get("title")
                if "unit" not in item and "sub" in item:
                    item["unit"] = item.get("sub")
    return payload


def _apply_kg_enhancements(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply KG enhancements (trends, patterns, centrality) to response payload.
    Runs non-blocking - if enhancements fail, returns payload unchanged.
    Currently disabled pending RAI integration improvements.
    """
    if not _KG_ENHANCEMENT_ENABLED:
        return payload
    
    # KG enhancements temporarily disabled during request handling
    # They will be re-enabled as background tasks or in parallel execution
    # Current issue: RAI KG functions require session context that's difficult
    # to manage within the request/response cycle.
    # 
    # To re-enable: move to background task or separate worker thread
    return payload


def run_orchestrate(
    session,
    question: str,
    views: Optional[List[str]] = None,
    months_window: Optional[int] = None,
    widen_time_on_insufficient: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Full orchestration flow:
    1. Load RAI specs and build ontology context
    2. Define Cortex LLM completion function
    3. Convert NL question → RAI spec JSON via LLM
    4. Execute spec against RAI knowledge graph model
    5. Format results and return response
    """
    if session is None:
        raise RuntimeError("Snowflake session is required for orchestration")
    
    # Reset transient UI hints
    _ORCHESTRATION_STATE["thinking"] = ""
    request_id = _get_request_id()
    if request_id:
        req_state = _ORCHESTRATION_STATE.setdefault("requests", {}).setdefault(request_id, {})
        req_state["thinking"] = ""
        req_state["stage"] = "planning"
        req_state["updated_at"] = time.time()

    # Verify session has warehouse configured
    try:
        current_wh = session.get_current_warehouse()
        print(f"[DEBUG] Current warehouse: {current_wh}")
    except Exception as e:
        if _is_auth_expired_error(e):
            raise AuthExpiredError(str(e))
        print(f"[WARNING] Could not get current warehouse: {e}")
        # Set warehouse explicitly from environment
        wh_env = os.environ.get("SNOWFLAKE_WAREHOUSE")
        if wh_env:
            try:
                session.sql(f"USE WAREHOUSE {wh_env}").collect()
                print(f"[DEBUG] Switched to warehouse: {wh_env}")
            except Exception as wh_err:
                if _is_auth_expired_error(wh_err):
                    raise AuthExpiredError(str(wh_err))
                print(f"[WARNING] Could not switch warehouse: {wh_err}")
    
    print(f"[DEBUG] run_orchestrate: START - question='{question}'")
    import sys
    sys.stdout.flush()
    
    # ============================================================
    # STEP 0: IMPORT AND VALIDATION
    # ============================================================
    try:
        from rai_dynamic_reasoner import generate_dynamic_query_spec
        from rai_dynamic_query import run_dynamic_query
        from rai_ai_insights_ontology import (
            build_ai_insights_builder,
            infer_join_keys,
            render_ai_insights_ontology_text,
            load_ai_insights_specs,
        )
        try:
            from rai_reasoner_orchestrator import apply_all_relevant_reasoners
        except Exception:
            apply_all_relevant_reasoners = None
    except ImportError as e:
        print(f"[ERROR] Failed to import RAI modules: {e}", file=sys.stderr)
        return {
            "frames": {},
            "insights": [f"Failed to initialize RAI modules: {e}"],
            "kpis": {},
        }
    
    try:
        # ============================================================
        # STEP 1: BUILD RAI CONTEXT (ONTOLOGY + SCHEMA)
        # ============================================================
        _update_stage("planning")
        print("[DEBUG] Step 1: Loading RAI specs and building ontology...")
        
        # Load specs (no session needed)
        specs = load_ai_insights_specs()
        print(f"[DEBUG] Loaded {len(specs)} semantic specs")

        spec_entity_names = {str(s.name).lower() for s in specs if getattr(s, "name", None)}
        print(f"[DEBUG] Loaded entities: {sorted(spec_entity_names)}")
        
        # Render ontology text describing all entities and relationships
        ontology_text = render_ai_insights_ontology_text(specs)
        print(f"[DEBUG] Rendered ontology text ({len(ontology_text)} chars)")
        
        # Extract join keys for entity linking
        join_keys = infer_join_keys(specs)
        print(f"[DEBUG] Inferred join keys: {join_keys}")
        
        # Build RAI knowledge graph builder (no session needed; uses cached model)
        builder = build_ai_insights_builder()
        print("[DEBUG] Built RAI knowledge graph builder")
        
        # ============================================================
        # STEP 2: DEFINE CORTEX LLM COMPLETION FUNCTION
        # ============================================================
        print("[DEBUG] Step 2: Defining Cortex LLM completion function...")
        
        def cortex_complete(prompt: str) -> str:
            """Call Snowflake Cortex LLM to process prompt with timeout"""
            try:
                import signal
                import threading
                
                llm_model = os.environ.get("CORTEX_LLM_MODEL", "openai-gpt-5-chat")
                print(f"[DEBUG] Cortex call: model={llm_model}, prompt_len={len(prompt)}")
                sys.stdout.flush()
                
                # Create a timeout wrapper - Cortex calls can hang indefinitely
                cortex_timeout = int(os.environ.get("CORTEX_TIMEOUT_SECONDS", "120"))
                
                result_holder = [None]
                exception_holder = [None]
                
                def cortex_thread():
                    try:
                        # Use parameters to avoid SQL injection and escaping issues
                        # Snowflake parameterized queries handle all special characters safely
                        result_holder[0] = session.sql(
                            "SELECT snowflake.cortex.complete(?, ?) as response",
                            params=[llm_model, prompt]
                        ).collect()
                    except Exception as e:
                        exception_holder[0] = e
                
                # Run Cortex call in thread with timeout
                t = threading.Thread(target=cortex_thread, daemon=True)
                t.start()
                t.join(timeout=cortex_timeout)
                
                if t.is_alive():
                    print(f"[ERROR] Cortex LLM call timed out after {cortex_timeout}s", file=sys.stderr)
                    sys.stderr.flush()
                    return ""
                
                if exception_holder[0]:
                    raise exception_holder[0]
                
                result = result_holder[0]
                if result and len(result) > 0:
                    response_text = str(result[0][0])
                    print(f"[DEBUG] Cortex response ({len(response_text)} chars):")
                    print(response_text)
                    sys.stdout.flush()
                    try:
                        parsed = _coerce_json_response(response_text)
                        if isinstance(parsed, dict):
                            reasoning = parsed.get("reasoning")
                            if isinstance(reasoning, str) and reasoning.strip():
                                _ORCHESTRATION_STATE["thinking"] = reasoning.strip()
                                request_id = _get_request_id()
                                if request_id:
                                    req_state = _ORCHESTRATION_STATE.setdefault("requests", {}).setdefault(request_id, {})
                                    req_state["thinking"] = reasoning.strip()
                                    req_state["updated_at"] = time.time()
                    except Exception:
                        pass
                    return response_text
                
                print("[WARNING] Cortex returned empty result")
                sys.stdout.flush()
                return ""
                
            except Exception as e:
                error_msg = str(e)
                if _is_auth_expired_error(e):
                    raise AuthExpiredError(error_msg)
                print(f"[ERROR] Cortex LLM call failed: {error_msg}", file=sys.stderr)
                
                # Log specific error types
                if "NETWORK_ERROR" in error_msg or "timeout" in error_msg.lower():
                    print(f"[ERROR] Network/timeout issue - warehouse may be unavailable", file=sys.stderr)
                elif "SYNTAX_ERROR" in error_msg or "invalid" in error_msg.lower():
                    print(f"[ERROR] SQL syntax error - prompt escaping issue", file=sys.stderr)
                elif "PERMISSION" in error_msg or "access" in error_msg.lower():
                    print(f"[ERROR] Permission error - may need role/warehouse rights", file=sys.stderr)
                
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
                return ""
        
        # ============================================================
        # STEP 3: CONVERT NL QUESTION → RAI SPEC VIA LLM
        # ============================================================
        print(f"[DEBUG] Step 3: Converting question to RAI spec: '{question}'")
        _update_stage("planning")
        
        # Call LLM to generate spec from question with retry logic
        spec = None
        reasoner_ids = []
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries and spec is None:
            try:
                spec, reasoner_ids = generate_dynamic_query_spec(
                    cortex_complete=cortex_complete,
                    question=question,
                    ontology_text=ontology_text,
                    join_keys=join_keys,
                    default_limit=2000,
                    max_retries=6,
                )
                if spec is not None:
                    print(f"[DEBUG] Generated spec on attempt {retry_count + 1}")
                    break
            except Exception as e:
                if isinstance(e, AuthExpiredError):
                    raise
                last_error = e
                retry_count += 1
                print(f"[WARNING] Spec generation failed (attempt {retry_count}/{max_retries}): {str(e)[:200]}")
                
                # Wait before retry (exponential backoff)
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # 2s, 4s, 8s
                    print(f"[DEBUG] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        if spec is None:
            raise RuntimeError(f"Failed to generate spec after {max_retries} attempts: {last_error}")
        
        print(f"[DEBUG] Generated spec: {json.dumps(spec, default=str, indent=2)[:500]}...")
        print(f"[DEBUG] Selected reasoners: {reasoner_ids}")
        
        def _is_fallback_spec(candidate: dict) -> bool:
            if not isinstance(candidate, dict):
                return True
            binds = candidate.get("bind")
            selects = candidate.get("select")
            return not (isinstance(binds, list) and len(binds) > 0 and isinstance(selects, list) and len(selects) > 0)

        fallback_specs: List[Dict[str, Any]] = []
        try:
            from rai_semantic_registry import load_reasoners
        except Exception:
            load_reasoners = None

        reasoner_specs = load_reasoners() if load_reasoners else []
        reasoner_map = {r.id: r for r in reasoner_specs if getattr(r, "id", None)}

        specs_by_name = {s.name: s for s in specs if getattr(s, "name", None)}
        specs_by_type: Dict[str, List[Any]] = {}
        for s in specs:
            et = str(getattr(s, "entity_type", "") or "generic").lower()
            specs_by_type.setdefault(et, []).append(s)

        def _field_map(entity_spec) -> Dict[str, Any]:
            return {str(f.name): f for f in (getattr(entity_spec, "fields", None) or []) if getattr(f, "name", None)}

        def _agg_op_for_field(field_spec: Any) -> str:
            op = str(getattr(field_spec, "default_agg", "") or "").strip()
            if op:
                return op
            role = str(getattr(field_spec, "role", "") or "").strip().lower()
            if role in ("metric", "derived"):
                return "avg"
            return "count"

        def _expand_required_fields(entity_spec: Any, fields: List[str]) -> List[str]:
            fmap = _field_map(entity_spec)
            expanded: List[str] = []
            for name in fields:
                fs = fmap.get(name)
                if fs is None:
                    continue
                if getattr(fs, "derived", False) and getattr(fs, "depends_on", None):
                    for dep in fs.depends_on:
                        if dep in fmap and dep not in expanded:
                            expanded.append(dep)
                elif name not in expanded:
                    expanded.append(name)
            return expanded

        def _pick_entity_for_reasoner(reasoner_spec: Any) -> Optional[Any]:
            if reasoner_spec is None:
                return None
            et = str(getattr(reasoner_spec, "entity_type", "") or "").lower()
            candidates = specs_by_type.get(et) or []
            if not candidates:
                return None
            target_fields = set(str(x) for x in (getattr(reasoner_spec, "outputs", None) or []))
            for sig in (getattr(reasoner_spec, "signals", None) or []):
                name = str(getattr(sig, "metric_field", "") or "").strip()
                if name:
                    target_fields.add(name)
            if target_fields:
                scored = []
                for cand in candidates:
                    fields = set(_field_map(cand).keys())
                    score = len(fields & target_fields)
                    scored.append((score, cand))
                scored.sort(key=lambda x: x[0], reverse=True)
                return scored[0][1]
            return candidates[0]

        def _build_spec_for_entity(entity_spec: Any, required_fields: List[str]) -> Dict[str, Any]:
            alias = "e0"
            fmap = _field_map(entity_spec)
            join_keys = list(getattr(entity_spec, "join_keys", None) or [])
            dims = [jk for jk in join_keys if jk in fmap]
            if not dims:
                dims = [
                    f.name for f in (getattr(entity_spec, "fields", None) or [])
                    if str(getattr(f, "role", "")).strip().lower() == "dimension"
                ][:3]

            select = [{"alias": alias, "prop": d, "as": d} for d in dims]
            metrics = _expand_required_fields(entity_spec, required_fields or [])
            if not metrics:
                default_metric = str(getattr(entity_spec, "default_metric", "") or "")
                if default_metric and default_metric in fmap:
                    metrics = [default_metric]
                else:
                    metrics = [
                        f.name for f in (getattr(entity_spec, "fields", None) or [])
                        if str(getattr(f, "role", "")).strip().lower() in ("metric", "derived")
                    ][:1]

            aggregations = []
            for m in metrics:
                fs = fmap.get(m)
                if fs is None:
                    continue
                op = _agg_op_for_field(fs)
                aggregations.append({"op": op, "term": {"alias": alias, "prop": m}, "as": m})

            spec = {"bind": [{"alias": alias, "entity": entity_spec.name}], "select": select, "limit": 2000}
            if aggregations:
                spec["group_by"] = select
                spec["aggregations"] = aggregations
                spec["order_by"] = [{"term": {"value": aggregations[0]["as"]}, "dir": "desc"}]
            return spec

        if _is_fallback_spec(spec):
            required_by_entity: Dict[str, List[str]] = {}
            for rid in (reasoner_ids or []):
                reasoner_spec = reasoner_map.get(rid)
                ent = _pick_entity_for_reasoner(reasoner_spec)
                if ent is None:
                    continue
                needed = []
                needed.extend(list(getattr(reasoner_spec, "outputs", None) or []))
                for sig in (getattr(reasoner_spec, "signals", None) or []):
                    mf = str(getattr(sig, "metric_field", "") or "").strip()
                    if mf:
                        needed.append(mf)
                required_by_entity.setdefault(ent.name, [])
                for nf in needed:
                    if nf not in required_by_entity[ent.name]:
                        required_by_entity[ent.name].append(nf)

            if not required_by_entity and specs:
                primary = specs[0]
                required_by_entity[primary.name] = []

            for ent_name, needed in required_by_entity.items():
                ent = specs_by_name.get(ent_name)
                if ent is None:
                    continue
                fallback_specs.append(_build_spec_for_entity(ent, needed))

            if fallback_specs:
                spec = fallback_specs[0]

        def _run_spec_with_timeout(local_spec: Dict[str, Any]) -> Optional[pd.DataFrame]:
            rai_timeout = int(os.environ.get("RAI_QUERY_TIMEOUT_SECONDS", "300"))
            result_holder = [None]
            exception_holder = [None]

            def rai_thread():
                try:
                    result_holder[0] = run_dynamic_query(builder, local_spec)
                except Exception as e:
                    exception_holder[0] = e

            t = threading.Thread(target=rai_thread, daemon=True)
            t.start()
            t.join(timeout=rai_timeout)

            if t.is_alive():
                raise TimeoutError(f"RAI query timeout after {rai_timeout}s")
            if exception_holder[0]:
                raise exception_holder[0]
            return result_holder[0]

        def _execute_and_merge_fallback_specs(spec_list: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
            frames: List[pd.DataFrame] = []
            debug: Dict[str, List[Dict[str, Any]]] = {}
            for idx, sp in enumerate(spec_list):
                try:
                    df = _run_spec_with_timeout(sp) or pd.DataFrame()
                except Exception as exc:
                    df = pd.DataFrame()
                    debug[f"_debug_spec_{idx + 1}_error"] = [{"error": str(exc)}]
                frames.append(df)
                debug[f"_debug_spec_{idx + 1}"] = df.to_dict("records") if not df.empty else []

            if not frames:
                return pd.DataFrame(), debug
            if len(frames) == 1:
                return frames[0], debug

            common = set(frames[0].columns)
            for df in frames[1:]:
                common &= set(df.columns)
            merge_keys = [k for k in join_keys if k in common] or list(common)
            if not merge_keys:
                return frames[0], debug

            merged = frames[0]
            for df in frames[1:]:
                merged = merged.merge(df, on=merge_keys, how="outer")
            return merged, debug

        # ============================================================
        # STEP 4: EXECUTE SPEC AGAINST RAI MODEL
        # ============================================================
        print("[DEBUG] Step 4: Executing RAI spec with timeout...")
        _update_stage("fetching")

        debug_frames: Dict[str, List[Dict[str, Any]]] = {}
        if fallback_specs and len(fallback_specs) > 1:
            results_df, debug_frames = _execute_and_merge_fallback_specs(fallback_specs)
            print("[DEBUG] Fallback multi-spec merge executed")
            sys.stdout.flush()
        else:
            # Execute the spec with retry logic for transient failures and timeout protection
            def _should_retry_plan_on_exec_error(err: Any) -> bool:
                msg = str(err or "")
                return "Shadowed variable" in msg

            results_df = None
            last_error = None
            plan_attempt = 0
            max_plan_attempts = 2

            while plan_attempt < max_plan_attempts:
                retry_count = 0
                max_retries = 2
                rai_timeout = int(os.environ.get("RAI_QUERY_TIMEOUT_SECONDS", "300"))  # 5 minutes default
                results_df = None

                while retry_count < max_retries and results_df is None:
                    try:
                        import threading

                        result_holder = [None]
                        exception_holder = [None]

                        def rai_thread():
                            try:
                                result_holder[0] = run_dynamic_query(builder, spec)
                            except Exception as e:
                                exception_holder[0] = e

                        # Run RAI query in thread with timeout
                        t = threading.Thread(target=rai_thread, daemon=True)
                        t.start()
                        t.join(timeout=rai_timeout)

                        if t.is_alive():
                            print(f"[ERROR] RAI query execution timed out after {rai_timeout}s", file=sys.stderr)
                            sys.stderr.flush()
                            last_error = f"RAI query timeout after {rai_timeout}s"
                            retry_count += 1
                            if retry_count < max_retries:
                                time.sleep(1)
                            continue

                        if exception_holder[0]:
                            raise exception_holder[0]

                        results_df = result_holder[0]
                        if results_df is not None:
                            print(f"[DEBUG] Query executed successfully on attempt {retry_count + 1}")
                            break

                    except Exception as e:
                        if isinstance(e, AuthExpiredError):
                            raise
                        last_error = e
                        retry_count += 1
                        print(f"[WARNING] Query execution failed (attempt {retry_count}/{max_retries}): {str(e)[:200]}")
                        sys.stdout.flush()

                        # Wait before retry
                        if retry_count < max_retries:
                            time.sleep(1)

                if results_df is not None:
                    break

                # Optionally re-plan on specific execution errors
                if plan_attempt == 0 and _should_retry_plan_on_exec_error(last_error):
                    print(f"[WARNING] Replanning due to execution error: {last_error}")
                    sys.stdout.flush()
                    try:
                        spec, reasoner_ids = generate_dynamic_query_spec(
                            cortex_complete=cortex_complete,
                            question=question,
                            ontology_text=ontology_text,
                            join_keys=join_keys,
                            default_limit=2000,
                            max_retries=6,
                            failure_context=(
                                "ExecutionError: Shadowed variable. "
                                "Avoid grouping by and aggregating the same field. "
                                "Use dimensions in group_by; aggregate measures only."
                            ),
                        )
                    except Exception as replan_err:
                        last_error = replan_err
                    plan_attempt += 1
                    continue

                break

            if results_df is None:
                print(f"[WARNING] Query returned None after {max_retries} attempts: {last_error}")
                sys.stdout.flush()
                results_df = pd.DataFrame()  # Return empty DataFrame instead of None

            print(f"[DEBUG] Query returned {len(results_df) if results_df is not None else 0} rows")
            sys.stdout.flush()

        reasoner_results = {}
        reasoning_context = ""
        if apply_all_relevant_reasoners is not None:
            try:
                reasoner_payload = apply_all_relevant_reasoners(
                    builder=builder,
                    spec=spec,
                    df=results_df if results_df is not None else pd.DataFrame(),
                    reasoner_ids=reasoner_ids,
                ) or {}
                if isinstance(reasoner_payload, dict):
                    results_df = reasoner_payload.get("dataframe") or results_df
                    reasoner_results = reasoner_payload.get("reasoner_results") or {}
                    reasoning_context = reasoner_payload.get("reasoning_context") or ""
            except Exception as e:
                print(f"[WARNING] Reasoner orchestration failed (non-critical): {e}", file=sys.stderr)
                sys.stderr.flush()

        # ============================================================
        # STEP 5: FORMAT RESULTS AND BUILD RESPONSE
        # ============================================================
        _update_stage("linking")
        print("[DEBUG] Step 5: Formatting results...")
        
        # Handle empty or None results
        if results_df is None or results_df.empty:
            insights = [f"Query executed but returned no results for: {question}"]
            rows = []
        else:
            # Convert DataFrame to list of dicts for JSON serialization
            rows = results_df.to_dict("records")
            
            # Generate insights from results
            insights = [
                f"Found {len(rows)} results for: {question}",
                f"Columns: {', '.join(results_df.columns.tolist())}",
            ]
            if len(rows) > 0 and len(str(rows[0])) < 500:
                insights.append(f"Top result: {rows[0]}")
        
        # ============================================================
        # STEP 6: BUILD RESPONSE PAYLOAD
        # ============================================================
        print("[DEBUG] Step 6: Building response payload...")
        
        payload = {
            "question": question,
            "frames": {"results": rows} if rows else {},
            "insights": insights,
            "kpis": [{"label": "Result Count", "value": len(rows)}],
            "plan": {"spec": spec, "reasoner_ids": reasoner_ids},
            "chart": {},  # Initialize empty chart, will be populated if generation succeeds
        }
        if reasoner_results:
            payload["frames"]["reasoners"] = _jsonable(reasoner_results)
        if reasoning_context:
            payload["plan"]["reasoning_context"] = reasoning_context
        if debug_frames:
            for key, value in debug_frames.items():
                if key and value:
                    payload["frames"][key] = value
        
        # ============================================================
        # STEP 7: GENERATE LLM NARRATIVE SUMMARY AND KPIs
        # ============================================================
        _update_stage("summarizing")
        print("[DEBUG] Step 7: Generating LLM narrative summary and KPIs...")
        sys.stderr.flush()
        try:
            # Build narrative via Cortex LLM
            print(f"[DEBUG] About to call _generate_narrative_summary with {len(insights)} insights", file=sys.stderr)
            sys.stderr.flush()
            
            summary_result = _generate_narrative_summary(
                session=session,
                question=question,
                results_df=results_df if results_df is not None and not results_df.empty else None,
                insights=insights
            )
            
            narrative = summary_result.get("narrative", "\n".join(insights))
            kpis_from_llm = summary_result.get("kpis", [])
            thinking_text = summary_result.get("reasoning", "")
            
            print(f"[DEBUG] Returned narrative length: {len(narrative)}, KPIs: {len(kpis_from_llm)}", file=sys.stderr)
            sys.stderr.flush()
            
            payload["narrative"] = narrative
            if thinking_text:
                payload["thinking"] = thinking_text
                _ORCHESTRATION_STATE["thinking"] = thinking_text
            
            # Merge with existing KPIs or use LLM-generated ones
            if kpis_from_llm:
                payload["kpis"] = kpis_from_llm
            
            print(f"[DEBUG] OK Set payload['narrative'] and KPIs, payload now has keys: {list(payload.keys())}", file=sys.stderr)
            sys.stderr.flush()
            
        except Exception as e:
            print(f"[ERROR] LLM narrative generation failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            payload["narrative"] = "\n".join(insights)
        
        # ============================================================
        # STEP 8: GENERATE PLOTLY CHART VISUALIZATION
        # ============================================================
        print("[DEBUG] Step 8: Generating Plotly chart...")
        sys.stderr.flush()
        try:
            # Convert rows to DataFrame for chart generation
            chart_df = None
            if rows:
                try:
                    chart_df = pd.DataFrame(rows)
                except Exception as e:
                    print(f"[WARNING] Failed to convert rows to DataFrame: {e}", file=sys.stderr)
                    sys.stderr.flush()
            
            chart_dict = _generate_plotly_chart(
                session=session,
                question=question,
                results_df=chart_df,
                narrative_summary=payload.get("narrative", None)
            )
            
            if chart_dict:
                payload["chart"] = chart_dict
                print(f"[DEBUG] OK Chart generated", file=sys.stderr)
                sys.stderr.flush()
            else:
                print("[DEBUG] Chart generation returned None", file=sys.stderr)
                sys.stderr.flush()
                
        except Exception as e:
            if isinstance(e, AuthExpiredError):
                raise
            print(f"[WARNING] Chart generation failed (non-critical): {e}", file=sys.stderr)
            sys.stderr.flush()
        
        # Apply KG enhancements if enabled
        payload = _apply_kg_enhancements(payload)

        request_id = _get_request_id()
        if request_id:
            req_state = _ORCHESTRATION_STATE.setdefault("requests", {}).setdefault(request_id, {})
            req_state["result"] = payload
            req_state["stage"] = "done"
            req_state["updated_at"] = time.time()

        _update_stage("done")
        print("[DEBUG] run_orchestrate: COMPLETE")
        return payload
        
    except Exception as e:
        if isinstance(e, AuthExpiredError):
            raise
        print(f"[ERROR] Orchestration failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        _update_stage("error")
        
        return {
            "frames": {},
            "insights": [f"Orchestration error: {str(e)[:500]}"],
            "kpis": {},
            "error": str(e),
        }
