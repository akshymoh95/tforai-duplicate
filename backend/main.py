from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
import uuid
import queue
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
from datetime import datetime
from logging.handlers import QueueHandler

# Add parent directory to path for RAI modules FIRST, before other imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ai_insights_orchestrator import run_orchestrate, set_stage_callback, _ORCHESTRATION_STATE, set_request_id
from snowflake.snowpark import Session
from dotenv import load_dotenv
from relationalai.semantics.rel.rel_utils import sanitize_identifier

from rai_ai_insights_ontology import (
    build_ai_insights_builder,
    ensure_rai_config,
    infer_join_keys,
    load_ai_insights_relationships,
    load_ai_insights_specs,
    registry_entities,
    registry_relationships,
    render_ai_insights_ontology_text,
)
from rai_semantic_registry import (
    load_registry as registry_load_entities,
    load_relationships as registry_load_relationships,
    load_reasoners as registry_load_reasoners,
    load_derived_rel_rules,
    load_prompt_templates,
    load_registry_config,
    load_analysis_config,
    load_kg_spec,
)
from rai_dynamic_query import run_dynamic_query
from rai_dynamic_reasoner import generate_dynamic_query_spec
from prompt_defaults import DEFAULT_PROMPT_TEMPLATES

ENV_PATH = Path(os.environ.get("AI_INSIGHTS_ENV", Path(__file__).with_name(".env")))
load_dotenv(ENV_PATH, override=False)

# Validate required environment variables
_REQUIRED_ENV_VARS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_ROLE",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_DATABASE",
    "SNOWFLAKE_SCHEMA",
]

def _validate_required_env() -> None:
    """Validate that all required environment variables are set."""
    missing = []
    for var in _REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing.append(var)
    
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Please set these in your .env file or environment.\n"
            f"Checked .env file: {ENV_PATH}"
        )

# Run validation at module load time
try:
    _validate_required_env()
except RuntimeError as e:
    import sys
    print(f"[ERROR] Environment validation failed: {e}", file=sys.stderr)
    raise

# ============================================================================
# RETRY LOGIC FOR CONNECTION RESILIENCE
# ============================================================================
from functools import wraps

def retry_on_connection_error(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator to retry a function on Snowflake connection errors.
    Resets the session on failure to allow reconnection.
    
    Args:
        max_retries: Number of retry attempts (default 3)
        backoff_factor: Exponential backoff multiplier (default 1.0 = 1s, 2s, 4s...)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _session
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (TimeoutError, ConnectionError, RuntimeError) as e:
                    error_msg = str(e).lower()
                    # Retry on connection/timeout errors
                    if any(x in error_msg for x in ["timeout", "connection", "expired", "closed", "disconnected"]):
                        last_exception = e
                        if attempt < max_retries - 1:
                            wait_time = (backoff_factor ** attempt)
                            print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}", file=sys.stderr)
                            print(f"[RETRY] Resetting session and retrying in {wait_time}s...", file=sys.stderr)
                            sys.stderr.flush()
                            
                            # Reset session to force reconnection
                            _session = None
                            time.sleep(wait_time)
                        else:
                            print(f"[ERROR] All {max_retries} retry attempts exhausted", file=sys.stderr)
                            sys.stderr.flush()
                    else:
                        # Don't retry on other exceptions
                        raise
                except Exception as e:
                    # Don't retry on non-connection errors
                    raise
            
            # All retries exhausted
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


DEFAULT_LLM = os.environ.get("CORTEX_LLM_MODEL", "openai-gpt-5-chat")
DRAFT_REGISTRY_LLM = os.environ.get("RAI_DRAFT_REGISTRY_LLM", "openai-gpt-4.1")
DRAFT_REGISTRY_DEBUG = os.environ.get("RAI_DRAFT_REGISTRY_DEBUG", "").lower() in ("1", "true", "yes")
SAMPLE_ROWS_LIMIT = int(os.environ.get("RAI_DRAFT_SAMPLE_ROWS", "5"))
SAMPLE_COLS_LIMIT = int(os.environ.get("RAI_DRAFT_SAMPLE_COLS", "0"))

app = FastAPI(title="AI Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOGGING SETUP FOR LOG STREAMING
# ============================================================================
_log_queue: queue.Queue = queue.Queue(maxsize=1000)
_log_history: List[Dict[str, Any]] = []
_log_history_lock = threading.Lock()
_max_log_history = 500

_original_stdout = sys.stdout
_original_stderr = sys.stderr

_LEVEL_PREFIX_RE = re.compile(r'^\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s*')


class _ExcludeLoggerFilter(logging.Filter):
    """Filter out logs from specific logger names."""

    def __init__(self, names: set[str]):
        super().__init__()
        self._names = names

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name not in self._names


class StreamToLogger:
    """File-like object that redirects writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int, stream) -> None:
        self.logger = logger
        self.level = level
        self.stream = stream
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0

        # Preserve terminal output.
        try:
            if self.stream:
                self.stream.write(message)
        except Exception:
            pass

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            level = self.level
            match = _LEVEL_PREFIX_RE.match(line)
            if match:
                level = logging._nameToLevel.get(match.group(1), self.level)
                line = line[match.end():].lstrip()
                if not line:
                    continue
            self.logger.log(level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""
        try:
            if self.stream:
                self.stream.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return False

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)


class QueueListenerHandler(logging.Handler):
    """Custom handler to send logs to queue for streaming."""
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
            }
            
            # Add to queue (non-blocking, drop oldest if full)
            try:
                _log_queue.put_nowait(log_entry)
            except queue.Full:
                try:
                    _log_queue.get_nowait()
                    _log_queue.put_nowait(log_entry)
                except queue.Empty:
                    pass
            
            # Add to history for recent logs endpoint
            with _log_history_lock:
                _log_history.append(log_entry)
                if len(_log_history) > _max_log_history:
                    _log_history.pop(0)
        except Exception:
            pass


def _setup_logging() -> None:
    """Configure logging with queue handler for streaming."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Queue handler for streaming
    queue_handler = QueueListenerHandler()
    queue_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    queue_handler.setFormatter(formatter)
    root_logger.addHandler(queue_handler)
    
    # Also add console handler for local debugging
    console_handler = logging.StreamHandler(_original_stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ExcludeLoggerFilter({"stdout", "stderr"}))
    root_logger.addHandler(console_handler)

    # Ensure common server loggers propagate to root
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.DEBUG)

    # Capture warnings from the warnings module
    logging.captureWarnings(True)

    # Redirect stdout/stderr to logging so prints are streamed
    sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO, _original_stdout)
    sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.ERROR, _original_stderr)


# Initialize logging
_setup_logging()
_logger = logging.getLogger(__name__)
_logger.info("API Server initialized")


_session: Optional[Session] = None
_current_stage: Dict[str, str] = {}
_request_store: Dict[str, Dict[str, Any]] = {}
_request_lock = threading.Lock()
_session_pool: "queue.Queue[Session]" = queue.Queue()
_session_pool_lock = threading.Lock()
_session_pool_total = 0
_session_pool_size = int(os.environ.get("SNOWFLAKE_SESSION_POOL_SIZE", "2"))


class AskRequest(BaseModel):
    question: str
    request_id: Optional[str] = None


class AskResponse(BaseModel):
    question: str
    request_id: Optional[str] = None
    plan: Dict[str, Any] = Field(default_factory=dict)
    frames: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    insights: str = ""
    narrative: Optional[str] = None
    thinking: Optional[str] = None
    kpis: List[Dict[str, Any]] = Field(default_factory=list)
    chart: Dict[str, Any] = Field(default_factory=dict)
    log: List[Any] = Field(default_factory=list)
    sqls: List[List[str]] = Field(default_factory=list)
    followups: List[Any] = Field(default_factory=list)
    merges: List[Dict[str, Any]] = Field(default_factory=list)
    catalog: Dict[str, Any] = Field(default_factory=dict)
    needs_more: Optional[bool] = None
    followup_prompt: Optional[str] = None
    required: Optional[Dict[str, Any]] = None
    summary_obj: Optional[Dict[str, Any]] = None
    view: Optional[str] = None
    sql: Optional[str] = None
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    narrative: Optional[str] = None


class RegistryDescribeRequest(BaseModel):
    table: str


class RegistryDraftRequest(BaseModel):
    tables: List[str] = Field(default_factory=list)
    instructions: Optional[str] = None


class RegistrySaveRequest(BaseModel):
    registry: Dict[str, Any]


class RegistryRollbackRequest(BaseModel):
    version: str


class RegistryOntologyExportRequest(BaseModel):
    registry: Optional[Dict[str, Any]] = None
    format: str = "json"
    write_file: bool = True


class DerivedRuleGenerateRequest(BaseModel):
    entity: str
    field: str
    instruction: str
    registry: Optional[Dict[str, Any]] = None


class DerivedRuleValidateRequest(BaseModel):
    rule: Dict[str, Any]
    registry: Optional[Dict[str, Any]] = None


class PromptCustomizationRequest(BaseModel):
    business_context: Optional[str] = None
    data_context: Optional[str] = None
    expectations: Optional[str] = None
    prompt_keys: Optional[List[str]] = None


class DerivedRuleSqlRequest(BaseModel):
    rule: Dict[str, Any]
    registry: Optional[Dict[str, Any]] = None


class ReasonerPlanGenerateRequest(BaseModel):
    reasoner_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    entity_type: Optional[str] = None
    outputs: List[str] = Field(default_factory=list)
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    instruction: Optional[str] = None
    registry: Optional[Dict[str, Any]] = None


class ReasonerSanityRequest(BaseModel):
    reasoner_id: str
    step_id: str
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    limit: int = 5


class KgGenerateRequest(BaseModel):
    registry: Optional[Dict[str, Any]] = None
    instructions: Optional[str] = None


def _session_params() -> Dict[str, Any]:
    params = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT"),
        "user": os.environ.get("SNOWFLAKE_USER"),
        "role": os.environ.get("SNOWFLAKE_ROLE"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE"),
        "database": os.environ.get("SNOWFLAKE_DATABASE"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA"),
    }
    auth = os.environ.get("SNOWFLAKE_AUTHENTICATOR")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    passcode = os.environ.get("SNOWFLAKE_PASSCODE")
    passcode_in_password = os.environ.get("SNOWFLAKE_PASSCODE_IN_PASSWORD")
    if auth:
        params["authenticator"] = auth
    if password:
        params["password"] = password
    if passcode:
        params["passcode"] = passcode
    if passcode_in_password:
        params["passcode_in_password"] = passcode_in_password
    return {k: v for k, v in params.items() if v}


def _registry_path() -> Path:
    env_path = os.environ.get("RAI_REGISTRY_PATH")
    if env_path:
        return Path(env_path)
    reg_dir = Path(__file__).resolve().parent.parent / "registry"
    rel_path = reg_dir / "registry.rel"
    if rel_path.exists():
        return rel_path
    return reg_dir / "semantic_registry.json"


def _is_rel_registry(path: Path) -> bool:
    return path.suffix.lower() == ".rel"


def _registry_runtime_path(primary: Path) -> Optional[Path]:
    env_path = os.environ.get("RAI_REGISTRY_RUNTIME_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None
    if primary.suffix.lower() == ".rel" and primary.name == "registry.rel":
        candidate = primary.with_name("registry_runtime.rel")
        if candidate.exists():
            return candidate
    return None


def _load_registry_payload_from_file(path: Path) -> Dict[str, Any]:
    """
    Load a registry payload from disk.

    Supports:
    - `.json` semantic registry (native format for the visual editor)
    - `.rel` registry (Rel-first source-of-truth); converted to semantic registry JSON
    """
    if not path.exists():
        return {}
    if _is_rel_registry(path):
        from registry.rel_to_semantic_registry import (
            build_semantic_registry_payload,
            extract_rel_meta,
            parse_registry_rel,
        )

        text = path.read_text(encoding="utf-8")
        entities, relations, streams, entity_stream, field_expr, reasoners = parse_registry_rel(text)

        # Support split registry mode:
        # - registry.rel => schema DSL
        # - registry_runtime.rel => pure Rel defs/reasoners for execute_raw
        runtime_path = _registry_runtime_path(path)
        if runtime_path is not None:
            runtime_text = runtime_path.read_text(encoding="utf-8")
            (
                _runtime_entities,
                _runtime_relations,
                runtime_streams,
                runtime_entity_stream,
                runtime_field_expr,
                runtime_reasoners,
            ) = parse_registry_rel(runtime_text)
            streams = {**streams, **runtime_streams}
            entity_stream = {**entity_stream, **runtime_entity_stream}
            field_expr = {**field_expr, **runtime_field_expr}
            if runtime_reasoners:
                seen_reasoners: set[str] = set()
                merged_reasoners: List[Dict[str, Any]] = []
                for r in list(reasoners or []) + list(runtime_reasoners or []):
                    if not isinstance(r, dict):
                        continue
                    rid = str(r.get("id") or "")
                    key = rid if rid else json.dumps(r, sort_keys=True)
                    if key in seen_reasoners:
                        continue
                    seen_reasoners.add(key)
                    merged_reasoners.append(r)
                reasoners = merged_reasoners

        payload = build_semantic_registry_payload(
            entities,
            relations,
            streams,
            entity_stream,
            field_expr,
            reasoners,
            emit_inverse_relationships=False,
            emit_kg=True,
        )
        payload["rel_meta"] = extract_rel_meta(text)
        if runtime_path is not None:
            payload["rel_meta"]["runtime_rel_path"] = str(runtime_path)
        return payload

    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry_payload_to_file(path: Path, payload: Dict[str, Any]) -> None:
    """
    Persist a semantic registry payload to disk.

    If `path` ends with `.rel`, we render a `registry.rel` file.
    Otherwise we write JSON.
    """
    if _is_rel_registry(path):
        from registry.semantic_registry_to_rel import render_registry_rel

        path.write_text(render_registry_rel(payload), encoding="utf-8")
        return

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _registry_history_dir() -> Path:
    return _registry_path().parent / "history"


def _registry_export_dir() -> Path:
    return _registry_path().parent / "exports"


def _write_registry_version(payload: Dict[str, Any]) -> Optional[Path]:
    max_versions = int(os.environ.get("RAI_REGISTRY_MAX_VERSIONS", "50"))
    history_dir = _registry_history_dir()
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"semantic_registry_{timestamp}.json"
    version_path = history_dir / base_name
    counter = 1
    while version_path.exists():
        counter += 1
        version_path = history_dir / f"semantic_registry_{timestamp}_{counter}.json"
    try:
        version_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    except Exception:
        return None
    versions = sorted(history_dir.glob("semantic_registry_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in versions[max_versions:]:
        try:
            old.unlink()
        except Exception:
            pass
    return version_path


def _list_registry_versions() -> List[Dict[str, Any]]:
    history_dir = _registry_history_dir()
    if not history_dir.exists():
        return []
    versions = sorted(history_dir.glob("semantic_registry_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for path in versions:
        out.append(
            {
                "name": path.name,
                "path": str(path),
                "created_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z",
            }
        )
    return out


def _registry_to_triples(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    triples: List[Dict[str, Any]] = []
    entities = payload.get("entities") or []
    relationships = payload.get("relationships") or []
    reasoners = payload.get("reasoners") or []
    kg = payload.get("kg") or {}
    for entity in entities:
        name = entity.get("name") or ""
        if not name:
            continue
        subject = f"entity:{name}"
        triples.append({"subject": subject, "predicate": "type", "object": "Entity"})
        table = ".".join([p for p in [entity.get("database"), entity.get("schema"), entity.get("table")] if p])
        if table:
            triples.append({"subject": subject, "predicate": "table", "object": table})
        triples.append({"subject": subject, "predicate": "entity_type", "object": entity.get("entity_type", "")})
        for field in entity.get("fields") or []:
            field_name = field.get("name") or ""
            if not field_name:
                continue
            field_subject = f"field:{name}.{field_name}"
            triples.append({"subject": subject, "predicate": "hasField", "object": field_subject})
            triples.append({"subject": field_subject, "predicate": "dtype", "object": field.get("dtype", "")})
            triples.append({"subject": field_subject, "predicate": "role", "object": field.get("role", "")})
            triples.append({"subject": field_subject, "predicate": "derived", "object": str(field.get("derived", False))})
            if field.get("expr"):
                triples.append({"subject": field_subject, "predicate": "expr", "object": field.get("expr", "")})
            for dep in field.get("depends_on") or []:
                triples.append({"subject": field_subject, "predicate": "depends_on", "object": dep})
        for key in entity.get("join_keys") or []:
            triples.append({"subject": subject, "predicate": "join_key", "object": key})
        for reasoner_id in entity.get("applicable_reasoners") or []:
            triples.append({"subject": subject, "predicate": "applies_reasoner", "object": reasoner_id})
    for rel in relationships:
        name = rel.get("name") or ""
        if not name:
            continue
        rel_subject = f"relationship:{name}"
        triples.append({"subject": rel_subject, "predicate": "type", "object": "Relationship"})
        triples.append({"subject": rel_subject, "predicate": "from", "object": rel.get("from_entity", "")})
        triples.append({"subject": rel_subject, "predicate": "to", "object": rel.get("to_entity", "")})
        for join in rel.get("join_on") or []:
            if len(join) >= 2:
                triples.append({"subject": rel_subject, "predicate": "join_on", "object": f"{join[0]}={join[1]}"})
    for reasoner in reasoners:
        rid = reasoner.get("id") or ""
        if not rid:
            continue
        r_subject = f"reasoner:{rid}"
        triples.append({"subject": r_subject, "predicate": "type", "object": "Reasoner"})
        triples.append({"subject": r_subject, "predicate": "name", "object": reasoner.get("name", "")})
        triples.append({"subject": r_subject, "predicate": "entity_type", "object": reasoner.get("entity_type", "")})
        triples.append({"subject": r_subject, "predicate": "reasoner_type", "object": reasoner.get("type", "")})
        for out in reasoner.get("outputs") or []:
            triples.append({"subject": r_subject, "predicate": "produces", "object": out})
        for signal in reasoner.get("signals") or []:
            s_name = signal.get("name") or ""
            if not s_name:
                continue
            s_subject = f"signal:{rid}.{s_name}"
            triples.append({"subject": r_subject, "predicate": "hasSignal", "object": s_subject})
            triples.append({"subject": s_subject, "predicate": "metric_field", "object": signal.get("metric_field", "")})
            triples.append({"subject": s_subject, "predicate": "threshold", "object": str(signal.get("threshold", ""))})
            triples.append({"subject": s_subject, "predicate": "direction", "object": signal.get("direction", "")})
            if signal.get("weight") is not None:
                triples.append({"subject": s_subject, "predicate": "weight", "object": str(signal.get("weight"))})
    if isinstance(kg, dict):
        nodes = kg.get("nodes") or []
        edges = kg.get("edges") or []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            entity = node.get("entity") or ""
            if not entity:
                continue
            n_subject = f"kg_node:{entity}"
            triples.append({"subject": n_subject, "predicate": "type", "object": "KGNode"})
            triples.append({"subject": n_subject, "predicate": "entity", "object": entity})
            if node.get("node_type"):
                triples.append({"subject": n_subject, "predicate": "node_type", "object": node.get("node_type", "")})
            if node.get("key_field"):
                triples.append({"subject": n_subject, "predicate": "key_field", "object": node.get("key_field", "")})
            if node.get("label_field"):
                triples.append({"subject": n_subject, "predicate": "label_field", "object": node.get("label_field", "")})
            for prop in node.get("properties") or []:
                triples.append({"subject": n_subject, "predicate": "property", "object": prop})
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            name = edge.get("name") or ""
            if not name:
                continue
            e_subject = f"kg_edge:{name}"
            triples.append({"subject": e_subject, "predicate": "type", "object": "KGEdge"})
            triples.append({"subject": e_subject, "predicate": "from", "object": edge.get("from_entity", "")})
            triples.append({"subject": e_subject, "predicate": "to", "object": edge.get("to_entity", "")})
            if edge.get("edge_type"):
                triples.append({"subject": e_subject, "predicate": "edge_type", "object": edge.get("edge_type", "")})
            for join in edge.get("join_on") or []:
                if len(join) >= 2:
                    triples.append({"subject": e_subject, "predicate": "join_on", "object": f"{join[0]}={join[1]}"})
    return triples


def _is_safe_table_ident(value: str) -> bool:
    if not value:
        return False
    return re.fullmatch(r"[A-Za-z0-9_.$\"]+", value) is not None


def _describe_table(session: Session, table: str) -> List[Dict[str, Any]]:
    if not _is_safe_table_ident(table):
        raise HTTPException(status_code=400, detail="Invalid table identifier")
    rows = session.sql(f"describe table {table}").collect()
    out = []
    for row in rows:
        data = row.asDict() if hasattr(row, "asDict") else dict(row)
        name = data.get("name") or data.get("NAME")
        dtype = data.get("type") or data.get("TYPE")
        kind = data.get("kind") or data.get("KIND")
        nullable = data.get("null?") or data.get("NULL?") or data.get("nullable") or data.get("NULLABLE")
        comment = data.get("comment") or data.get("COMMENT")
        if name:
            out.append(
                {
                    "name": str(name),
                    "type": str(dtype or ""),
                    "kind": str(kind or ""),
                    "nullable": str(nullable or ""),
                    "comment": str(comment or ""),
                }
            )
    return out


def _quote_ident(name: str) -> str:
    if not name:
        return name
    trimmed = name.strip()
    if trimmed.startswith('"') and trimmed.endswith('"'):
        return trimmed
    return f"\"{trimmed.replace('\"', '\"\"')}\""


def _quote_table(table: str) -> str:
    parts = [p for p in table.split(".") if p]
    if not parts:
        return table
    return ".".join(_quote_ident(part) for part in parts)


def _sample_table_rows(
    session: Session,
    table: str,
    columns: List[str],
    *,
    limit: int = 5,
    max_cols: int = 20,
) -> List[Dict[str, Any]]:
    if not _is_safe_table_ident(table):
        return []
    if limit <= 0:
        return []
    col_list = [c for c in (columns or []) if c]
    if not col_list:
        return []
    if max_cols and max_cols > 0:
        col_list = col_list[:max_cols]
    cols_sql = ", ".join(_quote_ident(c) for c in col_list)
    table_sql = _quote_table(table)
    try:
        df = session.sql(f"select {cols_sql} from {table_sql} limit {limit}").to_pandas()
    except Exception:
        return []
    if df is None or df.empty:
        return []
    try:
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")
    except Exception:
        return []
    cleaned = []
    for row in records:
        clean_row = {}
        for key, value in (row or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean_row[key] = value
            else:
                clean_row[key] = str(value)
        cleaned.append(clean_row)
    return cleaned


def _dtype_role(dtype: str) -> str:
    dtype_l = (dtype or "").lower()
    if any(token in dtype_l for token in ["char", "text", "string", "varchar", "date", "time", "timestamp"]):
        return "dimension"
    return "metric"


def _default_agg(field_name: str, dtype: str) -> str:
    name_l = (field_name or "").lower()
    dtype_l = (dtype or "").lower()
    if any(token in name_l for token in ["percent", "pct", "ratio", "rate", "avg", "mean", "score", "trend"]):
        return "avg"
    if any(token in dtype_l for token in ["float", "number", "decimal", "numeric", "int"]):
        return "sum"
    return ""


def _entity_name_from_table(table: str) -> str:
    base = table.split(".")[-1]
    base = re.sub(r"^rai_", "", base, flags=re.IGNORECASE)
    tokens = [t for t in re.split(r"[^A-Za-z0-9]+", base) if t]
    if not tokens:
        return base
    return "".join(t[:1].upper() + t[1:].lower() for t in tokens)


def _heuristic_registry(describes: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    entities = []
    for table, cols in describes.items():
        database = ""
        schema = ""
        parts = table.split(".")
        if len(parts) == 3:
            database, schema, _ = parts
        elif len(parts) == 2:
            schema, _ = parts
        fields = []
        join_keys = []
        for col in cols:
            name = str(col.get("name") or "")
            dtype = str(col.get("type") or "").lower()
            role = _dtype_role(dtype)
            expr = name
            if role == "metric":
                default_agg = _default_agg(name, dtype)
            else:
                default_agg = ""
            if name.lower().endswith("id") or name.lower() in {"rmid", "mandateid", "meeting_id"}:
                join_keys.append(name.lower())
            fields.append(
                {
                    "name": name.lower(),
                    "dtype": dtype,
                    "role": role,
                    "expr": expr,
                    "description": str(col.get("comment") or ""),
                    "derived": False,
                    "default_agg": default_agg,
                    "depends_on": [],
                }
            )
        metrics = [f for f in fields if f.get("role") == "metric"]
        default_metric = metrics[0]["name"] if metrics else ""
        entities.append(
            {
                "name": _entity_name_from_table(table),
                "description": "",
                "database": database,
                "schema": schema,
                "table": table.split(".")[-1],
                "fields": fields,
                "join_keys": sorted(set(join_keys)),
                "default_metric": default_metric,
                "entity_type": "generic",
                "applicable_reasoners": [],
                "applicable_derived_metrics": [],
            }
        )
    relationships = []
    for i, left in enumerate(entities):
        for right in entities[i + 1 :]:
            left_keys = set(left.get("join_keys") or [])
            right_keys = set(right.get("join_keys") or [])
            common = sorted(left_keys & right_keys)
            if not common:
                continue
            join_on = [[k, k] for k in common]
            relationships.append(
                {
                    "name": f"{left['name']}_to_{right['name']}",
                    "from_entity": left["name"],
                    "to_entity": right["name"],
                    "description": "Auto-linked by shared keys",
                    "join_on": join_on,
                }
            )
    return {"entities": entities, "relationships": relationships, "reasoners": []}


def _patch_reasoner_outputs(draft: Dict[str, Any]) -> None:
    reasoners = draft.get("reasoners") or []
    entities = draft.get("entities") or []
    if not isinstance(reasoners, list) or not isinstance(entities, list):
        return

    derived_fields: List[str] = []
    for entity in entities:
        for field in entity.get("fields") or []:
            if field.get("derived"):
                name = str(field.get("name") or "").strip()
                if name:
                    derived_fields.append(name)

    if not derived_fields:
        return

    def _candidate_fields(rid: str) -> List[str]:
        rid_norm = (rid or "").strip().lower()
        if not rid_norm:
            return []
        hits = [f for f in derived_fields if rid_norm in f.lower()]
        if hits:
            return hits
        # Fallback: common derived output suffixes
        suffixes = ("_score", "_condition", "_trend", "_priority", "_impact", "_factor")
        return [f for f in derived_fields if f.lower().endswith(suffixes)]

    for reasoner in reasoners:
        if not isinstance(reasoner, dict):
            continue
        outputs = reasoner.get("outputs")
        if isinstance(outputs, list) and outputs:
            continue
        rid = str(reasoner.get("id") or reasoner.get("name") or "").strip()
        inferred = _candidate_fields(rid)
        if inferred:
            reasoner["outputs"] = sorted(set(inferred))


def _ensure_graph_reasoners(draft: Dict[str, Any]) -> None:
    kg = draft.get("kg") or {}
    graphs = (kg.get("graphs") or []) if isinstance(kg, dict) else []
    if not isinstance(graphs, list):
        return
    reasoners = draft.get("reasoners") or []
    if not isinstance(reasoners, list):
        reasoners = []

    existing = {str(r.get("id") or "").strip(): r for r in reasoners if isinstance(r, dict)}
    for graph in graphs:
        if not isinstance(graph, dict):
            continue
        graph_id = str(graph.get("id") or "").strip()
        if not graph_id:
            continue
        if graph_id in existing:
            reasoner = existing[graph_id]
            reasoner.setdefault("type", "graph_reasoner")
            reasoner.setdefault("graph_id", graph_id)
            reasoner.setdefault(
                "params",
                [
                    {"name": "window_start", "dtype": "TIMESTAMP_NTZ(9)", "required": True, "description": "Window start"},
                    {"name": "window_end", "dtype": "TIMESTAMP_NTZ(9)", "required": True, "description": "Window end"},
                ],
            )
            continue
        nodes = graph.get("nodes") or []
        entity_type = "generic"
        if isinstance(nodes, list) and nodes:
            first = nodes[0] if isinstance(nodes[0], dict) else {}
            entity_type = str(first.get("entity") or "generic") or "generic"
        name = graph_id.replace("_", " ").title()
        reasoners.append(
            {
                "id": graph_id,
                "name": name,
                "description": graph.get("description") or f"Graph reasoner for {graph_id}.",
                "type": "graph_reasoner",
                "graph_id": graph_id,
                "entity_type": entity_type,
                "outputs": [],
                "signals": [],
                "params": [
                    {"name": "window_start", "dtype": "TIMESTAMP_NTZ(9)", "required": True, "description": "Window start"},
                    {"name": "window_end", "dtype": "TIMESTAMP_NTZ(9)", "required": True, "description": "Window end"},
                ],
            }
        )
    draft["reasoners"] = reasoners


def _draft_registry(session: Session, describes: Dict[str, List[Dict[str, Any]]], instructions: Optional[str]) -> Dict[str, Any]:
    describe_json = json.dumps(describes, indent=2, default=str)
    sample_rows: Dict[str, List[Dict[str, Any]]] = {}
    if SAMPLE_ROWS_LIMIT > 0:
        for table, cols in (describes or {}).items():
            col_names = [str(c.get("name") or "") for c in (cols or []) if c.get("name")]
            rows = _sample_table_rows(
                session,
                table,
                col_names,
                limit=SAMPLE_ROWS_LIMIT,
                max_cols=SAMPLE_COLS_LIMIT,
            )
            if rows:
                sample_rows[table] = rows
    sample_json = json.dumps(sample_rows, indent=2, default=str)
    extra = f"\nAdditional instructions:\n{instructions}\n" if instructions else ""

    prompt = """
You are generating a Semantic Registry JSON for the RAI pipeline (registry -> ontology -> KG -> dynamic reasoner execution).

Goal: produce a COMPLETE and LOGICALLY DEFENSIBLE registry using schema + sample evidence:
- represent every table as an entity
- infer a comprehensive set of defensible equi-join relationships
- create useful derived metrics backed by derived_rel_rules
- produce kg nodes/edges/graphs
- produce graph reasoners and drilldown plans

Do not be overly conservative (missing strong relations is bad), but do not fabricate weak/illogical joins (false relations are also bad).
When there are competing joins, use SAMPLE ROWS and schema heuristics to pick the most plausible. If evidence is weak, omit.

========================================================
STRICT OUTPUT FORMAT (HARD REQUIREMENT)
========================================================
- Output MUST be exactly ONE valid JSON object.
- No markdown, no code fences, no commentary, no extra top-level keys.
- The JSON MUST contain ONLY these top-level keys, and all must exist:
  - "entities": list
  - "relationships": list
  - "derived_rel_rules": list
  - "kg": object
  - "reasoners": list

========================================================
SOURCE OF TRUTH (HARD CONSTRAINT)
========================================================
- Base tables + base columns + their dtypes come ONLY from DESCRIBE OUTPUTS (JSON).
- For base (non-derived) fields, dtype MUST EXACTLY MATCH DESCRIBE (string match).
- SAMPLE ROWS are ONLY for inference/validation of joins/semantics; NEVER invent a base column from samples.

IMPORTANT: You ARE allowed to create derived metrics (derived=true fields) that do NOT exist as physical columns in DESCRIBE,
as long as they are computed strictly from existing base fields and are backed by derived_rel_rules.
(If a downstream validator rejects derived fields not present in DESCRIBE, then keep derived fields minimal and only
create them where clearly necessary for reasoners; otherwise prefer to create them comprehensively.)

Derived fields policy:
- You MAY add derived=true fields only if:
  - they depend only on existing base fields
  - you include derived_rel_rules entries for them
  - you only use evaluator-supported operations (avoid timestamp subtraction / null checks unless you are sure they work)

========================================================
NAMING + NORMALIZATION (HARD REQUIREMENT)
========================================================
- Entity.name MUST be lowercase (e.g., "dt_unit_config").
- Field.name MUST be lowercase (e.g., "pu_id").
- Field.expr MUST be the exact physical column name from DESCRIBE for base fields (often uppercase), OR "" if derived=true.
- Relationships.join_on MUST use Field.name (lowercase), not expr.

Define a DOMAIN-AGNOSTIC canonical comparator for matching candidate join keys:
- normalize(field_name):
  - lowercase
  - remove underscores
  - remove ONE trailing token among: "id", "key", "code", "num", "no" (only if present as a suffix)
Examples:
  "PU_ID" -> "puid" -> "pu"
  "plant_id" -> "plantid" -> "plant"
  "operator_code" -> "operatorcode" -> "operator"

========================================================
1) ENTITIES (REQUIRED: ONE PER TABLE IN DESCRIBE OUTPUTS)
========================================================
For EVERY table in DESCRIBE OUTPUTS, output exactly one entity:

{
  "name": <string>,
  "description": <string>,
  "database": <string>,
  "schema": <string>,
  "table": <string>,
  "fields": [ ... ],
  "join_keys": [ ... ],
  "default_metric": <string>,
  "entity_type": <string>,
  "applicable_reasoners": [ ... ],
  "applicable_derived_metrics": [ ... ]
}

Entity guidance:
- description: neutral, schema-derived, short.
- entity_type: neutral, derived from table name (no invented domain).
- applicable_reasoners: list reasoner IDs that plausibly apply to this entity (you will create these reasoners later).
- applicable_derived_metrics: list derived field names you create for this entity.

FIELDS (REQUIRED PER FIELD)
Base field (from DESCRIBE):
{
  "name": <lowercase>,
  "dtype": <DESCRIBE dtype EXACT>,
  "role": "dimension" | "metric",
  "expr": <DESCRIBE column name EXACT>,
  "description": <string>,
  "derived": false,
  "default_agg": <string>,
  "depends_on": []
}

Derived field (computed):
{
  "name": <lowercase>,
  "dtype": <valid Snowflake dtype string (e.g., "DOUBLE", "NUMBER(38,0)", "BOOLEAN", "VARCHAR")>,
  "role": "metric" or "dimension",
  "expr": "",
  "description": <string>,
  "derived": true,
  "default_agg": <string>,
  "depends_on": [<lowercase field names>]
}

Role + agg rules (domain-agnostic):
- dimension if:
  - name ends with _id/_key/_code/_type
  - looks like a label/description/name (contains name/desc/label)
  - looks like time (contains time/date/timestamp/start/end)
- metric if:
  - numeric-like dtype AND name suggests measure (count/qty/amount/value/score/rate/pct/percent/min/sec/duration)
- default_agg:
  - dimension -> ""
  - metrics:
    - durations/quantities -> "sum"
    - rates/percents -> "avg"
    - snapshot-like -> "last"

JOIN KEYS (important; used for relationship inference)
- Populate join_keys with high-likelihood identifiers:
  - Any field ending with _id, _key, _code
  - Prefer the most specific id-like fields in the table
  - Avoid adding timestamps as join_keys unless there are no id-like fields at all

DEFAULT METRIC (must be a metric, not an ID)
- Pick the most informative metric column:
  1) durations (minutes/seconds) > quantities > counts > rates/percents
  2) never pick an *_id
  3) if no metric exists, set "" (empty string)

========================================================
2) RELATIONSHIPS (STATIC EQUI-JOIN ONLY; COMPREHENSIVE BUT EVIDENCE-BASED)
========================================================
You MUST infer and emit a comprehensive set of plausible equi-join relationships across entities.
Do NOT restrict to exact column-name matches. Use canonical normalization + sample validation to infer joins.

Relationship structure (all required):
{
  "name": <string>,
  "from_entity": <string>,
  "to_entity": <string>,
  "description": <string>,
  "join_on": [[from_field, to_field], ...]
}

HARD RULES:
- Equi-join only (==).
- join_on fields MUST exist as lowercase Field.name on their entities.
- If join requires temporal overlap or effective dating, do NOT encode it here. That belongs in reasoner.graph_build.

RELATIONSHIP INFERENCE (DOMAIN-AGNOSTIC; SCHEMA + SAMPLE DRIVEN)
For each entity pair (A, B):

A) Candidate generation (schema-only):
Propose candidate join pairs (A.x == B.y) when:
1) Exact match: x == y (lowercase field names)
2) Canonical match: normalize(x) == normalize(y)
3) Join-key anchoring:
   - A contains x that matches (exact or canonical) a join_key field in B

Also propose composite candidates when 2+ candidate join pairs exist between A and B.

B) Candidate validation (use SAMPLE ROWS when available):
A join pair (A.x == B.y) is VALID if:
- Type compatibility from DESCRIBE:
  - both numeric-like OR both string-like OR both timestamp-like
  - timestamp joins require stronger evidence (prefer id-like joins)
- Sample overlap evidence:
  - at least one shared non-null value appears in both sample sets for those fields
  - OR, if sample overlap is unavailable due to sparse samples, allow exact-name joins for *_id/_key/_code ONLY,
    but mark description as "Schema-only inferred (no sample overlap observed)"
- No strong contradictions:
  - e.g., one side looks like free text (long strings) and the other looks like numeric IDs

Composite join emission:
- If 2+ VALID join pairs exist for the same (A,B), emit ONE composite relationship using all VALID pairs,
  unless samples suggest they are independent (in which case emit separate relationships).

Direction heuristic (domain-agnostic):
- Prefer from_entity = table with more metric/time fields (fact-like)
- Prefer to_entity   = table with more id/label fields (dimension-like)
If unclear, choose the direction that uses B.join_keys as the right side of join pairs.

Emission threshold (middle ground):
- Emit relationships that have at least 1 VALID join pair and are id-like or strongly evidenced by samples.
- Do NOT emit joins purely from canonical similarity if there is no sample overlap and fields are not id-like.

RELATIONSHIP NAMING:
- Unique and stable: "<from>_to_<to>__by__<key1>[_<key2>...]"
Description must state evidence:
- "Exact match + sample overlap"
- "Canonical match + sample overlap"
- "Schema-only inferred (no sample overlap observed)"

========================================================
3) DERIVED METRICS (CREATE MANY; BACK THEM WITH derived_rel_rules) — UPDATED
========================================================
Goal: create a useful set of derived metrics that can power analytics and reasoners, without inventing unsupported logic.

You MUST:
- Create derived=true metrics when they are plausible from existing fields.
- For every derived=true field, create exactly one derived_rel_rules entry.

Derived rule structure:
{
  "entity": <entity_name>,
  "field": <derived_field_name>,
  "rules": [
    { "when": <string>, "expr": <string> }
  ],
  "vars": {},
  "description": <string>
}

Evaluator expression rules (UPDATED):
- No SQL. No CASE. No COALESCE/NULLIF. No DATE_*.
- No object method calls (e.g., x.total_seconds()).
- Function-style helpers ARE allowed ONLY if they are in the allowed helper whitelist below.
- Use & and |, not and/or.
- Use parentheses around comparisons in compound conditions.
- Use single-quoted strings.
- Reference fields by bare lowercase name (no aliases).
- Boolean literals "true" and "false" are allowed in when/expr.

Allowed helper whitelist:
- ilike(field_expr, pattern_string)
  - Example: ilike(erc_desc, '%unplanned%')
  - pattern_string must be a single-quoted string and may include % wildcards

Zero-division fallback for ratios is REQUIRED:
- rule1: when="(denom > 0)" expr="numer / denom"
- rule2: when="(denom <= 0)" expr="0.0"

DERIVED METRIC INFERENCE (schema-driven; use samples for text flags)
Create derived metrics when you see these patterns (only if inputs exist):
A) Totals / sums:
- If you see multiple numeric components, create total_* as pure sums.
B) Ratios / rates:
- If you see numerator/denominator pairs, create ratio_* or pct_* with denom guards.
C) Normalizations:
- If you see qty + time/units numeric fields, create per_* rates with denom guards.
D) Text classification flags (using ilike):
- If an entity has a description-like field (contains desc/name/label/type),
  you MAY create boolean flags like is_* using ilike(),
  but ONLY for tokens/patterns that appear in SAMPLE ROWS for that field.

Avoid:
- timestamp subtraction unless you are confident it is supported
- null checks unless you are confident they are supported
- arbitrary thresholds not present in data

Also populate:
- entity.applicable_derived_metrics to include all derived field names created for that entity.

========================================================
4) KG (REQUIRED; MUST BE COMPLETE AND CONSISTENT)
========================================================
kg MUST be:
{
  "nodes": [ ... ],
  "edges": [ ... ],
  "graphs": [ ... ]
}

kg.nodes:
- Create a node for EVERY entity (yes, all).
- key_field:
  - use join_keys if there is a clear primary ID; else use a composite of the most specific IDs.
- label_field:
  - prefer *_desc, *_name, *_label; else use key_field (first key).
- properties:
  - include join_keys + 5–15 most informative fields (mix of dims + core metrics)
  - do NOT include every field blindly; pick informative ones.

kg.edges:
- MUST mirror relationships 1:1:
  - same name/from/to/join_on
  - edge_type: semantic label like "rel" or derived from relationship name (neutral)

kg.graphs (CREATE ENOUGH TO BE USEFUL, BUT NOT RANDOM):
- Create hub graphs around entities with many relationships (highest degree).
- Also create a few multi-hop graphs if there are clear bridges (A->B->C).
- For every graph edge:
  - relation MUST be exactly "from_entity.relationship_name" matching an entry in relationships
  - If relationship is from A->B, relation must be "A.<relationship_name>"

========================================================
5) REASONERS (CREATE GRAPH REASONERS + EVIDENCE DRILLDOWNS)
========================================================
Each reasoner:
{
  "id": <string>,
  "name": <string>,
  "description": <string>,
  "type": "graph_reasoner" | "signal_based" | "custom",
  "graph_id": <string>,          // required for graph_reasoner
  "entity_type": <string>,
  "outputs": [ ... ],
  "signals": [ ... ],
  "params": [ ... ],
  "graph_build": { ... },
  "drilldown_plan": { ... }
}

REASONER GENERATION (balanced):
- Create at least ONE graph_reasoner per kg.graphs entry.
- Also create a few non-graph reasoners for evidence queries when tables have time fields + metrics.

Params:
- For time-aware reasoners include:
  - window_start (TIMESTAMP_NTZ(9), required=true)
  - window_end (TIMESTAMP_NTZ(9), required=true)

graph_build (preferred edge_sets):
- For each graph_reasoner, include graph_build with edge_sets that:
  - builds edges based on equi-joins consistent with relationships
  - AND, if time fields exist, restrict edges to the window using where predicates with {"param":"window_start"/"window_end"}.
- Temporal overlap logic:
  - If both sides have start/end, you may use overlap blocks.
  - Otherwise, apply where filters on whichever entity has timestamps.

drilldown_plan:
- For each reasoner, add at least 1–3 drilldown steps for evidence.
- Use batch-friendly inputs:
  - inputs map from row.<field> and context.window_start/end
- Queries must use real entities/fields only.

Outputs/signals:
- outputs should be real fields relevant to the reasoner’s entity_type; if uncertain use [] but try to populate.
- signals: include derived metric names you created (if any) that help data-quality/trend reasoning.

========================================================
6) SELF-CONSISTENCY CHECK (DO THIS BEFORE FINAL OUTPUT)
========================================================
Before outputting JSON, ensure:
- Every DESCRIBE table has an entity.
- Every base field dtype exactly matches DESCRIBE.
- All referenced fields in relationships/kg/reasoners exist as lowercase field names.
- Every relationship has at least one join_on pair; no empty join_on.
- kg.edges mirrors relationships exactly.
- kg.graphs[*].edges[*].relation matches "from_entity.relationship_name" for an existing relationship.
- Every derived=true field has a derived_rel_rules entry.
- Every graph_reasoner.graph_id matches a kg.graphs.id.

If anything is uncertain:
- do NOT omit obviously strong relationships (exact id matches or validated by sample overlap)
- omit weak relationships (canonical-only without evidence), and state why in description when you include schema-only joins.

EXTRA INSTRUCTIONS (if any):
""" + (extra or "") + """

DESCRIBE OUTPUTS (JSON):
""" + (describe_json or "") + """

SAMPLE ROWS (JSON, up to """ + str(SAMPLE_ROWS_LIMIT) + """ rows per table):
""" + (sample_json or "") + """
""".strip()

    raw = _call_complete(session, prompt, model=DRAFT_REGISTRY_LLM)
    if DRAFT_REGISTRY_DEBUG:
        print("[DRAFT_REGISTRY] Call raw:\n", raw)
    _write_draft_registry_debug("draft-registry-entities-only", raw or "")
    parsed = _extract_json_with_keys(raw or "", ["entities", "relationships", "reasoners", "kg"])
    if not isinstance(parsed, dict) or not parsed.get("entities"):
        return {
            "entities": [],
            "relationships": [],
            "reasoners": [],
            "derived_rel_rules": [],
            "kg": {"nodes": [], "edges": [], "graphs": []},
        }
    parsed.setdefault("relationships", [])
    parsed.setdefault("reasoners", [])
    parsed.setdefault("derived_rel_rules", [])
    parsed.setdefault("kg", {"nodes": [], "edges": [], "graphs": []})
    _ensure_graph_reasoners(parsed)
    _patch_reasoner_outputs(parsed)
    return parsed


def _summarize_registry_for_plan(registry: Dict[str, Any]) -> Dict[str, Any]:
    entities = []
    for ent in registry.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        fields = []
        for f in ent.get("fields") or []:
            if not isinstance(f, dict):
                continue
            fields.append(
                {
                    "name": f.get("name"),
                    "dtype": f.get("dtype"),
                    "role": f.get("role"),
                }
            )
        entities.append(
            {
                "name": ent.get("name"),
                "entity_type": ent.get("entity_type"),
                "fields": fields,
                "join_keys": ent.get("join_keys") or [],
            }
        )
    relationships = []
    for rel in registry.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        relationships.append(
            {
                "name": rel.get("name"),
                "from_entity": rel.get("from_entity"),
                "to_entity": rel.get("to_entity"),
                "join_on": rel.get("join_on") or [],
            }
        )
    return {"entities": entities, "relationships": relationships}


def _summarize_registry_for_kg(registry: Dict[str, Any]) -> Dict[str, Any]:
    entities = []
    for ent in registry.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        fields = []
        for f in ent.get("fields") or []:
            if not isinstance(f, dict):
                continue
            fields.append(
                {
                    "name": f.get("name"),
                    "dtype": f.get("dtype"),
                    "role": f.get("role"),
                    "derived": f.get("derived", False),
                }
            )
        entities.append(
            {
                "name": ent.get("name"),
                "entity_type": ent.get("entity_type"),
                "fields": fields,
                "join_keys": ent.get("join_keys") or [],
                "default_metric": ent.get("default_metric"),
            }
        )
    relationships = []
    for rel in registry.get("relationships") or []:
        if not isinstance(rel, dict):
            continue
        relationships.append(
            {
                "name": rel.get("name"),
                "from_entity": rel.get("from_entity"),
                "to_entity": rel.get("to_entity"),
                "join_on": rel.get("join_on") or [],
            }
        )
    return {"entities": entities, "relationships": relationships}


def _draft_reasoner_plan(session: Session, payload: ReasonerPlanGenerateRequest) -> Dict[str, Any]:
    registry_payload = payload.registry or _registry_payload_for_helpers()
    registry_summary = _summarize_registry_for_plan(registry_payload)
    registry_json = json.dumps(registry_summary, indent=2, default=str)

    reasoner_info = {
        "id": payload.reasoner_id,
        "name": payload.name or payload.reasoner_id,
        "description": payload.description or "",
        "entity_type": payload.entity_type or "generic",
        "outputs": payload.outputs or [],
        "signals": payload.signals or [],
    }
    reasoner_json = json.dumps(reasoner_info, indent=2, default=str)

    instruction = (payload.instruction or "").strip()
    extra = f"\nAdditional instructions:\n{instruction}\n" if instruction else ""

    prompt = (
        "You are generating a DRILLDOWN PLAN for a registry-defined reasoner.\n"
        "Return ONE JSON object only. No markdown, no code fences, no extra text.\n"
        "Return JSON with key: drilldown_plan.\n"
        "drilldown_plan schema:\n"
        "{\n"
        "  \"steps\": [\n"
        "    {\n"
        "      \"id\": \"step_id\",\n"
        "      \"from\": \"base\" | \"<previous_step_id>\",\n"
        "      \"limit\": 1-200 (max rows to iterate from the source),\n"
        "      \"inputs\": {\"var\": \"row.<field>\" | \"context.<name>\"},\n"
        "      \"query\": <dynamic query spec JSON>\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use ONLY entities/fields that exist in the registry summary.\n"
        "- The query spec must be valid (bind/select/where/order_by/limit), and MUST NOT include aggregations.\n"
        "- Use placeholders in query values: \"$var\" where var is defined in inputs.\n"
        "- For time windows, use context.window_start / context.window_end.\n"
        "- Keep query limit <= 200.\n"
        "- Prefer multi-hop: base -> worst operations -> events -> audits.\n"
        f"{extra}\n"
        "REASONER (JSON):\n"
        f"{reasoner_json}\n"
        "REGISTRY SUMMARY (JSON):\n"
        f"{registry_json}\n"
    )

    raw = _call_complete(session, prompt, model=DRAFT_REGISTRY_LLM)
    if DRAFT_REGISTRY_DEBUG:
        print("[REASONER_PLAN] Call raw:\n", raw)

    parsed = _extract_json_with_keys(raw or "", ["drilldown_plan"])
    if isinstance(parsed, dict) and isinstance(parsed.get("drilldown_plan"), dict):
        return {"drilldown_plan": parsed.get("drilldown_plan")}

    fallback = _extract_json_with_keys(raw or "", ["steps"])
    if isinstance(fallback, dict) and isinstance(fallback.get("steps"), list):
        return {"drilldown_plan": fallback}

    return {"drilldown_plan": {}}


def _draft_kg_config(session: Session, payload: KgGenerateRequest) -> Dict[str, Any]:
    registry_payload = payload.registry or _registry_payload_for_helpers()
    registry_summary = _summarize_registry_for_kg(registry_payload)
    registry_json = json.dumps(registry_summary, indent=2, default=str)
    extra = f"\nAdditional instructions:\n{payload.instructions}\n" if payload.instructions else ""

    prompt = (
        "You are generating a KG configuration for the RAI semantic registry.\n"
        "Return ONE JSON object only. No markdown, no code fences, no extra text.\n"
        "Return ONLY JSON with key: kg.\n"
        "kg schema:\n"
        "{\n"
        "  \"nodes\": [\n"
        "    {\"entity\": \"<entity>\", \"node_type\": \"<label>\", \"key_field\": \"<field>\" | [\"<field>\", ...], "
        "\"label_field\": \"<field>\", \"properties\": [\"<field>\", ...]}\n"
        "  ],\n"
        "  \"edges\": [\n"
        "    {\"name\": \"<edge_name>\", \"from_entity\": \"<entity>\", \"to_entity\": \"<entity>\", "
        "\"join_on\": [[\"from_field\", \"to_field\"]], \"edge_type\": \"<label>\"}\n"
        "  ],\n"
        "  \"graphs\": [\n"
        "    {\"id\": \"<graph_id>\", \"description\": \"<text>\", \"directed\": true, \"weighted\": false,\n"
        "     \"nodes\": [{\"entity\": \"<entity>\", \"label_field\": \"<field>\", \"properties\": [\"<field>\", ...]}],\n"
        "     \"edges\": [{\"relation\": \"<from_entity>.<relationship>\", \"weight_field\": \"<field>\"}]}\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use ONLY entities/fields that exist in the registry summary.\n"
        "- For nodes, choose a key_field that is a join_key or primary id field. Use a list for composite keys.\n"
        "- label_field should be a descriptive dimension if available.\n"
        "- properties should be 3-8 fields (metrics or important dimensions).\n"
        "- For edges, use relationships join_on pairs exactly.\n"
        "- If unsure, return empty nodes/edges arrays.\n"
        f"{extra}\n"
        "REGISTRY SUMMARY (JSON):\n"
        f"{registry_json}\n"
    )

    raw = _call_complete(session, prompt, model=DRAFT_REGISTRY_LLM)
    if DRAFT_REGISTRY_DEBUG:
        print("[KG_CONFIG] Call raw:\n", raw)

    parsed = _extract_json_with_keys(raw or "", ["kg"])
    if isinstance(parsed, dict) and isinstance(parsed.get("kg"), dict):
        return {"kg": parsed.get("kg")}
    return {"kg": {"nodes": [], "edges": [], "graphs": []}}


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(str(value))


def _to_yaml(value: Any, indent: int = 0) -> List[str]:
    space = " " * indent
    lines: List[str] = []
    if isinstance(value, dict):
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.extend(_to_yaml(val, indent + 2))
            else:
                lines.append(f"{space}{key}: {_yaml_scalar(val)}")
        return lines
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{space}-")
                lines.extend(_to_yaml(item, indent + 2))
            else:
                lines.append(f"{space}- {_yaml_scalar(item)}")
        return lines
    return [f"{space}{_yaml_scalar(value)}"]


def _registry_payload_from_current() -> Dict[str, Any]:
    entities = []
    for entity in registry_load_entities():
        fields = []
        for field in entity.fields:
            fields.append(
                {
                    "name": field.name,
                    "dtype": field.dtype,
                    "role": field.role,
                    "expr": field.expr,
                    "description": field.description,
                    "derived": field.derived,
                    "default_agg": field.default_agg,
                    "depends_on": list(field.depends_on or []),
                }
            )
        entities.append(
            {
                "name": entity.name,
                "description": entity.description,
                "database": entity.database,
                "schema": entity.schema,
                "table": entity.table,
                "fields": fields,
                "join_keys": list(entity.join_keys or []),
                "default_metric": entity.default_metric,
                "entity_type": entity.entity_type,
                "applicable_reasoners": list(entity.applicable_reasoners or []),
                "applicable_derived_metrics": list(entity.applicable_derived_metrics or []),
            }
        )

    relationships = []
    for rel in registry_load_relationships():
        relationships.append(
            {
                "name": rel.name,
                "from_entity": rel.from_entity,
                "to_entity": rel.to_entity,
                "description": rel.description,
                "join_on": [[pair[0], pair[1]] for pair in rel.join_on],
            }
        )

    reasoners = []
    for reasoner in registry_load_reasoners():
        signals = []
        for sig in reasoner.signals or []:
            signals.append(
                {
                    "name": sig.name,
                    "description": sig.description,
                    "metric_field": sig.metric_field,
                    "threshold": sig.threshold,
                    "direction": sig.direction,
                    "weight": sig.weight,
                }
            )
        reasoners.append(
            {
                "id": reasoner.id,
                "name": reasoner.name,
                "description": reasoner.description,
                "entity_type": reasoner.entity_type,
                "type": reasoner.type,
                "outputs": list(reasoner.outputs or []),
                "signals": signals,
                "drilldown_plan": dict(getattr(reasoner, "drilldown_plan", {}) or {}),
                "graph_id": str(getattr(reasoner, "graph_id", "") or ""),
                "params": list(getattr(reasoner, "params", []) or []),
                "graph_build": dict(getattr(reasoner, "graph_build", {}) or {}),
            }
        )

    derived_rel_rules = []
    for rule in load_derived_rel_rules():
        derived_rel_rules.append(
            {
                "entity": rule.entity,
                "field": rule.field,
                "description": rule.description,
                "vars": dict(rule.vars or {}),
                "rules": [
                    {
                        "when": clause.when,
                        "expr": clause.expr,
                    }
                    for clause in (rule.rules or [])
                ],
            }
        )

    cfg = load_registry_config()
    config_payload = {
        "post_compute_derived_metrics": bool(getattr(cfg, "post_compute_derived_metrics", False)),
        "allow_sql_derived_expr": bool(getattr(cfg, "allow_sql_derived_expr", False)),
    }

    kg_payload = load_kg_spec() or {"nodes": [], "edges": [], "graphs": []}

    return {
        "entities": entities,
        "relationships": relationships,
        "reasoners": reasoners,
        "derived_rel_rules": derived_rel_rules,
        "prompt_templates": load_prompt_templates() or {},
        "analysis_config": load_analysis_config() or {},
        "config": config_payload,
        "kg": kg_payload,
    }


def _registry_payload_for_helpers() -> Dict[str, Any]:
    path = _registry_path()
    if path.exists():
        try:
            return _load_registry_payload_from_file(path)
        except Exception:
            pass
    return _registry_payload_from_current()


def _registry_reasoner_plan(reasoner_id: str) -> Dict[str, Any]:
    registry = _registry_payload_for_helpers() or {}
    for r in registry.get("reasoners") or []:
        if isinstance(r, dict) and r.get("id") == reasoner_id:
            return dict(r.get("drilldown_plan") or {})
    return {}


def _reasoner_step_select_fields(reasoner_id: str, step_id: str) -> List[str]:
    plan = _registry_reasoner_plan(reasoner_id)
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("id") or "").strip() != step_id:
            continue
        query = step.get("query") if isinstance(step.get("query"), dict) else {}
        select = query.get("select") or []
        fields = []
        for term in select:
            if isinstance(term, dict) and term.get("prop"):
                fields.append(str(term.get("prop")))
        return [f for f in fields if f]
    return []


def _clear_registry_caches() -> None:
    for fn in (
        registry_entities,
        registry_relationships,
        load_ai_insights_relationships,
        load_ai_insights_specs,
        build_ai_insights_builder,
        _rai_builder_and_specs,
    ):
        try:
            fn.cache_clear()
        except Exception:
            pass


def _rewrite_spec_aliases(spec: dict) -> dict:
    alias_map = {"rm_score": "performance"}
    specs = load_ai_insights_specs()
    if specs and not getattr(specs[0], "field_exprs", None):
        load_ai_insights_specs.cache_clear()
        specs = load_ai_insights_specs()
    fields_by_entity = {s.name: list(s.fields or []) for s in specs}
    field_name_by_entity = {name: {str(f).lower(): f for f in fields} for name, fields in fields_by_entity.items()}
    exprs_by_entity = {}
    expr_to_field_by_entity = {}
    for s in specs:
        exprs = dict(getattr(s, "field_exprs", {}) or {})
        exprs_by_entity[s.name] = exprs
        expr_to_field_by_entity[s.name] = {
            str(expr).lower(): field for field, expr in (exprs or {}).items() if expr
        }
    alias_to_entity = {}
    for b in (spec or {}).get("bind") or []:
        if isinstance(b, dict) and b.get("alias") and b.get("entity"):
            alias_to_entity[b["alias"]] = b["entity"]

    def _rewrite(node):
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
            return {k: _rewrite(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_rewrite(v) for v in node]
        return node

    return _rewrite(spec) if isinstance(spec, dict) else spec


def _coerce_value_terms(spec: dict, specs: list) -> dict:
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
    for s in specs or []:
        fields = list(getattr(s, "fields", []) or [])
        field_name_by_entity[s.name] = {str(f).lower(): f for f in fields}
        exprs = dict(getattr(s, "field_exprs", {}) or {})
        expr_to_field_by_entity[s.name] = {
            str(expr).lower(): field for field, expr in (exprs or {}).items() if expr
        }
        types_by_entity[s.name] = dict(getattr(s, "field_types", {}) or {})

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
            return ""
        dtype = (types_by_entity.get(ent, {}) or {}).get(canonical)
        return str(dtype or "").lower()

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

    def _coerce_value(value, dtype: str):
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


def get_session() -> Session:
    global _session
    if _session is not None:
        return _session
    params = _session_params()
    if not params.get("account") or not params.get("user"):
        raise RuntimeError("Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER in environment")
    
    # Add timeout protection for session creation
    import threading
    import sys
    
    session_holder = [None]
    exception_holder = [None]
    
    def create_session():
        try:
            session_holder[0] = Session.builder.configs(params).create()
        except Exception as e:
            exception_holder[0] = e
    
    # Timeout in seconds - if session creation takes longer, it's likely hanging on MFA
    timeout_seconds = int(os.environ.get("SNOWFLAKE_SESSION_TIMEOUT", "120"))
    
    print(f"[DEBUG] Creating Snowflake session with {timeout_seconds}s timeout...", file=sys.stderr)
    sys.stderr.flush()
    
    t = threading.Thread(target=create_session, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)
    
    if t.is_alive():
        print(f"[ERROR] Snowflake session creation timed out after {timeout_seconds}s", file=sys.stderr)
        print(f"[ERROR] This usually means MFA is prompting or network is unreachable", file=sys.stderr)
        sys.stderr.flush()
        raise RuntimeError(
            f"Snowflake session creation timed out after {timeout_seconds}s. "
            "This usually indicates MFA is waiting for input or network is unavailable. "
            "Check SNOWFLAKE_AUTHENTICATOR setting - use 'username_password_mfa' with passcode for non-interactive auth."
        )
    
    if exception_holder[0]:
        raise exception_holder[0]
    
    _session = session_holder[0]
    if _session is None:
        raise RuntimeError("Failed to create Snowflake session")
    
    print(f"[DEBUG] Snowflake session created successfully", file=sys.stderr)
    sys.stderr.flush()
    return _session


def create_session() -> Session:
    """Create a new Snowflake session."""
    params = _session_params()
    if not params.get("account") or not params.get("user"):
        raise RuntimeError("Missing SNOWFLAKE_ACCOUNT or SNOWFLAKE_USER in environment")

    session_holder = [None]
    exception_holder = [None]

    def create_session_inner():
        try:
            session_holder[0] = Session.builder.configs(params).create()
        except Exception as e:
            exception_holder[0] = e

    timeout_seconds = int(os.environ.get("SNOWFLAKE_SESSION_TIMEOUT", "120"))
    print(f"[DEBUG] Creating Snowflake session (per-request) with {timeout_seconds}s timeout...", file=sys.stderr)
    sys.stderr.flush()

    t = threading.Thread(target=create_session_inner, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        print(f"[ERROR] Snowflake session creation timed out after {timeout_seconds}s", file=sys.stderr)
        sys.stderr.flush()
        raise RuntimeError(
            f"Snowflake session creation timed out after {timeout_seconds}s. "
            "This usually indicates MFA is waiting for input or network is unavailable. "
            "Check SNOWFLAKE_AUTHENTICATOR setting - use 'username_password_mfa' with passcode for non-interactive auth."
        )

    if exception_holder[0]:
        raise exception_holder[0]

    sess = session_holder[0]
    if sess is None:
        raise RuntimeError("Failed to create Snowflake session")

    print(f"[DEBUG] Snowflake session created successfully", file=sys.stderr)
    sys.stderr.flush()
    return sess


def acquire_session() -> Session:
    """Acquire a session from the pool or create a new one up to pool size."""
    global _session_pool_total
    try:
        return _session_pool.get_nowait()
    except Exception:
        pass
    with _session_pool_lock:
        if _session_pool_total < _session_pool_size:
            sess = create_session()
            _session_pool_total += 1
            return sess
    try:
        wait_s = int(os.environ.get("SNOWFLAKE_SESSION_POOL_WAIT", "30"))
        return _session_pool.get(timeout=wait_s)
    except Exception as e:
        raise RuntimeError(f"No Snowflake sessions available in pool: {e}")


def release_session(sess: Session) -> None:
    """Return a session to the pool."""
    if sess is None:
        return
    try:
        _session_pool.put_nowait(sess)
    except Exception:
        try:
            sess.close()
        except Exception:
            pass


@lru_cache(maxsize=1)
def _rai_builder_and_specs():
    debug_init = os.environ.get("AI_INSIGHTS_DEBUG_INIT", "").strip().lower() in ("1", "true", "yes")
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] backend _rai_builder_and_specs: start\n")
        except Exception:
            debug_init = False
    t0 = time.perf_counter() if debug_init else 0.0
    ensure_rai_config()
    load_ai_insights_specs.cache_clear()
    build_ai_insights_builder.cache_clear()
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] backend _rai_builder_and_specs: caches_cleared\n")
        except Exception:
            pass
    specs = load_ai_insights_specs()
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write(f"[INIT] backend _rai_builder_and_specs: specs={len(specs)}\n")
        except Exception:
            pass
    builder = build_ai_insights_builder()
    if debug_init:
        try:
            import sys as _sys
            elapsed = time.perf_counter() - t0
            _sys.stderr.write(f"[INIT] backend _rai_builder_and_specs: done in {elapsed:.2f}s\n")
        except Exception:
            pass
    return builder, specs


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return row
    for attr in ("asDict", "as_dict", "_asdict"):
        if hasattr(row, attr):
            try:
                return getattr(row, attr)()
            except Exception:
                pass
    if hasattr(row, "__dict__"):
        try:
            return dict(row.__dict__)
        except Exception:
            pass
    try:
        return dict(row)
    except Exception:
        return {"value": str(row)}


def _rows_to_dicts(rows: List[Any]) -> List[Dict[str, Any]]:
    return [_row_to_dict(r) for r in rows]


def _json_extract(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    if not text:
        return None
    m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _extract_json_candidates(text: str) -> List[Any]:
    if not text:
        return []
    candidates: List[Any] = []
    stack = []
    start = None
    for idx, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start = idx
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            opener = stack.pop()
            if (opener == "{" and ch != "}") or (opener == "[" and ch != "]"):
                stack = []
                start = None
                continue
            if not stack and start is not None:
                snippet = text[start : idx + 1]
                try:
                    parsed = json.loads(snippet)
                    candidates.append(parsed)
                except Exception:
                    pass
                start = None
    return candidates


def _extract_json_with_keys(text: str, required_keys: List[str]) -> Dict[str, Any]:
    required = set(required_keys or [])
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and required.issubset(parsed.keys()):
            return parsed
    except Exception:
        pass
    for candidate in _extract_json_candidates(text):
        if isinstance(candidate, dict) and required.issubset(candidate.keys()):
            return candidate
    return {}


def _write_draft_registry_debug(label: str, text: str) -> None:
    if not DRAFT_REGISTRY_DEBUG:
        return
    try:
        root = Path(__file__).parent.parent
        out_path = root / "registry" / "draft_registry_llm_debug.txt"
        timestamp = datetime.utcnow().isoformat()
        payload = f"\n=== {label} @ {timestamp}Z ===\n{text}\n"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass




_RULE_ENGINE_PREAMBLE = (
    "IMPORTANT: These rule strings are evaluated by our KG using Python eval into RelationalAI query-builder (Rel) expressions.\n"
    "Write expressions in Python syntax that *constructs Rel terms*, not general Python logic.\n"
    "Constraints:\n"
    "- Use ONLY entity field names (bare names). Do NOT invent fields.\n"
    "- Boolean logic MUST use '&' and '|' (NOT 'and'/'or').\n"
    "- Allowed ops: + - * /, comparisons (< > <= >= == !=), parentheses.\n"
    "- Do NOT use SQL/SQL functions/keywords (CASE, NULLIF, COALESCE, SELECT, FROM, JOIN, DATE_* etc).\n"
    "- Use std.datetime.date(y,m,d) for dates and std.math.minimum(a,b) for min.\n"
    "- String matching is allowed via ilike(field, pattern) (case-insensitive) or like(field, pattern).\n"
    "- 'when' may be omitted for unconditional mappings. 'when' may also be 'true'/'false' (case-insensitive).\n"
    "- Prior-row logic is NOT automatic: only use prev.* if vars includes {\\\"prev\\\":\\\"ref\\\"} and the when includes explicit join constraints.\n"
)

def _default_rule_prompt_generate() -> str:
    return _RULE_ENGINE_PREAMBLE + """You are building derived_rel_rules for the RAI registry -> ontology -> KG pipeline.
Convert the user's natural language into derived_rel_rules JSON for the given ENTITY and FIELD.

Return JSON only in this shape:
{"entity":"<Entity>","field":"<DerivedField>","vars":{},"rules":[{"when":"<condition optional>","expr":"<rel expr>"}]}

Rules guidance:
- Use explicit guards for divide-by-zero (two rules: guarded expr + fallback).
- Use '&' and '|' only; do NOT use 'and'/'or'.
- Wrap comparisons in parentheses when combining: (a>0.4)&(a<=0.75).
- Use single-quoted string literals in expr (e.g., 'at_risk').
- Use std.datetime.date(y,m,1) and std.math.minimum(a,b) when needed.
- If you need prior-row logic: vars={"prev":"ref"} and use prev.<field> with explicit join constraints in when.
"""


def _default_rule_prompt_validate() -> str:
    return _RULE_ENGINE_PREAMBLE + """Validate the derived_rel_rules JSON for the RAI pipeline.
Check: entity exists, field exists and is derived, referenced fields exist.
Reject SQL syntax/functions/keywords and Python 'and'/'or'.
Allow empty/omitted 'when' for unconditional rules.

Return JSON only:
{"valid":true|false,"issues":["..."]}
"""


_RULE_SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "case",
    "when",
    "then",
    "else",
    "end",
    "and",
    "or",
    "not",
    "nullif",
    "coalesce",
    "lag",
    "lead",
    "over",
    "partition",
    "order",
    "by",
    "as",
    "in",
    "on",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "sum",
    "avg",
    "min",
    "max",
    "count",
    "year",
    "month",
    "date_from_parts",
}


def _extract_rule_fields(expr: str) -> List[str]:
    if not expr:
        return []
    scrubbed = re.sub(r"'[^']*'", " ", expr)
    scrubbed = re.sub(r'"[^"]*"', " ", scrubbed)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", scrubbed)
    fields = []
    for token in tokens:
        key = token.lower()
        if key in _RULE_SQL_KEYWORDS:
            continue
        if key in ("true","false","none"):
            continue
        fields.append(token)
    return fields


def _local_validate_rule(rule: Dict[str, Any], registry: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not isinstance(rule, dict):
        return ["Rule payload is not an object."]
    entity_name = str(rule.get("entity") or "")
    field_name = str(rule.get("field") or "")
    if not entity_name:
        issues.append("Missing entity.")
        return issues
    if not field_name:
        issues.append("Missing field.")
        return issues
    entity = next((e for e in (registry.get("entities") or []) if e.get("name") == entity_name), None)
    if not entity:
        issues.append(f"Unknown entity: {entity_name}.")
        return issues
    fields = {str(f.get("name") or ""): f for f in (entity.get("fields") or [])}
    if field_name not in fields:
        issues.append(f"Unknown field: {entity_name}.{field_name}.")
        return issues
    if not fields[field_name].get("derived"):
        issues.append(f"Field {entity_name}.{field_name} is not marked derived.")
    rules = rule.get("rules") or []
    if not isinstance(rules, list) or not rules:
        issues.append("No rules provided.")
        return issues
    known = set(name.lower() for name in fields.keys())
    vars_map = rule.get("vars") or {}
    if vars_map and not isinstance(vars_map, dict):
        issues.append("vars must be an object when provided.")
    for clause in rules:
        if not isinstance(clause, dict):
            issues.append("Rule clause must be an object.")
            continue
        expr = str(clause.get("expr") or "").strip()
        if not expr:
            issues.append("Rule clause missing expr.")
            continue
        for token in _extract_rule_fields(expr):
            if token in ("m", "std", "qb"):
                continue
            if token in vars_map:
                continue
            if token.lower() not in known:
                # allow dotted references like prev.field
                if "." in token:
                    head = token.split(".")[0]
                    if head in vars_map:
                        continue
                issues.append(f"Unknown field reference in expr: {token}")
        when_expr = str(clause.get("when") or "").strip()
        if when_expr:
            for token in _extract_rule_fields(when_expr):
                if token in ("m", "std", "qb"):
                    continue
                if token in vars_map:
                    continue
                if token.lower() not in known:
                    if "." in token:
                        head = token.split(".")[0]
                        if head in vars_map:
                            continue
                    issues.append(f"Unknown field reference in when: {token}")
    return issues


def _extract_rule_dependencies(rule: Dict[str, Any], field_map: Dict[str, Dict[str, Any]]) -> List[str]:
    tokens: List[str] = []
    for clause in rule.get("rules") or []:
        tokens.extend(_extract_rule_fields(str(clause.get("expr") or "")))
        tokens.extend(_extract_rule_fields(str(clause.get("when") or "")))
    deps = []
    ignore = {"std", "qb", "m", "prev"}
    for token in tokens:
        if token in ignore:
            continue
        if token in field_map:
            deps.append(token)
    return sorted(set(deps))


def _collect_rule_dependency_closure(
    entity_name: str,
    target_rule: Dict[str, Any],
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    entity = next((e for e in (registry.get("entities") or []) if e.get("name") == entity_name), None)
    if not entity:
        return {"rules": [target_rule], "derived_missing": [], "base_fields": []}
    field_map = {
        str(field.get("name")): field
        for field in (entity.get("fields") or [])
        if field.get("name")
    }
    derived_rules_map = {
        (rule.get("entity"), rule.get("field")): rule
        for rule in (registry.get("derived_rel_rules") or [])
        if rule.get("entity") == entity_name
    }
    closure: List[Dict[str, Any]] = []
    derived_missing: List[Dict[str, Any]] = []
    visited: set[tuple[str, str]] = set()
    queue = [target_rule]
    while queue:
        current = queue.pop(0)
        current_entity = current.get("entity") or entity_name
        current_field = current.get("field") or ""
        key = (current_entity, current_field)
        if key in visited:
            continue
        visited.add(key)
        closure.append(current)
        deps = _extract_rule_dependencies(current, field_map)
        for dep in deps:
            field = field_map.get(dep) or {}
            if not field.get("derived"):
                continue
            dep_key = (entity_name, dep)
            dep_rule = derived_rules_map.get(dep_key)
            if dep_rule:
                queue.append(dep_rule)
            else:
                derived_missing.append(
                    {
                        "field": dep,
                        "expr": str(field.get("expr") or ""),
                        "depends_on": field.get("depends_on") or [],
                    }
                )
    base_fields = []
    for name, field in field_map.items():
        if field.get("derived"):
            continue
        base_fields.append({"name": name, "expr": str(field.get("expr") or "")})
    return {"rules": closure, "derived_missing": derived_missing, "base_fields": base_fields}


def _call_complete(session: Session, prompt: str, model: Optional[str] = None) -> str:
    safe = prompt.replace("$$", "$ $")
    llm = model or DEFAULT_LLM
    sql = f"select snowflake.cortex.complete('{llm}', $$" + safe + "$$) as response"
    df = session.sql(sql).collect()
    if not df:
        return ""
    return str(df[0][0])


def _semantic_kind(view_name: str = "") -> str:
    forced = os.environ.get("AI_INSIGHTS_SEMANTIC_KIND", "view").strip().lower()
    name_u = (view_name or "").upper()
    if forced.startswith("model"):
        return "model"
    if forced.startswith("view"):
        if name_u.endswith("_MODEL"):
            return "model"
        return "view"
    if name_u.endswith("_MODEL"):
        return "model"
    return "view"


def _analyst_rest_url() -> Optional[str]:
    url = os.environ.get("ANALYST_REST_URL")
    if url:
        return url
    acct = os.environ.get("SNOWFLAKE_ACCOUNT", "").strip()
    if not acct:
        return None
    if acct.endswith(".snowflakecomputing.com"):
        return f"https://{acct}/api/v2/cortex/analyst/message"
    return f"https://{acct}.snowflakecomputing.com/api/v2/cortex/analyst/message"


def _call_analyst_rest(user_text: str, view_name: str) -> Dict[str, Any]:
    token = os.environ.get("ANALYST_REST_TOKEN")
    if not token:
        return {"ok": False, "error": "Missing ANALYST_REST_TOKEN"}
    url = _analyst_rest_url()
    if not url:
        return {"ok": False, "error": "Missing ANALYST_REST_URL or SNOWFLAKE_ACCOUNT"}
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Snowflake-Authorization-Token-Type": os.environ.get("ANALYST_REST_TOKEN_TYPE", "PROGRAMMATIC_ACCESS_TOKEN"),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        "stream": False,
    }
    if _semantic_kind(view_name) == "model":
        body["semantic_model"] = view_name
    else:
        body["semantic_view"] = view_name
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        data = resp.json() if resp.content else {}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": bool(resp.ok), "status": resp.status_code, "json": data, "text": resp.text}


def _parse_analyst_sql(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    if payload.get("sql"):
        return str(payload.get("sql"))
    msg = payload.get("message") or {}
    for block in msg.get("content") or []:
        if (block.get("type") or "").lower() in {"sql", "semantic_sql"}:
            stmt = block.get("statement") or block.get("sql") or block.get("query")
            if stmt:
                return str(stmt)
    return ""


def analyst_sql(session: Session, view_name: str, prompt: str) -> str:
    allow_fallback = os.environ.get("ANALYST_ALLOW_FALLBACK", "1").strip() == "1"
    rest = _call_analyst_rest(prompt, view_name)
    if rest.get("ok"):
        sql = _parse_analyst_sql(rest.get("json") or {})
        if sql:
            return sql
    if not allow_fallback:
        return ""
    fallback_prompt = (
        "Return ONLY a SQL statement that answers the task. "
        f"Task: {prompt}"
    )
    out = _call_complete(session, fallback_prompt)
    if out.strip().lower().startswith("select"):
        return out.strip()
    parsed = _json_extract(out)
    if isinstance(parsed, dict) and parsed.get("sql"):
        return str(parsed.get("sql")).strip()
    return ""


def _base_table_name(semantic_name: str) -> str:
    if semantic_name.upper().endswith("_MODEL"):
        return semantic_name[:-6]
    return semantic_name


def _insights_tables(views: List[str]) -> List[str]:
    tables_env = os.environ.get("AI_INSIGHTS_TABLES", "").strip()
    if tables_env:
        return [t.strip() for t in tables_env.split(",") if t.strip()]
    return [_base_table_name(v) for v in views]


def _best_table_match(alias: str, tables: List[str]) -> Optional[str]:
    alias_u = alias.upper()
    alias_tokens = set(re.split(r"[^A-Z0-9]+", alias_u)) - {""}
    if not alias_tokens:
        return None
    best = None
    best_score = 0
    for table in tables:
        short = table.split(".")[-1].upper()
        table_tokens = set(short.split("_")) - {""}
        score = len(alias_tokens & table_tokens)
        if score > best_score:
            best = table
            best_score = score
    return best if best_score >= 2 else None


def _fallback_sql_for_tables(sql: str, views: List[str]) -> str:
    updated = sql
    for view in views:
        base = _base_table_name(view)
        if base == view:
            continue
        updated = re.sub(re.escape(view), base, updated, flags=re.IGNORECASE)
        short_view = view.split(".")[-1]
        short_base = base.split(".")[-1]
        updated = re.sub(rf"\b{re.escape(short_view)}\b", short_base, updated, flags=re.IGNORECASE)

    tables = _insights_tables(views)
    # Replace unqualified logical table names in FROM/JOIN clauses with base tables.
    def _replace_match(match: re.Match) -> str:
        keyword = match.group(1)
        ident = match.group(2)
        if "." in ident:
            return match.group(0)
        best = _best_table_match(ident, tables)
        if not best:
            return match.group(0)
        return f"{keyword} {best}"

    updated = re.sub(r"\b(FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]+)\b", _replace_match, updated, flags=re.IGNORECASE)
    return updated


def _raw_fallback_sql(question: str, base_table: str) -> str:
    years = re.findall(r"\b(20\\d{2})\b", question or "")
    where = ""
    if years:
        year_list = ", ".join(years)
        where = f"WHERE YEAR(MONTH_DATE) IN ({year_list})"
    return f"SELECT * FROM {base_table} {where} LIMIT 2000"


def select_view(session: Session, question: str, views: List[str]) -> str:
    if len(views) == 1:
        return views[0]
    view_list = "\n".join(f"- {v}" for v in views)
    prompt = (
        "Pick the single best semantic view for the question. "
        "Return ONLY the view name.\n\n"
        f"Question: {question}\n\n"
        f"Available views:\n{view_list}\n"
    )
    raw = _call_complete(session, prompt).strip()
    for v in views:
        if v.lower() in raw.lower():
            return v
    return views[0]


def summarize(session: Session, question: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    preview = rows[:200]
    columns = list(preview[0].keys()) if preview else []
    if len(columns) > 30:
        columns = columns[:30]
    if columns and preview:
        slim_preview = [{k: row.get(k) for k in columns} for row in preview]
    else:
        slim_preview = preview
    prompt = (
        "You are a BI analyst. Create a concise answer and a Plotly chart spec. "
        "Return ONLY JSON with keys: narrative (string), kpis (list of {label,value,unit?}), "
        "chart (Plotly JSON with keys data/layout). If chart is not possible, return {}.\n\n"
        f"Question: {question}\n\n"
        f"Columns: {json.dumps(columns)}\n\n"
        f"Rows (preview): {json.dumps(slim_preview, default=str)}\n"
    )
    try:
        raw = _call_complete(session, prompt)
        parsed = _json_extract(raw) or {}
        if not isinstance(parsed, dict):
            parsed = {}
    except Exception:
        parsed = {}
    return {
        "narrative": str(parsed.get("narrative") or "").strip(),
        "kpis": parsed.get("kpis") or [],
        "chart": parsed.get("chart") or {},
    }


def _ensure_kpis(summary: Dict[str, Any], row_count: int) -> List[Dict[str, Any]]:
    kpis = summary.get("kpis") if isinstance(summary, dict) else None
    if isinstance(kpis, list) and kpis:
        return kpis
    return [{"label": "Rows", "value": row_count}]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest, async_mode: bool = Query(False, alias="async")) -> AskResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    request_id = req.request_id or str(uuid.uuid4())
    _logger.info(f"Processing question [{request_id}]: {question}")
    
    if async_mode:
        with _request_lock:
            _request_store[request_id] = {
                "status": "running",
                "stage": "planning",
                "thinking": "",
                "result": None,
                "error": None,
                "updated_at": time.time(),
            }

        def _worker():
            session = None
            set_request_id(request_id)
            try:
                session = acquire_session()
                payload = run_orchestrate(session, question)
                with _request_lock:
                    _request_store[request_id]["status"] = "done"
                    _request_store[request_id]["result"] = payload
                    _request_store[request_id]["stage"] = _ORCHESTRATION_STATE.get("requests", {}).get(request_id, {}).get("stage", "done")
                    _request_store[request_id]["thinking"] = _ORCHESTRATION_STATE.get("requests", {}).get(request_id, {}).get("thinking", "")
                    _request_store[request_id]["updated_at"] = time.time()
            except Exception as exc:
                detail = getattr(exc, "raw_content", None) or getattr(exc, "content", None)
                if not detail:
                    table_objects = getattr(exc, "table_objects", None)
                    if table_objects:
                        try:
                            detail = "\n".join([f"{o.source}: {o.message}" for o in table_objects])
                        except Exception:
                            detail = None
                detail_text = f"\n{detail}" if detail else ""
                with _request_lock:
                    _request_store[request_id]["status"] = "error"
                    _request_store[request_id]["error"] = f"AI Insights orchestration failed: {exc}{detail_text}"
                    _request_store[request_id]["stage"] = "error"
                    _request_store[request_id]["updated_at"] = time.time()
            finally:
                try:
                    if session is not None:
                        release_session(session)
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True).start()

        return AskResponse(
            question=question,
            request_id=request_id,
        )

    session = acquire_session()
    try:
        # Start with planning stage (provisioning/initializing will be detected from orchestrator output)
        _ORCHESTRATION_STATE['current_stage'] = 'planning'
        _current_stage['current'] = 'planning'
        set_request_id(request_id)
        payload = run_orchestrate(session, question)
        _current_stage['current'] = _ORCHESTRATION_STATE.get('current_stage', 'done')
    except Exception as exc:
        _current_stage['current'] = 'error'
        detail = getattr(exc, "raw_content", None) or getattr(exc, "content", None)
        if not detail:
            table_objects = getattr(exc, "table_objects", None)
            if table_objects:
                try:
                    detail = "\n".join([f"{o.source}: {o.message}" for o in table_objects])
                except Exception:
                    detail = None
        detail_text = f"\n{detail}" if detail else ""
        raise HTTPException(status_code=400, detail=f"AI Insights orchestration failed: {exc}{detail_text}")
    finally:
        try:
            if session is not None:
                release_session(session)
        except Exception:
            pass

    # Ensure payload is a dict
    if not isinstance(payload, dict):
        payload = {}
    
    # Log payload structure for debugging
    import sys
    print(f"[DEBUG][ask] Response payload keys: {list(payload.keys())}", file=sys.stderr)
    
    # Helper functions for type safety
    def get_as_dict(val, default=None):
        if isinstance(val, dict):
            return val
        return default if default is not None else {}
    
    def get_as_list(val, default=None):
        if isinstance(val, list):
            return val
        return default if default is not None else []
    
    def get_as_str(val, default=""):
        if isinstance(val, str):
            return val
        if val is None:
            return default
        return str(val) if val else default
    
    # Clean and validate all payload fields
    cleaned_payload = {
        "question": question,
        "request_id": request_id,
        "plan": get_as_dict(payload.get("plan")),
        "frames": get_as_dict(payload.get("frames")),
        "insights": get_as_str(payload.get("insights")),
        "kpis": get_as_list(payload.get("kpis")),
        "chart": get_as_dict(payload.get("chart")),
        "log": get_as_list(payload.get("log")),
        "sqls": get_as_list(payload.get("sqls")),
        "followups": get_as_list(payload.get("followups")),
        "merges": get_as_list(payload.get("merges")),
        "catalog": get_as_dict(payload.get("catalog")),
        "needs_more": payload.get("needs_more"),
        "followup_prompt": payload.get("followup_prompt"),
        "required": get_as_dict(payload.get("required")),
        "summary_obj": get_as_dict(payload.get("summary_obj")),
        "view": payload.get("view"),
        "sql": payload.get("sql"),
        "rows": get_as_list(payload.get("rows")),
        "narrative": payload.get("narrative"),
        "thinking": payload.get("thinking"),
    }
    
    # Ensure frames is properly formatted (dict of lists, not dict of dicts)
    frames = cleaned_payload.get("frames", {})
    if isinstance(frames, dict):
        fixed_frames = {}
        for k, v in frames.items():
            if isinstance(v, list):
                # Already a list - ensure it's a list of dicts
                fixed_frames[k] = v
            elif isinstance(v, dict):
                # Single dict - wrap in list
                fixed_frames[k] = [v]
            else:
                # Other type - try to convert or skip
                try:
                    if hasattr(v, 'to_dict'):  # pandas DataFrame
                        fixed_frames[k] = [v.to_dict()]
                    else:
                        fixed_frames[k] = []
                except Exception:
                    fixed_frames[k] = []
        cleaned_payload["frames"] = fixed_frames
    
    # Try to construct AskResponse with comprehensive error handling
    try:
        print(f"[DEBUG][ask] Creating AskResponse with cleaned payload", file=sys.stderr)
        response = AskResponse(**cleaned_payload)
        return response
    except Exception as exc:
        import traceback
        print(f"[ERROR][ask] AskResponse validation failed: {exc}", file=sys.stderr)
        print(f"[ERROR][ask] Cleaned payload structure:", file=sys.stderr)
        for key in cleaned_payload.keys():
            val = cleaned_payload[key]
            print(f"  {key}: {type(val).__name__} = {repr(val)[:100]}", file=sys.stderr)
        print(f"[ERROR][ask] Full traceback: {traceback.format_exc()}", file=sys.stderr)
        
        # Return a minimal but valid response instead of crashing
        # This ensures the client gets a response even if the response structure is unexpected
        return AskResponse(
            question=question,
            plan={},
            frames=cleaned_payload.get("frames", {}),
            insights=cleaned_payload.get("insights", ""),
            kpis=cleaned_payload.get("kpis", []),
            chart=cleaned_payload.get("chart", {}),
            log=cleaned_payload.get("log", []),
            sqls=cleaned_payload.get("sqls", []),
            followups=cleaned_payload.get("followups", []),
            merges=cleaned_payload.get("merges", []),
            catalog=cleaned_payload.get("catalog", {}),
        )


class SpecTestRequest(BaseModel):
    """Submit a RAI spec directly for testing without NL input"""
    spec: Dict[str, Any]


class SpecTestResponse(BaseModel):
    """Response from spec test endpoint"""
    status: str  # "valid", "error"
    message: str
    spec_validation: Dict[str, Any] = Field(default_factory=dict)
    execution_result: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None


@app.post("/api/spec/test", response_model=SpecTestResponse)
def spec_test(req: SpecTestRequest) -> SpecTestResponse:
    """
    Test a RAI spec directly without natural language processing.
    
    This allows you to:
    1. Validate spec structure against registry
    2. Execute the spec against Snowflake (if available)
    3. Inspect the results without NL interpretation
    
    Example spec for machine downtime summary (see /api/spec/example)
    """
    spec = req.spec
    if not isinstance(spec, dict):
        raise HTTPException(status_code=400, detail="spec must be a JSON object")
    
    # Validate spec structure
    validation_result = _validate_spec_structure(spec)
    
    if not validation_result["valid"]:
        return SpecTestResponse(
            status="error",
            message=f"Spec validation failed: {validation_result['errors']}",
            spec_validation=validation_result
        )
    
    # If we can't connect to Snowflake, still return validation results
    try:
        session = get_session()
        # If we got here, Snowflake is available - execute the spec
        result = _execute_spec_against_snowflake(session, spec)
        return SpecTestResponse(
            status="valid",
            message="Spec executed successfully",
            spec_validation=validation_result,
            execution_result=result
        )
    except RuntimeError as e:
        if "Session" in str(e):
            # Snowflake not available, but spec is valid
            return SpecTestResponse(
                status="valid",
                message="Spec structure is valid, but Snowflake connection not available for execution",
                spec_validation=validation_result,
                error_details=str(e)
            )
        raise


def _validate_spec_structure(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate spec structure against registry"""
    errors = []
    warnings = []
    
    # Check required sections
    if not spec.get("bind"):
        errors.append("Missing 'bind' section")
    else:
        bind = spec["bind"]
        if not isinstance(bind, list) or len(bind) == 0:
            errors.append("'bind' must be non-empty array")
        else:
            for i, b in enumerate(bind):
                entity = b.get("entity")
                alias = b.get("alias")
                if not entity:
                    errors.append(f"bind[{i}] missing 'entity'")
                if not alias:
                    errors.append(f"bind[{i}] missing 'alias'")
                else:
                    # Check entity exists in registry
                    registry = _registry_payload_for_helpers()
                    entities = {e["name"]: e for e in (registry.get("entities") or [])}
                    if entity not in entities:
                        errors.append(f"Entity '{entity}' not found in registry")
                    else:
                        # Check all referenced properties exist
                        entity_def = entities[entity]
                        available_props = {f["name"] for f in (entity_def.get("fields") or [])}
                        
                        # Collect all properties used
                        props_used = set()
                        for sel in (spec.get("select") or []):
                            if sel.get("alias") == alias:
                                props_used.add(sel.get("prop"))
                        for grp in (spec.get("group_by") or []):
                            if grp.get("alias") == alias:
                                props_used.add(grp.get("prop"))
                        for agg in (spec.get("aggregations") or []):
                            term = agg.get("term", {})
                            if isinstance(term, dict) and term.get("alias") == alias:
                                props_used.add(term.get("prop"))
                        
                        missing_props = props_used - available_props
                        if missing_props:
                            errors.append(f"Entity '{entity}' missing properties: {missing_props}")
    
    # Check select section
    if not spec.get("select"):
        warnings.append("No 'select' clause - query may not return results")
    
    # Check aggregations + group_by consistency
    has_agg = bool(spec.get("aggregations"))
    has_grp = bool(spec.get("group_by"))
    if has_agg and not has_grp:
        warnings.append("Has aggregations but no group_by - may produce single row")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "spec_structure": {
            "has_bind": bool(spec.get("bind")),
            "has_where": bool(spec.get("where")),
            "has_select": bool(spec.get("select")),
            "has_group_by": bool(spec.get("group_by")),
            "has_aggregations": bool(spec.get("aggregations")),
            "has_order_by": bool(spec.get("order_by")),
            "has_limit": bool(spec.get("limit")),
        }
    }


def _execute_spec_against_snowflake(session, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a spec against Snowflake (stub - actual execution would go here)"""
    return {
        "rows": [],
        "column_count": len(spec.get("select", [])),
        "query_time_ms": 0,
        "status": "executed"
    }


@app.get("/api/spec/example")
def spec_example() -> Dict[str, Any]:
    """Get an example spec for machine downtime summary"""
    return {
        "description": "Machine downtime summary - groups faults by machine with aggregations",
        "bind": [{"alias": "f", "entity": "dt_fault_details"}],
        "where": [
            {
                "and": [
                    {
                        "op": "<=",
                        "left": {"alias": "f", "prop": "start_time"},
                        "right": {"value": "2026-02-02 23:59:59"}
                    },
                    {
                        "op": ">=",
                        "left": {"alias": "f", "prop": "end_time"},
                        "right": {"value": "2026-01-26 00:00:00"}
                    }
                ]
            }
        ],
        "select": [
            {"alias": "f", "prop": "pu_id", "as": "machine_id"},
            {"alias": "f", "prop": "pu_desc", "as": "machine_name"}
        ],
        "group_by": [
            {"alias": "f", "prop": "pu_id", "as": "machine_id"},
            {"alias": "f", "prop": "pu_desc", "as": "machine_name"}
        ],
        "aggregations": [
            {
                "op": "sum",
                "term": {"alias": "f", "prop": "duration"},
                "as": "total_downtime_duration"
            },
            {
                "op": "count",
                "term": {"alias": "f", "prop": "tedet_id"},
                "as": "downtime_event_count"
            }
        ],
        "order_by": [
            {
                "term": {"value": "total_downtime_duration"},
                "dir": "desc"
            }
        ],
        "limit": 2000,
        "meta": {
            "intent": "machine_downtime_summary",
            "time_window": {
                "start": "2026-01-26 00:00:00",
                "end": "2026-02-02 23:59:59",
                "mode": "overlap"
            }
        }
    }


@app.get("/api/stages")
def get_stages(request_id: Optional[str] = None) -> Dict[str, str]:
    """Get current orchestration stage for polling"""
    if request_id:
        req_state = _ORCHESTRATION_STATE.get("requests", {}).get(request_id, {})
        stage = req_state.get("stage", "planning")
        thinking = req_state.get("thinking", "")
        return {
            "stage": stage,
            "current": stage,
            "thinking": thinking,
        }
    return {
        "stage": _ORCHESTRATION_STATE.get('current_stage', 'planning'),
        "current": _current_stage.get('current', 'planning'),
        "thinking": _ORCHESTRATION_STATE.get("thinking", "")
    }


@app.get("/api/result")
def get_result(request_id: str) -> Dict[str, Any]:
    """Get async orchestration result by request_id."""
    with _request_lock:
        entry = _request_store.get(request_id)
    if not entry:
        # Fallback to orchestrator state (may still hold result/stage)
        req_state = _ORCHESTRATION_STATE.get("requests", {}).get(request_id, {})
        if req_state.get("result") is not None:
            return {"status": "done", "result": req_state.get("result")}
        if req_state.get("stage") in ("planning", "fetching", "linking", "summarizing", "executing"):
            return {"status": "running"}
        return {"status": "unknown", "error": "request_id not found (server may have restarted)"}
    status = entry.get("status")
    if status == "done":
        return {"status": "done", "result": entry.get("result")}
    if status == "error":
        return {"status": "error", "error": entry.get("error")}
    return {"status": "running"}


@app.post("/api/registry/describe")
def registry_describe(req: RegistryDescribeRequest) -> Dict[str, Any]:
    table = (req.table or "").strip()
    if not table:
        raise HTTPException(status_code=400, detail="table is required")
    session = get_session()
    columns = _describe_table(session, table)
    return {"table": table, "columns": columns}


@app.post("/api/registry/rules/generate")
def registry_rule_generate(req: DerivedRuleGenerateRequest) -> Dict[str, Any]:
    registry = req.registry or _registry_payload_for_helpers()
    entities = registry.get("entities") or []
    entity = next((e for e in entities if e.get("name") == req.entity), None)
    if not entity:
        raise HTTPException(status_code=400, detail=f"Unknown entity: {req.entity}")
    field = next((f for f in (entity.get("fields") or []) if f.get("name") == req.field), None)
    if not field:
        raise HTTPException(status_code=400, detail=f"Unknown field: {req.entity}.{req.field}")

    fields = [f.get("name") for f in (entity.get("fields") or []) if f.get("name")]
    derived_fields = [f.get("name") for f in (entity.get("fields") or []) if f.get("derived")]

    templates = registry.get("prompt_templates") or load_prompt_templates() or {}
    template = templates.get("visual_builder_logic_to_rules") or _default_rule_prompt_generate()
    if "IMPORTANT: These rule strings are evaluated" not in template:
        template = _RULE_ENGINE_PREAMBLE + "\n" + template

    prompt = (
        f"{template}\n"
        f"\nENTITY: {req.entity}\n"
        f"FIELD: {req.field}\n"
        f"FIELDS: {', '.join(fields)}\n"
        f"DERIVED_FIELDS: {', '.join(derived_fields)}\n"
        f"INSTRUCTION: {req.instruction}\n"
        "Return JSON only.\n"
    )

    session = get_session()
    raw = _call_complete(session, prompt)
    parsed = _json_extract(raw) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    parsed.setdefault("entity", req.entity)
    parsed.setdefault("field", req.field)
    return {"rule": parsed, "raw": raw}


@app.post("/api/registry/rules/validate")
def registry_rule_validate(req: DerivedRuleValidateRequest) -> Dict[str, Any]:
    registry = req.registry or _registry_payload_for_helpers()
    local_issues = _local_validate_rule(req.rule, registry)

    templates = load_prompt_templates() or {}
    template = templates.get("visual_builder_rule_validate") or _default_rule_prompt_validate()
    if "IMPORTANT: These rule strings are evaluated" not in template:
        template = _RULE_ENGINE_PREAMBLE + "\n" + template
    prompt = (
        f"{template}\n"
        f"\nRULE_JSON:\n{json.dumps(req.rule, indent=2)}\n"
    )

    llm_payload: Dict[str, Any] = {}
    try:
        session = get_session()
        raw = _call_complete(session, prompt)
        parsed = _json_extract(raw) or {}
        if isinstance(parsed, dict):
            llm_payload = parsed
    except Exception:
        llm_payload = {}

    issues = list(llm_payload.get("issues") or []) if isinstance(llm_payload.get("issues"), list) else []

    suppress_if_local_ok = any(
        "Unknown entity" in item or "Unknown field" in item for item in local_issues
    ) is False

    if suppress_if_local_ok:
        issues = [
            issue
            for issue in issues
            if "cannot be verified" not in str(issue).lower()
            and "cannot be confirmed" not in str(issue).lower()
        ]

    for item in local_issues:
        if item not in issues:
            issues.append(item)
    valid = bool(llm_payload.get("valid")) if "valid" in llm_payload else not issues
    return {"valid": valid and not issues, "issues": issues}


@app.post("/api/registry/rules/sql")
def registry_rule_sql(req: DerivedRuleSqlRequest) -> Dict[str, Any]:
    registry = req.registry or _registry_payload_for_helpers()
    rule = req.rule or {}
    if not isinstance(rule, dict):
        raise HTTPException(status_code=400, detail="rule must be an object")
    entity_name = str(rule.get("entity") or "")
    field_name = str(rule.get("field") or "")
    if not entity_name or not field_name:
        raise HTTPException(status_code=400, detail="rule must include entity and field")
    entity = next((e for e in (registry.get("entities") or []) if e.get("name") == entity_name), None)
    if not entity:
        raise HTTPException(status_code=400, detail=f"Unknown entity: {entity_name}")
    if not any(f.get("name") == field_name for f in (entity.get("fields") or [])):
        raise HTTPException(status_code=400, detail=f"Unknown field: {entity_name}.{field_name}")

    database = str(entity.get("database") or "").strip()
    schema = str(entity.get("schema") or "").strip()
    table = str(entity.get("table") or "").strip()
    if database and schema and table:
        source_table = f"{database}.{schema}.{table}"
    else:
        source_table = table or entity_name

    closure = _collect_rule_dependency_closure(entity_name, rule, registry)
    field_specs = [
        {
            "name": str(field.get("name") or ""),
            "derived": bool(field.get("derived")),
            "expr": str(field.get("expr") or ""),
            "depends_on": field.get("depends_on") or [],
        }
        for field in (entity.get("fields") or [])
        if field.get("name")
    ]

    prompt = (
        "You convert derived_rel_rules JSON into executable Snowflake SQL for validation.\n"
        "Use BASE_TABLE as the source and produce a single SQL query.\n"
        "Requirements:\n"
        "- Use only base fields in the FROM table; expand derived fields via CTEs in dependency order.\n"
        "- Convert rule clauses to SQL CASE WHEN. If 'when' is empty/omitted/true, treat as ELSE.\n"
        "- Never use booleans in arithmetic. Replace boolean guards with CASE WHEN or IFF.\n"
        "  Example: (cond) * expr -> CASE WHEN cond THEN expr ELSE 0 END.\n"
        "- Convert '&' to AND, '|' to OR, '==' to '='. Keep !=, >=, <=.\n"
        "- std.datetime.date(y,m,d) -> DATE_FROM_PARTS(y,m,d).\n"
        "- std.math.minimum(a,b) -> LEAST(a,b). std.math.maximum(a,b) -> GREATEST(a,b).\n"
        "- Return SQL ONLY (no markdown).\n\n"
        f"BASE_TABLE: {source_table}\n"
        f"TARGET_FIELD: {entity_name}.{field_name}\n\n"
        f"FIELDS_JSON:\n{json.dumps(field_specs, ensure_ascii=True)}\n\n"
        f"DERIVED_RULES_JSON:\n{json.dumps(closure.get('rules') or [], ensure_ascii=True)}\n\n"
        f"DERIVED_EXPR_FALLBACKS_JSON:\n{json.dumps(closure.get('derived_missing') or [], ensure_ascii=True)}\n\n"
        f"BASE_FIELDS_JSON:\n{json.dumps(closure.get('base_fields') or [], ensure_ascii=True)}\n"
    )

    session = get_session()
    raw = _call_complete(session, prompt)
    sql = (raw or "").strip()
    if not sql:
        raise HTTPException(status_code=500, detail="SQL generation failed")
    return {"sql": sql, "raw": raw}


@app.post("/api/registry/reasoners/drilldown-plan")
def registry_reasoner_plan(req: ReasonerPlanGenerateRequest) -> Dict[str, Any]:
    if not req.reasoner_id:
        raise HTTPException(status_code=400, detail="reasoner_id is required")
    session = get_session()
    return _draft_reasoner_plan(session, req)


@app.post("/api/registry/reasoners/sanity")
def registry_reasoner_sanity(req: ReasonerSanityRequest) -> Dict[str, Any]:
    reasoner_id = (req.reasoner_id or "").strip()
    step_id = (req.step_id or "").strip()
    if not reasoner_id or not step_id:
        raise HTTPException(status_code=400, detail="reasoner_id and step_id are required")

    relation = sanitize_identifier(f"Reasoner_{reasoner_id}_{step_id}")
    select_fields = _reasoner_step_select_fields(reasoner_id, step_id)
    if not select_fields:
        select_fields = ["pu_id", "window_start", "window_end"]

    where = []
    if req.window_start:
        where.append({"op": "==", "left": {"alias": "r", "prop": "window_start"}, "right": {"value": req.window_start}})
    if req.window_end:
        where.append({"op": "==", "left": {"alias": "r", "prop": "window_end"}, "right": {"value": req.window_end}})

    spec = {
        "bind": [{"alias": "r", "entity": relation}],
        "where": where,
        "select": [{"alias": "r", "prop": field, "as": field} for field in select_fields],
        "limit": max(1, min(int(req.limit or 5), 50)),
    }

    try:
        builder = build_ai_insights_builder()
        df = run_dynamic_query(builder, spec)
        rows = df.to_dict(orient="records") if df is not None else []
    except Exception as exc:
        detail = getattr(exc, "raw_content", None) or getattr(exc, "content", None)
        detail_text = f"\n{detail}" if detail else ""
        raise HTTPException(status_code=400, detail=f"Reasoner sanity query failed: {exc}{detail_text}")

    return {
        "relation": relation,
        "select": select_fields,
        "rows": rows,
        "row_count": len(rows),
    }


@app.post("/api/registry/kg/generate")
def registry_kg_generate(req: KgGenerateRequest) -> Dict[str, Any]:
    session = get_session()
    return _draft_kg_config(session, req)


@app.post("/api/kg/rebuild")
def kg_rebuild() -> Dict[str, Any]:
    try:
        _clear_registry_caches()
        builder, specs = _rai_builder_and_specs()
        reasoners = registry_load_reasoners()
    except Exception as exc:
        detail = getattr(exc, "raw_content", None) or getattr(exc, "content", None)
        detail_text = f"\n{detail}" if detail else ""
        raise HTTPException(status_code=400, detail=f"KG rebuild failed: {exc}{detail_text}")

    return {
        "status": "ok",
        "entities": len(specs or []),
        "reasoners": len([r for r in (reasoners or []) if getattr(r, "id", None)]),
    }


@app.post("/api/registry/prompts/customize")
def registry_prompts_customize(req: PromptCustomizationRequest) -> Dict[str, Any]:
    base_templates = dict(DEFAULT_PROMPT_TEMPLATES)
    keys = [key for key in (req.prompt_keys or list(base_templates.keys())) if key in base_templates]
    if not keys:
        raise HTTPException(status_code=400, detail="No valid prompt keys supplied.")

    business = (req.business_context or "").strip()
    data_ctx = (req.data_context or "").strip()
    expectations = (req.expectations or "").strip()
    if not any([business, data_ctx, expectations]):
        return {"templates": {key: base_templates[key] for key in keys}, "used_defaults": True}

    base_subset = {key: base_templates[key] for key in keys}
    prompt = (
        "You customize prompt templates for a BI assistant.\n"
        "Use BASE_TEMPLATES_JSON as the foundation. Keep required schemas, keys, placeholders, and guardrails.\n"
        "Only add or adjust domain-specific wording, examples, and expectations based on the context.\n"
        "Do NOT remove or weaken constraints like output JSON shape, allowed chart types, or safety rules.\n"
        "Return JSON ONLY with the same keys and string values.\n\n"
        f"BUSINESS_CONTEXT:\n{business or 'N/A'}\n\n"
        f"DATA_CONTEXT:\n{data_ctx or 'N/A'}\n\n"
        f"EXPECTATIONS:\n{expectations or 'N/A'}\n\n"
        f"BASE_TEMPLATES_JSON:\n{json.dumps(base_subset, ensure_ascii=True)}\n"
    )

    session = get_session()
    raw = _call_complete(session, prompt)
    parsed = _extract_json_with_keys(raw, keys)

    templates: Dict[str, str] = {}
    for key in keys:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            templates[key] = value.strip()
        else:
            templates[key] = base_templates[key]

    return {"templates": templates, "used_defaults": False}


@app.post("/api/registry/draft")
def registry_draft(req: RegistryDraftRequest) -> Dict[str, Any]:
    tables = [t.strip() for t in (req.tables or []) if t and t.strip()]
    if not tables:
        raise HTTPException(status_code=400, detail="tables is required")
    session = get_session()
    describes: Dict[str, List[Dict[str, Any]]] = {}
    for table in tables:
        describes[table] = _describe_table(session, table)
    registry = _draft_registry(session, describes, req.instructions)
    return {"registry": registry, "describes": describes}


@app.get("/api/registry/load")
def registry_load() -> Dict[str, Any]:
    path = _registry_path()
    if path.exists():
        try:
            payload = _load_registry_payload_from_file(path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read registry file: {exc}")
        return {"exists": True, "path": str(path), "registry": payload, "source": "file"}
    payload = _registry_payload_from_current()
    return {"exists": False, "path": str(path), "registry": payload, "source": "active"}


@app.post("/api/registry/save")
def registry_save(req: RegistrySaveRequest) -> Dict[str, Any]:
    payload = req.registry or {}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="registry must be an object")
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_registry_payload_to_file(path, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write registry file: {exc}")
    _write_registry_version(payload)
    yaml_text = "\n".join(_to_yaml(payload))
    return {"ok": True, "path": str(path), "yaml": yaml_text}


@app.post("/api/registry/reload")
def registry_reload() -> Dict[str, Any]:
    _clear_registry_caches()
    return {"ok": True}


@app.get("/api/registry/versions")
def registry_versions() -> Dict[str, Any]:
    return {"versions": _list_registry_versions()}


@app.post("/api/registry/rollback")
def registry_rollback(req: RegistryRollbackRequest) -> Dict[str, Any]:
    version = (req.version or "").strip()
    if not version:
        raise HTTPException(status_code=400, detail="version is required")
    history_dir = _registry_history_dir()
    target = history_dir / version
    if not target.exists():
        raise HTTPException(status_code=404, detail="version not found")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read version: {exc}")
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_registry_payload_to_file(path, payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to restore registry: {exc}")
    _clear_registry_caches()
    return {"ok": True, "path": str(path)}


@app.post("/api/registry/ontology/export")
def registry_ontology_export(req: RegistryOntologyExportRequest) -> Dict[str, Any]:
    payload = req.registry
    if not payload:
        path = _registry_path()
        if path.exists():
            payload = _load_registry_payload_from_file(path)
        else:
            payload = _registry_payload_from_current()
    triples = _registry_to_triples(payload)
    export_path = ""
    if req.write_file:
        export_dir = _registry_export_dir()
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_path = str(export_dir / f"ontology_triples_{timestamp}.json")
        Path(export_path).write_text(json.dumps(triples, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"ok": True, "triples": triples, "path": export_path}


# ============================================================================
# LOG STREAMING ENDPOINTS
# ============================================================================

def _log_stream_generator() -> Generator[str, None, None]:
    """Generator that streams logs from the queue."""
    try:
        while True:
            try:
                log_entry = _log_queue.get(timeout=1)
                yield json.dumps(log_entry) + "\n"
            except queue.Empty:
                # Send a heartbeat to keep connection alive
                yield json.dumps({"type": "heartbeat"}) + "\n"
    except GeneratorExit:
        pass
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"


@app.get("/api/logs/stream")
def stream_logs() -> StreamingResponse:
    """
    Stream real-time logs from the API server.
    
    Returns: SSE-like streaming response with JSON log entries.
    Each line is a JSON object with: timestamp, level, logger, message, module.
    Use ?filter=<level> to filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    return StreamingResponse(
        _log_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/logs/recent")
def get_recent_logs(limit: int = Query(100, ge=1, le=1000)) -> Dict[str, Any]:
    """
    Get recent logs from history.
    
    Parameters:
    - limit: Number of recent logs to return (max 1000)
    
    Returns: List of log entries with timestamp, level, logger, message, module.
    """
    with _log_history_lock:
        recent = _log_history[-limit:] if limit > 0 else _log_history.copy()
    
    return {
        "ok": True,
        "count": len(recent),
        "logs": recent
    }


@app.get("/api/logs/clear")
def clear_logs() -> Dict[str, Any]:
    """Clear the log history."""
    with _log_history_lock:
        _log_history.clear()
    
    return {"ok": True, "message": "Log history cleared"}


@app.get("/api/logs/stats")
def get_log_stats() -> Dict[str, Any]:
    """Get statistics about collected logs."""
    with _log_history_lock:
        levels = {}
        for log_entry in _log_history:
            level = log_entry.get("level", "UNKNOWN")
            levels[level] = levels.get(level, 0) + 1
    
    return {
        "ok": True,
        "total_logs": len(_log_history),
        "queue_size": _log_queue.qsize(),
        "by_level": levels,
        "max_history": _max_log_history,
    }


# ============================================================================
# KG ENHANCEMENT ENDPOINTS (Generic utilities for advanced analysis)
# ============================================================================

class TemporalJoinRequest(BaseModel):
    entity1_name: str = Field(..., description="First entity to join")
    entity1_time_start: str = Field(..., description="Start time column in entity1")
    entity1_time_end: str = Field(..., description="End time column in entity1")
    entity2_name: str = Field(..., description="Second entity to join")
    entity2_time_start: str = Field(..., description="Start time column in entity2")
    entity2_time_end: str = Field(..., description="End time column in entity2")
    join_key: Optional[str] = Field(None, description="Optional common dimension to match (e.g., unit_id)")
    overlap_required: bool = Field(True, description="If True, time windows must overlap. If False, can be sequential.")
    max_gap_minutes: Optional[int] = Field(None, description="If set, only join if gap < this (for sequential joins)")
    limit: int = Field(1000, description="Max result rows")


class HierarchyAggregationRequest(BaseModel):
    entity_name: str = Field(..., description="Entity to aggregate")
    hierarchy_path: List[str] = Field(..., description="Dimension columns defining hierarchy (e.g., ['unit_id', 'line_id', 'facility_id'])")
    metric_column: str = Field(..., description="Column to aggregate (e.g., fault_count, duration)")
    metric_agg: str = Field("sum", description="Aggregation: sum, avg, min, max, count")
    filter_expr: Optional[Dict[str, Any]] = Field(None, description="Optional filter dict")


class SequenceDetectionRequest(BaseModel):
    entity_name: str = Field(..., description="Entity to analyze")
    sequence_key: str = Field(..., description="Grouping dimension (e.g., unit_id)")
    event_dimension: str = Field(..., description="Column identifying event type (e.g., fault_name)")
    time_column: str = Field(..., description="Timestamp column")
    max_gap_minutes: int = Field(360, description="Events > this apart are NOT in same sequence")
    min_occurrences: int = Field(5, description="Only report sequences occurring this many times")


class TrendAnalysisRequest(BaseModel):
    entity_name: str = Field(..., description="Entity to analyze")
    grouping_dimensions: List[str] = Field(..., description="Dimensions to group by (e.g., ['unit_id'])")
    metric_column: str = Field(..., description="Metric to track (e.g., fault_count)")
    time_column: str = Field(..., description="Timestamp column")
    time_period: str = Field("month", description="Grouping period: day, week, month, quarter, year")
    anomaly_threshold_pct: float = Field(50.0, description="Flag as anomaly if > this % above moving average")


@app.post("/api/kg/temporal-join")
def kg_temporal_join(req: TemporalJoinRequest) -> Dict[str, Any]:
    """
    Join two entities based on time window overlap or proximity.
    
    Use case: Correlate events (faults, executions, incidents) based on timing.
    Example: Find all faults that occurred during batch executions.
    """
    try:
        from rai_kg_enhancements import temporal_join_entities
        
        result_df = temporal_join_entities(
            entity1_name=req.entity1_name,
            entity1_time_start=req.entity1_time_start,
            entity1_time_end=req.entity1_time_end,
            entity2_name=req.entity2_name,
            entity2_time_start=req.entity2_time_start,
            entity2_time_end=req.entity2_time_end,
            join_key=req.join_key,
            overlap_required=req.overlap_required,
            max_gap_minutes=req.max_gap_minutes,
        )
        
        result_df = result_df.head(req.limit)
        return {
            "ok": True,
            "result_count": len(result_df),
            "data": result_df.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Temporal join failed: {str(exc)}")


@app.post("/api/kg/hierarchical-aggregation")
def kg_hierarchical_aggregation(req: HierarchyAggregationRequest) -> Dict[str, Any]:
    """
    Aggregate metrics up a hierarchy (e.g., Unit → Line → Facility → Plant).
    
    Use case: Drill-down analytics at different organizational/asset levels.
    Example: Aggregate fault duration by unit, line, and facility separately.
    """
    try:
        from rai_kg_enhancements import hierarchical_aggregation
        
        result_df = hierarchical_aggregation(
            entity_name=req.entity_name,
            hierarchy_path=req.hierarchy_path,
            metric_column=req.metric_column,
            metric_agg=req.metric_agg,
            filter_expr=req.filter_expr,
        )
        
        return {
            "ok": True,
            "result_count": len(result_df),
            "data": result_df.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Hierarchical aggregation failed: {str(exc)}")


@app.post("/api/kg/event-sequences")
def kg_event_sequences(req: SequenceDetectionRequest) -> Dict[str, Any]:
    """
    Detect recurring patterns/sequences of events within a grouping.
    
    Use case: Find fault cascades, incident chains, production failure sequences.
    Example: Find that Bearing_Wear always leads to Temp_High within 1 hour.
    """
    try:
        from rai_kg_enhancements import detect_event_sequences
        
        result_df = detect_event_sequences(
            entity_name=req.entity_name,
            sequence_key=req.sequence_key,
            event_dimension=req.event_dimension,
            time_column=req.time_column,
            max_gap_minutes=req.max_gap_minutes,
            min_occurrences=req.min_occurrences,
        )
        
        return {
            "ok": True,
            "result_count": len(result_df),
            "data": result_df.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Sequence detection failed: {str(exc)}")


@app.post("/api/kg/trend-analysis")
def kg_trend_analysis(req: TrendAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze trends over time, detect anomalies.
    
    Use case: Detect degradation, anomalies, forecast failures.
    Example: Track fault count by unit, identify when a unit starts having issues.
    """
    try:
        from rai_kg_enhancements import analyze_trends
        
        result_df = analyze_trends(
            entity_name=req.entity_name,
            grouping_dimensions=req.grouping_dimensions,
            metric_column=req.metric_column,
            time_column=req.time_column,
            time_period=req.time_period,
            anomaly_threshold_pct=req.anomaly_threshold_pct,
        )
        
        return {
            "ok": True,
            "result_count": len(result_df),
            "data": result_df.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Trend analysis failed: {str(exc)}")


STATIC_DIR = Path(__file__).with_name("static")
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
