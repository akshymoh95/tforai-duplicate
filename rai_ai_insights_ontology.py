from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

import relationalai.semantics as qb
from relationalai.semantics import std
from relationalai.semantics.std import strings as rstr
from relationalai.semantics.snowflake import Table
from relationalai.semantics.rel.rel_utils import sanitize_identifier
from relationalai.semantics.reasoners.graph import Graph
from rai_semantic_registry import (
    load_registry,
    load_relationships,
    load_derived_rel_rules,
    load_reasoners,
    load_kg_spec,
)


@dataclass(frozen=True)
class SemanticSpec:
    name: str
    description: str
    database: str
    schema: str
    table: str
    fields: List[str]
    join_keys: List[str]
    derived_fields: List[str]
    field_types: Dict[str, str]
    field_roles: Dict[str, str]
    field_aggs: Dict[str, str]
    field_exprs: Dict[str, str]
    field_descs: Dict[str, str]
    default_metric: str
    entity_type: str


@dataclass(frozen=True)
class RelationshipView:
    name: str
    from_entity: str
    to_entity: str
    description: str
    join_on: List[tuple[str, str]]


@lru_cache(maxsize=1)
def registry_entities() -> Dict[str, object]:
    return {e.name: e for e in load_registry()}


@lru_cache(maxsize=1)
def registry_relationships() -> List[object]:
    return list(load_relationships())


@lru_cache(maxsize=1)
def load_ai_insights_relationships() -> List[object]:
    return list(load_relationships())


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=1)
def load_ai_insights_specs() -> List[SemanticSpec]:
    specs: List[SemanticSpec] = []
    entities = registry_entities()
    for name in sorted(entities):
        entity = entities[name]
        base_fields = [f.name for f in entity.fields if not f.derived]
        derived_fields = [f.name for f in entity.fields if f.derived]
        field_types = {f.name: f.dtype for f in entity.fields}
        field_roles = {f.name: f.role for f in entity.fields}
        field_aggs = {f.name: f.default_agg for f in entity.fields if f.default_agg}
        field_descs = {f.name: f.description for f in entity.fields if f.description}
        field_exprs = {}
        for f in entity.fields:
            if f.derived:
                continue
            # Normalize expression: use uppercase and ensure it's sanitized
            expr = (f.expr or "").strip().upper()
            if expr and re.match(r"^[A-Z0-9_]+$", expr):
                # Valid SQL identifier, keep it
                field_exprs[f.name] = expr
            elif expr:
                # Invalid characters, fallback to field name
                import sys
                print(f"[WARN] Entity {entity.name}, field {f.name}: expr '{expr}' contains invalid chars, using field name", file=sys.stderr)
                field_exprs[f.name] = f.name.upper()
            else:
                # No expr, use field name as fallback
                field_exprs[f.name] = f.name.upper()
        specs.append(
            SemanticSpec(
                name=entity.name,
                description=entity.description,
                database=entity.database,
                schema=entity.schema,
                table=entity.table,
                fields=base_fields,
                join_keys=list(entity.join_keys or []),
                derived_fields=derived_fields,
                field_types=field_types,
                field_roles=field_roles,
                field_aggs=field_aggs,
                field_exprs=field_exprs,
                field_descs=field_descs,
                default_metric=entity.default_metric,
                entity_type=entity.entity_type,
            )
        )
    specs.extend(_reasoner_step_specs())
    return specs


def _reasoner_step_concept_name(reasoner_id: str, step_id: str) -> str:
    return sanitize_identifier(f"Reasoner_{reasoner_id}_{step_id}")


def _lookup_field_dtype_role_any(field_name: str) -> tuple[str, str]:
    target = str(field_name or "").lower()
    if not target:
        return "", ""
    entities = registry_entities()
    for entity in entities.values():
        for field in entity.fields:
            name_norm = str(field.name or "").lower()
            if name_norm == target:
                return field.dtype or "", field.role or ""
            expr_norm = str(field.expr or "").lower()
            if expr_norm and expr_norm == target:
                return field.dtype or "", field.role or ""
    return "", ""


def _lookup_field_dtype_role(entity_name: str, field_name: str) -> tuple[str, str]:
    if not entity_name or not field_name:
        return "", ""
    entities = registry_entities()
    entity = entities.get(entity_name)
    if not entity:
        return "", ""
    target = str(field_name or "").lower()
    for field in entity.fields:
        name_norm = str(field.name or "").lower()
        if name_norm == target:
            return field.dtype or "", field.role or ""
        expr_norm = str(field.expr or "").lower()
        if expr_norm and expr_norm == target:
            return field.dtype or "", field.role or ""
    return "", ""


def _reasoner_step_specs() -> List[SemanticSpec]:
    specs: List[SemanticSpec] = []
    reasoners = load_reasoners()
    if not reasoners:
        return specs

    for reasoner in reasoners:
        reasoner_id = getattr(reasoner, "id", "") or ""
        plan = getattr(reasoner, "drilldown_plan", None) or {}
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if not reasoner_id or not isinstance(steps, list):
            continue

        for step in steps:
            if not isinstance(step, dict):
                continue
            step_id = str(step.get("id") or "").strip()
            if not step_id:
                continue

            query = step.get("query") if isinstance(step.get("query"), dict) else {}
            binds = query.get("bind") or []
            select = query.get("select") or []
            inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}

            alias_to_entity: Dict[str, str] = {}
            for b in binds:
                if isinstance(b, dict) and b.get("alias") and b.get("entity"):
                    alias_to_entity[b["alias"]] = b["entity"]

            fields: List[str] = []
            field_types: Dict[str, str] = {}
            field_roles: Dict[str, str] = {}
            field_aggs: Dict[str, str] = {}
            field_exprs: Dict[str, str] = {}
            field_descs: Dict[str, str] = {}

            def _add_field(name: str, dtype: str, role: str) -> None:
                if not name:
                    return
                if name not in fields:
                    fields.append(name)
                if dtype:
                    field_types[name] = dtype
                if role:
                    field_roles[name] = role

            for input_name in inputs.keys():
                dtype, role = _lookup_field_dtype_role_any(input_name)
                if not dtype and input_name in ("window_start", "window_end", "as_of_time"):
                    dtype, role = "TIMESTAMP_NTZ(9)", "dimension"
                if not dtype:
                    dtype = "VARCHAR(16777216)"
                if not role:
                    role = "dimension"
                _add_field(input_name, dtype, role)

            for term in select:
                if not isinstance(term, dict):
                    continue
                alias = term.get("alias")
                prop = term.get("prop")
                if not alias or not prop:
                    continue
                out_name = term.get("as") or prop
                ent_name = alias_to_entity.get(alias, "")
                dtype, role = _lookup_field_dtype_role(ent_name, prop) if ent_name else ("", "")
                if not dtype:
                    dtype, role = _lookup_field_dtype_role_any(prop)
                if not dtype:
                    dtype = "VARCHAR(16777216)"
                if not role:
                    role = "dimension"
                _add_field(out_name, dtype, role)

            if not fields:
                continue

            specs.append(
                SemanticSpec(
                    name=_reasoner_step_concept_name(reasoner_id, step_id),
                    description=f"Reasoner {reasoner_id} step {step_id}",
                    database="",
                    schema="",
                    table="",
                    fields=fields,
                    join_keys=[],
                    derived_fields=[],
                    field_types=field_types,
                    field_roles=field_roles,
                    field_aggs=field_aggs,
                    field_exprs=field_exprs,
                    field_descs=field_descs,
                    default_metric="",
                    entity_type="reasoner_step",
                )
            )

    return specs


def infer_join_keys(specs: Iterable[SemanticSpec]) -> List[str]:
    preferred = [
        "rmid",
        "rm_name",
        "mandateid",
        "mandateid_str",
        "month_date",
        "meet_year",
        "meet_mon",
        "posyear",
        "posmon",
    ]
    join_keys = []
    for s in specs:
        join_keys.extend(list(s.join_keys or []))
    if join_keys:
        ordered = []
        for k in preferred:
            if k in join_keys and k not in ordered:
                ordered.append(k)
        for k in join_keys:
            if k not in ordered:
                ordered.append(k)
        return ordered
    fields_by_spec = [set(s.fields) for s in specs]
    out = []
    for key in preferred:
        if sum(1 for fields in fields_by_spec if key in fields) >= 2:
            out.append(key)
    return out


def get_applicable_reasoners_for_entity(entity_name: str) -> List[str]:
    """Derive applicable reasoners from registry"""
    from rai_semantic_registry import load_registry
    
    registry = load_registry()
    for entity in registry:
        if entity.name == entity_name:
            return list(entity.applicable_reasoners or [])
    return []


def get_applicable_derived_metrics_for_entity(entity_name: str) -> List[str]:
    """Derive applicable derived metrics from registry"""
    from rai_semantic_registry import load_registry
    
    registry = load_registry()
    for entity in registry:
        if entity.name == entity_name:
            return list(entity.applicable_derived_metrics or [])
    return []


def _default_schema(specs: Iterable[SemanticSpec]) -> str:
    for s in specs:
        if s.database and s.schema:
            return f"{s.database}.{s.schema}"
    return os.environ.get("RAI_OUTPUT_SCHEMA", "TFO_TEST.ML_TEST")


def ensure_rai_config(config_path: Optional[str] = None) -> str:
    env_key = "RAI_CONFIG_FILE"
    if os.environ.get(env_key):
        return os.environ[env_key]
    if not config_path:
        local_config = os.path.join(_repo_root(), "rai_config", "raiconfig.toml")
        config_path = local_config if os.path.exists(local_config) else os.path.join(_repo_root(), "rai-getting-started", "raiconfig.toml")
    if os.path.exists(config_path):
        pat_env = os.environ.get("RAI_SNOWFLAKE_PAT", "").strip()
        if pat_env:
            secrets_path = "/home/site/secrets/snowflake_pat.txt"
            try:
                os.makedirs(os.path.dirname(secrets_path), exist_ok=True)
                with open(secrets_path, "w", encoding="utf-8") as f:
                    f.write(pat_env)
            except Exception:
                pass
        token_override = os.environ.get("RAI_TOKEN_FILE_PATH", "").strip()
        if not token_override:
            token_override = os.environ.get("RAI_SECRETS_FILE_PATH", "").strip()
        if token_override:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    raw = f.read()
                if "token_file_path" in raw:
                    raw = re.sub(
                        r'(?m)^(\\s*token_file_path\\s*=\\s*)\"[^\"]*\"\\s*$',
                        rf'\\1\"{token_override}\"',
                        raw,
                        count=1,
                    )
                else:
                    raw = raw + f"\n\ntoken_file_path = \"{token_override}\"\n"
                gen_dir = os.path.join(_repo_root(), "rai_config", ".generated")
                os.makedirs(gen_dir, exist_ok=True)
                gen_path = os.path.join(gen_dir, "raiconfig.env.toml")
                with open(gen_path, "w", encoding="utf-8") as f:
                    f.write(raw)
                os.environ[env_key] = gen_path
            except Exception:
                os.environ[env_key] = config_path
        else:
            os.environ[env_key] = config_path
        if not os.environ.get("RAI_PROFILE"):
            try:
                with open(os.environ.get(env_key, config_path), "r", encoding="utf-8") as f:
                    for line in f:
                        m = re.match(r'active_profile\s*=\s*"([^"]+)"', line.strip())
                        if m:
                            os.environ["RAI_PROFILE"] = m.group(1)
                            break
            except Exception:
                pass
    config_value = os.environ.get(env_key, config_path or "")
    try:
        import importlib
        from relationalai.clients import config as rai_config

        rai_config.CONFIG_FILE = config_value
        rai_config.CONFIG_FILE_SET_BY_USER = True
        importlib.reload(rai_config)
    except Exception:
        pass
    return config_value


class KGBuilder:
    def __init__(self, model: qb.Model, schema: str = ""):
        self._model = model
        self._schema = schema
        self._concepts: Dict[str, Dict[str, object]] = {}
        self._relationships: Dict[str, Dict[str, object]] = {}
        self._graphs: Dict[str, object] = {}

    def _attach_table_fields_as_relationships(
        self,
        concept,
        concept_name: str,
        table,
        key_cols: List[object],
    ) -> None:
        # Use relationships (not properties) to avoid implicit uniqueness constraints on CDC tables.
        table._lazy_init()
        key_dict = {sanitize_identifier(k._column_name.lower()): k for k in key_cols}
        if key_cols:
            me = concept.new(**key_dict)
            qb.define(me)
        else:
            me = table._rel._field_refs[0]
            qb.where(table).define(concept(me))
        for field in table._rel._fields[1:]:
            field_name = sanitize_identifier(field.name.lower())
            if field_name in key_dict:
                continue
            rai_type = self._resolve_base_field_type(concept_name, field_name, field.type_str)
            rel = qb.Relationship(
                f"{{{concept}}} has {{{field_name}:{rai_type}}}",
                parent=concept,
                short_name=field_name,
                model=self._model,
            )
            setattr(concept, field_name, rel)
            table_col = getattr(table, field.name)
            qb.define(rel(me, table_col))

        entities = registry_entities()
        entity = entities.get(concept_name)
        if not entity:
            return

        for field in entity.fields:
            if field.derived:
                continue
            field_name = sanitize_identifier(field.name.lower())
            if hasattr(concept, field_name):
                continue
            expr = (field.expr or "").strip()
            if not expr or not re.match(r"^[A-Za-z0-9_]+$", expr):
                continue
            try:
                table_col = getattr(table, expr)
            except Exception:
                continue
            rai_type = self._registry_dtype_to_rai_type(field.dtype or "", field.role or "")
            rel = qb.Relationship(
                f"{{{concept}}} has {{{field_name}:{rai_type}}}",
                parent=concept,
                short_name=field_name,
                model=self._model,
            )
            setattr(concept, field_name, rel)
            qb.define(rel(me, table_col))

    def add_concept(
        self,
        concept_name: str,
        *,
        table_name: str = "",
        desc: str = "",
        schema: Optional[Dict[str, str]] = None,
        keys: Optional[List[str]] = None,
    ):
        if concept_name in self._concepts:
            return self._concepts[concept_name]["concept"]
        concept = self._model.Concept(concept_name)
        if table_name:
            if "." in table_name:
                table = Table(table_name, schema=schema)
            else:
                table = Table(f"{self._schema}.{table_name}", schema=schema)
            key_cols = []
            for key in keys or []:
                if not key:
                    continue
                try:
                    key_cols.append(getattr(table, key))
                except Exception:
                    continue
            use_properties = os.environ.get("RAI_USE_PROPERTY_UNIQUENESS", "").lower() in ("1", "true", "yes")
            if use_properties:
                table.into(concept, keys=key_cols)
            else:
                self._attach_table_fields_as_relationships(concept, concept_name, table, key_cols)
            
            # AFTER attaching base fields, attach derived fields as real Rel relations
            self._attach_derived_fields_to_concept(concept, concept_name, table)
            
        self._concepts[concept_name] = {
            "concept": concept,
            "desc": desc,
        }
        return concept

    def _rai_type_ref(self, rai_type: str):
        type_name = str(rai_type or "").strip()
        ref_map = {
            "Int64": qb.Int64,
            "Int128": qb.Int128,
            "Integer": qb.Integer,
            "Decimal": qb.Decimal,
            "Float": qb.Float,
            "String": qb.String,
            "DateTime": qb.DateTime,
            "Date": qb.Date,
            "Bool": qb.Bool,
        }
        cls = ref_map.get(type_name, qb.String)
        return cls.ref()

    def _lookup_field_dtype_role_any(self, field_name: str) -> tuple[str, str]:
        target = str(field_name or "").lower()
        if not target:
            return "", ""
        entities = registry_entities()
        for entity in entities.values():
            for field in entity.fields:
                name_norm = str(field.name or "").lower()
                if name_norm == target:
                    return field.dtype or "", field.role or ""
                expr_norm = str(field.expr or "").lower()
                if expr_norm and expr_norm == target:
                    return field.dtype or "", field.role or ""
        return "", ""

    def _lookup_field_dtype_role(self, entity_name: str, field_name: str) -> tuple[str, str]:
        if not entity_name or not field_name:
            return "", ""
        entities = registry_entities()
        entity = entities.get(entity_name)
        if not entity:
            return "", ""
        target = str(field_name or "").lower()
        for field in entity.fields:
            name_norm = str(field.name or "").lower()
            if name_norm == target:
                return field.dtype or "", field.role or ""
            expr_norm = str(field.expr or "").lower()
            if expr_norm and expr_norm == target:
                return field.dtype or "", field.role or ""
        return "", ""

    def _normalize_rel_name(self, name: str) -> str:
        return sanitize_identifier(str(name or "").lower())

    def _reasoner_step_concept_name(self, reasoner_id: str, step_id: str) -> str:
        base = f"Reasoner_{reasoner_id}_{step_id}"
        return sanitize_identifier(base)

    def add_reasoner_step_relation(self, reasoner_id: str, step: Dict[str, Any]) -> None:
        if not isinstance(step, dict):
            return
        step_id = str(step.get("id") or "").strip()
        if not step_id:
            return
        query = step.get("query") if isinstance(step.get("query"), dict) else {}
        binds = query.get("bind") or []
        where = query.get("where") or []
        select = query.get("select") or []
        inputs = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}

        concept_name = self._reasoner_step_concept_name(reasoner_id, step_id)
        if concept_name in self._concepts:
            return
        concept = self._model.Concept(concept_name)

        aliases: Dict[str, object] = {}
        where_args: List[object] = []
        alias_to_entity: Dict[str, str] = {}
        for b in binds:
            if not isinstance(b, dict):
                continue
            entity_name = b.get("entity")
            alias = b.get("alias")
            if not entity_name or not alias:
                continue
            ent = self.get_concept(entity_name)
            if ent is None:
                continue
            ref = ent.ref()
            aliases[alias] = ref
            alias_to_entity[alias] = entity_name
            where_args.append(ref)

        var_map: Dict[str, object] = {}
        placeholder_types: Dict[str, str] = {}

        def _record_placeholder_type(var_name: str, alias: str, prop: str) -> None:
            if not var_name or not alias or not prop:
                return
            ent_name = alias_to_entity.get(alias)
            dtype, role = self._lookup_field_dtype_role(ent_name, prop) if ent_name else ("", "")
            rai_type = self._registry_dtype_to_rai_type(dtype, role)
            if rai_type:
                placeholder_types[var_name] = rai_type

        def _var_for(name: str):
            if name in var_map:
                return var_map[name]
            rai_type = placeholder_types.get(name, "")
            if not rai_type:
                if name in ("window_start", "window_end"):
                    rai_type = "DateTime"
                else:
                    dtype, role = self._lookup_field_dtype_role_any(name)
                    rai_type = self._registry_dtype_to_rai_type(dtype, role)
            var_map[name] = self._rai_type_ref(rai_type)
            return var_map[name]

        def _term(t: Dict[str, Any]):
            if "value" in t:
                v = t.get("value")
                if isinstance(v, str) and v.startswith("$"):
                    return _var_for(v[1:])
                return v
            alias = t.get("alias")
            if alias not in aliases:
                raise KeyError(f"Unknown alias {alias}")
            ref = aliases[alias]
            prop = t.get("prop")
            if not prop:
                return ref
            return getattr(ref, self._normalize_rel_name(prop))

        def _capture_placeholder_types(node: Dict[str, Any]) -> None:
            if not isinstance(node, dict):
                return
            if "op" in node:
                left = node.get("left") or {}
                right = node.get("right") or {}
                if isinstance(left, dict) and isinstance(right, dict):
                    lv = left.get("value")
                    rv = right.get("value")
                    if isinstance(lv, str) and lv.startswith("$") and right.get("alias") and right.get("prop"):
                        _record_placeholder_type(lv[1:], right.get("alias"), right.get("prop"))
                    if isinstance(rv, str) and rv.startswith("$") and left.get("alias") and left.get("prop"):
                        _record_placeholder_type(rv[1:], left.get("alias"), left.get("prop"))
            if "between" in node:
                s = node.get("between") or {}
                left = s.get("left") or {}
                low = s.get("low") or {}
                high = s.get("high") or {}
                if isinstance(low, dict) and isinstance(left, dict):
                    lv = low.get("value")
                    if isinstance(lv, str) and lv.startswith("$") and left.get("alias") and left.get("prop"):
                        _record_placeholder_type(lv[1:], left.get("alias"), left.get("prop"))
                if isinstance(high, dict) and isinstance(left, dict):
                    hv = high.get("value")
                    if isinstance(hv, str) and hv.startswith("$") and left.get("alias") and left.get("prop"):
                        _record_placeholder_type(hv[1:], left.get("alias"), left.get("prop"))
            if "in" in node:
                s = node.get("in") or {}
                left = s.get("left") or {}
                right = s.get("right") or {}
                if isinstance(right, dict) and isinstance(left, dict):
                    rv = right.get("value")
                    if isinstance(rv, str) and rv.startswith("$") and left.get("alias") and left.get("prop"):
                        _record_placeholder_type(rv[1:], left.get("alias"), left.get("prop"))
            for key in ("and", "or", "not"):
                if key in node:
                    child = node.get(key)
                    if isinstance(child, list):
                        for item in child:
                            _capture_placeholder_types(item)
                    else:
                        _capture_placeholder_types(child)

        def _predicate(node: Dict[str, Any]):
            if "op" in node:
                op = node["op"]
                if op in ("contains", "ilike", "like"):
                    left = _term(node["left"])
                    right = _term(node["right"])
                    left_s = rstr.string(left)
                    right_s = rstr.string(right)
                    if op == "contains":
                        return rstr.contains(rstr.lower(left_s), rstr.lower(right_s))
                    if op == "ilike":
                        return rstr.like(rstr.lower(left_s), rstr.lower(right_s))
                    return rstr.like(left_s, right_s)
                left = _term(node["left"])
                right = _term(node["right"])
                return {
                    "==": left == right,
                    "!=": left != right,
                    "<": left < right,
                    "<=": left <= right,
                    ">": left > right,
                    ">=": left >= right,
                }[op]
            if "between" in node:
                s = node["between"]
                x = _term(s["left"])
                low = _term(s["low"])
                high = _term(s["high"])
                return (x >= low) & (x <= high)
            if "in" in node:
                s = node["in"]
                x = _term(s["left"])
                vals = _term(s["right"])
                if isinstance(vals, (list, tuple, set)):
                    exprs = [x == v for v in vals]
                    if not exprs:
                        return x.in_(vals)
                    e = exprs[0]
                    for rest in exprs[1:]:
                        e = e | rest
                    return e
                return x.in_(vals)
            if "not" in node:
                return ~_predicate(node["not"])
            if "and" in node:
                exprs = [_predicate(p) for p in node["and"]]
                e = exprs[0]
                for rest in exprs[1:]:
                    e = e & rest
                return e
            if "or" in node:
                exprs = [_predicate(p) for p in node["or"]]
                e = exprs[0]
                for rest in exprs[1:]:
                    e = e | rest
                return e
            raise ValueError(f"Unsupported predicate node: {node}")

        for pred in where:
            if isinstance(pred, dict):
                _capture_placeholder_types(pred)

        for pred in where:
            if isinstance(pred, dict):
                where_args.append(_predicate(pred))

        relationships: Dict[str, object] = {}

        def _add_rel(field_name: str, rai_type: str):
            rel_name = self._normalize_rel_name(field_name)
            if hasattr(concept, rel_name):
                relationships[field_name] = getattr(concept, rel_name)
                return
            rel = qb.Relationship(
                f"{{{concept_name}}} has {{{rel_name}:{rai_type}}}",
                parent=concept,
                short_name=rel_name,
                model=self._model,
            )
            setattr(concept, rel_name, rel)
            relationships[field_name] = rel

        for input_name in inputs.keys():
            dtype, role = self._lookup_field_dtype_role_any(input_name)
            rai_type = self._registry_dtype_to_rai_type(dtype, role)
            _add_rel(input_name, rai_type)
            _var_for(input_name)

        output_terms: List[tuple[str, object]] = []
        for term in select:
            if not isinstance(term, dict):
                continue
            alias = term.get("alias")
            prop = term.get("prop")
            if not alias or not prop:
                continue
            entity_name = None
            for b in binds:
                if isinstance(b, dict) and b.get("alias") == alias:
                    entity_name = b.get("entity")
                    break
            dtype, role = self._lookup_field_dtype_role(entity_name, prop) if entity_name else ("", "")
            rai_type = self._registry_dtype_to_rai_type(dtype, role)
            out_name = term.get("as") or prop
            _add_rel(out_name, rai_type)
            output_terms.append((out_name, _term({"alias": alias, "prop": prop})))

        row = concept.new()
        qb.where(*where_args).define(row)
        for field_name, rel in relationships.items():
            if field_name in var_map:
                qb.where(*where_args).define(rel(row, var_map[field_name]))
        for field_name, expr in output_terms:
            rel = relationships.get(field_name)
            if rel is None:
                continue
            qb.where(*where_args).define(rel(row, expr))

        self._concepts[concept_name] = {
            "concept": concept,
            "desc": f"Reasoner {reasoner_id} step {step_id}",
        }

    def add_reasoner_relations(self, reasoners: Iterable[object]) -> None:
        for reasoner in reasoners or []:
            if not getattr(reasoner, "id", None):
                continue
            plan = getattr(reasoner, "drilldown_plan", None) or {}
            steps = plan.get("steps") if isinstance(plan, dict) else None
            if not isinstance(steps, list):
                continue
            for step in steps:
                if isinstance(step, dict):
                    self.add_reasoner_step_relation(reasoner.id, step)

    def add_graphs(self, graphs: Iterable[object]) -> None:
        debug = os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILDER", "").strip().lower() in ("1", "true", "yes")
        for graph in graphs or []:
            if not isinstance(graph, dict):
                continue
            graph_id = str(graph.get("id") or "").strip()
            if not graph_id:
                continue
            graph_key = sanitize_identifier(graph_id)
            if graph_key in self._graphs:
                continue
            directed = bool(graph.get("directed", True))
            weighted = bool(graph.get("weighted", False))
            try:
                g = Graph(self._model, directed=directed, weighted=weighted)
            except Exception as exc:
                if debug:
                    try:
                        import sys as _sys
                        _sys.stderr.write(f"[GRAPH_BUILDER] Graph() failed id={graph_id}: {exc}\n")
                    except Exception:
                        pass
                continue
            self._graphs[graph_key] = g

            nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
            edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []

            node_cfg: Dict[str, Dict[str, object]] = {}
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                entity_name = str(node.get("entity") or "").strip()
                if not entity_name:
                    continue
                node_cfg[entity_name] = {}

            if debug:
                node_total = len(nodes)
                edge_total = len(edges)
                node_attached = 0
                edge_attached = 0
                node_missing_concept = 0
                edge_missing_relation = 0

            # Ensure nodes exist even when the graph has no edges.
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                entity_name = str(node.get("entity") or "").strip()
                if not entity_name:
                    continue
                concept = self.get_concept(entity_name)
                if concept is None:
                    if debug:
                        node_missing_concept += 1
                    continue
                try:
                    a = concept.ref()
                    qb.where(a).define(g.Node.new(id=a))
                    if debug:
                        node_attached += 1
                except Exception:
                    pass

            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                relation_ref = str(edge.get("relation") or "").strip()
                if not relation_ref:
                    from_entity = str(edge.get("from_entity") or "").strip()
                    rel_name = str(edge.get("name") or "").strip()
                    if from_entity and rel_name:
                        relation_ref = f"{from_entity}.{rel_name}"
                if not relation_ref:
                    continue
                rel_meta = self._relationships.get(relation_ref)
                if not rel_meta:
                    if debug:
                        edge_missing_relation += 1
                    continue
                from_entity = rel_meta.get("from_concept")
                to_entity = rel_meta.get("to_concept")
                rel_name = rel_meta.get("relation")
                if not from_entity or not rel_name:
                    continue
                from_concept = self.get_concept(from_entity)
                if from_concept is None:
                    if debug:
                        node_missing_concept += 1
                    continue
                to_concept = self.get_concept(to_entity) if to_entity else None
                rel = getattr(from_concept, rel_name, None)
                if rel is None:
                    if debug:
                        edge_missing_relation += 1
                    continue
                try:
                    a = from_concept.ref()
                    b = to_concept.ref() if to_concept is not None else None
                    if b is None:
                        continue

                    edge_label = str(edge.get("label") or edge.get("name") or relation_ref)

                    qb.where(rel(a, b)).define(
                        na := g.Node.new(id=a),
                        nb := g.Node.new(id=b),
                        g.Edge.new(src=na, dst=nb, label=edge_label),
                    )
                    if debug:
                        edge_attached += 1
                except Exception:
                    pass
            if debug:
                try:
                    import sys as _sys
                    _sys.stderr.write(
                        f"[GRAPH_BUILDER] graph={graph_id} nodes={node_attached}/{node_total} "
                        f"edges={edge_attached}/{edge_total} "
                        f"missing_concepts={node_missing_concept} missing_relations={edge_missing_relation}\n"
                    )
                except Exception:
                    pass

    def _attach_derived_fields_to_concept(self, concept, concept_name: str, table):
        """
        Attach derived fields to a concept as real Rel predicates.
        
        This creates Rel relations/predicates that express the semantic meaning
        of derived fields. These become true KG relations that can be filtered, sorted, 
        and aggregated server-side by the query executor.
        
        ARCHITECTURE NOTE:
        
        Derived fields are defined as explicit Rel rules (see DERIVED_FIELDS_REL_RULES.md),
        so they are available directly in the KG for query-time filtering and aggregation.
        SQL conversion fallback is disabled to keep the registry as the single source of truth.
        """
        try:
            entities = registry_entities()
        except Exception:
            return
        
        entity = entities.get(concept_name)
        if not entity:
            return

        derived_fields = [f for f in entity.fields if f.derived]
        if not derived_fields:
            return

        derived_rels: Dict[str, object] = {}
        for field in derived_fields:
            field_name = sanitize_identifier(field.name.lower())
            field_type_str = field.dtype or "String"

            # Map dtype to RAI type (avoid generic Number, which is not a valid RAI type)
            rai_type = "String"
            field_type_norm = field_type_str.lower()
            num_match = re.match(
                r"^(number|numeric|decimal)\s*\((\d+)(?:\s*,\s*(\d+))?\)\s*$",
                field_type_norm,
            )
            if num_match:
                precision = int(num_match.group(2))
                scale = int(num_match.group(3) or 0)
                if scale == 0:
                    rai_type = "Int64" if precision <= 18 else "Int128"
                else:
                    rai_type = f"Decimal({precision},{scale})"
            elif field_type_norm in ("number", "numeric", "float", "double") or field_type_norm.startswith("float"):
                rai_type = "Float"
            elif field_type_norm in ("int", "integer", "int64", "int128"):
                rai_type = "Int128"
            elif field_type_norm.startswith("timestamp") or field_type_norm.startswith("datetime"):
                rai_type = "DateTime"
            elif field_type_norm.startswith("date"):
                rai_type = "Date"
            elif field_type_norm in ("bool", "boolean"):
                rai_type = "Bool"

            try:
                rel = qb.Relationship(
                    f"{{{concept_name}}} has {{{field_name}:{rai_type}}}",
                    parent=concept,
                    short_name=field_name,
                    model=self._model,
                )
                setattr(concept, field_name, rel)
                derived_rels[field.name] = rel
            except Exception:
                continue

        handled = self._apply_registry_derived_rules(concept, concept_name, derived_rels)

        # Remaining derived fields must be defined in registry rules.
        # SQL-derived fallbacks are intentionally disabled to keep logic source-of-truth in the registry.

    def _apply_registry_derived_rules(
        self,
        concept,
        concept_name: str,
        derived_rels: Dict[str, object],
    ) -> set[str]:
        handled: set[str] = set()
        rules = [r for r in load_derived_rel_rules() if r.entity == concept_name]
        if not rules:
            return handled

        m = concept.ref()
        base_ns = self._build_rel_namespace(concept_name, concept, m)

        for rule_spec in rules:
            rel = derived_rels.get(rule_spec.field) or derived_rels.get(rule_spec.field.lower())
            if rel is None:
                continue

            ns = dict(base_ns)
            # Helpers for null/missing checks in registry rules.
            # Use Missing sentinel rather than None literals to avoid IR typing errors.
            def _has_value(x):
                try:
                    from relationalai import dsl as _dsl
                    missing = getattr(getattr(_dsl, "rel", None), "Missing", None)
                    if missing is None:
                        return None
                    return x != missing
                except Exception:
                    return None

            def _is_missing(x):
                try:
                    from relationalai import dsl as _dsl
                    missing = getattr(getattr(_dsl, "rel", None), "Missing", None)
                    if missing is None:
                        return None
                    return x == missing
                except Exception:
                    return None

            ns.setdefault("has_value", _has_value)
            ns.setdefault("is_missing", _is_missing)
            # Allow rule-scoped variables (including extra refs).
            for var_name, var_expr in (rule_spec.vars or {}).items():
                if not var_name:
                    continue
                if str(var_expr).strip().lower() == "ref":
                    ns[var_name] = concept.ref()
                    continue
                val = self._safe_eval_rel_expr(str(var_expr), ns)
                if val is not None:
                    ns[var_name] = val

            for clause in rule_spec.rules:
                expr = self._safe_eval_rel_expr(clause.expr, ns)
                if expr is None:
                    continue
                cond_text = (clause.when or "").strip()
                cond_lower = cond_text.lower()
                if cond_text:
                    cond = self._safe_eval_rel_expr(cond_text, ns)
                    if cond is None:
                        continue
                    qb.where(cond).define(rel(m, expr))
                else:
                    qb.define(rel(m, expr))
            handled.add(rule_spec.field)

        return handled

    def _build_rel_namespace(self, concept_name: str, concept, ref) -> Dict[str, object]:
        ns: Dict[str, object] = {
            "m": ref,
            "std": std,
            "qb": qb,
        }
        try:
            def ilike(string, pattern):
                return rstr.like(rstr.lower(string), rstr.lower(pattern))

            ns.setdefault("ilike", ilike)
            ns.setdefault("like", rstr.like)
            ns.setdefault("lowercase", rstr.lower)
            ns.setdefault("uppercase", rstr.upper)
            ns.setdefault("contains", rstr.contains)
        except Exception:
            pass
        entities = registry_entities()
        entity = entities.get(concept_name)
        if not entity:
            return ns
        for field in entity.fields:
            field_name = sanitize_identifier(field.name.lower())
            if hasattr(ref, field_name):
                ns[field_name] = getattr(ref, field_name)
                if field.name != field_name:
                    ns[field.name] = getattr(ref, field_name)
        return ns

    def _safe_eval_rel_expr(self, expr: str, namespace: Dict[str, object]) -> Optional[object]:
        debug = os.environ.get("RAI_DERIVED_RULE_DEBUG", "").strip() not in ("", "0", "false", "False")
        if not expr:
            return None
        text = expr.strip()
        if not text:
            return None
        if "__" in text or "import" in text or "lambda" in text:
            return None
        if not re.match(r'^[A-Za-z0-9_().,<>!=+\-*/%&|\s\'"]+$', text):
            return None
        try:
            # Allow common literals used in registry rules.
            eval_ns = dict(namespace or {})
            eval_ns.setdefault("None", None)
            eval_ns.setdefault("True", True)
            eval_ns.setdefault("False", False)
            # Accept JSON-style lowercase literals in registry rules.
            eval_ns.setdefault("true", True)
            eval_ns.setdefault("false", False)
            eval_ns.setdefault("null", None)
            return eval(text, {"__builtins__": {}}, eval_ns)
        except Exception as e:
            if debug:
                print(f"[DERIVED_RULE_DEBUG] eval error: {text} -> {e}")
            return None
        
    def _resolve_base_field_type(self, concept_name: str, field_name: str, fallback: str) -> str:
        if fallback and fallback not in ("Any", "Number"):
            return fallback
        dtype, role = self._registry_field_dtype(concept_name, field_name)
        if dtype:
            return self._registry_dtype_to_rai_type(dtype, role)
        if fallback == "Number":
            return "Float"
        return "String"

    def _registry_field_dtype(self, concept_name: str, field_name: str) -> tuple[str, str]:
        entities = registry_entities()
        entity = entities.get(concept_name)
        if not entity:
            return "", ""
        for field in entity.fields:
            if field.name.lower() == field_name:
                return field.dtype or "", field.role or ""
            expr = (field.expr or "").strip()
            if expr and re.match(r"^[A-Za-z0-9_]+$", expr):
                expr_norm = sanitize_identifier(expr.lower())
                if expr_norm == field_name:
                    return field.dtype or "", field.role or ""
        return "", ""

    def _registry_dtype_to_rai_type(self, dtype: str, role: str) -> str:
        dt = str(dtype or "").strip().lower()
        num_match = re.match(r"^(number|numeric|decimal)\s*\((\d+)(?:\s*,\s*(\d+))?\)\s*$", dt)
        if num_match:
            precision = int(num_match.group(2))
            scale = int(num_match.group(3) or 0)
            if scale == 0:
                return "Int64" if precision <= 18 else "Int128"
            return f"Decimal({precision},{scale})"
        if dt in ("text", "string", "str") or dt.startswith("varchar"):
            return "String"
        if dt in ("float", "double") or dt.startswith("float"):
            return "Float"
        if dt in ("number", "numeric", "int", "int64", "int128", "decimal"):
            if str(role or "").lower() == "dimension":
                return "Int128"
            return "Decimal(38,14)"
        if dt.startswith("timestamp") or dt.startswith("datetime"):
            return "DateTime"
        if dt.startswith("date"):
            return "Date"
        if dt in ("bool", "boolean"):
            return "Bool"
        return "String"

    def get_concept(self, concept_name: str):
        entry = self._concepts.get(concept_name)
        return entry.get("concept") if entry else None

    def get_graph(self, graph_id: str):
        if not graph_id:
            return None
        graph_key = sanitize_identifier(str(graph_id))
        return self._graphs.get(graph_key)

    def add_relation(
        self,
        relation_ref: str,
        relation_def: str,
        desc: str = "",
        join_on: Optional[List[tuple[str, str]]] = None,
    ):
        if relation_ref in self._relationships:
            return
        if "." not in relation_ref:
            return
        ref_parts = relation_ref.strip().split(".")
        if len(ref_parts) != 2:
            return
        ref_from = ref_parts[0]
        ref_property = ref_parts[1]
        def_parts = re.findall(r"\{([^{}]+)\}", relation_def.strip())
        if len(def_parts) < 2:
            return
        def_from = def_parts[0].split(":")[1] if (":" in def_parts[0]) else def_parts[0]
        def_to = def_parts[1].split(":")[1] if (":" in def_parts[1]) else def_parts[1]
        if ref_from != def_from:
            return
        from_concept = self.get_concept(def_from)
        if from_concept is None:
            return
        rel = self._model.Relationship(relation_def)
        setattr(from_concept, ref_property, rel)
        self._relationships[relation_ref] = {
            "from_concept": def_from,
            "to_concept": def_to,
            "relation": ref_property,
            "desc": desc,
            "join_on": list(join_on or []),
        }
        self._define_join_relationship(def_from, def_to, rel, join_on or [])

    def _resolve_join_field_ref(self, entity_name: str, ref, field_name: str):
        if not field_name:
            return None
        field_norm = sanitize_identifier(str(field_name).lower())
        if hasattr(ref, field_norm):
            return getattr(ref, field_norm)
        entities = registry_entities()
        entity = entities.get(entity_name)
        if not entity:
            return None
        target = str(field_name).lower()
        for field in entity.fields:
            name_norm = str(field.name or "").lower()
            if name_norm and name_norm == target:
                candidate = sanitize_identifier(name_norm)
                if hasattr(ref, candidate):
                    return getattr(ref, candidate)
            expr = str(field.expr or "").lower()
            if expr and expr == target:
                candidate = sanitize_identifier(name_norm)
                if hasattr(ref, candidate):
                    return getattr(ref, candidate)
        if hasattr(ref, field_name):
            return getattr(ref, field_name)
        return None

    def _define_join_relationship(
        self,
        from_entity: str,
        to_entity: str,
        rel,
        join_on: List[tuple[str, str]],
    ) -> None:
        if not join_on:
            return
        from_concept = self.get_concept(from_entity)
        to_concept = self.get_concept(to_entity)
        if from_concept is None or to_concept is None:
            return
        a = from_concept.ref()
        b = to_concept.ref()
        join_exprs = []
        for left_key, right_key in join_on:
            left_ref = self._resolve_join_field_ref(from_entity, a, left_key)
            right_ref = self._resolve_join_field_ref(to_entity, b, right_key)
            if left_ref is None or right_ref is None:
                return
            join_exprs.append(left_ref == right_ref)
        qb.where(*join_exprs).define(rel(a, b))


def _resolve_use_lqp(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    env_flag = os.environ.get("RAI_USE_LQP", "").strip().lower()
    if env_flag in ("0", "false", "no"):
        return False
    if env_flag in ("1", "true", "yes"):
        return True
    return True


def _resolve_model_name(model_name: str, model_suffix: Optional[str] = None) -> str:
    env_model = os.environ.get("AI_INSIGHTS_MODEL_NAME", "").strip()
    if env_model:
        model_name = env_model
    suffix = (model_suffix or os.environ.get("AI_INSIGHTS_MODEL_NAME_SUFFIX", "")).strip()
    if suffix:
        model_name = f"{model_name}_{suffix}"
    return model_name


@lru_cache(maxsize=8)
def build_ai_insights_builder(
    model_name: str = "AIInsights",
    use_lqp: Optional[bool] = None,
    model_suffix: Optional[str] = None,
) -> KGBuilder:
    debug_init = os.environ.get("AI_INSIGHTS_DEBUG_INIT", "").strip().lower() in ("1", "true", "yes")
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] build_ai_insights_builder: start\n")
        except Exception:
            debug_init = False
    t0 = time.perf_counter() if debug_init else 0.0
    model_name = _resolve_model_name(model_name, model_suffix=model_suffix)
    specs = load_ai_insights_specs()
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write(
                f"[INIT] build_ai_insights_builder: specs={len(specs)} model_name={model_name}\n"
            )
        except Exception:
            pass
    schema = _default_schema(specs)
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] build_ai_insights_builder: schema_ready\n")
        except Exception:
            pass
    model = qb.Model(model_name, strict=False, use_lqp=_resolve_use_lqp(use_lqp))
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] build_ai_insights_builder: model_created\n")
        except Exception:
            pass
    builder = KGBuilder(model, schema=schema)
    use_join_keys_as_id = os.environ.get("RAI_USE_JOIN_KEYS_AS_ID", "").lower() in ("1", "true", "yes")
    kg = load_kg_spec() or {}
    kg_nodes = kg.get("nodes") if isinstance(kg, dict) else None
    kg_edges = kg.get("edges") if isinstance(kg, dict) else None
    kg_graphs = kg.get("graphs") if isinstance(kg, dict) else None
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write(
                f"[INIT] build_ai_insights_builder: kg_nodes={len(kg_nodes) if isinstance(kg_nodes, list) else 0} "
                f"kg_edges={len(kg_edges) if isinstance(kg_edges, list) else 0} "
                f"kg_graphs={len(kg_graphs) if isinstance(kg_graphs, list) else 0}\n"
            )
        except Exception:
            pass
    if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILDER", "").strip().lower() in ("1", "true", "yes"):
        try:
            import sys as _sys
            _sys.stderr.write(
                f"[GRAPH_BUILDER] kg_graphs={len(kg_graphs) if isinstance(kg_graphs, list) else 0}\n"
            )
        except Exception:
            pass
    kg_node_by_entity: Dict[str, Dict[str, object]] = {}
    if isinstance(kg_nodes, list):
        for node in kg_nodes:
            if not isinstance(node, dict):
                continue
            entity_name = str(node.get("entity") or "").strip()
            if not entity_name:
                continue
            kg_node_by_entity[entity_name] = node
    for spec in specs:
        # Reasoner step concepts are built via add_reasoner_relations to attach typed relations.
        if getattr(spec, "entity_type", "") == "reasoner_step":
            continue
        key_exprs: List[str] = []
        kg_node = kg_node_by_entity.get(spec.name)
        if kg_node:
            key_field = kg_node.get("key_field")
            if isinstance(key_field, list):
                key_exprs = [str(k) for k in key_field if k]
            elif key_field:
                key_exprs = [str(key_field)]
        if not key_exprs:
            field_exprs = spec.field_exprs or {}
            for key in spec.join_keys or []:
                expr = field_exprs.get(key) or key
                if expr and re.match(r"^[A-Za-z0-9_]+$", str(expr)):
                    key_exprs.append(str(expr))
        builder.add_concept(
            spec.name,
            table_name=spec.table,
            desc=spec.description,
            schema=_schema_for_spec(spec),
            keys=key_exprs if (use_join_keys_as_id or (kg_node and key_exprs)) else None,
        )
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] build_ai_insights_builder: concepts_added\n")
        except Exception:
            pass
    seen = set()
    for rel in load_ai_insights_relationships():
        relation_ref = f"{rel.from_entity}.{rel.name}"
        if relation_ref in seen:
            continue
        seen.add(relation_ref)
        rel_name = rel.name.replace("_", " ")
        relation_def = f"{{{rel.from_entity}}} {rel_name} {{{rel.to_entity}}}"
        builder.add_relation(
            relation_ref,
            relation_def,
            desc=rel.description,
            join_on=list(rel.join_on or []),
        )
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] build_ai_insights_builder: relationships_added\n")
        except Exception:
            pass
    if isinstance(kg_edges, list):
        for edge in kg_edges:
            if not isinstance(edge, dict):
                continue
            from_entity = str(edge.get("from_entity") or "").strip()
            to_entity = str(edge.get("to_entity") or "").strip()
            if not from_entity or not to_entity:
                continue
            edge_name = str(edge.get("name") or "").strip()
            if not edge_name:
                edge_name = f"{from_entity}_to_{to_entity}"
            relation_ref = f"{from_entity}.{edge_name}"
            if relation_ref in seen:
                continue
            seen.add(relation_ref)
            edge_label = edge_name.replace("_", " ")
            relation_def = f"{{{from_entity}}} {edge_label} {{{to_entity}}}"
            builder.add_relation(
                relation_ref,
                relation_def,
                desc=str(edge.get("description") or ""),
                join_on=list(edge.get("join_on") or []),
            )
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] build_ai_insights_builder: kg_edges_added\n")
        except Exception:
            pass
    try:
        builder.add_reasoner_relations(load_reasoners())
    except Exception:
        pass
    if debug_init:
        try:
            import sys as _sys
            _sys.stderr.write("[INIT] build_ai_insights_builder: reasoner_relations_added\n")
        except Exception:
            pass
    if isinstance(kg_graphs, list):
        try:
            builder.add_graphs(kg_graphs)
        except Exception as exc:
            if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILDER", "").strip().lower() in ("1", "true", "yes"):
                try:
                    import sys as _sys
                    _sys.stderr.write(f"[GRAPH_BUILDER] add_graphs failed: {exc}\n")
                except Exception:
                    pass
        if os.environ.get("AI_INSIGHTS_DEBUG_GRAPH_BUILDER", "").strip().lower() in ("1", "true", "yes"):
            try:
                import sys as _sys
                graph_keys = list(getattr(builder, "_graphs", {}).keys())
                _sys.stderr.write(f"[GRAPH_BUILDER] graphs_added={len(graph_keys)} keys={graph_keys[:6]}\n")
            except Exception:
                pass
    if debug_init:
        try:
            import sys as _sys
            elapsed = time.perf_counter() - t0
            _sys.stderr.write(f"[INIT] build_ai_insights_builder: done in {elapsed:.2f}s\n")
        except Exception:
            pass
    return builder


def _schema_for_spec(spec: SemanticSpec) -> Optional[Dict[str, str]]:
    use_registry_types = os.environ.get("RAI_USE_REGISTRY_SCHEMA_TYPES", "true").lower() in ("1", "true", "yes")
    if not use_registry_types:
        return None

    def _to_type(dtype: str, role: str) -> str:
        dt = str(dtype or "").strip().lower()
        num_match = re.match(r"^(number|numeric|decimal)\s*\((\d+)(?:\s*,\s*(\d+))?\)\s*$", dt)
        if num_match:
            precision = int(num_match.group(2))
            scale = int(num_match.group(3) or 0)
            if scale == 0:
                return "Int64" if precision <= 18 else "Int128"
            return f"Decimal({precision},{scale})"
        if dt in ("text", "string", "str") or dt.startswith("varchar"):
            return "String"
        if dt in ("float", "double") or dt.startswith("float"):
            return "Float"
        if dt in ("number", "numeric", "int", "int64", "int128", "decimal"):
            if str(role or "").lower() == "dimension":
                return "Int128"
            # Use Decimal for metrics to avoid unresolved overloads on generic Number.
            return "Decimal(38,14)"
        if dt.startswith("timestamp") or dt.startswith("datetime"):
            return "DateTime"
        if dt.startswith("date"):
            return "Date"
        if dt in ("bool", "boolean"):
            return "Bool"
        return "String"

    schema: Dict[str, str] = {}
    for field in spec.fields:
        expr = (spec.field_exprs or {}).get(field)
        if not expr:
            # For synthetic/derived concepts (no physical table), map field -> type directly
            # to avoid unresolved types during reasoning.
            if not spec.table:
                expr = field
            else:
                continue
        expr = str(expr).strip()
        if not re.match(r"^[A-Za-z0-9_]+$", expr):
            continue
        schema[expr] = _to_type((spec.field_types or {}).get(field, ""), (spec.field_roles or {}).get(field, ""))
    return schema


def render_ai_insights_ontology_text(specs: Iterable[SemanticSpec]) -> str:
    lines = [
        "Entities and base fields (use these names exactly):",
    ]
    for spec in specs:
        desc = f" - {spec.description}" if spec.description else ""
        fields = ", ".join(
            f"{f}:{spec.field_types.get(f, 'unknown')}:{spec.field_roles.get(f, 'unknown')}"
            + (f":{spec.field_aggs.get(f)}" if spec.field_aggs.get(f) else "")
            + (f" | {spec.field_descs.get(f)}" if spec.field_descs.get(f) else "")
            for f in spec.fields
        )
        lines.append(f"- {spec.name}{desc}")
        lines.append(f"  fields: {fields}")
        if spec.join_keys:
            lines.append(f"  join_keys: {', '.join(spec.join_keys)}")
        if spec.default_metric:
            lines.append(f"  default_metric: {spec.default_metric}")
        if spec.derived_fields:
            derived = ", ".join(spec.derived_fields)
            lines.append(f"  derived_fields (computed in KG; queryable): {derived}")

    derived_rules = load_derived_rel_rules()
    if derived_rules:
        lines.append("Derived rules (expressions for derived_fields; use these names exactly):")
        for rule in derived_rules:
            if not rule.rules:
                continue
            lines.append(f"- {rule.entity}.{rule.field}")
            for clause in rule.rules:
                when = (clause.when or "").strip()
                expr = (clause.expr or "").strip()
                if when:
                    lines.append(f"  when {when} -> {expr}")
                else:
                    lines.append(f"  {expr}")

    reasoners = load_reasoners()
    if reasoners:
        lines.append("Reasoners (optional outputs you may include if asked to explain/diagnose):")
        for r in reasoners:
            if not r.id:
                continue
            desc = f" - {r.description}" if r.description else ""
            lines.append(f"- {r.id}{desc}")
            if r.entity_type:
                lines.append(f"  entity_type: {r.entity_type}")
            if r.outputs:
                lines.append(f"  outputs: {', '.join(r.outputs)}")
            if r.signals:
                sigs = ", ".join(
                    [
                        f"{s.name}({s.metric_field} {s.direction} {s.threshold})"
                        for s in r.signals
                        if s.name and s.metric_field
                    ]
                )
                if sigs:
                    lines.append(f"  signals: {sigs}")
            plan = getattr(r, "drilldown_plan", None) or {}
            steps = plan.get("steps") if isinstance(plan, dict) else None
            if isinstance(steps, list) and steps:
                summary = []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    step_id = step.get("id") or ""
                    from_ref = step.get("from") or "base"
                    query = step.get("query") or {}
                    entity = ""
                    binds = query.get("bind") if isinstance(query, dict) else None
                    if isinstance(binds, list) and binds and isinstance(binds[0], dict):
                        entity = binds[0].get("entity") or ""
                    limit = step.get("limit")
                    label = f"{step_id}:{from_ref}"
                    if entity:
                        label += f"->{entity}"
                    if limit:
                        label += f"(limit {limit})"
                    summary.append(label)
                if summary:
                    lines.append(f"  drilldown_steps: {', '.join(summary)}")

    kg = load_kg_spec()
    if isinstance(kg, dict) and (kg.get("nodes") or kg.get("edges") or kg.get("graphs")):
        lines.append("KG configuration (optional; used to materialize KG in RAI):")
        nodes = kg.get("nodes")
        if isinstance(nodes, list) and nodes:
            lines.append("  nodes:")
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                entity = node.get("entity") or ""
                node_type = node.get("node_type") or ""
                key_field = node.get("key_field") or ""
                label_field = node.get("label_field") or ""
                props = node.get("properties") or []
                prop_list = ", ".join([str(p) for p in props]) if isinstance(props, list) else ""
                key_text = ""
                if isinstance(key_field, list):
                    key_text = "[" + ", ".join([str(k) for k in key_field]) + "]"
                else:
                    key_text = str(key_field)
                lines.append(
                    f"    - entity={entity} node_type={node_type} key_field={key_text} label_field={label_field} properties={prop_list}"
                )
        edges = kg.get("edges")
        if isinstance(edges, list) and edges:
            lines.append("  edges:")
            for edge in edges:
                if not isinstance(edge, dict):
                    continue
                name = edge.get("name") or ""
                from_entity = edge.get("from_entity") or ""
                to_entity = edge.get("to_entity") or ""
                join_on = edge.get("join_on") or []
                join_pairs = ", ".join([f"{l}={r}" for l, r in join_on]) if isinstance(join_on, list) else ""
                edge_type = edge.get("edge_type") or ""
                lines.append(
                    f"    - {name}: {from_entity} -> {to_entity} join_on={join_pairs} edge_type={edge_type}"
                )
        graphs = kg.get("graphs")
        if isinstance(graphs, list) and graphs:
            lines.append("  graphs:")
            for graph in graphs:
                if not isinstance(graph, dict):
                    continue
                gid = graph.get("id") or ""
                directed = "directed" if graph.get("directed", True) else "undirected"
                nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
                edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
                lines.append(f"    - {gid} ({directed}) nodes={len(nodes)} edges={len(edges)}")
    relationships = load_ai_insights_relationships()
    if relationships:
        lines.append("Relationships (use join_on keys in where predicates):")
        for rel in relationships:
            join_pairs = ", ".join([f"{l}={r}" for l, r in rel.join_on])
            desc = f" - {rel.description}" if rel.description else ""
            lines.append(f"- {rel.from_entity}.{rel.name} -> {rel.to_entity}{desc}")
            lines.append(f"  join_on: {join_pairs}")
    return "\n".join(lines)
