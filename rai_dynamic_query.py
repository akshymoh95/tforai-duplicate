import os
import sys
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import pandas as pd
import relationalai.semantics as qb
from relationalai.semantics.std import strings as rstr
from rai_ai_insights_ontology import ensure_rai_config, load_ai_insights_specs
from rai_semantic_registry import load_registry_config

_DEBUG_AGG = os.environ.get("AI_INSIGHTS_DEBUG_AGG", "").strip().lower() in ("1", "true", "yes")

def _field_meta_for(alias_to_entity: dict, alias: str, prop: str) -> tuple[str, str, str]:
    """Return (dtype, role, default_agg) from the ontology spec for alias.prop, best-effort."""
    if not alias or not prop:
        return ("", "", "")
    ent = alias_to_entity.get(alias)
    if not ent:
        return ("", "", "")
    try:
        specs = load_ai_insights_specs()
    except Exception:
        specs = []
    spec = next((s for s in (specs or []) if getattr(s, "name", None) == ent), None)
    if not spec:
        return ("", "", "")
    # Canonicalize prop via expr mapping if needed
    prop_norm = str(prop).lower()
    try:
        # If prop is actually an expr alias, map back
        expr_to_field = {str(expr).lower(): field for field, expr in (getattr(spec, "field_exprs", {}) or {}).items() if expr}
        canonical = expr_to_field.get(prop_norm) or prop_norm
    except Exception:
        canonical = prop_norm
    dtype = str((getattr(spec, "field_types", {}) or {}).get(canonical, "") or "").lower()
    role = str((getattr(spec, "field_roles", {}) or {}).get(canonical, "") or "").lower()
    default_agg = str((getattr(spec, "field_aggs", {}) or {}).get(canonical, "") or "").lower()
    return (dtype, role, default_agg)

def _is_bool(dtype: str, role: str) -> bool:
    d = (dtype or "").lower()
    r = (role or "").lower()
    return ("bool" in d) or ("boolean" in d) or (r in ("flag", "signal", "indicator"))

def _is_dimension_like(role: str, dtype: str) -> bool:
    r = (role or "").lower()
    d = (dtype or "").lower()
    # Treat identifiers/time/category as dimensions.
    if r in ("dimension", "id", "key", "timestamp", "date", "category"):
        return True
    if "date" in d or "timestamp" in d or "time" in d:
        return True
    return False

def _normalize_grouped_spec(spec: dict, alias_to_entity: dict) -> dict:
    """When aggregations exist, ensure select contains only group_by dims; promote other selects into aggregations."""
    if not isinstance(spec, dict):
        return spec
    aggregations = spec.get("aggregations") or []
    if not aggregations:
        return spec
    group_by = list(spec.get("group_by") or [])
    select = list(spec.get("select") or [])
    # Keys
    group_keys = {(g.get("alias"), g.get("prop")) for g in group_by if isinstance(g, dict)}
    agg_term_keys = set()
    agg_as = set()
    for a in aggregations:
        if not isinstance(a, dict):
            continue
        agg_as.add(a.get("as"))
        term = a.get("term")
        if isinstance(term, dict):
            agg_term_keys.add((term.get("alias"), term.get("prop")))

    new_select = []
    new_aggs = list(aggregations)

    def _agg_op_for(alias: str, prop: str) -> str:
        dtype, role, default_agg = _field_meta_for(alias_to_entity, alias, prop)
        if _is_bool(dtype, role):
            # Avoid qb.max on Bool (can fail overload resolution); use count as a boolean-safe rollup.
            return "count"
        if default_agg in ("sum", "avg", "min", "max", "count"):
            return default_agg
        # numeric-ish defaults
        if any(tok in (dtype or "") for tok in ("number", "numeric", "decimal", "int", "float", "double")):
            return "sum"
        return "max"

    # Promote non-grouped selects into aggregations, keep only group_by dims in select.
    for s in select:
        if not isinstance(s, dict):
            continue
        alias = s.get("alias")
        prop = s.get("prop")
        as_name = s.get("as") or prop
        if not alias or not prop:
            continue
        key = (alias, prop)
        if key in group_keys:
            new_select.append(s)
            continue
        if key in agg_term_keys:
            # Don't keep raw select if same field is already aggregated.
            continue
        # If it's dimension-like, push into group_by instead of aggregation.
        dtype, role, _ = _field_meta_for(alias_to_entity, alias, prop)
        if _is_dimension_like(role, dtype):
            if key not in group_keys:
                group_by.append({"alias": alias, "prop": prop, "as": as_name})
                group_keys.add(key)
            new_select.append({"alias": alias, "prop": prop, "as": as_name})
            continue
        # Otherwise, make it an aggregation.
        if as_name in agg_as:
            continue
        op = _agg_op_for(alias, prop)
        new_aggs.append({"op": op, "term": {"alias": alias, "prop": prop}, "as": as_name})
        agg_as.add(as_name)

    # Ensure select has at least one item (schema requires select)
    if not new_select and group_by:
        for g in group_by:
            if isinstance(g, dict) and g.get("alias") and g.get("prop"):
                new_select.append({"alias": g.get("alias"), "prop": g.get("prop"), "as": g.get("as") or g.get("prop")})
        # still could be empty if group_by invalid

    spec = dict(spec)
    spec["group_by"] = group_by
    spec["select"] = new_select if new_select else select
    spec["aggregations"] = new_aggs
    return spec


from rai_derived_metrics import enrich_dataframe_with_derived_metrics


DYNAMIC_QUERY_SCHEMA = r"""{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/schemas/qb-query-spec.json",
  "title": "RelationalAI QB Dynamic Query Spec",
  "type": "object",
  "additionalProperties": false,
  "required": ["bind", "select"],
  "properties": {
    "bind": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/Bind" }
    },
    "where": {
      "type": "array",
      "items": { "$ref": "#/$defs/Predicate" },
      "default": []
    },
    "select": {
      "type": "array",
      "minItems": 1,
      "items": { "$ref": "#/$defs/SelectItem" }
    },
    "group_by": {
      "type": "array",
      "items": { "$ref": "#/$defs/GroupByItem" },
      "default": []
    },
    "aggregations": {
      "type": "array",
      "items": { "$ref": "#/$defs/Aggregation" },
      "default": []
    },
    "order_by": {
      "type": "array",
      "items": { "$ref": "#/$defs/OrderByItem" },
      "default": []
    },
    "limit": { "type": "integer", "minimum": 0 },
    "offset": { "type": "integer", "minimum": 0, "default": 0 },
    "distinct": { "type": "boolean", "default": false },

    "meta": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "intent": { "type": "string" },
        "time_window": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "start": { "type": "string" },
            "end": { "type": "string" },
            "mode": { "type": "string", "enum": ["point", "overlap"], "default": "point" }
          },
          "required": ["start", "end"]
        }
      },
      "default": {}
    }
  },

  "$defs": {
    "Alias": {
      "type": "string",
      "minLength": 1,
      "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"
    },

    "Term": {
      "oneOf": [
        { "$ref": "#/$defs/AliasTerm" },
        { "$ref": "#/$defs/ValueTerm" }
      ]
    },

    "AliasTerm": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "alias": { "$ref": "#/$defs/Alias" },
        "prop": { "type": "string", "minLength": 1 }
      },
      "required": ["alias"]
    },

    "ValueTerm": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "value": {}
      },
      "required": ["value"]
    },

    "Predicate": {
      "oneOf": [
        { "$ref": "#/$defs/BinaryOp" },
        { "$ref": "#/$defs/Between" },
        { "$ref": "#/$defs/InSet" },
        { "$ref": "#/$defs/Not" },
        { "$ref": "#/$defs/And" },
        { "$ref": "#/$defs/Or" }
      ]
    },

    "BinaryOp": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "op": {
          "type": "string",
          "enum": ["==", "!=", "<", "<=", ">", ">=", "contains", "ilike", "like"]
        },
        "left": { "$ref": "#/$defs/Term" },
        "right": { "$ref": "#/$defs/Term" }
      },
      "required": ["op", "left", "right"]
    },

    "Between": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "between": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "left": { "$ref": "#/$defs/Term" },
            "low": { "$ref": "#/$defs/Term" },
            "high": { "$ref": "#/$defs/Term" }
          },
          "required": ["left", "low", "high"]
        }
      },
      "required": ["between"]
    },

    "InSet": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "in": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "left": { "$ref": "#/$defs/Term" },
            "right": { "$ref": "#/$defs/Term" }
          },
          "required": ["left", "right"]
        }
      },
      "required": ["in"]
    },

    "Not": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "not": { "$ref": "#/$defs/Predicate" }
      },
      "required": ["not"]
    },

    "And": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "and": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/Predicate" }
        }
      },
      "required": ["and"]
    },

    "Or": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "or": {
          "type": "array",
          "minItems": 1,
          "items": { "$ref": "#/$defs/Predicate" }
        }
      },
      "required": ["or"]
    },

    "Bind": {
      "oneOf": [
        { "$ref": "#/$defs/BindEntity" },
        { "$ref": "#/$defs/BindPath" }
      ]
    },

    "BindEntity": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "alias": { "$ref": "#/$defs/Alias" },
        "entity": { "type": "string", "minLength": 1 },
        "extends": { "type": "string", "minLength": 1 }
      },
      "required": ["alias", "entity"]
    },

    "BindPath": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "alias": { "$ref": "#/$defs/Alias" },
        "from": { "$ref": "#/$defs/Alias" },
        "path": {
          "type": "array",
          "minItems": 1,
          "items": { "type": "string", "minLength": 1 }
        }
      },
      "required": ["alias", "from", "path"]
    },

    "SelectItem": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "alias": { "$ref": "#/$defs/Alias" },
        "prop": { "type": "string", "minLength": 1 },
        "as": { "type": "string", "minLength": 1 }
      },
      "required": ["alias"]
    },

    "GroupByItem": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "alias": { "$ref": "#/$defs/Alias" },
        "prop": { "type": "string", "minLength": 1 },
        "as": { "type": "string", "minLength": 1 }
      },
      "required": ["alias", "prop"]
    },

    "Aggregation": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "op": { "type": "string", "enum": ["sum", "avg", "min", "max", "count", "count_distinct"] },
        "term": { "$ref": "#/$defs/Term" },
        "as": { "type": "string", "minLength": 1 }
      },
      "required": ["op"]
    },

    "OrderByItem": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "term": {
          "oneOf": [
            { "$ref": "#/$defs/SelectItem" },
            { "$ref": "#/$defs/ValueTerm" }
          ]
        },
        "dir": { "type": "string", "enum": ["asc", "desc"], "default": "asc" }
      },
      "required": ["term"]
    }
  }
}
"""




def _normalize_from_key(b: dict) -> str | None:
    return b.get("from") or b.get("from_")


def _is_entity_bind(b: dict) -> bool:
    return "entity" in b and "alias" in b


def _is_path_bind(b: dict) -> bool:
    return "path" in b and _normalize_from_key(b) is not None and "alias" in b


def _infer_src_from_reverse_first_hop(rel_df: pd.DataFrame, dest_entity: str, relation: str) -> list[str]:
    if dest_entity is None or relation is None:
        return []
    mask = (rel_df["to_concept"] == dest_entity) & (rel_df["relation"] == relation)
    if not mask.any():
        return []
    return list(pd.unique(rel_df.loc[mask, "from_concept"]))


def _rewrite_spec_aliases(spec: dict) -> dict:
    if not isinstance(spec, dict):
        return spec
    alias_map = {
        "rm_score": "performance",
        "signal_high_cost": "signal_high_cost_ratio",
    }
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


def _coerce_value_terms(spec: dict) -> dict:
    if not isinstance(spec, dict):
        return spec
    try:
        import copy
        out = copy.deepcopy(spec)
    except Exception:
        out = json.loads(json.dumps(spec))

    alias_to_entity = {}
    for b in out.get("bind") or []:
        if isinstance(b, dict) and b.get("alias") and b.get("entity"):
            alias_to_entity[b["alias"]] = b["entity"]

    specs = load_ai_insights_specs()
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

    def _is_numeric_dtype(dtype: str) -> bool:
        if not dtype:
            return False
        return any(
            token in dtype
            for token in ("number", "numeric", "decimal", "int", "float", "double")
        )

    def _coerce_value(value, dtype: str):
        if not _is_numeric_dtype(dtype):
            return value
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "false"):
                return 1.0 if lowered == "true" else 0.0
        return value

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
                values = right.get("value")
                if isinstance(values, list):
                    right["value"] = [_coerce_value(v, dtype) for v in values]
                else:
                    right["value"] = _coerce_value(values, dtype)
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


def _dedupe_agg_aliases(spec: dict) -> dict:
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
    seen = set()
    for a in aggregations:
        if not isinstance(a, dict):
            continue
        a_as = a.get("as")
        if not a_as:
            continue
        base = a_as
        candidate = a_as
        i = 2
        while candidate in used or candidate in seen:
            candidate = f"{base}_{i}"
            i += 1
        if candidate != a_as:
            a["as"] = candidate
            renamed[a_as] = candidate
        seen.add(candidate)

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


def _rename_group_by_agg_conflicts(spec: dict) -> dict:
    if not isinstance(spec, dict):
        return spec
    aggregations = list(spec.get("aggregations") or [])
    if not aggregations:
        return spec

    agg_terms = set()
    for a in aggregations:
        term = a.get("term") if isinstance(a, dict) else None
        if isinstance(term, dict):
            alias = term.get("alias")
            prop = term.get("prop")
            if alias and prop:
                agg_terms.add((alias, prop))

    if not agg_terms:
        return spec

    group_by = list(spec.get("group_by") or [])
    select = list(spec.get("select") or [])

    used = set()
    for t in group_by + select:
        if isinstance(t, dict) and t.get("as"):
            used.add(t["as"])

    renamed = {}

    def _rename_term(term: dict) -> None:
        cur_as = term.get("as")
        if not cur_as:
            return
        base = f"{cur_as}_dim"
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        term["as"] = candidate
        renamed[cur_as] = candidate
        used.add(candidate)

    for g in group_by:
        if isinstance(g, dict) and (g.get("alias"), g.get("prop")) in agg_terms:
            _rename_term(g)

    for s in select:
        if isinstance(s, dict) and (s.get("alias"), s.get("prop")) in agg_terms:
            _rename_term(s)

    spec["group_by"] = group_by
    spec["select"] = select

    if renamed and spec.get("order_by"):
        for ob in spec.get("order_by") or []:
            term = (ob or {}).get("term") or {}
            if isinstance(term, dict) and "value" in term:
                v = term.get("value")
                if v in renamed:
                    term["value"] = renamed[v]
                    ob["term"] = term
    return spec


def _has_group_by_agg_conflict(spec: dict) -> bool:
    if not isinstance(spec, dict):
        return False
    aggs = list(spec.get("aggregations") or [])
    if not aggs:
        return False
    agg_terms: dict[tuple, set] = {}
    for a in aggs:
        if not isinstance(a, dict):
            continue
        term = a.get("term")
        if not isinstance(term, dict):
            continue
        alias = term.get("alias")
        prop = term.get("prop")
        if not alias or not prop:
            continue
        key = (alias, prop)
        agg_terms.setdefault(key, set()).add(str(a.get("op") or "").lower())
    if not agg_terms:
        return False
    for g in (spec.get("group_by") or []):
        if not isinstance(g, dict):
            continue
        key = (g.get("alias"), g.get("prop"))
        if key in agg_terms:
            # Allow common safe pattern: group by a field and count it.
            ops = agg_terms.get(key) or set()
            if not ops.issubset({"count", "count_distinct"}):
                return True
    return False


def fix_reverse_binds(binds: list[dict], rel_df: pd.DataFrame) -> list[dict]:
    required_cols = {"from_concept", "relation", "to_concept"}
    if not required_cols.issubset(set(rel_df.columns)):
        raise ValueError(f"rel_df missing required columns {required_cols}; found {set(rel_df.columns)}")

    fixed_binds: list[dict] = []
    alias_to_entity: dict[str, str] = {}

    for b in binds:
        if _is_entity_bind(b):
            alias_to_entity[b["alias"]] = b["entity"]
            fixed_binds.append(b)

    for b in binds:
        if not _is_path_bind(b):
            continue

        from_alias = _normalize_from_key(b)
        to_alias = b["alias"]
        path = b["path"]
        if not path:
            fixed_binds.append(b)
            continue

        from_entity = alias_to_entity.get(from_alias)
        valid = True
        current = from_entity
        for step in path:
            candidates = rel_df[(rel_df["from_concept"] == current) & (rel_df["relation"] == step)]
            if candidates.empty:
                valid = False
                break
            current = candidates.iloc[0]["to_concept"]

        if valid:
            fixed_binds.append(b)
            continue

        reversed_path = list(reversed(path))
        reversed_b = {
            "alias": from_alias,
            "from": to_alias,
            "path": reversed_path,
        }

        new_from_alias = reversed_b["from"]
        if new_from_alias not in alias_to_entity:
            first_hop_relation = path[0]
            dest_entity = alias_to_entity.get(from_alias)
            candidates = _infer_src_from_reverse_first_hop(rel_df, dest_entity, first_hop_relation)

            if len(candidates) == 1:
                inferred_entity = candidates[0]
                reused = None
                for alias, ent in alias_to_entity.items():
                    if ent == inferred_entity:
                        reused = alias
                        break
                if reused is not None:
                    reversed_b["from"] = reused
                    new_from_alias = reused
                else:
                    entity_b = {"alias": new_from_alias, "entity": inferred_entity}
                    alias_to_entity[new_from_alias] = inferred_entity
                    fixed_binds.append(entity_b)

        fixed_binds.append(reversed_b)

    return fixed_binds


def run_dynamic_query(builder, spec: dict) -> pd.DataFrame:
    """
    Improvements:
    - Supports count_distinct aggregation
    - Order-by can reference aggregation alias via {"term":{"value":"<agg_alias>"}}
    - Robust timestamp parsing for TIMESTAMP_NTZ (avoid tz-aware values)
    - Automatic timed-event overlap semantics when a window is present
    - Proper derived-metric post-compute by passing entity name
    """
    # Ensure RAI client config is loaded (prevents missing role/profile errors)
    ensure_rai_config()
    if _DEBUG_AGG:
        print(f"[DEBUG][agg] run_dynamic_query module: {__file__}", file=sys.stderr)

    import json as _json
    spec = _coerce_value_terms(_rewrite_spec_aliases(spec))
    # Final alias normalization before execution
    spec = _rename_group_by_agg_conflicts(spec)
    spec = _dedupe_agg_aliases(spec)

    # -----------------------------
    # Helpers local to executor
    # -----------------------------
    def _entity_from_spec(spec_: dict) -> str:
        binds = spec_.get("bind") or []
        if isinstance(binds, list) and binds and isinstance(binds[0], dict):
            return str(binds[0].get("entity") or "")
        return ""

    def _alias_of_entity(spec_: dict, entity_name: str) -> Optional[str]:
        for b in (spec_.get("bind") or []):
            if isinstance(b, dict) and b.get("entity") == entity_name and b.get("alias"):
                return b["alias"]
        return None

    def _extract_window_from_meta(spec_: dict) -> tuple[Optional[str], Optional[str], str]:
        meta = spec_.get("meta") or {}
        tw = meta.get("time_window") if isinstance(meta, dict) else None
        if isinstance(tw, dict):
            return tw.get("start"), tw.get("end"), (tw.get("mode") or "point")
        return None, None, "point"

    def _extract_start_time_window(where_list: list, alias: str) -> tuple[Optional[str], Optional[str]]:
        lo = None
        hi = None
        for p in where_list:
            if not isinstance(p, dict) or "op" not in p:
                continue
            left = p.get("left") or {}
            right = p.get("right") or {}
            if not (isinstance(left, dict) and isinstance(right, dict) and "value" in right):
                continue
            if left.get("alias") == alias and left.get("prop") == "start_time":
                if p["op"] in (">=", ">"):
                    lo = right["value"]
                elif p["op"] in ("<=", "<"):
                    hi = right["value"]
        return lo, hi

    def _rewrite_timed_event_window_to_overlap(spec_: dict) -> dict:
        """
        If dt_timed_event_denorm is involved and we have a time window,
        enforce overlap semantics:
          start_time <= window_end AND end_time >= window_start
        """
        te_alias = _alias_of_entity(spec_, "dt_timed_event_denorm")
        if not te_alias:
            return spec_

        where_list = list(spec_.get("where") or [])

        # Prefer meta.time_window if present
        win_start, win_end, mode = _extract_window_from_meta(spec_)
        if not (win_start and win_end):
            # fallback: infer from start_time predicates
            win_start, win_end = _extract_start_time_window(where_list, te_alias)
            mode = "point"  # inferred window usually point-based in specs

        if not (win_start and win_end):
            spec_["where"] = where_list
            return spec_

        # If already overlap mode or already has end_time predicate, keep as-is
        already_end_time = any(
            isinstance(p, dict)
            and p.get("op") in (">=", ">", "<=", "<")
            and isinstance(p.get("left"), dict)
            and p["left"].get("alias") == te_alias
            and p["left"].get("prop") == "end_time"
            for p in where_list
        )
        if already_end_time:
            return spec_

        # Remove start_time-only window predicates (>= start_time / <= start_time)
        def _is_start_time_window_pred(p):
            return (
                isinstance(p, dict)
                and p.get("op") in (">=", ">", "<=", "<")
                and isinstance(p.get("left"), dict)
                and isinstance(p.get("right"), dict)
                and p["left"].get("alias") == te_alias
                and p["left"].get("prop") == "start_time"
                and "value" in p["right"]
            )

        where_list = [p for p in where_list if not _is_start_time_window_pred(p)]

        # Add overlap predicates
        where_list.append({"op": "<=", "left": {"alias": te_alias, "prop": "start_time"}, "right": {"value": win_end}})
        where_list.append({"op": ">=", "left": {"alias": te_alias, "prop": "end_time"}, "right": {"value": win_start}})

        spec_ = dict(spec_)
        spec_["where"] = where_list
        # Promote meta.time_window.mode to overlap for transparency
        meta = dict(spec_.get("meta") or {})
        tw = dict(meta.get("time_window") or {})
        if win_start and win_end:
            tw.setdefault("start", win_start)
            tw.setdefault("end", win_end)
            tw["mode"] = "overlap"
            meta["time_window"] = tw
        spec_["meta"] = meta
        return spec_

    # Enforce grouped shape (your existing normalizer)
    try:
        alias_to_entity = {b.get("alias"): b.get("entity") for b in (spec.get("bind") or []) if isinstance(b, dict) and b.get("alias") and b.get("entity")}
        spec = _normalize_grouped_spec(spec, alias_to_entity)
    except Exception:
        pass

    # ✅ Semantic fix: downtime windows should be overlap
    try:
        spec = _rewrite_timed_event_window_to_overlap(spec)
    except Exception:
        pass

    aliases: dict[str, object] = {}
    where_args: list = []
    group_terms: list = []

    def _now_ntz():
        # TIMESTAMP_NTZ wants naive datetime
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def _parse_datetime_value(v):
        if not isinstance(v, str):
            return v
        s = v.strip()

        if re.fullmatch(r"CURRENT_TIMESTAMP\(\)|CURRENT_TIMESTAMP", s, flags=re.I):
            return _now_ntz()

        m = re.fullmatch(
            r"DATEADD\s*\(\s*day\s*,\s*([+-]?\d+)\s*,\s*CURRENT_TIMESTAMP\(\)\s*\)\s*",
            s,
            flags=re.I,
        )
        if m:
            return _now_ntz() + timedelta(days=int(m.group(1)))

        try:
            iso = s.replace(" ", "T")
            if iso.endswith("Z"):
                iso = iso[:-1] + "+00:00"
            dt = datetime.fromisoformat(iso)
            # normalize to naive UTC
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return v

    def _term(t, parse_datetime: bool = True):
        if "value" in t:
            return _parse_datetime_value(t["value"]) if parse_datetime else t["value"]
        alias = t["alias"]
        ref = aliases.get(alias)
        if ref is None:
            raise KeyError(f"Unknown alias {alias}")
        return getattr(ref, t["prop"]) if "prop" in t else ref

    def _predicate(node):
        if "op" in node:
            op = node["op"]
            if op in ("contains", "ilike", "like"):
                left = _term(node["left"])
                right = _term(node["right"], parse_datetime=False)
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

    relation_df = pd.DataFrame(list(builder._relationships.values()))
    if relation_df.empty:
        binds_fixed = spec.get("bind", [])
    else:
        binds_fixed = fix_reverse_binds(spec.get("bind", []), relation_df)

    # Build (from_entity, relation_name) -> relationship metadata for BindPath traversal
    rel_index = {}
    for meta in builder._relationships.values():
        if not isinstance(meta, dict):
            continue
        fe = meta.get("from_concept")
        rn = meta.get("relation")
        if fe and rn:
            rel_index[(fe, rn)] = meta

    alias_to_entity = {}
    for b in binds_fixed:
        if "entity" in b:
            Ent = builder.get_concept(b["entity"])
            if Ent is None:
                raise KeyError(f"Unknown entity {b['entity']}")
            ref = Ent.ref()
            aliases[b["alias"]] = ref
            alias_to_entity[b["alias"]] = b["entity"]
            where_args.append(ref)
        elif "from" in b and "path" in b:
            src_alias = b["from"]
            if src_alias not in aliases:
                raise KeyError(f"BindPath 'from' alias not bound: {src_alias}")
            if src_alias not in alias_to_entity:
                raise KeyError(f"BindPath cannot resolve entity for alias: {src_alias}")

            cur_alias = src_alias
            cur_entity = alias_to_entity[cur_alias]
            cur_ref = aliases[cur_alias]

            steps = b.get("path") or []
            if not steps:
                raise ValueError(f"BindPath has empty path: {b}")

            for i, step in enumerate(steps):
                meta = rel_index.get((cur_entity, step))
                if meta is None:
                    known = sorted({k[1] for k in rel_index.keys() if k[0] == cur_entity})
                    raise KeyError(
                        f"Unknown relationship step '{step}' from entity '{cur_entity}'. "
                        f"Known steps from '{cur_entity}': {known}"
                    )

                next_entity = meta["to_concept"]
                next_alias = b["alias"] if i == (len(steps) - 1) else f"{b['alias']}__hop{i}"

                if next_alias not in aliases:
                    Next = builder.get_concept(next_entity)
                    if Next is None:
                        raise KeyError(
                            f"Unknown entity {next_entity} while expanding BindPath step '{step}'"
                        )
                    next_ref = Next.ref()
                    aliases[next_alias] = next_ref
                    alias_to_entity[next_alias] = next_entity
                    where_args.append(next_ref)
                else:
                    next_ref = aliases[next_alias]

                From = builder.get_concept(cur_entity)
                edge_fn = getattr(From, step)
                where_args.append(edge_fn(cur_ref, next_ref))

                cur_entity = next_entity
                cur_alias = next_alias
                cur_ref = next_ref
        else:
            raise ValueError(f"Invalid bind: {b}")

    for pred in spec.get("where", []):
        where_args.append(_predicate(pred))

    q = qb.where(*where_args)

    select_terms = []
    group_by = spec.get("group_by") or []
    aggregations = spec.get("aggregations") or []

    # Group terms
    if group_by:
        seen_group = set()
        for g in group_by:
            if not isinstance(g, dict):
                continue
            key = (g.get("alias"), g.get("prop"))
            if key in seen_group:
                continue
            seen_group.add(key)
            base = {"alias": g.get("alias"), "prop": g.get("prop")}
            group_term = _term(base)
            group_terms.append(group_term)
            term = group_term
            if "as" in g and hasattr(term, "alias"):
                term = term.alias(g["as"])
            select_terms.append(term)

    # Aggregations with count_distinct
    if aggregations:
        def _count_distinct(expr):
            # Try best available patterns
            if hasattr(qb, "count_distinct"):
                if _DEBUG_AGG:
                    print("[DEBUG][agg] count_distinct via qb.count_distinct", file=sys.stderr)
                return qb.count_distinct(expr)
            if hasattr(qb, "distinct"):
                if _DEBUG_AGG:
                    print("[DEBUG][agg] count_distinct via qb.distinct(expr, expr)", file=sys.stderr)
                # RelationalAI distinct expects 2 args; use expr twice to count distinct values.
                return qb.count(qb.distinct(expr, expr))
            if hasattr(expr, "distinct"):
                try:
                    if _DEBUG_AGG:
                        print("[DEBUG][agg] count_distinct via expr.distinct()", file=sys.stderr)
                    return qb.count(expr.distinct())
                except Exception:
                    if _DEBUG_AGG:
                        print("[DEBUG][agg] expr.distinct() failed; falling back to count(expr)", file=sys.stderr)
            if _DEBUG_AGG:
                print("[DEBUG][agg] count_distinct fallback to qb.count(expr)", file=sys.stderr)
            return qb.count(expr)

        agg_map = {
            "sum": qb.sum,
            "avg": qb.avg,
            "min": qb.min,
            "max": qb.max,
            "count": qb.count,
            "count_distinct": _count_distinct,
        }

        for a in aggregations:
            op = a.get("op")
            term_spec = a.get("term")
            fn = agg_map.get(op)
            if fn is None:
                raise ValueError(f"Unsupported aggregation op: {op}")

            if term_spec is None:
                agg_expr = fn()
            else:
                agg_expr = fn(_term(term_spec))

            if group_terms:
                agg_expr = agg_expr.per(*group_terms)

            if "as" in a and hasattr(agg_expr, "alias"):
                agg_expr = agg_expr.alias(a["as"])
            select_terms.append(agg_expr)

    # Regular selects: only add if NOT using group_by or aggregations
    # (group_by and agg terms already added to select_terms above)
    if not group_by and not aggregations:
        seen_select = set()
        for s in spec.get("select", []):
            if not isinstance(s, dict):
                continue
            key = (s.get("alias"), s.get("prop"))
            if key in seen_select:
                continue
            seen_select.add(key)
            term = _term(s)
            if "as" in s and hasattr(term, "alias"):
                term = term.alias(s["as"])
            select_terms.append(term)

    if _has_group_by_agg_conflict(spec):
        raise ValueError(
            "Invalid spec: group_by includes fields that are also aggregated. "
            "Rename does not resolve this; choose different group_by fields or remove the conflict."
        )

    # CRITICAL: Wrap RAI query execution with timeout
    # The .to_df() call can hang indefinitely if RAI query compilation is stuck
    import threading
    import json
    
    rai_timeout = int(os.environ.get("RAI_TO_DF_TIMEOUT_SECONDS", "300"))  # 5 min default
    
    # Log the spec and select terms for debugging
    print(f"[DEBUG] RAI Query Spec (JSON):", file=sys.stderr)
    print(json.dumps(spec, indent=2, default=str), file=sys.stderr)
    print(f"[DEBUG] RAI Select Terms Count: {len(select_terms)}", file=sys.stderr)
    print(f"[DEBUG] RAI Timeout: {rai_timeout}s", file=sys.stderr)
    sys.stderr.flush()
    
    df_holder = [None]
    exception_holder = [None]
    
    def execute_query():
        try:
            # Ensure config is set in this worker thread too
            ensure_rai_config()
            try:
                from relationalai.clients import config as rai_config
                cfg_path = getattr(rai_config, "CONFIG_FILE", None)
            except Exception:
                cfg_path = None
            print(
                f"[DEBUG] RAI config: env.RAI_CONFIG_FILE={os.environ.get('RAI_CONFIG_FILE')} "
                f"env.RAI_PROFILE={os.environ.get('RAI_PROFILE')} "
                f"rai_config.CONFIG_FILE={cfg_path}",
                file=sys.stderr,
            )
            sys.stderr.flush()
            print(f"[DEBUG] RAI: Starting q.select(...).to_df() call", file=sys.stderr)
            sys.stderr.flush()
            try:
                df_holder[0] = q.select(*select_terms).to_df()
            except Exception as inner:
                # Retry once if config got stale
                if "Missing config value for 'role'" in str(inner):
                    ensure_rai_config()
                    df_holder[0] = q.select(*select_terms).to_df()
                else:
                    raise
            print(f"[DEBUG] RAI: Query returned successfully with {len(df_holder[0])} rows", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            exception_holder[0] = e
            print(f"[ERROR] RAI query execution failed: {e}", file=sys.stderr)
            sys.stderr.flush()
    
    # Run in thread with timeout
    t = threading.Thread(target=execute_query, daemon=True)
    t.start()
    t.join(timeout=rai_timeout)
    
    if t.is_alive():
        print(f"[ERROR] RAI query .to_df() timed out after {rai_timeout}s", file=sys.stderr)
        print(f"[ERROR] Query compilation or execution is stuck in RelationalAI", file=sys.stderr)
        sys.stderr.flush()
        raise TimeoutError(
            f"RAI query execution timed out after {rai_timeout}s. "
            "This may indicate an issue with the query specification or RAI compiler. "
            f"Increase RAI_TO_DF_TIMEOUT_SECONDS (currently {rai_timeout}s) if needed."
        )
    
    if exception_holder[0]:
        raise exception_holder[0]
    
    df = df_holder[0]
    if df is None:
        raise RuntimeError("RAI query returned None")

    # distinct / dedupe
    if spec.get("distinct"):
        df = df.drop_duplicates()
    elif group_by and not aggregations:
        df = df.drop_duplicates()

    # order_by (supports {"term":{"value":"agg_alias"}})
    if spec.get("order_by"):
        cols, ascending = [], []
        for ob in spec["order_by"]:
            term = ob.get("term") or {}
            dir_asc = (ob.get("dir", "asc") or "asc").lower() == "asc"
            if isinstance(term, dict) and "value" in term:
                name = term.get("value")
            else:
                name = term.get("as") or term.get("prop")
            cols.append(name)
            ascending.append(dir_asc)

        valid = [(c, a) for c, a in zip(cols, ascending) if c in df.columns]
        if valid:
            df = df.sort_values(by=[c for c, _ in valid], ascending=[a for _, a in valid])

    # limit/offset
    if "limit" in spec:
        off = int(spec.get("offset", 0) or 0)
        lim = int(spec.get("limit", 0) or 0)
        if lim > 0:
            df = df.iloc[off: off + lim]
        else:
            df = df.iloc[off:]

    # ✅ Derived metrics post-compute: pass entity
    try:
        cfg = load_registry_config()
        if getattr(cfg, "post_compute_derived_metrics", False):
            entity = _entity_from_spec(spec)
            if entity:
                try:
                    df = enrich_dataframe_with_derived_metrics(df, entity)
                except TypeError:
                    # backward compat if signature is enrich(df)
                    df = enrich_dataframe_with_derived_metrics(df)
    except Exception:
        pass

    return df
