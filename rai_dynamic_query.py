import re
import pandas as pd
import relationalai.semantics as qb
from rai_ai_insights_ontology import load_ai_insights_specs
from rai_derived_metrics import enrich_dataframe_with_derived_metrics


DYNAMIC_QUERY_SCHEMA = """{
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
    "distinct": { "type": "boolean", "default": false }
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
          "enum": ["==", "!=", "<", "<=", ">", ">="]
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
        "op": { "type": "string", "enum": ["sum", "avg", "min", "max", "count"] },
        "term": { "$ref": "#/$defs/Term" },
        "as": { "type": "string", "minLength": 1 }
      },
      "required": ["op"]
    },

    "OrderByItem": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "term": { "$ref": "#/$defs/SelectItem" },
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
    spec = _coerce_value_terms(_rewrite_spec_aliases(spec))
    aliases: dict[str, object] = {}
    where_args: list = []
    group_terms: list = []

    def _term(t):
        if "value" in t:
            return t["value"]
        alias = t["alias"]
        ref = aliases.get(alias)
        if ref is None:
            raise KeyError(f"Unknown alias {alias}")
        return getattr(ref, t["prop"]) if "prop" in t else ref

    def _predicate(node):
        if "op" in node:
            left = _term(node["left"])
            right = _term(node["right"])
            return {
                "==": left == right,
                "!=": left != right,
                "<": left < right,
                "<=": left <= right,
                ">": left > right,
                ">=": left >= right,
            }[node["op"]]

        if "between" in node:
            spec = node["between"]
            x = _term(spec["left"])
            low = _term(spec["low"])
            high = _term(spec["high"])
            return (x >= low) & (x <= high)

        if "in" in node:
            spec = node["in"]
            x = _term(spec["left"])
            vals = _term(spec["right"])
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

    for b in binds_fixed:
        if "entity" in b:
            Ent = builder.get_concept(b["entity"])
            if Ent is None:
                raise KeyError(f"Unknown entity {b['entity']}")
            ref = Ent.ref()
            aliases[b["alias"]] = ref
            where_args.append(ref)
        elif "from" in b and "path" in b:
            src = aliases[b["from"]]
            expr = src
            for step in b["path"]:
                expr = getattr(expr, step)
            aliases[b["alias"]] = expr
            where_args.append(expr)
        else:
            raise ValueError(f"Invalid bind: {b}")

    for pred in spec.get("where", []):
        where_args.append(_predicate(pred))

    q = qb.where(*where_args)

    select_terms = []
    group_by = spec.get("group_by") or []
    aggregations = spec.get("aggregations") or []

    # If aggregations are present, treat selected non-aggregate fields as grouping keys
    # so they can be returned alongside aggregates.
    if aggregations and spec.get("select"):
        group_by = list(group_by)
        original_keys = {(g.get("alias"), g.get("prop")) for g in group_by if isinstance(g, dict)}
        agg_keys = set()
        for a in aggregations:
            term = a.get("term")
            if isinstance(term, dict):
                agg_keys.add((term.get("alias"), term.get("prop")))
        existing_keys = set(original_keys)
        for s in spec.get("select", []):
            if not isinstance(s, dict):
                continue
            key = (s.get("alias"), s.get("prop"))
            if key in agg_keys and key not in original_keys:
                # Avoid shadowed variables when a field is both selected and aggregated.
                continue
            if key in existing_keys or s.get("prop") is None:
                continue
            group_by.append({"alias": s.get("alias"), "prop": s.get("prop"), "as": s.get("as")})
            existing_keys.add(key)

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

    if aggregations:
        agg_map = {
            "sum": qb.sum,
            "avg": qb.avg,
            "min": qb.min,
            "max": qb.max,
            "count": qb.count,
        }
        for a in aggregations:
            op = a.get("op")
            fn = agg_map.get(op)
            if fn is None:
                raise ValueError(f"Unsupported aggregation op: {op}")
            term_spec = a.get("term")
            if term_spec is None:
                agg_expr = fn()
            else:
                agg_expr = fn(_term(term_spec))
            if group_terms:
                agg_expr = agg_expr.per(*group_terms)
            if "as" in a and hasattr(agg_expr, "alias"):
                agg_expr = agg_expr.alias(a["as"])
            select_terms.append(agg_expr)

    # Only add regular select fields if we don't have aggregations
    # (with aggregations, only return the aggregated values)
    if not select_terms:
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

    df = q.select(*select_terms).to_df()

    if spec.get("distinct"):
        df = df.drop_duplicates()
    elif group_by and not aggregations:
        df = df.drop_duplicates()

    if spec.get("order_by"):
        cols, ascending = [], []
        for ob in spec["order_by"]:
            t = ob["term"]
            name = t.get("as") or t.get("prop")
            cols.append(name)
            ascending.append(ob.get("dir", "asc").lower() == "asc")
        valid = [(c, a) for c, a in zip(cols, ascending) if c in df.columns]
        if valid:
            df = df.sort_values(by=[c for c, _ in valid], ascending=[a for _, a in valid])

    if "limit" in spec:
        off = spec.get("offset", 0)
        lim = spec["limit"]
        df = df.iloc[off: off + lim]

    # Post-process: compute derived metrics
    # Check if any derived metrics were requested in select
    requested_derived = []
    for s in spec.get("select", []):
        if isinstance(s, dict) and "prop" in s:
            prop = s["prop"].lower()
            # Common derived field names
            if prop in ["profit_margin", "cost_to_revenue", "revenue_per_cost", 
                        "aum_per_cost", "month_date", "aum_trend", "revenue_trend"]:
                requested_derived.append(prop)
    
    # Also check aggregations for derived field references
    for a in spec.get("aggregations", []):
        term = a.get("term", {})
        if isinstance(term, dict) and "prop" in term:
            prop = term["prop"].lower()
            if prop in ["profit_margin", "cost_to_revenue", "revenue_per_cost", 
                        "aum_per_cost", "aum_trend", "revenue_trend"]:
                requested_derived.append(prop)
    
    # Always compute common derived metrics if dependencies are available
    # This enriches the dataframe even if not explicitly requested
    df = enrich_dataframe_with_derived_metrics(df)

    return df
