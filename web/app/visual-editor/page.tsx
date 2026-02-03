"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";

type NodeKind = "entity" | "derived" | "reasoner" | "kg" | "graph";
type CanvasMode = "data" | "kg";

const ENABLE_REASONERS = true;
const AVAILABLE_NODE_KINDS: NodeKind[] = ENABLE_REASONERS
  ? ["entity", "derived", "reasoner", "kg", "graph"]
  : ["entity", "derived", "kg", "graph"];
const DEFAULT_VISIBLE_KINDS: Record<NodeKind, boolean> = {
  entity: true,
  derived: true,
  reasoner: ENABLE_REASONERS,
  kg: true,
  graph: true,
};
const CANVAS_MODE_KINDS: Record<CanvasMode, NodeKind[]> = {
  data: ["entity", "derived"],
  kg: ["kg", "graph", "reasoner"],
};

interface RegistryField {
  name: string;
  dtype: string;
  role: string;
  expr: string;
  description?: string;
  derived?: boolean;
  default_agg?: string;
  depends_on?: string[];
  order_by?: Array<{ field: string; direction?: "asc" | "desc" }>;
}

interface RegistryEntity {
  name: string;
  description: string;
  database: string;
  schema: string;
  table: string;
  fields: RegistryField[];
  join_keys: string[];
  default_metric: string;
  entity_type: string;
  applicable_reasoners?: string[];
  applicable_derived_metrics?: string[];
}

interface RegistryRelationship {
  name: string;
  from_entity: string;
  to_entity: string;
  description: string;
  join_on: Array<[string, string]>;
}

interface ReasonerSignal {
  name: string;
  description?: string;
  metric_field: string;
  threshold: number;
  direction: string;
  weight?: number;
}

interface DrilldownPlanStep {
  id: string;
  from?: string;
  description?: string;
  limit?: number;
  inputs?: Record<string, string>;
  query?: Record<string, unknown>;
}

interface DrilldownPlan {
  steps?: DrilldownPlanStep[];
}

interface RegistryReasoner {
  id: string;
  name: string;
  description: string;
  entity_type: string;
  type?: string;
  graph_id?: string;
  signals?: ReasonerSignal[];
  outputs?: string[];
  drilldown_plan?: DrilldownPlan;
}

interface DerivedRuleClause {
  when?: string;
  expr: string;
}

interface DerivedRuleSpec {
  entity: string;
  field: string;
  rules: DerivedRuleClause[];
  vars?: Record<string, string>;
  description?: string;
}

interface RegistryPayload {
  entities: RegistryEntity[];
  relationships: RegistryRelationship[];
  reasoners: RegistryReasoner[];
  derived_rel_rules?: DerivedRuleSpec[];
  prompt_templates?: Record<string, string>;
  analysis_config?: Record<string, unknown>;
  config?: Record<string, unknown>;
  kg?: {
    nodes?: KgNodeSpec[];
    edges?: KgEdgeSpec[];
    graphs?: KgGraphSpec[];
  };
}

interface KgNodeSpec {
  entity: string;
  node_type?: string;
  key_field?: string | string[];
  label_field?: string;
  properties?: string[];
  description?: string;
}

interface KgEdgeSpec {
  name: string;
  from_entity: string;
  to_entity: string;
  join_on?: Array<[string, string]>;
  edge_type?: string;
  description?: string;
}

interface KgGraphNodeSpec {
  entity: string;
  label_field?: string;
  properties?: string[];
}

interface KgGraphEdgeSpec {
  relation?: string;
  weight_field?: string;
  name?: string;
  from_entity?: string;
  to_entity?: string;
}

interface KgGraphSpec {
  id: string;
  description?: string;
  directed?: boolean;
  weighted?: boolean;
  nodes?: KgGraphNodeSpec[];
  edges?: KgGraphEdgeSpec[];
}

interface RegistryVersion {
  name: string;
  created_at: string;
  path?: string;
}

interface ValidationIssue {
  level: "error" | "warn";
  message: string;
  context?: string;
}

interface NodeRef {
  type: NodeKind;
  entityName?: string;
  fieldName?: string;
  reasonerId?: string;
  graphId?: string;
}

interface GraphNode {
  id: string;
  label: string;
  kind: NodeKind;
  description: string;
  tags: string[];
  inputs: string[];
  outputs: string[];
  x: number;
  y: number;
  ref: NodeRef;
}

interface GraphEdge {
  id: string;
  from: string;
  to: string;
  label: string;
  type:
    | "relation"
    | "derived"
    | "reasoner"
    | "reasoner_signal"
    | "reasoner_output"
    | "dependency"
    | "kg"
    | "graph";
}

interface RelationshipDraft {
  from: string;
  to: string;
  name: string;
  description: string;
  joinText: string;
  mode?: "create" | "edit";
  originalName?: string;
}

const NODE_WIDTH = 240;
const NODE_HEIGHT = 148;

const KIND_STYLES: Record<NodeKind, string> = {
  entity: "from-[#0f172a] to-[#1e293b] text-white",
  derived: "from-[#0f766e] to-[#14b8a6] text-white",
  reasoner: "from-[#b45309] to-[#f59e0b] text-white",
  kg: "from-[#1e3a8a] to-[#3b82f6] text-white",
  graph: "from-[#111827] to-[#4b5563] text-white",
};

const EDGE_COLORS: Record<GraphEdge["type"], string> = {
  relation: "#0ea5e9",
  derived: "#14b8a6",
  reasoner: "#f59e0b",
  reasoner_signal: "#f97316",
  reasoner_output: "#ec4899",
  dependency: "#64748b",
  kg: "#22c55e",
  graph: "#6366f1",
};

const EDGE_PADDING = 0;
const JOIN_LABEL_MAX = 80;
const PROMPT_TEMPLATE_KEYS = [
  "summary_full",
  "summary_full_single",
  "summary_chunk",
  "summary_reduce",
];

const derivedNodeId = (entityName: string, fieldName: string) =>
  `derived::${entityName}::${fieldName}`;

const reasonerNodeId = (reasonerId: string) => `reasoner::${reasonerId}`;
const kgNodeId = (entityName: string) => `kg::${entityName}`;
const graphNodeId = (graphId: string) => `graph::${graphId}`;

const normalizeRegistry = (payload: RegistryPayload | null): RegistryPayload => {
  if (!payload) {
    return {
      entities: [],
      relationships: [],
      reasoners: [],
      derived_rel_rules: [],
      prompt_templates: {},
      analysis_config: {},
      config: {},
      kg: { nodes: [], edges: [], graphs: [] },
    };
  }
  return {
    entities: payload.entities || [],
    relationships: (payload.relationships || []).map((rel) => ({
      ...rel,
      join_on: Array.isArray(rel.join_on) ? rel.join_on : [],
    })),
    reasoners: ENABLE_REASONERS ? payload.reasoners || [] : [],
    derived_rel_rules: payload.derived_rel_rules || [],
    prompt_templates: payload.prompt_templates || {},
    analysis_config: payload.analysis_config || {},
    config: payload.config || {},
    kg: payload.kg || { nodes: [], edges: [], graphs: [] },
  };
};

const formatJoinLabel = (rel: RegistryRelationship) => {
  const pairs = (rel.join_on || [])
    .filter((pair) => Array.isArray(pair) && pair.length >= 2)
    .map((pair) => `${pair[0]}=${pair[1]}`)
    .join(", ");
  if (!pairs) return "";
  if (pairs.length <= JOIN_LABEL_MAX) return pairs;
  return `${pairs.slice(0, JOIN_LABEL_MAX - 3)}...`;
};

const formatKgJoinLabel = (edge: KgEdgeSpec) => {
  const pairs = (edge.join_on || [])
    .filter((pair) => Array.isArray(pair) && pair.length >= 2)
    .map((pair) => `${pair[0]}=${pair[1]}`)
    .join(", ");
  if (!pairs) return edge.name || "";
  if (pairs.length <= JOIN_LABEL_MAX) return pairs;
  return `${pairs.slice(0, JOIN_LABEL_MAX - 3)}...`;
};

const isGraphReasoner = (reasoner?: RegistryReasoner | null) => reasoner?.type === "graph_reasoner";

const graphFromRegistry = (payload: RegistryPayload): { nodes: GraphNode[]; edges: GraphEdge[] } => {
  const registry = normalizeRegistry(payload);
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const edgeIds = new Set<string>();
  const entityMap = new Map<string, RegistryEntity>();
  const derivedRules = registry.derived_rel_rules || [];
  const kgSpec = registry.kg || {};
  const kgNodesRaw = Array.isArray(kgSpec.nodes) ? (kgSpec.nodes as KgNodeSpec[]) : [];
  const kgEdgesRaw = Array.isArray(kgSpec.edges) ? (kgSpec.edges as KgEdgeSpec[]) : [];
  const kgGraphsRaw = Array.isArray((kgSpec as { graphs?: KgGraphSpec[] }).graphs)
    ? ((kgSpec as { graphs?: KgGraphSpec[] }).graphs as KgGraphSpec[])
    : [];
  const graphIds = new Set(
    kgGraphsRaw
      .map((graph) => String(graph?.id || "").trim())
      .filter(Boolean)
  );

  const pushEdge = (edge: GraphEdge) => {
    if (edgeIds.has(edge.id)) return;
    edgeIds.add(edge.id);
    edges.push(edge);
  };

  const entityNodes: GraphNode[] = [];
  const derivedNodes: GraphNode[] = [];
  const reasonerNodes: GraphNode[] = [];
  const kgNodes: GraphNode[] = [];
  const graphNodes: GraphNode[] = [];
  const kgNodeIds = new Map<string, string>();
  const graphNodeIds = new Map<string, string>();
  const reasonerOutputsByField = new Map<string, string[]>();
  const addReasonerOutput = (entityName: string, fieldName: string, reasonerId: string) => {
    const key = `${entityName}::${fieldName}`;
    const list = reasonerOutputsByField.get(key) || [];
    if (!list.includes(reasonerId)) {
      list.push(reasonerId);
      reasonerOutputsByField.set(key, list);
    }
  };

  if (ENABLE_REASONERS) {
    registry.reasoners.forEach((reasoner) => {
      if (isGraphReasoner(reasoner)) return;
      const outputs = reasoner.outputs || [];
      if (!outputs.length) return;
      registry.entities.forEach((entity) => {
        const applicable =
          (entity.applicable_reasoners || []).includes(reasoner.id) ||
          (!!entity.entity_type && entity.entity_type === reasoner.entity_type);
        if (!applicable) return;
        outputs.forEach((output) => {
          const field = (entity.fields || []).find((candidate) => candidate.name === output);
          if (!field || !field.derived) return;
          addReasonerOutput(entity.name, output, reasoner.id);
          pushEdge({
            id: `reasoner-output:${reasoner.id}:${entity.name}:${output}`,
            from: reasonerNodeId(reasoner.id),
            to: derivedNodeId(entity.name, output),
            label: "output",
            type: "reasoner_output",
          });
        });
      });
    });
  }

  registry.entities.forEach((entity) => {
    entityMap.set(entity.name, entity);
    entityNodes.push({
      id: entity.name,
      label: entity.name,
      kind: "entity",
      description: entity.description || "",
      tags: [entity.entity_type || "entity", entity.table || ""].filter(Boolean),
      inputs: entity.join_keys || [],
      outputs: (entity.fields || []).map((f) => f.name),
      x: 0,
      y: 0,
      ref: { type: "entity", entityName: entity.name },
    });

    const derivedFields = (entity.fields || []).filter((field) => field.derived);
    derivedFields
      .filter((field) => field.derived)
      .forEach((field) => {
        const outputReasoners =
          reasonerOutputsByField.get(`${entity.name}::${field.name}`) || [];
        const outputTags = outputReasoners.length ? ["reasoner-output"] : [];
        const hasRule = derivedRules.some(
          (rule) => rule.entity === entity.name && rule.field === field.name
        );
        derivedNodes.push({
          id: derivedNodeId(entity.name, field.name),
          label: field.name,
          kind: "derived",
          description: field.description || "",
          tags: [
            "derived",
            field.default_agg || field.role,
            hasRule ? "rule:ok" : "rule:missing",
            ...outputTags,
          ].filter(Boolean),
          inputs: field.depends_on || [],
          outputs: [field.name],
          x: 0,
          y: 0,
          ref: { type: "derived", entityName: entity.name, fieldName: field.name },
        });
        pushEdge({
          id: `derived:${entity.name}:${field.name}`,
          from: entity.name,
          to: derivedNodeId(entity.name, field.name),
          label: "derived",
          type: "derived",
        });
      });

    derivedFields.forEach((field) => {
      (field.depends_on || []).forEach((dep) => {
        if (!derivedFields.some((candidate) => candidate.name === dep)) return;
        pushEdge({
          id: `dep:${entity.name}:${dep}:${field.name}`,
          from: derivedNodeId(entity.name, dep),
          to: derivedNodeId(entity.name, field.name),
          label: "depends_on",
          type: "dependency",
        });
      });
    });
  });

  if (ENABLE_REASONERS) {
    registry.reasoners.forEach((reasoner) => {
      const graphId = String((reasoner as RegistryReasoner).graph_id || "").trim();
      const graphTag = graphId ? `graph:${graphId}` : "graph:unbound";
      const graphStatusTag = graphId && !graphIds.has(graphId) ? "graph:missing" : "";
      const reasonerTypeTag = isGraphReasoner(reasoner) ? "graph-reasoner" : "legacy-reasoner";
      reasonerNodes.push({
        id: reasonerNodeId(reasoner.id),
        label: reasoner.name || reasoner.id,
        kind: "reasoner",
        description: reasoner.description || "",
        tags: ["reasoner", reasonerTypeTag, reasoner.entity_type, graphTag, graphStatusTag].filter(Boolean),
        inputs: (reasoner.signals || []).map((signal) => signal.metric_field),
        outputs: reasoner.outputs || [],
        x: 0,
        y: 0,
        ref: { type: "reasoner", reasonerId: reasoner.id },
      });
    });
  }

  kgNodesRaw.forEach((node) => {
    if (!node || typeof node !== "object") return;
    const entity = String(node.entity || "").trim();
    if (!entity) return;
    const nodeType = String(node.node_type || "node");
    const keyField = node.key_field;
    const keyFields = Array.isArray(keyField)
      ? keyField.map((entry) => String(entry)).filter(Boolean)
      : keyField
        ? [String(keyField)]
        : [];
    const labelField = node.label_field ? String(node.label_field) : "";
    const props = Array.isArray(node.properties)
      ? node.properties.map((entry) => String(entry)).filter(Boolean)
      : [];
    const nodeId = kgNodeId(entity);
    kgNodeIds.set(entity, nodeId);
    kgNodes.push({
      id: nodeId,
      label: entity,
      kind: "kg",
      description: node.description || `KG node: ${nodeType}`,
      tags: ["kg", nodeType].filter(Boolean),
      inputs: keyFields,
      outputs: [labelField, ...props].filter(Boolean),
      x: 0,
      y: 0,
      ref: { type: "kg", entityName: entity },
    });
  });

  kgEdgesRaw.forEach((edge) => {
    if (!edge || typeof edge !== "object") return;
    const fromEntity = String(edge.from_entity || "").trim();
    const toEntity = String(edge.to_entity || "").trim();
    if (!fromEntity || !toEntity) return;
    const fromId = kgNodeIds.get(fromEntity);
    const toId = kgNodeIds.get(toEntity);
    if (!fromId || !toId) return;
    pushEdge({
      id: `kg:${edge.name || "edge"}:${fromEntity}:${toEntity}`,
      from: fromId,
      to: toId,
      label: formatKgJoinLabel(edge),
      type: "kg",
    });
  });

  const getGraphTargetId = (entityName: string) => {
    if (!entityName) return "";
    return kgNodeIds.get(entityName) || (entityMap.has(entityName) ? entityName : "");
  };

  kgGraphsRaw.forEach((graph) => {
    if (!graph || !graph.id) return;
    const graphId = String(graph.id).trim();
    if (!graphId) return;
    const nodeId = graphNodeId(graphId);
    graphNodeIds.set(graphId, nodeId);
    graphNodes.push({
      id: nodeId,
      label: graphId,
      kind: "graph",
      description: graph.description || "",
      tags: ["graph", graph.directed === false ? "undirected" : "directed"].filter(Boolean),
      inputs: [],
      outputs: [],
      x: 0,
      y: 0,
      ref: { type: "graph", graphId },
    });

    (graph.nodes || []).forEach((node) => {
      if (!node?.entity) return;
      const entityId = getGraphTargetId(String(node.entity || "").trim());
      if (!entityId) return;
      pushEdge({
        id: `graph:${graphId}:node:${node.entity}`,
        from: nodeId,
        to: entityId,
        label: "node",
        type: "graph",
      });
    });

    (graph.edges || []).forEach((edge) => {
      const relation = (edge?.relation || "").toString();
      if (!relation || !relation.includes(".")) return;
      const [fromEntity, relName] = relation.split(".", 2);
      const fromId = getGraphTargetId(String(fromEntity || "").trim());
      if (!fromId) return;
      pushEdge({
        id: `graph:${graphId}:edge:${relation}`,
        from: nodeId,
        to: fromId,
        label: relName || "edge",
        type: "graph",
      });
    });
  });

  if (ENABLE_REASONERS) {
    registry.reasoners.forEach((reasoner) => {
      if (!isGraphReasoner(reasoner)) return;
      const graphId = String(reasoner.graph_id || "").trim();
      if (!graphId) return;
      const graphNode = graphNodeIds.get(graphId);
      if (!graphNode) return;
      pushEdge({
        id: `reasoner-graph:${reasoner.id}:${graphId}`,
        from: reasonerNodeId(reasoner.id),
        to: graphNode,
        label: "uses graph",
        type: "graph",
      });
    });
  }

  registry.relationships.forEach((rel) => {
    if (!entityMap.has(rel.from_entity) || !entityMap.has(rel.to_entity)) return;
    pushEdge({
      id: `rel:${rel.name}:${rel.from_entity}:${rel.to_entity}`,
      from: rel.from_entity,
      to: rel.to_entity,
      label: formatJoinLabel(rel),
      type: "relation",
    });
  });

  if (ENABLE_REASONERS) {
    registry.entities.forEach((entity) => {
      registry.reasoners.forEach((reasoner) => {
        const applicable =
          (entity.applicable_reasoners || []).includes(reasoner.id) ||
          (!!entity.entity_type && entity.entity_type === reasoner.entity_type);
        if (!applicable) return;
        pushEdge({
          id: `reasoner:${entity.name}:${reasoner.id}`,
          from: entity.name,
          to: reasonerNodeId(reasoner.id),
          label: reasoner.id,
          type: "reasoner",
        });
      });
    });

    registry.reasoners.forEach((reasoner) => {
      const signals = reasoner.signals || [];
      if (!signals.length) return;
      registry.entities.forEach((entity) => {
        const applicable =
          (entity.applicable_reasoners || []).includes(reasoner.id) ||
          (!!entity.entity_type && entity.entity_type === reasoner.entity_type);
        if (!applicable) return;
        const derivedFields = (entity.fields || []).filter((field) => field.derived);
        const derivedNames = new Set(derivedFields.map((field) => field.name));
        signals.forEach((signal) => {
          const metric = (signal.metric_field || "").trim();
          if (!metric) return;
          let entityName: string | null = null;
          let fieldName = metric;
          if (metric.includes(".")) {
            const parts = metric.split(".");
            entityName = parts.slice(0, -1).join(".");
            fieldName = parts[parts.length - 1];
          }
          if (entityName && entityName !== entity.name) return;
          if (!derivedNames.has(fieldName)) return;
          pushEdge({
            id: `reasoner-signal:${entity.name}:${fieldName}:${reasoner.id}`,
            from: derivedNodeId(entity.name, fieldName),
            to: reasonerNodeId(reasoner.id),
            label: signal.name || fieldName,
            type: "reasoner_signal",
          });
        });
      });
    });
  }

  nodes.push(
    ...entityNodes,
    ...kgNodes,
    ...graphNodes,
    ...derivedNodes,
    ...(ENABLE_REASONERS ? reasonerNodes : [])
  );
  return { nodes, edges };
};

const layoutNodes = (nodes: GraphNode[], positionMap: Map<string, { x: number; y: number }>) => {
  const order: NodeKind[] = ["entity", "kg", "graph", "derived", "reasoner"];
  const sorted = [...nodes].sort((a, b) => {
    const kindOrder = order.indexOf(a.kind) - order.indexOf(b.kind);
    if (kindOrder !== 0) return kindOrder;
    return a.label.localeCompare(b.label);
  });
  const counters: Record<NodeKind, number> = {
    entity: 0,
    kg: 0,
    graph: 0,
    derived: 0,
    reasoner: 0,
  };
  const xMap: Record<NodeKind, number> = {
    entity: 120,
    kg: 380,
    graph: 640,
    derived: 900,
    reasoner: 1160,
  };
  const startY = 90;
  const gap = 190;

  return sorted.map((node) => {
    const existing = positionMap.get(node.id);
    if (existing) {
      return { ...node, x: existing.x, y: existing.y };
    }
    const idx = counters[node.kind]++;
    return { ...node, x: xMap[node.kind], y: startY + idx * gap };
  });
};

const layoutJoinNodes = (nodes: GraphNode[], edges: GraphEdge[]) => {
  const entityNodes = nodes.filter((node) => node.kind === "entity");
  const entityMap = new Map(entityNodes.map((node) => [node.id, node]));
  const adjacency = new Map<string, Set<string>>();
  entityNodes.forEach((node) => adjacency.set(node.id, new Set()));
  edges
    .filter((edge) => edge.type === "relation")
    .forEach((edge) => {
      if (!adjacency.has(edge.from) || !adjacency.has(edge.to)) return;
      adjacency.get(edge.from)!.add(edge.to);
      adjacency.get(edge.to)!.add(edge.from);
    });

  const placed = new Map<string, { x: number; y: number }>();
  const visited = new Set<string>();
  const startX = 120;
  const xGap = 320;
  const yGap = 180;
  const componentGap = 220;
  let offsetY = 90;

  const nodesByDegree = [...entityNodes].sort((a, b) => {
    const degA = (adjacency.get(a.id)?.size || 0);
    const degB = (adjacency.get(b.id)?.size || 0);
    if (degA !== degB) return degB - degA;
    return a.label.localeCompare(b.label);
  });

  nodesByDegree.forEach((start) => {
    if (visited.has(start.id)) return;
    const queue = [start.id];
    const component: string[] = [];
    visited.add(start.id);
    while (queue.length) {
      const current = queue.shift()!;
      component.push(current);
      (adjacency.get(current) || new Set()).forEach((neighbor) => {
        if (visited.has(neighbor)) return;
        visited.add(neighbor);
        queue.push(neighbor);
      });
    }

    const root = component
      .map((id) => ({
        id,
        degree: adjacency.get(id)?.size || 0,
        label: entityMap.get(id)?.label || id,
      }))
      .sort((a, b) => (b.degree - a.degree) || a.label.localeCompare(b.label))[0];

    const depthMap = new Map<string, number>();
    const bfs = [root.id];
    depthMap.set(root.id, 0);
    while (bfs.length) {
      const current = bfs.shift()!;
      const depth = depthMap.get(current) || 0;
      (adjacency.get(current) || new Set()).forEach((neighbor) => {
        if (depthMap.has(neighbor)) return;
        depthMap.set(neighbor, depth + 1);
        bfs.push(neighbor);
      });
    }
    component.forEach((id) => {
      if (!depthMap.has(id)) depthMap.set(id, 0);
    });

    const depthGroups = new Map<number, string[]>();
    depthMap.forEach((depth, id) => {
      const group = depthGroups.get(depth) || [];
      group.push(id);
      depthGroups.set(depth, group);
    });

    let maxY = offsetY;
    Array.from(depthGroups.keys())
      .sort((a, b) => a - b)
      .forEach((depth) => {
        const group = depthGroups.get(depth) || [];
        group.sort((a, b) => {
          const labelA = entityMap.get(a)?.label || a;
          const labelB = entityMap.get(b)?.label || b;
          return labelA.localeCompare(labelB);
        });
        group.forEach((id, index) => {
          const x = startX + depth * xGap;
          const y = offsetY + index * yGap;
          placed.set(id, { x, y });
          if (y > maxY) {
            maxY = y;
          }
        });
      });
    offsetY = maxY + componentGap;
  });

  return nodes.map((node) => {
    if (node.kind === "entity") {
      const pos = placed.get(node.id);
      if (pos) return { ...node, x: pos.x, y: pos.y };
    }
    return node;
  });
};

const mergeNodePositions = (nodes: GraphNode[], updates: GraphNode[]) => {
  const pos = new Map(updates.map((node) => [node.id, { x: node.x, y: node.y }]));
  return nodes.map((node) => {
    const next = pos.get(node.id);
    if (!next) return node;
    if (node.x === next.x && node.y === next.y) return node;
    return { ...node, x: next.x, y: next.y };
  });
};

const uniqueName = (base: string, existing: Set<string>) => {
  let idx = 1;
  let candidate = base;
  while (existing.has(candidate)) {
    idx += 1;
    candidate = `${base}_${idx}`;
  }
  return candidate;
};

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const joinPairsToText = (pairs: Array<[string, string]>) =>
  pairs.map(([left, right]) => `${left}=${right}`).join("\n");

const textToJoinPairs = (text: string) =>
  text
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(/[=,:]/).map((part) => part.trim()))
    .filter((parts) => parts.length >= 2 && parts[0] && parts[1])
    .map((parts) => [parts[0], parts[1]] as [string, string]);

const SQL_KEYWORDS = new Set(
  [
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
  ].map((token) => token.toLowerCase())
);

const extractDependencies = (expr: string, fieldNames: string[], exclude?: string) => {
  const tokens = [
    ...(expr.match(/[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+/g) || []),
    ...(expr.match(/[A-Za-z_][A-Za-z0-9_]*/g) || []),
  ];
  const fields = new Set(fieldNames.map((name) => name.toLowerCase()));
  const excludeKey = (exclude || "").toLowerCase();
  const deps = new Set<string>();
  tokens.forEach((token) => {
    const key = token.split(".").pop()?.toLowerCase() || "";
    if (!fields.has(key)) return;
    if (key === excludeKey) return;
    if (SQL_KEYWORDS.has(key)) return;
    deps.add(key);
  });
  return Array.from(deps);
};

const DTYPE_PRESETS = [
  { label: "number", value: "number" },
  { label: "string", value: "varchar" },
  { label: "date", value: "date" },
  { label: "timestamp", value: "timestamp" },
  { label: "boolean", value: "boolean" },
];

const AGG_PRESETS = ["", "sum", "avg", "min", "max", "last"];

const mergeDependencies = (
  deps: string[],
  orderBy: Array<{ field: string; direction?: "asc" | "desc" }> | undefined
) => {
  const orderDeps = (orderBy || []).map((item) => item.field).filter(Boolean);
  return Array.from(new Set([...deps, ...orderDeps]));
};

const getDerivedRule = (payload: RegistryPayload | null, entityName: string, fieldName: string) => {
  if (!payload) return null;
  return (
    (payload.derived_rel_rules || []).find(
    (rule) => rule.entity === entityName && rule.field === fieldName
  ) || null
  );
};

const upsertDerivedRule = (payload: RegistryPayload, rule: DerivedRuleSpec) => {
  const list = payload.derived_rel_rules || [];
  const next = list.filter(
    (item) => !(item.entity === rule.entity && item.field === rule.field)
  );
  next.push(rule);
  return next;
};

const summarizeRuleClauses = (rule: DerivedRuleSpec | null) => {
  if (!rule || !Array.isArray(rule.rules)) return "";
  const parts = rule.rules
    .map((clause) => {
      const expr = String(clause.expr || "").trim();
      if (!expr) return "";
      const when = String(clause.when || "").trim();
      return when ? `${when} -> ${expr}` : expr;
    })
    .filter(Boolean);
  const summary = parts.join("; ");
  if (!summary) return "";
  return summary.length > 200 ? `${summary.slice(0, 197)}...` : summary;
};

const fieldExpressionPreview = (
  payload: RegistryPayload | null,
  entityName: string,
  field: RegistryField
) => {
  const direct = String(field.expr || "").trim();
  if (direct) return direct;
  const rule = getDerivedRule(payload, entityName, field.name);
  const summary = summarizeRuleClauses(rule);
  return summary ? `rule: ${summary}` : "No expression.";
};

const buildRuleFromExpression = (entity: string, field: string, expr: string): DerivedRuleSpec => ({
  entity,
  field,
  rules: [{ expr }],
});

const extractDependenciesFromRule = (
  rule: DerivedRuleSpec,
  fieldNames: string[],
  exclude?: string
) => {
  const combined = (rule.rules || [])
    .map((clause) => [clause.expr, clause.when].filter(Boolean).join(" "))
    .join(" ");
  return extractDependencies(combined, fieldNames, exclude);
};

const inferRole = (dtype: string) => {
  const value = (dtype || "").toLowerCase();
  if (["char", "text", "string", "varchar", "date", "time", "timestamp", "boolean"].some((t) => value.includes(t))) {
    return "dimension";
  }
  return "metric";
};

const inferDefaultAgg = (fieldName: string, dtype: string) => {
  const name = (fieldName || "").toLowerCase();
  const type = (dtype || "").toLowerCase();
  if (["percent", "pct", "ratio", "rate", "avg", "mean", "score", "trend"].some((t) => name.includes(t))) {
    return "avg";
  }
  if (["float", "number", "decimal", "numeric", "int"].some((t) => type.includes(t))) {
    return "sum";
  }
  return "";
};

const buildQualifiedTable = (entity: RegistryEntity) => {
  const parts = [entity.database, entity.schema, entity.table].filter(Boolean);
  return parts.join(".");
};

export default function VisualEditorPage() {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const canvasShellRef = useRef<HTMLDivElement | null>(null);
  const apiUrl = useMemo(() => {
    const envUrl = process.env.NEXT_PUBLIC_API_URL;
    if (envUrl) return envUrl;
    if (typeof window !== "undefined") return window.location.origin;
    return "http://localhost:8000";
  }, []);

  const [registry, setRegistry] = useState<RegistryPayload | null>(null);
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [linkFrom, setLinkFrom] = useState<string | null>(null);
  const [dragging, setDragging] = useState<{ id: string; offsetX: number; offsetY: number } | null>(null);
  const [canvasMode, setCanvasMode] = useState<CanvasMode>("data");
  const [visibleKinds, setVisibleKinds] = useState<Record<NodeKind, boolean>>(DEFAULT_VISIBLE_KINDS);
  const [tablesInput, setTablesInput] = useState("");
  const [draftNotes, setDraftNotes] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [registryPath, setRegistryPath] = useState("");
  const [kgDraft, setKgDraft] = useState("");
  const [kgStatus, setKgStatus] = useState("");
  const [yamlPreview, setYamlPreview] = useState("");
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [panning, setPanning] = useState<{ startX: number; startY: number; originX: number; originY: number } | null>(
    null
  );
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [logicDraft, setLogicDraft] = useState("");
  const [logicDependsOn, setLogicDependsOn] = useState("");
  const [ruleJsonDraft, setRuleJsonDraft] = useState("");
  const [ruleInstruction, setRuleInstruction] = useState("");
  const [ruleHelperStatus, setRuleHelperStatus] = useState("");
  const [ruleValidationIssues, setRuleValidationIssues] = useState<string[]>([]);
  const [ruleSqlDraft, setRuleSqlDraft] = useState("");
  const [ruleSqlStatus, setRuleSqlStatus] = useState("");
  const [showPromptTemplates, setShowPromptTemplates] = useState(false);
  const [importJsonDraft, setImportJsonDraft] = useState("");
  const [reasonerSignalsDraft, setReasonerSignalsDraft] = useState<ReasonerSignal[]>([]);
  const [reasonerPlanDraft, setReasonerPlanDraft] = useState("");
  const [reasonerPlanInstruction, setReasonerPlanInstruction] = useState("");
  const [reasonerPlanStatus, setReasonerPlanStatus] = useState("");
  const [graphDescriptionDraft, setGraphDescriptionDraft] = useState("");
  const [graphDirectedDraft, setGraphDirectedDraft] = useState(true);
  const [graphWeightedDraft, setGraphWeightedDraft] = useState(false);
  const [graphNodesDraft, setGraphNodesDraft] = useState("");
  const [graphEdgesDraft, setGraphEdgesDraft] = useState("");
  const [graphDraftStatus, setGraphDraftStatus] = useState("");
  const [reasonerSanityStepId, setReasonerSanityStepId] = useState("");
  const [reasonerSanityWindowStart, setReasonerSanityWindowStart] = useState("");
  const [reasonerSanityWindowEnd, setReasonerSanityWindowEnd] = useState("");
  const [reasonerSanityLimit, setReasonerSanityLimit] = useState("5");
  const [reasonerSanityStatus, setReasonerSanityStatus] = useState("");
  const [reasonerSanityResult, setReasonerSanityResult] = useState("");
  const [relationshipDraft, setRelationshipDraft] = useState<RelationshipDraft | null>(null);
  const [fieldView, setFieldView] = useState<"base" | "derived" | "all">("base");
  const [joinView, setJoinView] = useState(false);
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(true);
  const [versions, setVersions] = useState<RegistryVersion[]>([]);
  const [draftRegistry, setDraftRegistry] = useState<RegistryPayload | null>(null);
  const [mergeMode, setMergeMode] = useState<"add" | "replace">("add");
  const [prevFieldSelection, setPrevFieldSelection] = useState<Record<string, string[]>>({});
  const [focusNodeId, setFocusNodeId] = useState<string | null>(null);
  const [mergeSelection, setMergeSelection] = useState<{
    entities: Record<string, boolean>;
    relationships: Record<string, boolean>;
    reasoners: Record<string, boolean>;
  }>({ entities: {}, relationships: {}, reasoners: {} });
  const lastSavedRef = useRef<string>("");
  const contentSize = useMemo(() => {
    if (!nodes.length) {
      return { width: 1600, height: 900 };
    }
    const maxX = Math.max(...nodes.map((node) => node.x + NODE_WIDTH));
    const maxY = Math.max(...nodes.map((node) => node.y + NODE_HEIGHT));
    const padding = 420;
    return {
      width: Math.max(maxX + padding, 1600),
      height: Math.max(maxY + padding, 900),
    };
  }, [nodes]);

  const selectedNode = useMemo(
    () => nodes.find((node) => node.id === selectedId) || null,
    [nodes, selectedId]
  );
  const focusNode = useMemo(() => nodes.find((node) => node.id === focusNodeId) || null, [focusNodeId, nodes]);

  const registryLookup = useMemo(() => {
    const safe = normalizeRegistry(registry);
    return {
      entities: safe.entities,
      reasoners: safe.reasoners,
      relationships: safe.relationships,
      entityMap: new Map(safe.entities.map((entity) => [entity.name, entity])),
      reasonerMap: new Map(safe.reasoners.map((reasoner) => [reasoner.id, reasoner])),
      kgNodes: (safe.kg?.nodes as KgNodeSpec[]) || [],
      kgEdges: (safe.kg?.edges as KgEdgeSpec[]) || [],
      kgGraphs: (safe.kg?.graphs as KgGraphSpec[]) || [],
      kgNodeMap: new Map(
        ((safe.kg?.nodes as KgNodeSpec[]) || [])
          .filter((node) => node && typeof node === "object" && node.entity)
          .map((node) => [String(node.entity), node])
      ),
    };
  }, [registry]);

  useEffect(() => {
    if (!registry) return;
    setKgDraft(JSON.stringify(registry.kg || { nodes: [], edges: [], graphs: [] }, null, 2));
  }, [registry?.kg]);

  const promptTemplates = useMemo(() => registry?.prompt_templates || {}, [registry]);
  const promptTemplateKeys = useMemo(() => [...PROMPT_TEMPLATE_KEYS].sort(), []);

  const promptContext = useMemo(() => {
    const cfg = registry?.analysis_config || {};
    const ctx = (cfg as Record<string, unknown>).prompt_context as Record<string, unknown> | undefined;
    return {
      business: String(ctx?.business_context ?? ctx?.business ?? ""),
      data: String(ctx?.data_context ?? ctx?.data ?? ""),
      expectations: String(ctx?.expectations ?? ""),
    };
  }, [registry]);

  const applyKgDraft = useCallback(() => {
    if (!registry) return;
    try {
      const parsed = JSON.parse(kgDraft || "{}");
      if (!parsed || typeof parsed !== "object") {
        setKgStatus("KG config must be a JSON object.");
        return;
      }
      setRegistry({ ...registry, kg: parsed });
      setKgStatus("KG config applied.");
    } catch (err: any) {
      setKgStatus(err?.message || "Failed to parse KG JSON.");
    }
  }, [kgDraft, registry]);

  const applyGraphDraft = useCallback(() => {
    if (!registry || !selectedNode?.ref.graphId) return;
    let nodesParsed: KgGraphNodeSpec[] = [];
    let edgesParsed: KgGraphEdgeSpec[] = [];
    try {
      const rawNodes = graphNodesDraft.trim() ? JSON.parse(graphNodesDraft) : [];
      const rawEdges = graphEdgesDraft.trim() ? JSON.parse(graphEdgesDraft) : [];
      if (!Array.isArray(rawNodes) || !Array.isArray(rawEdges)) {
        setGraphDraftStatus("Graph nodes/edges must be JSON arrays.");
        return;
      }
      nodesParsed = rawNodes;
      edgesParsed = rawEdges;
    } catch (err: any) {
      setGraphDraftStatus(err?.message || "Invalid graph JSON.");
      return;
    }
    const kg = registry.kg || { nodes: [], edges: [], graphs: [] };
    const graphs = Array.isArray(kg.graphs) ? kg.graphs : [];
    const nextGraphs = graphs.map((graph) =>
      graph.id === selectedNode.ref.graphId
        ? {
            ...graph,
            description: graphDescriptionDraft,
            directed: graphDirectedDraft,
            weighted: graphWeightedDraft,
            nodes: nodesParsed,
            edges: edgesParsed,
          }
        : graph
    );
    setRegistry({ ...registry, kg: { ...kg, graphs: nextGraphs } });
    setGraphDraftStatus("Graph config applied.");
  }, [
    registry,
    selectedNode,
    graphNodesDraft,
    graphEdgesDraft,
    graphDescriptionDraft,
    graphDirectedDraft,
    graphWeightedDraft,
  ]);

  const updatePromptContext = useCallback(
    (field: "business" | "data" | "expectations", value: string) => {
      if (!registry) return;
      const nextAnalysis = { ...(registry.analysis_config || {}) } as Record<string, unknown>;
      const current = (nextAnalysis.prompt_context as Record<string, unknown> | undefined) || {};
      const nextContext = {
        business_context: String(current.business_context ?? current.business ?? ""),
        data_context: String(current.data_context ?? current.data ?? ""),
        expectations: String(current.expectations ?? ""),
      };
      if (field === "business") nextContext.business_context = value;
      if (field === "data") nextContext.data_context = value;
      if (field === "expectations") nextContext.expectations = value;
      nextAnalysis.prompt_context = nextContext;
      setRegistry({ ...registry, analysis_config: nextAnalysis });
    },
    [registry]
  );

  const removePromptTemplate = useCallback(
    (key: string) => {
      if (!registry) return;
      const nextTemplates = { ...(registry.prompt_templates || {}) };
      if (!(key in nextTemplates)) return;
      delete nextTemplates[key];
      setRegistry({ ...registry, prompt_templates: nextTemplates });
    },
    [registry]
  );

  const resetPromptTemplates = useCallback(() => {
    if (!registry) return;
    const nextTemplates = { ...(registry.prompt_templates || {}) };
    PROMPT_TEMPLATE_KEYS.forEach((key) => {
      delete nextTemplates[key];
    });
    setRegistry({ ...registry, prompt_templates: nextTemplates });
    setStatus("Prompt templates reset to defaults.");
  }, [registry]);

  const generatePromptTemplates = useCallback(async () => {
    if (!registry) return;
    const hasContext = !!(
      promptContext.business.trim() ||
      promptContext.data.trim() ||
      promptContext.expectations.trim()
    );
    if (!hasContext) {
      setStatus("Add business/data/expectations to generate prompts.");
      return;
    }
    setBusy(true);
    setStatus("Generating prompt templates...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/prompts/customize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          business_context: promptContext.business,
          data_context: promptContext.data,
          expectations: promptContext.expectations,
          prompt_keys: PROMPT_TEMPLATE_KEYS,
        }),
      });
      if (!resp.ok) {
        setStatus("Prompt generation failed.");
        return;
      }
      const data = await resp.json();
      const templates = data.templates || {};
      const nextTemplates = { ...(registry.prompt_templates || {}), ...templates };
      setRegistry({ ...registry, prompt_templates: nextTemplates });
      setStatus("Prompt templates generated.");
    } catch (err) {
      setStatus("Prompt generation failed.");
    } finally {
      setBusy(false);
    }
  }, [apiUrl, promptContext.business, promptContext.data, promptContext.expectations, registry]);

  const generateKgConfig = useCallback(async () => {
    if (!registry) return;
    setKgStatus("Generating KG config...");
    setBusy(true);
    try {
      const resp = await fetch(`${apiUrl}/api/registry/kg/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ registry, instructions: draftNotes }),
      });
      if (!resp.ok) {
        setKgStatus("KG generation failed.");
        return;
      }
      const data = await resp.json();
      const kg = data.kg || {};
      setRegistry({ ...registry, kg });
      setKgDraft(JSON.stringify(kg, null, 2));
      setKgStatus("KG config generated.");
    } catch (err: any) {
      setKgStatus(err?.message || "KG generation failed.");
    } finally {
      setBusy(false);
    }
  }, [apiUrl, registry, draftNotes]);


  const draftDiff = useMemo(() => {
    if (!draftRegistry) {
      return { entities: [], relationships: [], reasoners: [] };
    }
    const current = normalizeRegistry(registry);
    const compare = (a: object, b: object) => JSON.stringify(a) !== JSON.stringify(b);
    const toMap = <T extends { name?: string; id?: string }>(items: T[], key: "name" | "id") =>
      new Map(items.map((item) => [String(item[key] || ""), item]));
    const entityMap = toMap(current.entities, "name");
    const relMap = toMap(current.relationships as RegistryRelationship[], "name");
    const reasonerMap = toMap(current.reasoners, "id");
    const entityDiff = (draftRegistry.entities || []).map((entity) => ({
      name: entity.name,
      status: entityMap.has(entity.name) ? (compare(entityMap.get(entity.name)!, entity) ? "update" : "same") : "add",
    }));
    const relDiff = (draftRegistry.relationships || []).map((rel) => ({
      name: rel.name,
      status: relMap.has(rel.name) ? (compare(relMap.get(rel.name)!, rel) ? "update" : "same") : "add",
    }));
    const reasonerDiff = (draftRegistry.reasoners || []).map((reasoner) => ({
      name: reasoner.id,
      status: reasonerMap.has(reasoner.id)
        ? (compare(reasonerMap.get(reasoner.id)!, reasoner) ? "update" : "same")
        : "add",
    }));
    return { entities: entityDiff, relationships: relDiff, reasoners: reasonerDiff };
  }, [draftRegistry, registry]);

  const joinEntityIds = useMemo(() => {
    const ids = new Set<string>();
    (registry?.relationships || []).forEach((rel) => {
      ids.add(rel.from_entity);
      ids.add(rel.to_entity);
    });
    return ids;
  }, [registry]);

  const modeKinds = useMemo(() => CANVAS_MODE_KINDS[canvasMode], [canvasMode]);
  const isJoinView = canvasMode === "data" && joinView;
  const setKindsVisible = useCallback((kinds: NodeKind[]) => {
    setVisibleKinds((prev) => {
      const next = { ...prev };
      (Object.keys(next) as NodeKind[]).forEach((kind) => {
        next[kind] = false;
      });
      kinds.forEach((kind) => {
        next[kind] = true;
      });
      return next;
    });
  }, []);

  useEffect(() => {
    setKindsVisible(modeKinds);
    if (canvasMode === "kg") {
      setJoinView(false);
    }
  }, [canvasMode, modeKinds, setKindsVisible]);

  const focusGraph = useMemo(() => {
    if (!focusNodeId) return null;
    if (!nodes.some((node) => node.id === focusNodeId)) return null;
    const forward = new Map<string, Set<string>>();
    const backward = new Map<string, Set<string>>();
    const addEdge = (from: string, to: string) => {
      if (!forward.has(from)) forward.set(from, new Set());
      if (!backward.has(to)) backward.set(to, new Set());
      forward.get(from)!.add(to);
      backward.get(to)!.add(from);
    };
    edges.forEach((edge) => {
      addEdge(edge.from, edge.to);
      if (edge.type === "relation") {
        addEdge(edge.to, edge.from);
      }
    });
    const walk = (start: string, graph: Map<string, Set<string>>) => {
      const visited = new Set<string>();
      const stack = [start];
      visited.add(start);
      while (stack.length) {
        const current = stack.pop()!;
        const next = graph.get(current);
        if (!next) continue;
        next.forEach((node) => {
          if (visited.has(node)) return;
          visited.add(node);
          stack.push(node);
        });
      }
      return visited;
    };
    const downstream = walk(focusNodeId, forward);
    const upstream = walk(focusNodeId, backward);
    const nodeSet = new Set<string>([...(downstream || []), ...(upstream || [])]);
    return { nodes: nodeSet };
  }, [edges, focusNodeId, nodes]);

  const filteredNodes = useMemo(() => {
    let baseNodes: GraphNode[];
    if (isJoinView) {
      const entities = nodes.filter((node) => node.kind === "entity");
      baseNodes = joinEntityIds.size === 0 ? entities : entities.filter((node) => joinEntityIds.has(node.id));
    } else {
      baseNodes = nodes.filter((node) => visibleKinds[node.kind]);
    }
    if (canvasMode === "kg") {
      baseNodes = baseNodes.filter(
        (node) => node.kind !== "reasoner" || node.tags.includes("graph-reasoner")
      );
    }
    if (focusGraph) {
      return baseNodes.filter((node) => focusGraph.nodes.has(node.id));
    }
    return baseNodes;
  }, [canvasMode, focusGraph, isJoinView, joinEntityIds, nodes, visibleKinds]);

  const visibleNodeIds = useMemo(() => new Set(filteredNodes.map((node) => node.id)), [filteredNodes]);
  const activeKinds = useMemo(
    () => modeKinds.filter((kind) => visibleKinds[kind]),
    [modeKinds, visibleKinds]
  );
  const allKindsVisible = useMemo(
    () => activeKinds.length === modeKinds.length && modeKinds.length > 0,
    [activeKinds.length, modeKinds.length]
  );
  const activeKindLabel = useMemo(() => {
    if (allKindsVisible) return "all";
    return activeKinds.join(", ");
  }, [activeKinds, allKindsVisible]);

  const validationIssues = useMemo<ValidationIssue[]>(() => {
    if (!registry) return [];
    const issues: ValidationIssue[] = [];
    const entityMap = new Map(registry.entities.map((entity) => [entity.name, entity]));
    const reasonerIds = new Set(registry.reasoners.map((reasoner) => reasoner.id));
    registry.entities.forEach((entity) => {
      const fieldNames = new Set(entity.fields.map((field) => field.name));
      entity.fields.forEach((field) => {
        const dtype = (field.dtype || "").toLowerCase();
        if (!dtype || dtype === "any" || dtype === "unknown") {
          issues.push({
            level: "warn",
            message: `Missing dtype for ${entity.name}.${field.name}`,
          });
        }
        if (field.derived) {
          (field.depends_on || []).forEach((dep) => {
            if (!fieldNames.has(dep)) {
              issues.push({
                level: "error",
                message: `Missing dependency ${dep} for ${entity.name}.${field.name}`,
              });
            }
          });
        }
        if (field.default_agg === "last") {
          const orderBy = field.order_by || [];
          if (!orderBy.length) {
            issues.push({
              level: "warn",
              message: `Missing order_by for ${entity.name}.${field.name} (last agg)`,
            });
          }
          orderBy.forEach((entry) => {
            if (entry.field && !fieldNames.has(entry.field)) {
              issues.push({
                level: "error",
                message: `Invalid order_by ${entry.field} for ${entity.name}.${field.name}`,
              });
            }
          });
        }
      });
    });
    registry.relationships.forEach((rel) => {
      const fromEntity = entityMap.get(rel.from_entity);
      const toEntity = entityMap.get(rel.to_entity);
      if (!fromEntity || !toEntity) {
        issues.push({
          level: "error",
          message: `Relationship ${rel.name} references missing entity`,
        });
        return;
      }
      if (!rel.join_on || rel.join_on.length === 0) {
        issues.push({
          level: "error",
          message: `Relationship ${rel.name} is missing join_on pairs`,
        });
        return;
      }
      const fromFields = new Set(fromEntity.fields.map((field) => field.name));
      const toFields = new Set(toEntity.fields.map((field) => field.name));
      (rel.join_on || []).forEach(([left, right]) => {
        if (!fromFields.has(left) || !toFields.has(right)) {
          issues.push({
            level: "error",
            message: `Invalid join ${rel.name}: ${left}=${right}`,
          });
        }
      });
    });
    const entityTypes = new Set(registry.entities.map((entity) => entity.entity_type));
    registry.reasoners.forEach((reasoner) => {
      if (isGraphReasoner(reasoner)) {
        if (!reasoner.graph_id) {
          issues.push({
            level: "warn",
            message: `Graph reasoner ${reasoner.id} missing graph_id`,
          });
        }
        return;
      }
      const referenced =
        registry.entities.some((entity) =>
          (entity.applicable_reasoners || []).includes(reasoner.id)
        ) || entityTypes.has(reasoner.entity_type);
      if (!referenced) {
        issues.push({
          level: "warn",
          message: `Orphan reasoner ${reasoner.id}`,
        });
      }
    });
    registry.entities.forEach((entity) => {
      const hasRel = registry.relationships.some(
        (rel) => rel.from_entity === entity.name || rel.to_entity === entity.name
      );
      const hasReasoner = (entity.applicable_reasoners || []).length > 0;
      if (!hasRel && !hasReasoner) {
        issues.push({
          level: "warn",
          message: `Unused entity ${entity.name}`,
        });
      }
    });
    return issues;
  }, [registry]);

  const dependencyHighlight = useMemo(() => {
    if (!selectedNode || selectedNode.kind !== "derived") return null;
    const depEdges = edges.filter((edge) => edge.type === "dependency");
    const upstream = new Set<string>();
    const downstream = new Set<string>();
    const edgeSet = new Set<string>();
    const forward = new Map<string, string[]>();
    const backward = new Map<string, string[]>();
    depEdges.forEach((edge) => {
      forward.set(edge.from, [...(forward.get(edge.from) || []), edge.to]);
      backward.set(edge.to, [...(backward.get(edge.to) || []), edge.from]);
    });
    const walk = (start: string, graph: Map<string, string[]>, collector: Set<string>) => {
      const stack = [start];
      while (stack.length) {
        const current = stack.pop()!;
        const next = graph.get(current) || [];
        next.forEach((node) => {
          if (collector.has(node)) return;
          collector.add(node);
          stack.push(node);
        });
      }
    };
    walk(selectedNode.id, forward, downstream);
    walk(selectedNode.id, backward, upstream);
    depEdges.forEach((edge) => {
      if (
        (edge.from === selectedNode.id || upstream.has(edge.from) || downstream.has(edge.from)) &&
        (edge.to === selectedNode.id || upstream.has(edge.to) || downstream.has(edge.to))
      ) {
        edgeSet.add(edge.id);
      }
    });
    const nodeSet = new Set<string>([selectedNode.id, ...upstream, ...downstream]);
    return { nodes: nodeSet, edges: edgeSet };
  }, [edges, selectedNode]);

  const fitToNodes = useCallback((nodeList: GraphNode[]) => {
    const canvas = canvasRef.current;
    if (!canvas || nodeList.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const minX = Math.min(...nodeList.map((node) => node.x));
    const minY = Math.min(...nodeList.map((node) => node.y));
    const maxX = Math.max(...nodeList.map((node) => node.x + NODE_WIDTH));
    const maxY = Math.max(...nodeList.map((node) => node.y + NODE_HEIGHT));
    const width = Math.max(1, maxX - minX);
    const height = Math.max(1, maxY - minY);
    const padding = 120;
    const nextZoom = clamp(
      Math.min((rect.width - padding) / width, (rect.height - padding) / height),
      0.4,
      1.6
    );
    const nextPanX = (rect.width - width * nextZoom) / 2 - minX * nextZoom;
    const nextPanY = (rect.height - height * nextZoom) / 2 - minY * nextZoom;
    setZoom(nextZoom);
    setPan({ x: nextPanX, y: nextPanY });
  }, []);

  const applyRegistry = (payload: RegistryPayload, preservePositions = true) => {
    const normalized = normalizeRegistry(payload);
    const { nodes: nextNodes, edges: nextEdges } = graphFromRegistry(normalized);
    const positionMap = preservePositions
      ? new Map(nodes.map((node) => [node.id, { x: node.x, y: node.y }]))
      : new Map();
    const laidOut = layoutNodes(nextNodes, positionMap);
    setRegistry(normalized);
    setNodes(laidOut);
    setEdges(nextEdges);
    if (!preservePositions) {
      requestAnimationFrame(() => fitToNodes(laidOut));
    }
    if (!laidOut.some((node) => node.id === selectedId)) {
      setSelectedId(laidOut[0]?.id || null);
    }
  };

  const applyImportedRegistry = useCallback(() => {
    if (!importJsonDraft.trim()) {
      setStatus("Paste registry JSON to import.");
      return;
    }
    try {
      const parsed = JSON.parse(importJsonDraft);
      if (!parsed || typeof parsed !== "object") {
        setStatus("Registry JSON is invalid.");
        return;
      }
      applyRegistry(parsed as RegistryPayload, false);
      setDraftRegistry(null);
      setStatus("Registry imported from JSON.");
    } catch {
      setStatus("Registry JSON is invalid.");
    }
  }, [importJsonDraft]);

  const loadRegistry = async () => {
    setBusy(true);
    setStatus("Loading registry...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/load`);
      if (!resp.ok) {
        setStatus("Failed to load registry.");
        return;
      }
      const data = await resp.json();
      setRegistryPath(data.path || "");
      if (data.registry) {
        applyRegistry(data.registry, false);
        const source = data.source || (data.exists ? "file" : "active");
        setStatus(`Registry loaded (${source}).`);
        setDraftRegistry(null);
        lastSavedRef.current = JSON.stringify(data.registry);
      } else {
        setRegistry(null);
        setNodes([]);
        setEdges([]);
        setStatus("No registry data available yet.");
      }
      await loadVersions();
    } catch (err) {
      setStatus("Registry load failed.");
    } finally {
      setBusy(false);
    }
  };

  const loadVersions = async () => {
    try {
      const resp = await fetch(`${apiUrl}/api/registry/versions`);
      if (!resp.ok) return;
      const data = await resp.json();
      setVersions(data.versions || []);
    } catch {
      // ignore
    }
  };

  const reloadApi = async () => {
    setBusy(true);
    setStatus("Reloading API registry...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/reload`, { method: "POST" });
      if (!resp.ok) {
        setStatus("Reload failed.");
        return;
      }
      setStatus("API registry reloaded.");
    } catch (err) {
      setStatus("Reload failed.");
    } finally {
      setBusy(false);
    }
  };

  const rebuildKg = async () => {
    setBusy(true);
    setStatus("Rebuilding KG...");
    try {
      const resp = await fetch(`${apiUrl}/api/kg/rebuild`, { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) {
        setStatus(data?.detail || "KG rebuild failed.");
        return;
      }
      setStatus(`KG rebuilt (entities=${data?.entities ?? 0}, reasoners=${data?.reasoners ?? 0}).`);
    } catch (err) {
      setStatus("KG rebuild failed.");
    } finally {
      setBusy(false);
    }
  };

  const startFresh = () => {
    const emptyRegistry: RegistryPayload = {
      entities: [],
      relationships: [],
      reasoners: [],
      derived_rel_rules: [],
      prompt_templates: {},
      analysis_config: {},
      config: {},
    };
    setRegistry(emptyRegistry);
    setNodes([]);
    setEdges([]);
    setSelectedId(null);
    setYamlPreview("");
    setDraftRegistry(null);
    setStatus("Started fresh registry.");
  };

  const draftRegistryFromSnowflake = async () => {
    const tables = tablesInput
      .split(/[\n,]+/)
      .map((t) => t.trim())
      .filter(Boolean);
    if (!tables.length) {
      setStatus("Enter at least one Snowflake table.");
      return;
    }
    setBusy(true);
    setStatus("Drafting registry from Snowflake...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/draft`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tables, instructions: draftNotes }),
      });
      if (!resp.ok) {
        setStatus("Draft failed.");
        return;
      }
      const data = await resp.json();
      if (data.registry) {
        const hasExisting = !!(registry?.entities?.length || registry?.relationships?.length || registry?.reasoners?.length);
        if (hasExisting) {
          setDraftRegistry(data.registry);
          setStatus("Draft ready for merge.");
        } else {
          applyRegistry(data.registry, false);
          setStatus("Draft loaded into canvas.");
        }
      } else {
        setStatus("Draft returned empty registry.");
      }
    } catch (err) {
      setStatus("Draft failed.");
    } finally {
      setBusy(false);
    }
  };

  const saveRegistry = async (silent = false) => {
    if (!registry) {
      if (!silent) setStatus("Nothing to save yet.");
      return;
    }
    if (!silent) {
      setBusy(true);
      setStatus("Saving registry...");
    }
    try {
      const resp = await fetch(`${apiUrl}/api/registry/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ registry }),
      });
      if (!resp.ok) {
        if (!silent) setStatus("Save failed.");
        return;
      }
      const data = await resp.json();
      setRegistryPath(data.path || "");
      setYamlPreview(data.yaml || "");
      lastSavedRef.current = JSON.stringify(registry);
      if (!silent) setStatus("Registry saved.");
      await loadVersions();
      if (!silent) {
        await reloadApi();
      }
    } catch (err) {
      if (!silent) setStatus("Save failed.");
    } finally {
      if (!silent) setBusy(false);
    }
  };

  useEffect(() => {
    loadRegistry();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiUrl]);

  useEffect(() => {
    if (!autoSaveEnabled || !registry) return;
    const payload = JSON.stringify(registry);
    if (payload === lastSavedRef.current) return;
    const timer = window.setTimeout(async () => {
      lastSavedRef.current = payload;
      await saveRegistry(true);
      setStatus("Auto-saved.");
    }, 1500);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoSaveEnabled, registry]);

  useEffect(() => {
    if (!nodes.length) {
      setSelectedId(null);
      return;
    }
    if (!selectedId || !nodes.find((node) => node.id === selectedId)) {
      setSelectedId(nodes[0].id);
    }
  }, [nodes, selectedId]);

  useEffect(() => {
    if (!draftRegistry) return;
    const entitySelections: Record<string, boolean> = {};
    draftDiff.entities.forEach((entry) => {
      entitySelections[entry.name] = entry.status !== "same";
    });
    const relSelections: Record<string, boolean> = {};
    draftDiff.relationships.forEach((entry) => {
      relSelections[entry.name] = entry.status !== "same";
    });
    const reasonerSelections: Record<string, boolean> = {};
    draftDiff.reasoners.forEach((entry) => {
      reasonerSelections[entry.name] = entry.status !== "same";
    });
    setMergeSelection({
      entities: entitySelections,
      relationships: relSelections,
      reasoners: reasonerSelections,
    });
  }, [draftDiff, draftRegistry]);

  useEffect(() => {
    if (!registry || !selectedNode) {
      setLogicDraft("");
      setLogicDependsOn("");
      setReasonerSignalsDraft([]);
      setReasonerPlanDraft("");
      setReasonerPlanInstruction("");
      setReasonerPlanStatus("");
      setRuleJsonDraft("");
      setRuleValidationIssues([]);
      setRuleHelperStatus("");
      setRuleSqlDraft("");
      setRuleSqlStatus("");
      setRuleInstruction("");
      return;
    }
    if (selectedNode.kind === "derived") {
      const entity = registry.entities.find((e) => e.name === selectedNode.ref.entityName);
      const field = entity?.fields.find((f) => f.name === selectedNode.ref.fieldName);
      setLogicDraft(field?.expr || "");
      setLogicDependsOn((field?.depends_on || []).join(", "));
      const rule = getDerivedRule(registry, selectedNode.ref.entityName || "", selectedNode.ref.fieldName || "");
      setRuleJsonDraft(rule ? JSON.stringify(rule, null, 2) : "");
      setRuleValidationIssues([]);
      setRuleHelperStatus("");
      setRuleSqlDraft("");
      setRuleSqlStatus("");
      setRuleInstruction("");
      setReasonerSignalsDraft([]);
      setReasonerPlanDraft("");
      setReasonerPlanInstruction("");
      setReasonerPlanStatus("");
      return;
    }
    if (ENABLE_REASONERS && selectedNode.kind === "reasoner") {
      const reasoner = registry.reasoners.find((r) => r.id === selectedNode.ref.reasonerId);
      setReasonerSignalsDraft(reasoner?.signals || []);
      setReasonerPlanDraft(
        reasoner?.drilldown_plan ? JSON.stringify(reasoner.drilldown_plan, null, 2) : ""
      );
      setReasonerPlanInstruction("");
      setReasonerPlanStatus("");
      setLogicDraft("");
      setLogicDependsOn("");
      setRuleJsonDraft("");
      setRuleValidationIssues([]);
      setRuleHelperStatus("");
      setRuleSqlDraft("");
      setRuleSqlStatus("");
      setRuleInstruction("");
      setGraphDescriptionDraft("");
      setGraphDirectedDraft(true);
      setGraphWeightedDraft(false);
      setGraphNodesDraft("");
      setGraphEdgesDraft("");
      setGraphDraftStatus("");
      return;
    }
    if (selectedNode.kind === "graph") {
      const graphs = (registry.kg?.graphs as KgGraphSpec[]) || [];
      const graph = graphs.find((entry) => entry.id === selectedNode.ref.graphId);
      setGraphDescriptionDraft(graph?.description || "");
      setGraphDirectedDraft(graph?.directed !== false);
      setGraphWeightedDraft(!!graph?.weighted);
      setGraphNodesDraft(graph?.nodes ? JSON.stringify(graph.nodes, null, 2) : "[]");
      setGraphEdgesDraft(graph?.edges ? JSON.stringify(graph.edges, null, 2) : "[]");
      setGraphDraftStatus("");
      setLogicDraft("");
      setLogicDependsOn("");
      setReasonerSignalsDraft([]);
      setReasonerPlanDraft("");
      setReasonerPlanInstruction("");
      setReasonerPlanStatus("");
      setRuleJsonDraft("");
      setRuleValidationIssues([]);
      setRuleHelperStatus("");
      setRuleSqlDraft("");
      setRuleSqlStatus("");
      setRuleInstruction("");
      return;
    }
    setLogicDraft("");
    setLogicDependsOn("");
    setReasonerSignalsDraft([]);
    setReasonerPlanDraft("");
    setReasonerPlanInstruction("");
    setReasonerPlanStatus("");
    setGraphDescriptionDraft("");
    setGraphDirectedDraft(true);
    setGraphWeightedDraft(false);
    setGraphNodesDraft("");
    setGraphEdgesDraft("");
    setGraphDraftStatus("");
    setRuleJsonDraft("");
    setRuleValidationIssues([]);
    setRuleHelperStatus("");
    setRuleSqlDraft("");
    setRuleSqlStatus("");
    setRuleInstruction("");
  }, [registry, selectedNode]);

  const toGraphCoords = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left - pan.x) / zoom,
      y: (clientY - rect.top - pan.y) / zoom,
    };
  };

  const handleCanvasPointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest("[data-node]")) return;
    setPanning({
      startX: event.clientX,
      startY: event.clientY,
      originX: pan.x,
      originY: pan.y,
    });
  };

  const handlePointerDown = (event: React.PointerEvent, nodeId: string) => {
    event.stopPropagation();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const point = toGraphCoords(event.clientX, event.clientY);
    const offsetX = point.x - node.x;
    const offsetY = point.y - node.y;
    setDragging({ id: nodeId, offsetX, offsetY });
  };

  useEffect(() => {
    if (!dragging) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleMove = (event: PointerEvent) => {
      const point = toGraphCoords(event.clientX, event.clientY);
      const nextX = point.x - dragging.offsetX;
      const nextY = point.y - dragging.offsetY;
      setNodes((prev) =>
        prev.map((node) =>
          node.id === dragging.id
            ? {
                ...node,
                x: Math.max(24, nextX),
                y: Math.max(24, nextY),
              }
            : node
        )
      );
    };

    const handleUp = () => setDragging(null);

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
  }, [dragging]);

  useEffect(() => {
    if (!panning) return;
    const handleMove = (event: PointerEvent) => {
      setPan({
        x: panning.originX + (event.clientX - panning.startX),
        y: panning.originY + (event.clientY - panning.startY),
      });
    };
    const handleUp = () => setPanning(null);
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
  }, [panning]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === canvasShellRef.current);
      requestAnimationFrame(() => fitToNodes(nodes));
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
    };
  }, [fitToNodes, nodes]);

  const edgePaths = useMemo(() => {
    const fromGroups = new Map<string, GraphEdge[]>();
    edges.forEach((edge) => {
      const list = fromGroups.get(edge.from) || [];
      list.push(edge);
      fromGroups.set(edge.from, list);
    });
    const offsetMap = new Map<string, number>();
    fromGroups.forEach((group) => {
      const mid = (group.length - 1) / 2;
      group.forEach((edge, index) => {
        offsetMap.set(edge.id, (index - mid) * 10);
      });
    });

    const baseEdges = isJoinView ? edges.filter((edge) => edge.type === "relation") : edges;
    const visibleEdges = isJoinView || allKindsVisible
      ? baseEdges.filter((edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to))
      : baseEdges.filter((edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to));
    return visibleEdges
      .map((edge) => {
        const from = nodes.find((n) => n.id === edge.from);
        const to = nodes.find((n) => n.id === edge.to);
        if (!from || !to) return null;
        const startY = from.y + NODE_HEIGHT / 2;
        const endY = to.y + NODE_HEIGHT / 2;
        const fromCenterX = from.x + NODE_WIDTH / 2;
        const toCenterX = to.x + NODE_WIDTH / 2;
        const dx = toCenterX - fromCenterX;
        const sameColumn = Math.abs(dx) < NODE_WIDTH * 0.6;
        const laneOffset = offsetMap.get(edge.id) || 0;

        let startX = 0;
        let endX = 0;
        let controlX1 = 0;
        let controlX2 = 0;

        if (sameColumn) {
          startX = from.x + NODE_WIDTH + EDGE_PADDING;
          endX = to.x + NODE_WIDTH + EDGE_PADDING;
          const anchorX = Math.max(from.x, to.x) + NODE_WIDTH + 140 + laneOffset;
          controlX1 = anchorX;
          controlX2 = anchorX;
        } else if (dx > 0) {
          startX = from.x + NODE_WIDTH + EDGE_PADDING;
          endX = to.x - EDGE_PADDING;
          const spread = Math.min(160, Math.abs(dx) * 0.4);
          controlX1 = startX + spread + laneOffset;
          controlX2 = endX - spread + laneOffset;
        } else {
          startX = from.x - EDGE_PADDING;
          endX = to.x + NODE_WIDTH + EDGE_PADDING;
          const spread = Math.min(160, Math.abs(dx) * 0.4);
          controlX1 = startX - spread + laneOffset;
          controlX2 = endX + spread + laneOffset;
        }

        const path = `M ${startX} ${startY} C ${controlX1} ${startY}, ${controlX2} ${endY}, ${endX} ${endY}`;
        const labelX = (startX + endX) / 2;
        const labelY = (startY + endY) / 2 - 12;
        return { ...edge, path, labelX, labelY };
      })
      .filter(Boolean) as Array<GraphEdge & { path: string; labelX: number; labelY: number }>;
  }, [allKindsVisible, edges, isJoinView, nodes, visibleNodeIds]);

  const startLink = (nodeId: string) => {
    setLinkFrom(nodeId);
  };

  const clearFocus = () => setFocusNodeId(null);

  const handleCanvasDoubleClick = (event: React.MouseEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement | null;
    if (target?.closest('[data-node="true"]')) return;
    clearFocus();
  };

  const handleNodeClick = (nodeId: string, event?: React.MouseEvent) => {
    if (event?.detail === 2) {
      handleNodeFocus(nodeId);
      return;
    }
    if (event?.shiftKey) {
      handleNodeFocus(nodeId);
      return;
    }
    if (linkFrom && linkFrom !== nodeId) {
      const fromNode = nodes.find((node) => node.id === linkFrom);
      const toNode = nodes.find((node) => node.id === nodeId);
      const isEntityLink = fromNode?.kind === "entity" && toNode?.kind === "entity";
      if (isEntityLink && registry) {
        const fromEntity = registryLookup.entityMap.get(fromNode!.id);
        const toEntity = registryLookup.entityMap.get(toNode!.id);
        const joinKeys = (fromEntity?.join_keys || []).filter((key) =>
          (toEntity?.join_keys || []).includes(key)
        );
        const nameBase = `${fromNode!.id}_to_${toNode!.id}`;
        setRelationshipDraft({
          from: fromNode!.id,
          to: toNode!.id,
          name: nameBase,
          description: "User defined relationship",
          joinText: joinPairsToText(joinKeys.map((key) => [key, key])),
          mode: "create",
        });
      } else {
        setEdges((prev) => [
          ...prev,
          {
            id: `edge-${Date.now()}`,
            from: linkFrom,
            to: nodeId,
            label: "custom_link",
            type: "relation",
          },
        ]);
      }
      setLinkFrom(null);
    }
    setSelectedId(nodeId);
  };

  const handleNodeFocus = (nodeId: string) => {
    setFocusNodeId((prev) => (prev === nodeId ? null : nodeId));
  };

  const updateSelected = (patch: Partial<GraphNode>) => {
    if (!selectedNode) return;
    setNodes((prev) => prev.map((node) => (node.id === selectedNode.id ? { ...node, ...patch } : node)));
  };

  const updateDescription = (node: GraphNode, value: string) => {
    updateSelected({ description: value });
    if (!registry) return;
    if (node.kind === "entity" && node.ref.entityName) {
      const next = {
        ...registry,
        entities: registry.entities.map((entity) =>
          entity.name === node.ref.entityName ? { ...entity, description: value } : entity
        ),
      };
      setRegistry(next);
      return;
    }
    if (node.kind === "derived" && node.ref.entityName && node.ref.fieldName) {
      const next = {
        ...registry,
        entities: registry.entities.map((entity) => {
          if (entity.name !== node.ref.entityName) return entity;
          return {
            ...entity,
            fields: entity.fields.map((field) =>
              field.name === node.ref.fieldName ? { ...field, description: value } : field
            ),
          };
        }),
      };
      setRegistry(next);
      return;
    }
    if (ENABLE_REASONERS && node.kind === "reasoner" && node.ref.reasonerId) {
      const next = {
        ...registry,
        reasoners: registry.reasoners.map((reasoner) =>
          reasoner.id === node.ref.reasonerId ? { ...reasoner, description: value } : reasoner
        ),
      };
      setRegistry(next);
      return;
    }
    if (node.kind === "graph" && node.ref.graphId) {
      const kg = registry.kg || { nodes: [], edges: [], graphs: [] };
      const graphs = Array.isArray(kg.graphs) ? kg.graphs : [];
      const nextGraphs = graphs.map((graph) =>
        graph.id === node.ref.graphId ? { ...graph, description: value } : graph
      );
      setRegistry({ ...registry, kg: { ...kg, graphs: nextGraphs } });
    }
  };

  const renameEntity = (node: GraphNode, newLabel: string) => {
    if (!registry || !node.ref.entityName) return;
    const trimmed = newLabel.trim();
    if (!trimmed) return;
    const oldName = node.ref.entityName;
    if (trimmed === oldName) {
      updateSelected({ label: trimmed });
      return;
    }
    const idMap = new Map<string, string>();
    idMap.set(oldName, trimmed);
    const updatedNodes = nodes.map((n) => {
      if (n.ref.entityName !== oldName) return n;
      const nextRef = { ...n.ref, entityName: trimmed };
      if (n.kind === "entity") {
        return { ...n, id: trimmed, label: trimmed, ref: nextRef };
      }
      if (n.kind === "derived" && n.ref.fieldName) {
        const nextId = derivedNodeId(trimmed, n.ref.fieldName);
        idMap.set(n.id, nextId);
        return { ...n, id: nextId, ref: nextRef };
      }
      return { ...n, ref: nextRef };
    });
    const updatedEdges = edges.map((edge) => ({
      ...edge,
      from: idMap.get(edge.from) || edge.from,
      to: idMap.get(edge.to) || edge.to,
    }));
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((entity) =>
        entity.name === oldName ? { ...entity, name: trimmed } : entity
      ),
      relationships: registry.relationships.map((rel) => ({
        ...rel,
        from_entity: rel.from_entity === oldName ? trimmed : rel.from_entity,
        to_entity: rel.to_entity === oldName ? trimmed : rel.to_entity,
      })),
    };
    setRegistry(nextRegistry);
    setNodes(updatedNodes);
    setEdges(updatedEdges);
    setSelectedId(trimmed);
  };

  const renameDerived = (node: GraphNode, newLabel: string) => {
    if (!registry || !node.ref.entityName || !node.ref.fieldName) return;
    const trimmed = newLabel.trim();
    if (!trimmed) return;
    const oldField = node.ref.fieldName;
    if (trimmed === oldField) {
      updateSelected({ label: trimmed });
      return;
    }
    const updatedNodes = nodes.map((n) => {
      if (n.id !== node.id) return n;
      return {
        ...n,
        id: derivedNodeId(node.ref.entityName!, trimmed),
        label: trimmed,
        ref: { ...n.ref, fieldName: trimmed },
      };
    });
    const updatedEdges = edges.map((edge) => {
      if (edge.from !== node.id && edge.to !== node.id) return edge;
      return {
        ...edge,
        from: edge.from === node.id ? derivedNodeId(node.ref.entityName!, trimmed) : edge.from,
        to: edge.to === node.id ? derivedNodeId(node.ref.entityName!, trimmed) : edge.to,
      };
    });
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((entity) => {
        if (entity.name !== node.ref.entityName) return entity;
        return {
          ...entity,
          default_metric: entity.default_metric === oldField ? trimmed : entity.default_metric,
          fields: entity.fields.map((field) => {
            const dependsOn = (field.depends_on || []).map((dep) => (dep === oldField ? trimmed : dep));
            if (field.name !== oldField) {
              return dependsOn === field.depends_on ? field : { ...field, depends_on: dependsOn };
            }
            const nextField = { ...field, name: trimmed, depends_on: dependsOn };
            if (field.expr === oldField) {
              nextField.expr = trimmed;
            }
            return nextField;
          }),
        };
      }),
    };
    setRegistry(nextRegistry);
    setNodes(updatedNodes);
    setEdges(updatedEdges);
    setSelectedId(derivedNodeId(node.ref.entityName, trimmed));
  };

  const renameReasoner = (node: GraphNode, newLabel: string) => {
    if (!registry || !node.ref.reasonerId) return;
    const trimmed = newLabel.trim();
    if (!trimmed) return;
    updateSelected({ label: trimmed });
    const nextRegistry: RegistryPayload = {
      ...registry,
      reasoners: registry.reasoners.map((reasoner) =>
        reasoner.id === node.ref.reasonerId ? { ...reasoner, name: trimmed } : reasoner
      ),
    };
    setRegistry(nextRegistry);
  };

  const renameGraph = (node: GraphNode, newLabel: string) => {
    if (!registry || !node.ref.graphId) return;
    const trimmed = newLabel.trim();
    if (!trimmed) return;
    updateSelected({ label: trimmed });
    const kg = registry.kg || { nodes: [], edges: [], graphs: [] };
    const graphs = Array.isArray(kg.graphs) ? kg.graphs : [];
    const nextGraphs = graphs.map((graph) =>
      graph.id === node.ref.graphId ? { ...graph, id: trimmed } : graph
    );
    const nextReasoners = registry.reasoners.map((reasoner) =>
      (reasoner as any).graph_id === node.ref.graphId ? { ...reasoner, graph_id: trimmed } : reasoner
    );
    const nextRegistry: RegistryPayload = {
      ...registry,
      reasoners: nextReasoners,
      kg: { ...kg, graphs: nextGraphs },
    };
    applyRegistry(nextRegistry, true);
    setSelectedId(graphNodeId(trimmed));
  };

  const handleLabelChange = (value: string) => {
    if (!selectedNode) return;
    if (selectedNode.kind === "kg") return;
    if (selectedNode.kind === "graph") {
      renameGraph(selectedNode, value);
      return;
    }
    if (selectedNode.kind === "entity") {
      renameEntity(selectedNode, value);
      return;
    }
    if (selectedNode.kind === "derived") {
      renameDerived(selectedNode, value);
      return;
    }
    renameReasoner(selectedNode, value);
  };

  const addNode = (kind: NodeKind) => {
    if (!registry) {
      setStatus("Load or draft a registry first.");
      return;
    }
    if (kind === "kg") {
      setStatus("KG nodes are generated from the KG config.");
      return;
    }
    if (kind === "entity") {
      const existing = new Set(registry.entities.map((entity) => entity.name));
      const name = uniqueName("NewEntity", existing);
      const nextRegistry: RegistryPayload = {
        ...registry,
        entities: [
          ...registry.entities,
          {
            name,
            description: "Describe this entity.",
            database: "",
            schema: "",
            table: "",
            fields: [],
            join_keys: [],
            default_metric: "",
            entity_type: "generic",
            applicable_reasoners: [],
            applicable_derived_metrics: [],
          },
        ],
      };
      applyRegistry(nextRegistry, true);
      setSelectedId(name);
      return;
    }
    if (kind === "derived") {
      const targetEntity = selectedNode?.ref.entityName || registry.entities[0]?.name;
      if (!targetEntity) return;
      const entity = registry.entities.find((e) => e.name === targetEntity);
      if (!entity) return;
      const existing = new Set(entity.fields.map((field) => field.name));
      const fieldName = uniqueName("derived_metric", existing);
      const nextRegistry: RegistryPayload = {
        ...registry,
        entities: registry.entities.map((entitySpec) => {
          if (entitySpec.name !== targetEntity) return entitySpec;
          return {
            ...entitySpec,
            fields: [
              ...entitySpec.fields,
              {
                name: fieldName,
                dtype: "number",
                role: "metric",
                expr: "",
                description: "Describe the derived metric.",
                derived: true,
                default_agg: "avg",
                depends_on: [],
              },
            ],
          };
        }),
      };
      applyRegistry(nextRegistry, true);
      setSelectedId(derivedNodeId(targetEntity, fieldName));
      return;
    }
    if (kind === "graph") {
      const kg = registry.kg || { nodes: [], edges: [], graphs: [] };
      const graphs = Array.isArray(kg.graphs) ? kg.graphs : [];
      const existing = new Set(graphs.map((g) => g.id));
      const graphId = uniqueName("graph", existing);
      const nextRegistry: RegistryPayload = {
        ...registry,
        kg: {
          ...kg,
          graphs: [
            ...graphs,
            {
              id: graphId,
              description: "Describe this graph reasoner.",
              directed: true,
              weighted: false,
              nodes: [],
              edges: [],
            },
          ],
        },
      };
      applyRegistry(nextRegistry, true);
      setSelectedId(graphNodeId(graphId));
      return;
    }
    const existing = new Set(registry.reasoners.map((reasoner) => reasoner.id));
    const reasonerId = uniqueName("reasoner", existing);
    const nextRegistry: RegistryPayload = {
      ...registry,
      reasoners: [
        ...registry.reasoners,
        {
          id: reasonerId,
          name: "New Reasoner",
          description: "Describe this reasoner.",
          entity_type: "generic",
          type: "signal_based",
          signals: [],
          outputs: [],
          drilldown_plan: { steps: [] },
        },
      ],
    };
    applyRegistry(nextRegistry, true);
    setSelectedId(reasonerNodeId(reasonerId));
  };

  const applyLayout = () => {
    const visibleIds = new Set(filteredNodes.map((node) => node.id));
    const visibleEdges = edges.filter((edge) => visibleIds.has(edge.from) && visibleIds.has(edge.to));
    if (isJoinView) {
      const laidOut = layoutJoinNodes(filteredNodes, visibleEdges);
      setNodes((prev) => mergeNodePositions(prev, laidOut));
      return;
    }
    const positionMap = new Map<string, { x: number; y: number }>();
    const laidOut = layoutNodes(filteredNodes, positionMap);
    setNodes((prev) => mergeNodePositions(prev, laidOut));
  };

  const zoomIn = () => setZoom((prev) => clamp(prev + 0.1, 0.4, 2));
  const zoomOut = () => setZoom((prev) => clamp(prev - 0.1, 0.4, 2));
  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const fitToView = () => fitToNodes(filteredNodes.length ? filteredNodes : nodes);

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const delta = -event.deltaY;
    const nextZoom = clamp(zoom + delta * 0.0015, 0.4, 2);
    const canvas = canvasRef.current;
    if (!canvas) {
      setZoom(nextZoom);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const cx = event.clientX - rect.left;
    const cy = event.clientY - rect.top;
    const worldX = (cx - pan.x) / zoom;
    const worldY = (cy - pan.y) / zoom;
    const nextPanX = cx - worldX * nextZoom;
    const nextPanY = cy - worldY * nextZoom;
    setZoom(nextZoom);
    setPan({ x: nextPanX, y: nextPanY });
  };

  const toggleFullscreen = async () => {
    const target = canvasShellRef.current;
    if (!target) return;
    try {
      if (document.fullscreenElement === target) {
        await document.exitFullscreen();
      } else {
        await target.requestFullscreen();
      }
    } catch {
      setStatus("Fullscreen request failed.");
    }
  };

  const applyLogic = () => {
    if (!registry || !selectedNode) return;
    if (selectedNode.kind === "derived") {
      if (!selectedNode.ref.entityName || !selectedNode.ref.fieldName) return;
      const entity = registry.entities.find((e) => e.name === selectedNode.ref.entityName);
      const targetField = entity?.fields.find((field) => field.name === selectedNode.ref.fieldName);
      const fieldNames = (entity?.fields || []).map((field) => field.name);
      const ruleSpec = buildRuleFromExpression(
        selectedNode.ref.entityName,
        selectedNode.ref.fieldName,
        logicDraft || ""
      );
      const dependsOn = mergeDependencies(
        extractDependenciesFromRule(ruleSpec, fieldNames, selectedNode.ref.fieldName),
        targetField?.order_by
      );
      const nextRegistry: RegistryPayload = {
        ...registry,
        derived_rel_rules: upsertDerivedRule(registry, ruleSpec),
        entities: registry.entities.map((entity) => {
          if (entity.name !== selectedNode.ref.entityName) return entity;
          return {
            ...entity,
            fields: entity.fields.map((field) =>
              field.name === selectedNode.ref.fieldName
                ? { ...field, expr: logicDraft, depends_on: dependsOn }
                : field
            ),
          };
        }),
      };
      setRegistry(nextRegistry);
      setLogicDependsOn(dependsOn.join(", "));
      setRuleJsonDraft(JSON.stringify(ruleSpec, null, 2));
      setRuleSqlDraft("");
      setRuleSqlStatus("");
      setStatus("Derived rule updated.");
      return;
    }
    if (ENABLE_REASONERS && selectedNode.kind === "reasoner") {
      if (!selectedNode.ref.reasonerId) return;
      const cleaned = reasonerSignalsDraft.filter((signal) => signal.name && signal.metric_field);
      const nextRegistry: RegistryPayload = {
        ...registry,
        reasoners: registry.reasoners.map((reasoner) =>
          reasoner.id === selectedNode.ref.reasonerId ? { ...reasoner, signals: cleaned } : reasoner
        ),
      };
      setRegistry(nextRegistry);
      setStatus("Reasoner signals updated.");
    }
  };

  const applyReasonerPlanJson = () => {
    if (!ENABLE_REASONERS) return;
    if (!registry || !selectedNode?.ref.reasonerId) return;
    if (selectedNode.kind !== "reasoner") return;
    if (!reasonerPlanDraft.trim()) {
      setReasonerPlanStatus("Drilldown plan JSON is empty.");
      return;
    }
    try {
      const parsed = JSON.parse(reasonerPlanDraft);
      updateSelectedReasoner({ drilldown_plan: parsed });
      setReasonerPlanStatus("Drilldown plan saved.");
    } catch {
      setReasonerPlanStatus("Drilldown plan JSON is invalid.");
    }
  };

  const generateReasonerPlanWithLLM = async () => {
    if (!registry || !selectedNode?.ref.reasonerId) return;
    if (selectedNode.kind !== "reasoner") return;
    const reasoner = registry.reasoners.find((r) => r.id === selectedNode.ref.reasonerId);
    if (!reasoner) return;
    const instruction = reasonerPlanInstruction.trim();
    setBusy(true);
    setReasonerPlanStatus("Generating drilldown plan...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/reasoners/drilldown-plan`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reasoner_id: reasoner.id,
          name: reasoner.name,
          description: reasoner.description,
          entity_type: reasoner.entity_type,
          outputs: reasoner.outputs || [],
          signals: reasoner.signals || [],
          instruction,
          registry,
        }),
      });
      if (!resp.ok) {
        setReasonerPlanStatus("Plan generation failed.");
        return;
      }
      const data = await resp.json();
      if (data?.drilldown_plan) {
        setReasonerPlanDraft(JSON.stringify(data.drilldown_plan, null, 2));
        setReasonerPlanStatus("Plan generated. Review and apply.");
      } else {
        setReasonerPlanStatus("No plan generated.");
      }
    } catch {
      setReasonerPlanStatus("Plan generation failed.");
    } finally {
      setBusy(false);
    }
  };

  const syncRuleJsonFromExpression = () => {
    if (!registry || !selectedNode?.ref.entityName || !selectedNode?.ref.fieldName) return;
    if (selectedNode.kind !== "derived") return;
    const ruleSpec = buildRuleFromExpression(
      selectedNode.ref.entityName,
      selectedNode.ref.fieldName,
      logicDraft || ""
    );
    setRuleJsonDraft(JSON.stringify(ruleSpec, null, 2));
    setRuleValidationIssues([]);
    setRuleHelperStatus("Rule JSON synced from expression.");
    setRuleSqlDraft("");
    setRuleSqlStatus("");
  };

  const applyRuleJson = () => {
    if (!registry || !selectedNode?.ref.entityName || !selectedNode?.ref.fieldName) return;
    if (selectedNode.kind !== "derived") return;
    let parsed: DerivedRuleSpec | null = null;
    try {
      parsed = JSON.parse(ruleJsonDraft || "");
    } catch {
      setStatus("Rule JSON is invalid.");
      return;
    }
    if (!parsed || typeof parsed !== "object") {
      setStatus("Rule JSON is invalid.");
      return;
    }
    const ruleSpec: DerivedRuleSpec = {
      entity: parsed.entity || selectedNode.ref.entityName,
      field: parsed.field || selectedNode.ref.fieldName,
      rules: parsed.rules || [],
      vars: parsed.vars || {},
      description: parsed.description || "",
    };
    if (!ruleSpec.rules || ruleSpec.rules.length === 0) {
      setStatus("Rule JSON must include at least one rule.");
      return;
    }
    const entity = registry.entities.find((e) => e.name === ruleSpec.entity);
    const fieldNames = (entity?.fields || []).map((field) => field.name);
    const targetField = entity?.fields.find((field) => field.name === ruleSpec.field);
    const dependsOn = mergeDependencies(
      extractDependenciesFromRule(ruleSpec, fieldNames, ruleSpec.field),
      targetField?.order_by
    );
    const exprPreview =
      ruleSpec.rules.length === 1 && !ruleSpec.rules[0].when
        ? ruleSpec.rules[0].expr
        : logicDraft;
    const nextRegistry: RegistryPayload = {
      ...registry,
      derived_rel_rules: upsertDerivedRule(registry, ruleSpec),
      entities: registry.entities.map((entity) => {
        if (entity.name !== ruleSpec.entity) return entity;
        return {
          ...entity,
          fields: entity.fields.map((field) =>
            field.name === ruleSpec.field
              ? { ...field, expr: exprPreview, depends_on: dependsOn }
              : field
          ),
        };
      }),
    };
    setRegistry(nextRegistry);
    setLogicDraft(exprPreview || "");
    setLogicDependsOn(dependsOn.join(", "));
    setRuleJsonDraft(JSON.stringify(ruleSpec, null, 2));
    setRuleValidationIssues([]);
    setRuleHelperStatus("Rule JSON applied.");
    setRuleSqlDraft("");
    setRuleSqlStatus("");
    setStatus("Derived rule updated.");
  };

  const generateRuleWithLLM = async () => {
    if (!registry || !selectedNode?.ref.entityName || !selectedNode?.ref.fieldName) return;
    if (selectedNode.kind !== "derived") return;
    if (!ruleInstruction.trim()) {
      setStatus("Enter a natural-language rule description.");
      return;
    }
    setBusy(true);
    setRuleHelperStatus("Generating rule with LLM...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/rules/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entity: selectedNode.ref.entityName,
          field: selectedNode.ref.fieldName,
          instruction: ruleInstruction,
          registry,
        }),
      });
      if (!resp.ok) {
        setRuleHelperStatus("Rule generation failed.");
        return;
      }
      const data = await resp.json();
      if (data.rule) {
        setRuleJsonDraft(JSON.stringify(data.rule, null, 2));
        setRuleHelperStatus("Rule generated. Review and apply.");
        setRuleSqlDraft("");
        setRuleSqlStatus("");
      } else {
        setRuleHelperStatus("No rule generated.");
      }
    } catch {
      setRuleHelperStatus("Rule generation failed.");
    } finally {
      setBusy(false);
    }
  };

  const validateRuleJson = async () => {
    if (!ruleJsonDraft.trim()) {
      setRuleValidationIssues(["Rule JSON is empty."]);
      return;
    }
    let parsed: DerivedRuleSpec | null = null;
    try {
      parsed = JSON.parse(ruleJsonDraft || "");
    } catch {
      setRuleValidationIssues(["Rule JSON is invalid."]);
      return;
    }
    setBusy(true);
    setRuleHelperStatus("Validating rule...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/rules/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rule: parsed, registry }),
      });
      if (!resp.ok) {
        setRuleHelperStatus("Validation failed.");
        return;
      }
      const data = await resp.json();
      const issues = Array.isArray(data.issues) ? data.issues : [];
      setRuleValidationIssues(issues);
      setRuleHelperStatus(data.valid ? "Rule is valid." : "Rule has issues.");
    } catch {
      setRuleHelperStatus("Validation failed.");
    } finally {
      setBusy(false);
    }
  };

  const generateRuleSql = async () => {
    if (!registry) return;
    if (!ruleJsonDraft.trim()) {
      setRuleSqlStatus("Rule JSON is empty.");
      return;
    }
    let parsed: DerivedRuleSpec | null = null;
    try {
      parsed = JSON.parse(ruleJsonDraft || "");
    } catch {
      setRuleSqlStatus("Rule JSON is invalid.");
      return;
    }
    setBusy(true);
    setRuleSqlStatus("Generating Snowflake SQL...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/rules/sql`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rule: parsed, registry }),
      });
      if (!resp.ok) {
        setRuleSqlStatus("SQL generation failed.");
        return;
      }
      const data = await resp.json();
      setRuleSqlDraft(data.sql || "");
      setRuleSqlStatus(data.sql ? "SQL generated." : "SQL generation failed.");
    } catch {
      setRuleSqlStatus("SQL generation failed.");
    } finally {
      setBusy(false);
    }
  };

  const updateSelectedEntity = (patch: Partial<RegistryEntity>) => {
    if (!registry || !selectedNode?.ref.entityName) return;
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((entity) =>
        entity.name === selectedNode.ref.entityName ? { ...entity, ...patch } : entity
      ),
    };
    applyRegistry(nextRegistry, true);
  };

  const updateSelectedDerived = (patch: Partial<RegistryField>) => {
    if (!registry || !selectedNode?.ref.entityName || !selectedNode?.ref.fieldName) return;
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((entity) => {
        if (entity.name !== selectedNode.ref.entityName) return entity;
        return {
          ...entity,
          fields: entity.fields.map((field) =>
            field.name === selectedNode.ref.fieldName ? { ...field, ...patch } : field
          ),
        };
      }),
    };
    applyRegistry(nextRegistry, true);
  };

  const updateSelectedReasoner = (patch: Partial<RegistryReasoner>) => {
    if (!registry || !selectedNode?.ref.reasonerId) return;
    const nextRegistry: RegistryPayload = {
      ...registry,
      reasoners: registry.reasoners.map((reasoner) =>
        reasoner.id === selectedNode.ref.reasonerId ? { ...reasoner, ...patch } : reasoner
      ),
    };
    applyRegistry(nextRegistry, true);
  };

  const updateEntityField = (entityName: string, fieldName: string, patch: Partial<RegistryField>) => {
    if (!registry) return;
    const trimmedName = (patch.name || fieldName).trim();
    const rename = patch.name && trimmedName && trimmedName !== fieldName;
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((entity) => {
        if (entity.name !== entityName) return entity;
        const nextFields = entity.fields.map((field) => {
          if (field.name === fieldName) {
            return {
              ...field,
              ...patch,
              name: trimmedName || field.name,
            };
          }
          if (rename) {
            const deps = (field.depends_on || []).map((dep) => (dep === fieldName ? trimmedName : dep));
            return deps === field.depends_on ? field : { ...field, depends_on: deps };
          }
          return field;
        });
        const nextJoinKeys = rename
          ? (entity.join_keys || []).map((key) => (key === fieldName ? trimmedName : key))
          : entity.join_keys;
        const nextDefaultMetric =
          rename && entity.default_metric === fieldName ? trimmedName : entity.default_metric;
        const nextApplicable = rename
          ? (entity.applicable_derived_metrics || []).map((key) => (key === fieldName ? trimmedName : key))
          : entity.applicable_derived_metrics;
        return {
          ...entity,
          fields: nextFields,
          join_keys: nextJoinKeys,
          default_metric: nextDefaultMetric,
          applicable_derived_metrics: nextApplicable,
        };
      }),
      derived_rel_rules: rename
        ? (registry.derived_rel_rules || []).map((rule) =>
            rule.entity === entityName && rule.field === fieldName ? { ...rule, field: trimmedName } : rule
          )
        : registry.derived_rel_rules,
      relationships: rename
        ? registry.relationships.map((rel) => {
            if (rel.from_entity !== entityName && rel.to_entity !== entityName) return rel;
            const nextJoin = (rel.join_on || []).map(
              ([left, right]) =>
                [
                  rel.from_entity === entityName && left === fieldName ? trimmedName : left,
                  rel.to_entity === entityName && right === fieldName ? trimmedName : right,
                ] as [string, string]
            );
            return { ...rel, join_on: nextJoin };
          })
        : registry.relationships,
    };
    applyRegistry(nextRegistry, true);
  };

  const updateFieldOrderBy = (entityName: string, fieldName: string, order_by: RegistryField["order_by"]) => {
    if (!registry) return;
    const entity = registry.entities.find((e) => e.name === entityName);
    const field = entity?.fields.find((f) => f.name === fieldName);
    if (!entity || !field) return;
    let nextDepends = field.depends_on;
    if (field.derived) {
      const fieldNames = entity.fields.map((f) => f.name);
      nextDepends = mergeDependencies(extractDependencies(field.expr || "", fieldNames, field.name), order_by);
    }
    updateEntityField(entityName, fieldName, {
      order_by,
      depends_on: nextDepends,
    });
    if (
      selectedNode?.kind === "derived" &&
      selectedNode.ref.entityName === entityName &&
      selectedNode.ref.fieldName === fieldName &&
      field.derived
    ) {
      setLogicDependsOn((nextDepends || []).join(", "));
    }
  };

  const addEntityField = (entityName: string) => {
    if (!registry) return;
    const entity = registry.entities.find((e) => e.name === entityName);
    if (!entity) return;
    const existing = new Set(entity.fields.map((field) => field.name));
    const name = uniqueName("new_column", existing);
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((e) => {
        if (e.name !== entityName) return e;
        return {
          ...e,
          fields: [
            ...e.fields,
            {
              name,
              dtype: "",
              role: "dimension",
              expr: "",
              description: "",
              derived: false,
              default_agg: "",
              depends_on: [],
            },
          ],
        };
      }),
    };
    applyRegistry(nextRegistry, true);
  };

  const deleteEntityField = (entityName: string, fieldName: string) => {
    if (!registry) return;
    if (!window.confirm(`Delete field "${fieldName}"?`)) return;
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((entity) => {
        if (entity.name !== entityName) return entity;
        return {
          ...entity,
          fields: entity.fields.filter((field) => field.name !== fieldName),
          join_keys: (entity.join_keys || []).filter((key) => key !== fieldName),
          default_metric: entity.default_metric === fieldName ? "" : entity.default_metric,
          applicable_derived_metrics: (entity.applicable_derived_metrics || []).filter((key) => key !== fieldName),
        };
      }),
    };
    applyRegistry(nextRegistry, true);
  };

  const generatePrevFields = (entityName: string, selectedMetrics: string[]) => {
    if (!registry) return;
    const entity = registry.entities.find((e) => e.name === entityName);
    if (!entity) return;
    const fields = entity.fields || [];
    const fieldMap = new Map(fields.map((field) => [field.name, field]));
    const timeKeys = ["posyear", "posmon", "meet_year", "meet_mon", "month_date", "positiondate", "month_date"];
    const orderKeys = timeKeys.filter((key) => fieldMap.has(key));
    const partitionCandidates = ["rmid", "mandateid", "meeting_id", "meetingid"];
    const partitionKeys = partitionCandidates.filter((key) => fieldMap.has(key));
    const partition = partitionKeys.length
      ? partitionKeys
      : (entity.join_keys || []).filter((key) => !orderKeys.includes(key));
    if (!orderKeys.length) {
      setStatus("No time keys found (posyear/posmon/meet_year/meet_mon/month_date).");
      return;
    }
    const metrics = fields.filter(
      (field) => field.role === "metric" && !field.derived && !field.name.startsWith("prev_")
    );
    const selectedSet = new Set(selectedMetrics);
    const targetMetrics = metrics.filter((field) => selectedSet.has(field.name));
    if (!targetMetrics.length) {
      setStatus("Select one or more metrics for prev_ fields.");
      return;
    }
    const additions: RegistryField[] = [];
    for (const metric of targetMetrics) {
      const prevName = `prev_${metric.name}`;
      if (fieldMap.has(prevName)) continue;
      const baseExpr = metric.expr || metric.name.toUpperCase();
      const partitionExprs = partition
        .map((key) => fieldMap.get(key)?.expr || key.toUpperCase())
        .filter(Boolean);
      const orderExprs = orderKeys
        .map((key) => fieldMap.get(key)?.expr || key.toUpperCase())
        .filter(Boolean);
      const partitionClause = partitionExprs.length ? `PARTITION BY ${partitionExprs.join(", ")}` : "";
      const orderClause = orderExprs.length ? `ORDER BY ${orderExprs.join(", ")}` : "";
      const windowClause = [partitionClause, orderClause].filter(Boolean).join(" ");
      additions.push({
        name: prevName,
        dtype: metric.dtype,
        role: "metric",
        expr: `LAG(${baseExpr}) OVER (${windowClause})`,
        description: `Previous ${metric.name} by ${partition.join(", ") || "entity"} over time.`,
        derived: true,
        default_agg: "avg",
        depends_on: [metric.name, ...partition, ...orderKeys],
      });
    }
    if (!additions.length) {
      setStatus("No new prev_ fields to add.");
      return;
    }
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((e) =>
        e.name === entityName ? { ...e, fields: [...e.fields, ...additions] } : e
      ),
    };
    applyRegistry(nextRegistry, true);
    setStatus(`Added ${additions.length} prev_ fields.`);
  };

  const importFieldsForEntity = async (entityName: string) => {
    if (!registry) return;
    const entity = registry.entities.find((e) => e.name === entityName);
    if (!entity) return;
    const table = buildQualifiedTable(entity);
    if (!table) {
      setStatus("Set database/schema/table before importing.");
      return;
    }
    setBusy(true);
    setStatus(`Importing fields from ${table}...`);
    try {
      const resp = await fetch(`${apiUrl}/api/registry/describe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ table }),
      });
      if (!resp.ok) {
        setStatus("Describe failed.");
        return;
      }
      const data = await resp.json();
      const columns = data.columns || [];
      const baseFields: RegistryField[] = columns.map((col: any) => {
        const name = String(col.name || "").toLowerCase();
        const dtype = String(col.type || "");
        const role = inferRole(dtype);
        return {
          name,
          dtype,
          role,
          expr: name,
          description: String(col.comment || ""),
          derived: false,
          default_agg: role === "metric" ? inferDefaultAgg(name, dtype) : "",
          depends_on: [],
        };
      });
      const derivedFields = (entity.fields || []).filter((field) => field.derived);
      const joinCandidates = baseFields
        .map((field) => field.name)
        .filter((name) => name.endsWith("id") || ["rmid", "mandateid", "meeting_id"].includes(name));
      const joinKeys = Array.from(new Set([...(entity.join_keys || []), ...joinCandidates]));
      const metrics = baseFields.filter((field) => field.role === "metric");
      const defaultMetric = metrics.length ? metrics[0].name : entity.default_metric;
      const nextRegistry: RegistryPayload = {
        ...registry,
        entities: registry.entities.map((e) =>
          e.name === entityName
            ? {
                ...e,
                fields: [...baseFields, ...derivedFields],
                join_keys: joinKeys,
                default_metric: defaultMetric || "",
              }
            : e
        ),
      };
      applyRegistry(nextRegistry, true);
      setStatus(`Imported ${baseFields.length} fields.`);
    } catch {
      setStatus("Describe failed.");
    } finally {
      setBusy(false);
    }
  };

  const updateReasonerSignal = (index: number, patch: Partial<ReasonerSignal>) => {
    setReasonerSignalsDraft((prev) =>
      prev.map((signal, idx) => (idx === index ? { ...signal, ...patch } : signal))
    );
  };

  const addReasonerOutput = () => {
    if (!selectedNode?.ref.reasonerId) return;
    const reasoner = registryLookup.reasonerMap.get(selectedNode.ref.reasonerId);
    const outputs = [...(reasoner?.outputs || []), ""];
    updateSelectedReasoner({ outputs });
  };

  const updateReasonerOutput = (index: number, value: string) => {
    if (!selectedNode?.ref.reasonerId) return;
    const reasoner = registryLookup.reasonerMap.get(selectedNode.ref.reasonerId);
    const outputs = [...(reasoner?.outputs || [])];
    outputs[index] = value.trim();
    updateSelectedReasoner({ outputs });
  };

  const removeReasonerOutput = (index: number) => {
    if (!selectedNode?.ref.reasonerId) return;
    const reasoner = registryLookup.reasonerMap.get(selectedNode.ref.reasonerId);
    const outputs = (reasoner?.outputs || []).filter((_, idx) => idx != index);
    updateSelectedReasoner({ outputs });
  };

  const togglePrevMetric = (entityName: string, metricName: string) => {
    setPrevFieldSelection((prev) => {
      const current = new Set(prev[entityName] || []);
      if (current.has(metricName)) {
        current.delete(metricName);
      } else {
        current.add(metricName);
      }
      return { ...prev, [entityName]: Array.from(current) };
    });
  };

  const setPrevMetrics = (entityName: string, metrics: string[]) => {
    setPrevFieldSelection((prev) => ({ ...prev, [entityName]: metrics }));
  };

  const addReasonerSignal = () => {
    setReasonerSignalsDraft((prev) => [
      ...prev,
      { name: "", description: "", metric_field: "", threshold: 0, direction: "below" },
    ]);
  };

  const removeReasonerSignal = (index: number) => {
    setReasonerSignalsDraft((prev) => prev.filter((_, idx) => idx !== index));
  };

  const confirmRelationship = () => {
    if (!registry || !relationshipDraft) return;
    const join_on = textToJoinPairs(relationshipDraft.joinText);
    if (!join_on.length) {
      setStatus("Relationship join keys are required.");
      return;
    }
    const nextRelationships = registry.relationships.map((rel) => ({ ...rel }));
    if (relationshipDraft.mode === "edit" && relationshipDraft.originalName) {
      const idx = nextRelationships.findIndex((rel) => rel.name === relationshipDraft.originalName);
      if (idx >= 0) {
        nextRelationships[idx] = {
          name: relationshipDraft.name || relationshipDraft.originalName,
          from_entity: relationshipDraft.from,
          to_entity: relationshipDraft.to,
          description: relationshipDraft.description || "User defined relationship",
          join_on,
        };
      }
    } else {
      const existing = new Set(nextRelationships.map((rel) => rel.name));
      const relName = uniqueName(relationshipDraft.name || "relationship", existing);
      nextRelationships.push({
        name: relName,
        from_entity: relationshipDraft.from,
        to_entity: relationshipDraft.to,
        description: relationshipDraft.description || "User defined relationship",
        join_on,
      });
    }
    const nextRegistry: RegistryPayload = { ...registry, relationships: nextRelationships };
    applyRegistry(nextRegistry, true);
    setRelationshipDraft(null);
    setStatus(relationshipDraft.mode === "edit" ? "Relationship updated." : "Relationship added.");
  };

  const editRelationship = (rel: RegistryRelationship) => {
    setRelationshipDraft({
      from: rel.from_entity,
      to: rel.to_entity,
      name: rel.name,
      description: rel.description || "",
      joinText: joinPairsToText(rel.join_on || []),
      mode: "edit",
      originalName: rel.name,
    });
  };

  const deleteRelationship = (name: string) => {
    if (!registry) return;
    if (!window.confirm(`Delete relationship "${name}"?`)) return;
    const nextRegistry: RegistryPayload = {
      ...registry,
      relationships: registry.relationships.filter((rel) => rel.name !== name),
    };
    applyRegistry(nextRegistry, true);
  };

  const computeDependenciesForEntity = (entity: RegistryEntity) => {
    const fieldNames = (entity.fields || []).map((field) => field.name);
    return (entity.fields || []).map((field) => {
      if (!field.derived) return field;
      const deps = mergeDependencies(
        extractDependencies(field.expr || "", fieldNames, field.name),
        field.order_by
      );
      return { ...field, depends_on: deps };
    });
  };

  const computeDependenciesForAll = () => {
    if (!registry) return;
    const nextRegistry: RegistryPayload = {
      ...registry,
      entities: registry.entities.map((entity) => ({
        ...entity,
        fields: computeDependenciesForEntity(entity),
      })),
    };
    applyRegistry(nextRegistry, true);
    setStatus("Dependencies computed for all derived fields.");
  };

  const startRelationshipDraft = () => {
    if (!registry) return;
    if (registry.entities.length < 2) {
      setStatus("Add at least two entities before creating relationships.");
      return;
    }
    const [from, to] = registry.entities.map((e) => e.name);
    setRelationshipDraft({
      from,
      to,
      name: `${from}_to_${to}`,
      description: "",
      joinText: "",
      mode: "create",
    });
  };

  const applyDraftReplace = () => {
    if (!draftRegistry) return;
    applyRegistry(draftRegistry, false);
    setDraftRegistry(null);
    setStatus("Draft applied (replaced).");
  };

  const applyDraftMerge = () => {
    if (!registry || !draftRegistry) return;
    const next: RegistryPayload = normalizeRegistry(registry);
    const draft = normalizeRegistry(draftRegistry);
    const entityMap = new Map(next.entities.map((entity) => [entity.name, entity]));
    const relMap = new Map(next.relationships.map((rel) => [rel.name, rel]));
    const reasonerMap = new Map(next.reasoners.map((reasoner) => [reasoner.id, reasoner]));
    (draft.entities || []).forEach((entity) => {
      if (!mergeSelection.entities[entity.name]) return;
      if (entityMap.has(entity.name)) {
        if (mergeMode === "replace") {
          entityMap.set(entity.name, entity);
        }
      } else {
        entityMap.set(entity.name, entity);
      }
    });
    (draft.relationships || []).forEach((rel) => {
      if (!mergeSelection.relationships[rel.name]) return;
      if (relMap.has(rel.name)) {
        if (mergeMode === "replace") {
          relMap.set(rel.name, rel);
        }
      } else {
        relMap.set(rel.name, rel);
      }
    });
    (draft.reasoners || []).forEach((reasoner) => {
      if (!mergeSelection.reasoners[reasoner.id]) return;
      if (reasonerMap.has(reasoner.id)) {
        if (mergeMode === "replace") {
          reasonerMap.set(reasoner.id, reasoner);
        }
      } else {
        reasonerMap.set(reasoner.id, reasoner);
      }
    });
    if (draft.kg && Object.keys(draft.kg).length) {
      next.kg = mergeMode === "replace" ? draft.kg : next.kg || draft.kg;
    }
    const merged: RegistryPayload = {
      entities: Array.from(entityMap.values()),
      relationships: Array.from(relMap.values()),
      reasoners: Array.from(reasonerMap.values()),
    };
    applyRegistry(merged, true);
    setDraftRegistry(null);
    setStatus("Draft merged.");
  };

  const discardDraft = () => {
    setDraftRegistry(null);
    setStatus("Draft discarded.");
  };

  const rollbackToVersion = async (version: string) => {
    setBusy(true);
    setStatus(`Rolling back to ${version}...`);
    try {
      const resp = await fetch(`${apiUrl}/api/registry/rollback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version }),
      });
      if (!resp.ok) {
        setStatus("Rollback failed.");
        return;
      }
      await loadRegistry();
      setStatus("Rollback complete.");
    } catch {
      setStatus("Rollback failed.");
    } finally {
      setBusy(false);
    }
  };

  const exportOntology = async () => {
    setBusy(true);
    setStatus("Exporting ontology...");
    try {
      const resp = await fetch(`${apiUrl}/api/registry/ontology/export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ registry, format: "json", write_file: true }),
      });
      if (!resp.ok) {
        setStatus("Export failed.");
        return;
      }
      const data = await resp.json();
      setStatus(`Ontology exported${data.path ? `: ${data.path}` : ""}.`);
    } catch {
      setStatus("Export failed.");
    } finally {
      setBusy(false);
    }
  };

  const deleteSelectedNode = () => {
    if (!registry || !selectedNode) return;
    if (selectedNode.kind === "kg") {
      setStatus("KG nodes are derived from the KG config. Edit the KG panel instead.");
      return;
    }
    const label = selectedNode.label || selectedNode.id;
    if (!window.confirm(`Delete ${selectedNode.kind} "${label}"?`)) return;
    if (selectedNode.kind === "entity" && selectedNode.ref.entityName) {
      const nextRegistry: RegistryPayload = {
        ...registry,
        entities: registry.entities.filter((entity) => entity.name !== selectedNode.ref.entityName),
        relationships: registry.relationships.filter(
          (rel) =>
            rel.from_entity !== selectedNode.ref.entityName && rel.to_entity !== selectedNode.ref.entityName
        ),
      };
      applyRegistry(nextRegistry, true);
      setSelectedId(null);
      return;
    }
    if (selectedNode.kind === "graph" && selectedNode.ref.graphId) {
      const kg = registry.kg || { nodes: [], edges: [], graphs: [] };
      const graphs = Array.isArray(kg.graphs) ? kg.graphs : [];
      const nextRegistry: RegistryPayload = {
        ...registry,
        kg: { ...kg, graphs: graphs.filter((graph) => graph.id !== selectedNode.ref.graphId) },
      };
      applyRegistry(nextRegistry, true);
      setSelectedId(null);
      return;
    }
    if (selectedNode.kind === "derived" && selectedNode.ref.entityName && selectedNode.ref.fieldName) {
      const nextRegistry: RegistryPayload = {
        ...registry,
        entities: registry.entities.map((entity) => {
          if (entity.name !== selectedNode.ref.entityName) return entity;
          return {
            ...entity,
            fields: entity.fields.filter((field) => field.name !== selectedNode.ref.fieldName),
          };
        }),
      };
      applyRegistry(nextRegistry, true);
      setSelectedId(null);
      return;
    }
    if (ENABLE_REASONERS && selectedNode.kind === "reasoner" && selectedNode.ref.reasonerId) {
      const nextRegistry: RegistryPayload = {
        ...registry,
        reasoners: registry.reasoners.filter((reasoner) => reasoner.id !== selectedNode.ref.reasonerId),
        entities: registry.entities.map((entity) => ({
          ...entity,
          applicable_reasoners: (entity.applicable_reasoners || []).filter(
            (id) => id !== selectedNode.ref.reasonerId
          ),
        })),
      };
      applyRegistry(nextRegistry, true);
      setSelectedId(null);
    }
  };

  const selectedEntity =
    selectedNode?.kind === "entity" ? registryLookup.entityMap.get(selectedNode.ref.entityName || "") : null;
  const prevCandidates = selectedEntity
    ? (selectedEntity.fields || []).filter(
        (field) => field.role === "metric" && !field.derived && !field.name.startsWith("prev_")
      )
    : [];
  const selectedPrevMetrics = selectedEntity ? prevFieldSelection[selectedEntity.name] || [] : [];
  const selectedReasoner =
    ENABLE_REASONERS && selectedNode?.kind === "reasoner"
      ? registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")
      : null;
  const selectedReasonerIsGraph = isGraphReasoner(selectedReasoner);
  const selectedReasonerGraphId = selectedReasoner?.graph_id || "";
  const reasonerOutputMeta = useMemo(() => {
    if (!registry || !selectedReasoner || isGraphReasoner(selectedReasoner)) return null;
    const entitiesForType = registry.entities.filter(
      (entity) => entity.entity_type === selectedReasoner.entity_type
    );
    const fieldMap = new Map<string, { field: RegistryField; entity: RegistryEntity }>();
    entitiesForType.forEach((entity) => {
      (entity.fields || []).forEach((field) => {
        if (!fieldMap.has(field.name)) {
          fieldMap.set(field.name, { field, entity });
        }
      });
    });
    return {
      entityNames: entitiesForType.map((entity) => entity.name),
      fieldMap,
      options: Array.from(fieldMap.keys()).sort(),
    };
  }, [registry, selectedReasoner]);

  const graphReasoners = useMemo(
    () => (registry?.reasoners || []).filter((reasoner) => isGraphReasoner(reasoner)),
    [registry]
  );
  const legacyReasoners = useMemo(
    () => (registry?.reasoners || []).filter((reasoner) => !isGraphReasoner(reasoner)),
    [registry]
  );

  const reasonerGraphStatus = useMemo(() => {
    if (!registry) {
      return { missing: [] as RegistryReasoner[], unbound: [] as RegistryReasoner[] };
    }
    const graphs = (registry.kg?.graphs as KgGraphSpec[]) || [];
    const graphIds = new Set(graphs.map((graph) => graph.id));
    const missing = registry.reasoners.filter(
      (reasoner) => isGraphReasoner(reasoner) && reasoner.graph_id && !graphIds.has(reasoner.graph_id)
    );
    const unbound = registry.reasoners.filter(
      (reasoner) => isGraphReasoner(reasoner) && !reasoner.graph_id
    );
    return { missing, unbound };
  }, [registry]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <datalist id="dtype-presets">
        {DTYPE_PRESETS.map((preset) => (
          <option key={preset.value} value={preset.value} />
        ))}
      </datalist>
      <div className="border-b border-slate-200/70 bg-white/60 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate-400">
              Registry Studio
            </p>
            <h1 className="mt-2 text-2xl font-semibold text-slate-900">
              Visual Semantic Builder
            </h1>
            <p className="mt-1 text-sm text-slate-500">
              Draft from Snowflake, refine visually, and export a registry for any client.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => saveRegistry()}
              disabled={busy}
              className="rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 shadow-sm"
              type="button"
            >
              Save registry
            </button>
            <button
              onClick={draftRegistryFromSnowflake}
              disabled={busy}
              className="rounded-full bg-slate-900 px-5 py-2 text-xs font-semibold text-white shadow-lg transition hover:-translate-y-0.5"
              type="button"
            >
              Draft from Snowflake
            </button>
          </div>
        </div>
      </div>

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-8 lg:flex-row">
        <motion.aside
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex w-full flex-col gap-5 rounded-3xl border border-white/60 bg-white/70 p-5 shadow-xl lg:w-80"
        >
          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
              How this fits together
            </p>
            <div className="mt-2 space-y-2 text-xs text-slate-600">
              <p>1) Entities + relationships define the schema surface.</p>
              <p>2) KG nodes/edges describe connectable topology.</p>
              <p>3) Graphs define traversal maps.</p>
              <p>4) Graph reasoners execute graphs inside RAI.</p>
            </div>
          </div>
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-[0.25em] text-slate-400">
              Snowflake Source
            </h2>
            <p className="mt-2 text-xs text-slate-500">
              Paste fully qualified tables (db.schema.table). Drafting will call DESCRIBE and Cortex.
            </p>
            <textarea
              value={tablesInput}
              onChange={(event) => setTablesInput(event.target.value)}
              placeholder="TFO_TEST.ML_TEST.RAI_MANDATE_MONTHLY_SUMMARY_MODEL"
              className="mt-3 h-24 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
            />
            <label className="mt-4 block text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
              Draft Notes
            </label>
            <textarea
              value={draftNotes}
              onChange={(event) => setDraftNotes(event.target.value)}
              placeholder="Focus on mandate risk + RM performance relationships."
              className="mt-2 h-20 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
            />
            <div className="mt-4 flex flex-wrap gap-2">
              <button
                onClick={draftRegistryFromSnowflake}
                disabled={busy}
                className="rounded-full bg-slate-900 px-4 py-2 text-xs font-semibold text-white shadow"
                type="button"
              >
                Draft registry
              </button>
              <button
                onClick={loadRegistry}
                disabled={busy}
                className="rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-600"
                type="button"
              >
                Reload
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Registry</p>
            <p className="mt-2 text-xs text-slate-500">
              {registryPath ? `Path: ${registryPath}` : "Registry path not set yet."}
            </p>
            <p className="mt-3 text-xs text-slate-500">
              {registry?.entities?.length || 0} entities - {registry?.relationships?.length || 0} relationships
              {ENABLE_REASONERS
                ? ` - ${graphReasoners.length} graph reasoners${
                    legacyReasoners.length ? ` - ${legacyReasoners.length} legacy reasoners` : ""
                  }`
                : ""}
              {` - ${(registry?.kg?.graphs || []).length || 0} graphs`}
            </p>
            <button
              onClick={() => saveRegistry()}
              disabled={busy}
              className="mt-3 w-full rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
              type="button"
            >
              Save registry
            </button>
            <button
              onClick={computeDependenciesForAll}
              disabled={busy}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
              type="button"
            >
              Compute deps (all derived)
            </button>
            <button
              onClick={rebuildKg}
              disabled={busy}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
              type="button"
            >
              Rebuild KG
            </button>
            <button
              onClick={exportOntology}
              disabled={busy}
              className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
              type="button"
            >
              Export ontology triples
            </button>
            <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                KG Configuration
              </p>
              <textarea
                value={kgDraft}
                onChange={(event) => setKgDraft(event.target.value)}
                placeholder='{"nodes":[...],"edges":[...]}'
                className="mt-2 h-40 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
              />
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  onClick={applyKgDraft}
                  className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                >
                  Apply KG JSON
                </button>
                <button
                  type="button"
                  onClick={generateKgConfig}
                  disabled={busy}
                  className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                >
                  Generate with LLM
                </button>
              </div>
              {kgStatus && <p className="mt-2 text-xs text-slate-500">{kgStatus}</p>}
            </div>
            <div className="mt-3 flex items-center justify-between rounded-2xl border border-slate-200 bg-white px-3 py-2">
              <span className="text-xs font-semibold text-slate-600">Auto-save</span>
              <button
                type="button"
                onClick={() => setAutoSaveEnabled((prev) => !prev)}
                className={`rounded-full px-3 py-1 text-[11px] font-semibold ${
                  autoSaveEnabled ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500"
                }`}
              >
                {autoSaveEnabled ? "On" : "Off"}
              </button>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <button
                onClick={reloadApi}
                disabled={busy}
                className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                type="button"
              >
                Reload API
              </button>
              <button
                onClick={startFresh}
                disabled={busy}
                className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                type="button"
              >
                Start fresh
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Import JSON
              </p>
              <button
                type="button"
                onClick={() => setImportJsonDraft("")}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600"
              >
                Clear
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Paste a full registry JSON to replace the current draft.
            </p>
            <textarea
              value={importJsonDraft}
              onChange={(event) => setImportJsonDraft(event.target.value)}
              placeholder={
                ENABLE_REASONERS
                  ? '{"entities":[...],"relationships":[...],"reasoners":[...]}'
                  : '{"entities":[...],"relationships":[...]}'
              }
              className="mt-2 h-32 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
            />
            <button
              type="button"
              onClick={applyImportedRegistry}
              className="mt-2 w-full rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
            >
              Import registry JSON
            </button>
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Prompt Customization
              </p>
              <button
                type="button"
                onClick={() => setShowPromptTemplates((prev) => !prev)}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600"
              >
                {showPromptTemplates ? "Hide previews" : "Preview prompts"}
              </button>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Provide business, data, and expectations so the LLM can generate customized prompts from the defaults.
              Full prompt templates are not directly editable.
            </p>
            <div className="mt-3 space-y-3">
              <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
                <p className="text-xs font-semibold text-slate-700">Business context</p>
                <textarea
                  value={promptContext.business}
                  onChange={(event) => updatePromptContext("business", event.target.value)}
                  placeholder="e.g., Wealth management, mandate profitability, RM performance"
                  className="mt-2 h-20 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                />
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
                <p className="text-xs font-semibold text-slate-700">Data context</p>
                <textarea
                  value={promptContext.data}
                  onChange={(event) => updatePromptContext("data", event.target.value)}
                  placeholder="e.g., Monthly mandate snapshots, meeting sentiment, revenue/cost metrics"
                  className="mt-2 h-20 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                />
              </div>
              <div className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
                <p className="text-xs font-semibold text-slate-700">Expectations</p>
                <textarea
                  value={promptContext.expectations}
                  onChange={(event) => updatePromptContext("expectations", event.target.value)}
                  placeholder="e.g., Focus on executive summaries, highlight risk, avoid speculation"
                  className="mt-2 h-20 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                />
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={generatePromptTemplates}
                  disabled={busy}
                  className="rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                >
                  Generate prompts
                </button>
                <button
                  type="button"
                  onClick={resetPromptTemplates}
                  disabled={busy}
                  className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                >
                  Reset to defaults
                </button>
              </div>
            </div>
            {showPromptTemplates && (
              <div className="mt-3 space-y-3">
                {promptTemplateKeys.map((key) => {
                  const hasTemplate = Object.prototype.hasOwnProperty.call(promptTemplates, key);
                  return (
                    <div key={key} className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-semibold text-slate-700">{key}</p>
                        <div className="flex items-center gap-2">
                          <span
                            className={`rounded-full px-2 py-1 text-[10px] font-semibold ${
                              hasTemplate ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
                            }`}
                          >
                            {hasTemplate ? "Custom" : "Default"}
                          </span>
                          {hasTemplate && (
                            <button
                              type="button"
                              onClick={() => removePromptTemplate(key)}
                              className="rounded-full border border-rose-200 bg-rose-50 px-2 py-1 text-[11px] text-rose-600"
                            >
                              Clear
                            </button>
                          )}
                        </div>
                      </div>
                      {hasTemplate ? (
                        <textarea
                          value={promptTemplates[key] || ""}
                          readOnly
                          className="mt-2 h-28 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600"
                        />
                      ) : (
                        <p className="mt-2 text-[11px] text-slate-500">
                          Using default prompt (not shown here).
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Graph Topology
              </p>
              <button
                onClick={() => addNode("graph")}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600"
                type="button"
              >
                + Add
              </button>
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
              Graphs describe the traversal map. Graph reasoners execute these graphs inside RAI.
            </p>
            <div className="mt-3 max-h-40 space-y-2 overflow-auto pr-1">
              {(registry?.kg?.graphs || []).map((graph) => (
                <button
                  key={graph.id}
                  type="button"
                  onClick={() => setSelectedId(graphNodeId(graph.id))}
                  className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-left text-xs font-semibold text-slate-700"
                >
                  {graph.id}
                  <span className="ml-2 text-[10px] font-semibold text-slate-400">
                    {graph.directed === false ? "undirected" : "directed"}
                  </span>
                </button>
              ))}
              {!(registry?.kg?.graphs || []).length && (
                <p className="text-xs text-slate-500">No graphs yet.</p>
              )}
            </div>
            <div className="mt-4 border-t border-slate-200 pt-3">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                Graph Reasoners
              </p>
              <p className="mt-1 text-[11px] text-slate-500">
                Executable reasoners bound to graphs (type = graph_reasoner).
              </p>
              <div className="mt-2 max-h-32 space-y-2 overflow-auto pr-1">
                {graphReasoners.map((reasoner) => (
                  <button
                    key={reasoner.id}
                    type="button"
                    onClick={() => setSelectedId(reasonerNodeId(reasoner.id))}
                    className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-left text-xs font-semibold text-slate-700"
                  >
                    {reasoner.name || reasoner.id}
                    {reasoner.graph_id ? (
                      <span className="ml-2 text-[10px] font-semibold text-slate-400">
                        {reasoner.graph_id}
                      </span>
                    ) : null}
                  </button>
                ))}
                {!graphReasoners.length && (
                  <p className="text-xs text-slate-500">No graph reasoners yet.</p>
                )}
              </div>
            </div>
            {legacyReasoners.length > 0 && (
              <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                <p className="font-semibold">Legacy reasoners present</p>
                <p className="mt-1 text-[11px]">
                  These are non-graph reasoners and run outside graph traversal logic.
                </p>
              </div>
            )}
            {(reasonerGraphStatus.missing.length > 0 || reasonerGraphStatus.unbound.length > 0) && (
              <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                <p className="font-semibold">Reasoner bindings need attention</p>
                {reasonerGraphStatus.missing.length > 0 && (
                  <p className="mt-1">
                    Missing graph: {reasonerGraphStatus.missing.map((r) => r.id).join(", ")}
                  </p>
                )}
                {reasonerGraphStatus.unbound.length > 0 && (
                  <p className="mt-1">
                    Unbound: {reasonerGraphStatus.unbound.map((r) => r.id).join(", ")}
                  </p>
                )}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Relationships
              </p>
              <button
                onClick={startRelationshipDraft}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600"
                type="button"
              >
                + Add
              </button>
            </div>
            <div className="mt-3 max-h-40 space-y-2 overflow-auto pr-1">
              {(registry?.relationships || []).map((rel) => (
                <div
                  key={rel.name}
                  className="rounded-2xl border border-slate-200 bg-white px-3 py-2"
                >
                  <p className="text-xs font-semibold text-slate-700">{rel.name}</p>
                  <p className="text-[11px] text-slate-500">
                    {rel.from_entity} → {rel.to_entity}
                  </p>
                  <div className="mt-2 flex gap-2">
                    <button
                      type="button"
                      onClick={() => editRelationship(rel)}
                      className="rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600"
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      onClick={() => deleteRelationship(rel.name)}
                      className="rounded-full border border-rose-200 bg-rose-50 px-2 py-1 text-[11px] text-rose-600"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
              {!registry?.relationships?.length && (
                <p className="text-xs text-slate-500">No relationships yet.</p>
              )}
            </div>
          </div>

          {draftRegistry && (
            <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                Draft Diff
              </p>
              <p className="mt-2 text-xs text-slate-500">
                Select what to merge from the draft.
              </p>
              <div className="mt-3 space-y-3">
                <div>
                  <p className="text-xs font-semibold text-slate-600">Entities</p>
                  <div className="mt-2 space-y-1">
                    {draftDiff.entities.map((entry) => (
                      <label key={entry.name} className="flex items-center gap-2 text-xs text-slate-600">
                        <input
                          type="checkbox"
                          checked={mergeSelection.entities[entry.name] || false}
                          onChange={(event) =>
                            setMergeSelection((prev) => ({
                              ...prev,
                              entities: { ...prev.entities, [entry.name]: event.target.checked },
                            }))
                          }
                        />
                        {entry.name} ({entry.status})
                      </label>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-600">Relationships</p>
                  <div className="mt-2 space-y-1">
                    {draftDiff.relationships.map((entry) => (
                      <label key={entry.name} className="flex items-center gap-2 text-xs text-slate-600">
                        <input
                          type="checkbox"
                          checked={mergeSelection.relationships[entry.name] || false}
                          onChange={(event) =>
                            setMergeSelection((prev) => ({
                              ...prev,
                              relationships: { ...prev.relationships, [entry.name]: event.target.checked },
                            }))
                          }
                        />
                        {entry.name} ({entry.status})
                      </label>
                    ))}
                  </div>
                </div>
                {ENABLE_REASONERS && (
                  <div>
                    <p className="text-xs font-semibold text-slate-600">Reasoners</p>
                    <div className="mt-2 space-y-1">
                      {draftDiff.reasoners.map((entry) => (
                        <label key={entry.name} className="flex items-center gap-2 text-xs text-slate-600">
                          <input
                            type="checkbox"
                            checked={mergeSelection.reasoners[entry.name] || false}
                            onChange={(event) =>
                              setMergeSelection((prev) => ({
                                ...prev,
                                reasoners: { ...prev.reasoners, [entry.name]: event.target.checked },
                              }))
                            }
                          />
                          {entry.name} ({entry.status})
                        </label>
                      ))}
                    </div>
                  </div>
                )}
                <div className="mt-2 flex items-center gap-2">
                  <label className="text-xs text-slate-600">Merge mode</label>
                  <select
                    value={mergeMode}
                    onChange={(event) => setMergeMode(event.target.value as "add" | "replace")}
                    className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                  >
                    <option value="add">add only</option>
                    <option value="replace">replace existing</option>
                  </select>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={applyDraftMerge}
                    className="flex-1 rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                  >
                    Merge selected
                  </button>
                  <button
                    type="button"
                    onClick={applyDraftReplace}
                    className="flex-1 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                  >
                    Replace
                  </button>
                </div>
                <button
                  type="button"
                  onClick={discardDraft}
                  className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                >
                  Discard draft
                </button>
              </div>
            </div>
          )}

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Validation</p>
            <p className="mt-2 text-xs text-slate-500">
              {validationIssues.length} issues
            </p>
            <div className="mt-3 max-h-40 space-y-2 overflow-auto pr-1">
              {validationIssues.map((issue, idx) => (
                <div key={`${issue.message}-${idx}`} className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
                  <p className={`text-xs font-semibold ${issue.level === "error" ? "text-rose-600" : "text-amber-600"}`}>
                    {issue.level.toUpperCase()}
                  </p>
                  <p className="text-[11px] text-slate-600">{issue.message}</p>
                </div>
              ))}
              {!validationIssues.length && (
                <p className="text-xs text-slate-500">No validation issues.</p>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Versions</p>
            <div className="mt-3 max-h-40 space-y-2 overflow-auto pr-1">
              {versions.map((version) => (
                <div key={version.name} className="rounded-2xl border border-slate-200 bg-white px-3 py-2">
                  <p className="text-xs font-semibold text-slate-700">{version.name}</p>
                  <p className="text-[11px] text-slate-500">{version.created_at}</p>
                  <button
                    type="button"
                    onClick={() => rollbackToVersion(version.name)}
                    className="mt-2 rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600"
                  >
                    Rollback
                  </button>
                </div>
              ))}
              {!versions.length && <p className="text-xs text-slate-500">No versions yet.</p>}
            </div>
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">YAML Preview</p>
            <textarea
              value={yamlPreview}
              readOnly
              placeholder="Save to generate YAML preview."
              className="mt-3 h-32 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
            />
          </div>

          <div className="rounded-2xl border border-white/60 bg-white/90 p-4 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Filters</p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => setCanvasMode("data")}
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  canvasMode === "data" ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500"
                }`}
              >
                Data model
              </button>
              <button
                type="button"
                onClick={() => setCanvasMode("kg")}
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  canvasMode === "kg" ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500"
                }`}
              >
                Knowledge graph
              </button>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {modeKinds.map((kind) => (
                <button
                  key={kind}
                  onClick={() => setVisibleKinds((prev) => ({ ...prev, [kind]: !prev[kind] }))}
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    visibleKinds[kind] ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-500"
                  }`}
                  type="button"
                >
                  {kind}
                </button>
              ))}
            </div>
            <button
              onClick={() => {
                if (canvasMode === "kg") return;
                setJoinView((prev) => !prev);
              }}
              className={`mt-3 rounded-full px-3 py-1 text-xs font-semibold ${
                isJoinView ? "bg-slate-900 text-white" : "border border-slate-200 bg-white text-slate-600"
              } ${canvasMode === "kg" ? "cursor-not-allowed opacity-50" : ""}`}
              type="button"
              disabled={canvasMode === "kg"}
            >
              Joins view
            </button>
            <button
              onClick={() => setKindsVisible(modeKinds)}
              className="mt-3 rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
              type="button"
            >
              Show all
            </button>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {modeKinds.filter((kind) => kind !== "kg").map((kind) => (
                <button
                  key={kind}
                  onClick={() => addNode(kind)}
                  className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                  type="button"
                >
                  Add {kind}
                </button>
              ))}
            </div>
          </div>

          {status && <p className="text-xs text-slate-500">{status}</p>}
        </motion.aside>

        <motion.section
          ref={canvasShellRef}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`flex min-h-[560px] w-full flex-1 flex-col gap-4 rounded-3xl border border-white/60 p-4 shadow-2xl ${
            isFullscreen ? "bg-white" : "bg-white/70"
          }`}
        >
          <div className="flex flex-wrap items-center justify-between gap-3 px-2">
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
              Canvas
              {linkFrom && (
                <span className="rounded-full bg-amber-100 px-3 py-1 text-[11px] font-semibold text-amber-700">
                  Linking mode
                </span>
              )}
              {!allKindsVisible && (
                <span className="rounded-full bg-slate-200 px-3 py-1 text-[11px] font-semibold text-slate-600">
                  Filter: {activeKindLabel || "none"}
                </span>
              )}
            </div>
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-1">
                  <button
                    onClick={() => setCanvasMode("data")}
                    className={`rounded-full px-2 py-1 text-[11px] font-semibold ${
                      canvasMode === "data" ? "bg-slate-900 text-white" : "text-slate-500"
                    }`}
                    type="button"
                  >
                    Data model
                  </button>
                  <button
                    onClick={() => setCanvasMode("kg")}
                    className={`rounded-full px-2 py-1 text-[11px] font-semibold ${
                      canvasMode === "kg" ? "bg-slate-900 text-white" : "text-slate-500"
                    }`}
                    type="button"
                  >
                    KG
                  </button>
                </div>
                <button
                  onClick={toggleFullscreen}
                  className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                  type="button"
                >
                  {isFullscreen ? "Exit full screen" : "Full screen"}
                </button>
                <button
                  onClick={() => {
                    if (canvasMode === "kg") return;
                    setJoinView((prev) => !prev);
                  }}
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    isJoinView ? "bg-slate-900 text-white" : "border border-slate-200 bg-white text-slate-600"
                  } ${canvasMode === "kg" ? "cursor-not-allowed opacity-50" : ""}`}
                  type="button"
                  disabled={canvasMode === "kg"}
                >
                  Joins view
                </button>
                <div className="flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-1">
                  {modeKinds.map((kind) => (
                    <button
                      key={kind}
                      onClick={() => setVisibleKinds((prev) => ({ ...prev, [kind]: !prev[kind] }))}
                    className={`rounded-full px-2 py-1 text-[11px] font-semibold ${
                      visibleKinds[kind] ? "bg-slate-900 text-white" : "text-slate-500"
                    }`}
                    type="button"
                  >
                    {kind}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-1">
                {modeKinds.filter((kind) => kind !== "kg").map((kind) => (
                  <button
                    key={kind}
                    onClick={() => addNode(kind)}
                    className="rounded-full px-2 py-1 text-[11px] font-semibold text-slate-600"
                    type="button"
                  >
                    + {kind}
                  </button>
                ))}
              </div>
              {!allKindsVisible && (
                <button
                  onClick={() => {
                    setKindsVisible(modeKinds);
                    requestAnimationFrame(() => fitToNodes(nodes));
                  }}
                  className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                  type="button"
                >
                  Show all
                </button>
              )}
              <button
                onClick={() => setLinkFrom(null)}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                type="button"
              >
                Cancel link
              </button>
              <button
                onClick={applyLayout}
                className="rounded-full bg-slate-900 px-3 py-1 text-xs font-semibold text-white shadow transition hover:-translate-y-0.5"
                type="button"
              >
                Auto layout
              </button>
              <button
                onClick={fitToView}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                type="button"
              >
                Fit
              </button>
              <button
                onClick={resetView}
                className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                type="button"
              >
                Reset
              </button>
              {focusNode && (
                <button
                  onClick={clearFocus}
                  className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700"
                  type="button"
                >
                  Clear focus
                </button>
              )}
              <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1">
                <button
                  onClick={zoomOut}
                  className="text-xs font-semibold text-slate-600"
                  type="button"
                >
                  -
                </button>
                <input
                  type="range"
                  min={0.4}
                  max={2}
                  step={0.05}
                  value={zoom}
                  onChange={(event) => setZoom(Number(event.target.value))}
                  className="h-1 w-24 accent-slate-700"
                />
                <button
                  onClick={zoomIn}
                  className="text-xs font-semibold text-slate-600"
                  type="button"
                >
                  +
                </button>
                <span className="text-[11px] font-semibold text-slate-500">
                  {Math.round(zoom * 100)}%
                </span>
              </div>
            </div>
          </div>

          <div
            ref={canvasRef}
            onPointerDown={handleCanvasPointerDown}
            onDoubleClick={handleCanvasDoubleClick}
            onClick={(event) => {
              if (event.detail === 2) {
                handleCanvasDoubleClick(event);
              }
            }}
            onWheel={handleWheel}
            className={`relative flex-1 overflow-hidden rounded-2xl border border-white/70 ${
              isFullscreen ? "bg-white" : "bg-gradient-to-br from-white/70 via-white/30 to-white/90"
            }`}
            style={{
              backgroundImage:
                "radial-gradient(circle at 1px 1px, rgba(15, 23, 42, 0.08) 1px, transparent 0)",
              backgroundSize: "28px 28px",
              cursor: panning ? "grabbing" : "grab",
            }}
          >
            <div
              className="absolute left-0 top-0"
              style={{
                width: contentSize.width,
                height: contentSize.height,
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: "0 0",
              }}
            >
              <svg
                className="pointer-events-none absolute left-0 top-0 z-20"
                width={contentSize.width}
                height={contentSize.height}
              >
                <defs>
                  <marker
                    id="arrow"
                    markerWidth="10"
                    markerHeight="10"
                    refX="10"
                    refY="3"
                    orient="auto"
                    markerUnits="strokeWidth"
                  >
                    <path d="M0,0 L0,6 L9,3 z" fill="#64748b" />
                  </marker>
                </defs>
                {edgePaths.map((edge) => (
                  <g key={edge.id}>
                    <path
                      d={edge.path}
                      fill="none"
                      stroke={EDGE_COLORS[edge.type]}
                      strokeWidth={2.2}
                      strokeDasharray={
                        edge.type === "dependency"
                          ? "5 6"
                          : edge.type === "reasoner_signal" || edge.type === "reasoner_output"
                            ? "3 6"
                            : undefined
                      }
                      markerEnd="url(#arrow)"
                      opacity={
                        dependencyHighlight
                          ? edge.type === "dependency"
                            ? dependencyHighlight.edges.has(edge.id)
                              ? 0.9
                              : 0.12
                            : 0.12
                          : 0.75
                      }
                    />
                    <text
                      x={edge.labelX}
                      y={edge.labelY}
                      textAnchor="middle"
                      className="fill-slate-500 text-[11px] font-semibold uppercase tracking-[0.15em]"
                      opacity={
                        dependencyHighlight
                          ? edge.type === "dependency"
                            ? dependencyHighlight.edges.has(edge.id)
                              ? 0.9
                              : 0.12
                            : 0.12
                          : 0.75
                      }
                    >
                      {edge.label}
                    </text>
                  </g>
                ))}
              </svg>

              {filteredNodes.map((node) => (
                <motion.div
                  key={node.id}
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.25 }}
                  onPointerDown={(event) => handlePointerDown(event, node.id)}
                  onClick={(event) => handleNodeClick(node.id, event)}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    handleNodeFocus(node.id);
                  }}
                  data-node="true"
                  className={`absolute z-10 cursor-grab rounded-2xl border border-white/80 bg-white/90 shadow-lg transition ${
                    selectedId === node.id ? "ring-2 ring-slate-900" : ""
                  }`}
                  style={{
                    left: node.x,
                    top: node.y,
                    width: NODE_WIDTH,
                    height: NODE_HEIGHT,
                    opacity:
                      dependencyHighlight && !dependencyHighlight.nodes.has(node.id) ? 0.25 : 1,
                  }}
                >
                  <button
                    type="button"
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={(event) => {
                      event.stopPropagation();
                      handleNodeFocus(node.id);
                    }}
                    className={`absolute right-2 top-2 rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
                      focusNodeId === node.id
                        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                        : "border-slate-200 bg-white text-slate-500"
                    }`}
                  >
                    {focusNodeId === node.id ? "Focused" : "Focus"}
                  </button>
                  <div
                    className={`rounded-2xl rounded-b-none bg-gradient-to-r px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] ${KIND_STYLES[node.kind]}`}
                  >
                    {node.kind}
                  </div>
                  <div className="flex h-[calc(100%-36px)] flex-col justify-between px-4 py-3">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-900">{node.label}</h3>
                      <p className="mt-1 text-xs text-slate-500 line-clamp-2">{node.description}</p>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      {node.tags.map((tag) => {
                        const tagClass =
                          tag === "reasoner-output"
                            ? "bg-rose-100 text-rose-700"
                            : "bg-slate-100 text-slate-500";
                        return (
                          <span
                            key={tag}
                            className={`rounded-full px-2 py-1 text-[10px] font-semibold ${tagClass}`}
                          >
                            {tag}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                  <button
                    onClick={(event) => {
                      event.stopPropagation();
                      startLink(node.id);
                    }}
                    className="absolute -right-2 top-1/2 h-7 w-7 -translate-y-1/2 rounded-full border border-white bg-slate-900 text-xs font-bold text-white shadow-md"
                    type="button"
                  >
                    +
                  </button>
                </motion.div>
              ))}
            </div>

            {isFullscreen && (
              <div className="pointer-events-auto absolute bottom-4 right-4 w-96 max-h-[calc(100vh-6rem)] overflow-y-auto rounded-2xl border border-white/60 bg-white/90 p-4 pr-3 shadow-xl backdrop-blur">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Logic</p>
                <p className="mt-2 text-xs text-slate-500">
                  {selectedNode ? selectedNode.label : "Select a node"}
                </p>
                {selectedNode?.kind === "derived" && (
                  <>
                    <textarea
                      value={logicDraft}
                      onChange={(event) => setLogicDraft(event.target.value)}
                      placeholder="Define formula or rule..."
                      className="mt-3 h-32 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                    />
                    <input
                      value={logicDependsOn}
                      readOnly
                      placeholder="Auto from expression"
                      className="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600"
                    />
                    <button
                      type="button"
                      onClick={applyLogic}
                      className="mt-3 w-full rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                    >
                      Apply rule
                    </button>
                    <button
                      type="button"
                      onClick={deleteSelectedNode}
                      className="mt-2 w-full rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-600"
                    >
                      Delete {selectedNode.kind}
                    </button>
                  </>
                )}
                {ENABLE_REASONERS && selectedNode?.kind === "reasoner" && (
                  <>
                    {reasonerOutputMeta && (
                      <datalist id={`reasoner-output-${selectedReasoner?.id || "unknown"}`}>
                        {reasonerOutputMeta.options.map((option) => (
                          <option key={option} value={option} />
                        ))}
                      </datalist>
                    )}
                    <div className="mt-3 flex items-center justify-between">
                      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                        Signals
                      </p>
                      <button
                        type="button"
                        onClick={addReasonerSignal}
                        className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                      >
                        + Signal
                      </button>
                    </div>
                    <div className="mt-3 flex items-center justify-between">
                      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                        Outputs
                      </p>
                      <button
                        type="button"
                        onClick={addReasonerOutput}
                        className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                      >
                        + Output
                      </button>
                    </div>
                    {reasonerOutputMeta?.entityNames?.length ? (
                      <p className="mt-2 text-xs text-slate-500">
                        Entity type: {selectedReasoner?.entity_type || "generic"} - Entities:{" "}
                        {reasonerOutputMeta.entityNames.join(", ")}
                      </p>
                    ) : (
                      <p className="mt-2 text-xs text-rose-500">
                        No entities found for entity_type {selectedReasoner?.entity_type || "generic"}.
                      </p>
                    )}
                    <div className="mt-2 space-y-2">
                      {(registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.outputs || []).length ===
                        0 && <p className="text-xs text-slate-500">No outputs yet.</p>}
                      {(registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.outputs || []).map(
                        (output, index) => (
                          <div key={`${output}-${index}`} className="rounded-2xl border border-slate-200 p-3">
                            <div className="flex items-center gap-2">
                              <input
                                value={output}
                                list={`reasoner-output-${selectedReasoner?.id || "unknown"}`}
                                onChange={(event) => updateReasonerOutput(index, event.target.value)}
                                placeholder="output_field"
                                className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                              />
                              <button
                                type="button"
                                onClick={() => removeReasonerOutput(index)}
                                className="rounded-xl border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-600"
                              >
                                Remove
                              </button>
                            </div>
                            {(() => {
                              const info = output ? reasonerOutputMeta?.fieldMap.get(output) : null;
                              if (!output) {
                                return (
                                  <p className="mt-2 text-xs text-slate-400">Select an output field.</p>
                                );
                              }
                              if (!info) {
                                return (
                                  <p className="mt-2 text-xs text-rose-500">
                                    Output not found on entity type {selectedReasoner?.entity_type || "generic"}.
                                  </p>
                                );
                              }
                              return (
                                <div className="mt-2 text-xs text-slate-500">
                                  <p>
                                    {info.field.derived ? "derived" : "base"} - {info.entity.name}
                                  </p>
                                  <p className="mt-1 text-slate-600">
                                    {fieldExpressionPreview(registry, info.entity.name, info.field)}
                                  </p>
                                  {info.field.depends_on?.length ? (
                                    <p className="mt-1 text-slate-400">
                                      depends_on: {info.field.depends_on.join(", ")}
                                    </p>
                                  ) : null}
                                </div>
                              );
                            })()}
                          </div>
                        )
                      )}
                    </div>
                    <div className="mt-2 max-h-56 space-y-3 overflow-auto pr-1">
                      {reasonerSignalsDraft.length === 0 && (
                        <p className="text-xs text-slate-500">No signals yet.</p>
                      )}
                      {reasonerSignalsDraft.map((signal, index) => (
                        <div key={`${signal.name}-${index}`} className="rounded-2xl border border-slate-200 p-3">
                          <div className="grid grid-cols-2 gap-2">
                            <input
                              value={signal.name}
                              onChange={(event) => updateReasonerSignal(index, { name: event.target.value })}
                              placeholder="name"
                              className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                            />
                            <input
                              value={signal.metric_field}
                              onChange={(event) =>
                                updateReasonerSignal(index, { metric_field: event.target.value })
                              }
                              placeholder="metric_field"
                              className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                            />
                            <input
                              value={signal.threshold}
                              type="number"
                              onChange={(event) =>
                                updateReasonerSignal(index, { threshold: Number(event.target.value) || 0 })
                              }
                              placeholder="threshold"
                              className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                            />
                            <select
                              value={signal.direction}
                              onChange={(event) => updateReasonerSignal(index, { direction: event.target.value })}
                              className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                            >
                              <option value="below">below</option>
                              <option value="above">above</option>
                              <option value="equals">equals</option>
                            </select>
                            <input
                              value={signal.weight ?? ""}
                              type="number"
                              onChange={(event) =>
                                updateReasonerSignal(index, {
                                  weight: event.target.value === "" ? undefined : Number(event.target.value),
                                })
                              }
                              placeholder="weight"
                              className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                            />
                            <button
                              type="button"
                              onClick={() => removeReasonerSignal(index)}
                              className="rounded-xl border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-600"
                            >
                              Remove
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-3">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                        Drilldown plan
                      </p>
                      <textarea
                        value={reasonerPlanDraft}
                        onChange={(event) => setReasonerPlanDraft(event.target.value)}
                        placeholder='{"steps":[...]}'
                        className="mt-2 h-40 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      <div className="mt-2 flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={applyReasonerPlanJson}
                          className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                        >
                          Apply plan JSON
                        </button>
                        <button
                          type="button"
                          onClick={generateReasonerPlanWithLLM}
                          disabled={busy}
                          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                        >
                          Generate plan
                        </button>
                      </div>
                      <textarea
                        value={reasonerPlanInstruction}
                        onChange={(event) => setReasonerPlanInstruction(event.target.value)}
                        placeholder="Describe the drilldown steps in natural language..."
                        className="mt-2 h-16 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      {reasonerPlanStatus && (
                        <p className="mt-2 text-xs text-slate-500">{reasonerPlanStatus}</p>
                      )}
                    </div>
                    <div className="mt-3 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-3">
                      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                        Reasoner sanity query
                      </p>
                      <div className="mt-2 grid grid-cols-2 gap-2">
                        <input
                          value={reasonerSanityStepId}
                          onChange={(event) => setReasonerSanityStepId(event.target.value)}
                          placeholder="step_id (e.g. worst_ops)"
                          className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                        />
                        <input
                          value={reasonerSanityLimit}
                          onChange={(event) => setReasonerSanityLimit(event.target.value)}
                          placeholder="limit"
                          className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                        />
                        <input
                          value={reasonerSanityWindowStart}
                          onChange={(event) => setReasonerSanityWindowStart(event.target.value)}
                          placeholder="window_start (optional)"
                          className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                        />
                        <input
                          value={reasonerSanityWindowEnd}
                          onChange={(event) => setReasonerSanityWindowEnd(event.target.value)}
                          placeholder="window_end (optional)"
                          className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={async () => {
                          if (!selectedNode?.ref.reasonerId) return;
                          if (!reasonerSanityStepId.trim()) {
                            setReasonerSanityStatus("step_id is required.");
                            return;
                          }
                          setReasonerSanityStatus("Running sanity query...");
                          setReasonerSanityResult("");
                          try {
                            const resp = await fetch(`${apiUrl}/api/registry/reasoners/sanity`, {
                              method: "POST",
                              headers: { "Content-Type": "application/json" },
                              body: JSON.stringify({
                                reasoner_id: selectedNode.ref.reasonerId,
                                step_id: reasonerSanityStepId.trim(),
                                window_start: reasonerSanityWindowStart || undefined,
                                window_end: reasonerSanityWindowEnd || undefined,
                                limit: Number(reasonerSanityLimit) || 5,
                              }),
                            });
                            const data = await resp.json();
                            if (!resp.ok) {
                              setReasonerSanityStatus(data?.detail || "Sanity query failed.");
                              return;
                            }
                            setReasonerSanityStatus(`OK (${data?.row_count || 0} rows)`);
                            setReasonerSanityResult(JSON.stringify(data, null, 2));
                          } catch (err) {
                            setReasonerSanityStatus("Sanity query failed.");
                          }
                        }}
                        className="mt-2 w-full rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                      >
                        Run sanity query
                      </button>
                      {reasonerSanityStatus && (
                        <p className="mt-2 text-xs text-slate-500">{reasonerSanityStatus}</p>
                      )}
                      {reasonerSanityResult && (
                        <textarea
                          readOnly
                          value={reasonerSanityResult}
                          className="mt-2 h-40 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                        />
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={applyLogic}
                      className="mt-3 w-full rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                    >
                      Apply reasoner rules
                    </button>
                    <button
                      type="button"
                      onClick={deleteSelectedNode}
                      className="mt-2 w-full rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-600"
                    >
                      Delete {selectedNode.kind}
                    </button>
                  </>
                )}
                {selectedNode?.kind === "entity" && (
                  <button
                    type="button"
                    onClick={deleteSelectedNode}
                    className="mt-3 w-full rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-600"
                  >
                    Delete {selectedNode.kind}
                  </button>
                )}
                <p className="mt-3 text-[11px] text-slate-400">
                  Link nodes: click + on a node, then click the target node.
                </p>
              </div>
            )}

            {relationshipDraft && (
              <div className="pointer-events-auto absolute left-1/2 top-6 w-[420px] -translate-x-1/2 rounded-3xl border border-white/60 bg-white/95 p-5 shadow-2xl backdrop-blur">
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  {relationshipDraft.mode === "edit" ? "Edit Relationship" : "Create Relationship"}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  {relationshipDraft.from} → {relationshipDraft.to}
                </p>
                <div className="mt-4 grid grid-cols-2 gap-2">
                  <select
                    value={relationshipDraft.from}
                    onChange={(event) =>
                      setRelationshipDraft((prev) => (prev ? { ...prev, from: event.target.value } : prev))
                    }
                    className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                  >
                    {(registry?.entities || []).map((entity) => (
                      <option key={entity.name} value={entity.name}>
                        {entity.name}
                      </option>
                    ))}
                  </select>
                  <select
                    value={relationshipDraft.to}
                    onChange={(event) =>
                      setRelationshipDraft((prev) => (prev ? { ...prev, to: event.target.value } : prev))
                    }
                    className="rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                  >
                    {(registry?.entities || []).map((entity) => (
                      <option key={entity.name} value={entity.name}>
                        {entity.name}
                      </option>
                    ))}
                  </select>
                </div>
                <label className="mt-4 block text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  Name
                </label>
                <input
                  value={relationshipDraft.name}
                  onChange={(event) =>
                    setRelationshipDraft((prev) => (prev ? { ...prev, name: event.target.value } : prev))
                  }
                  className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                />
                <label className="mt-4 block text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  Join Keys
                </label>
                <textarea
                  value={relationshipDraft.joinText}
                  onChange={(event) =>
                    setRelationshipDraft((prev) => (prev ? { ...prev, joinText: event.target.value } : prev))
                  }
                  placeholder="left_key=right_key"
                  className="mt-2 h-24 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                />
                <label className="mt-4 block text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                  Description
                </label>
                <textarea
                  value={relationshipDraft.description}
                  onChange={(event) =>
                    setRelationshipDraft((prev) => (prev ? { ...prev, description: event.target.value } : prev))
                  }
                  className="mt-2 h-20 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                />
                <div className="mt-4 flex gap-2">
                  <button
                    type="button"
                    onClick={() => setRelationshipDraft(null)}
                    className="flex-1 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={confirmRelationship}
                    className="flex-1 rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                  >
                    Create
                  </button>
                </div>
              </div>
            )}
          </div>
        </motion.section>

        <motion.aside
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex w-full flex-col gap-5 rounded-3xl border border-white/60 bg-white/70 p-5 shadow-xl lg:max-h-[calc(100vh-9rem)] lg:w-80 lg:overflow-y-auto lg:pr-4"
        >
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-[0.25em] text-slate-400">
              Inspector
            </h2>
            {selectedNode ? (
              <div className="mt-4 space-y-4">
                <div>
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Label
                  </label>
                  <input
                    value={selectedNode.label}
                    onChange={(event) => handleLabelChange(event.target.value)}
                    readOnly={selectedNode.kind === "kg"}
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                    Description
                  </label>
                  <textarea
                    value={selectedNode.description}
                    onChange={(event) => updateDescription(selectedNode, event.target.value)}
                    readOnly={selectedNode.kind === "kg" || selectedNode.kind === "graph"}
                    className="mt-2 h-20 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
                  />
                </div>
                {selectedNode.kind === "kg" ? (
                  <p className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-500">
                    KG nodes are read-only. Edit the KG config panel to update them.
                  </p>
                ) : (
                  <button
                    type="button"
                    onClick={deleteSelectedNode}
                    className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-600"
                  >
                    Delete {selectedNode.kind}
                  </button>
                )}

                <div className="rounded-2xl border border-slate-100 bg-slate-50 px-4 py-3">
                  <p className="text-xs font-semibold text-slate-500">Inputs</p>
                  <p className="mt-2 text-xs text-slate-600">
                    {selectedNode.inputs.length ? selectedNode.inputs.join(", ") : "None"}
                  </p>
                  <p className="mt-4 text-xs font-semibold text-slate-500">Outputs</p>
                  <p className="mt-2 text-xs text-slate-600">
                    {selectedNode.outputs.length ? selectedNode.outputs.join(", ") : "None"}
                  </p>
                </div>

                {selectedNode.kind === "kg" && (
                  <div className="rounded-2xl border border-white/60 bg-white/90 px-4 py-4 shadow-sm">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      KG Node Details
                    </p>
                    <div className="mt-3 space-y-2 text-xs text-slate-600">
                      <p>Entity: {selectedNode.ref.entityName || selectedNode.label}</p>
                      <p>
                        Type:{" "}
                        {registryLookup.kgNodeMap.get(selectedNode.ref.entityName || "")?.node_type || "node"}
                      </p>
                      <p>
                        Key field:{" "}
                        {(() => {
                          const keyField =
                            registryLookup.kgNodeMap.get(selectedNode.ref.entityName || "")?.key_field;
                          if (Array.isArray(keyField)) return keyField.join(", ");
                          return keyField || "None";
                        })()}
                      </p>
                      <p>
                        Label field:{" "}
                        {registryLookup.kgNodeMap.get(selectedNode.ref.entityName || "")?.label_field || "None"}
                      </p>
                      <p>
                        Properties:{" "}
                        {(registryLookup.kgNodeMap.get(selectedNode.ref.entityName || "")?.properties || []).length
                          ? (registryLookup.kgNodeMap.get(selectedNode.ref.entityName || "")?.properties || []).join(
                              ", "
                            )
                          : "None"}
                      </p>
                    </div>
                  </div>
                )}

                {selectedNode.kind === "graph" && (
                  <div className="rounded-2xl border border-white/60 bg-white/90 px-4 py-4 shadow-sm">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Graph Config
                    </p>
                    <div className="mt-3 space-y-3 text-xs text-slate-600">
                      <div className="flex items-center gap-3">
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={graphDirectedDraft}
                            onChange={(event) => setGraphDirectedDraft(event.target.checked)}
                          />
                          directed
                        </label>
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={graphWeightedDraft}
                            onChange={(event) => setGraphWeightedDraft(event.target.checked)}
                          />
                          weighted
                        </label>
                      </div>
                      <textarea
                        value={graphDescriptionDraft}
                        onChange={(event) => setGraphDescriptionDraft(event.target.value)}
                        placeholder="Describe what this graph captures."
                        className="h-16 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      <div>
                        <label className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                          Nodes (JSON array)
                        </label>
                        <textarea
                          value={graphNodesDraft}
                          onChange={(event) => setGraphNodesDraft(event.target.value)}
                          placeholder='[{"entity":"dt_operation_metrics","label_field":"pu_id"}]'
                          className="mt-2 h-28 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                        />
                      </div>
                      <div>
                        <label className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                          Edges (JSON array)
                        </label>
                        <textarea
                          value={graphEdgesDraft}
                          onChange={(event) => setGraphEdgesDraft(event.target.value)}
                          placeholder='[{"relation":"dt_operation_metrics.machine_id"}]'
                          className="mt-2 h-28 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                        />
                      </div>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={applyGraphDraft}
                          className="flex-1 rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                        >
                          Apply graph
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            const graphs = (registry?.kg?.graphs as KgGraphSpec[]) || [];
                            const graph = graphs.find((entry) => entry.id === selectedNode.ref.graphId);
                            setGraphDescriptionDraft(graph?.description || "");
                            setGraphDirectedDraft(graph?.directed !== false);
                            setGraphWeightedDraft(!!graph?.weighted);
                            setGraphNodesDraft(graph?.nodes ? JSON.stringify(graph.nodes, null, 2) : "[]");
                            setGraphEdgesDraft(graph?.edges ? JSON.stringify(graph.edges, null, 2) : "[]");
                            setGraphDraftStatus("");
                          }}
                          className="flex-1 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-600"
                        >
                          Reset
                        </button>
                      </div>
                      {graphDraftStatus && (
                        <p className="text-xs text-slate-500">{graphDraftStatus}</p>
                      )}
                    </div>
                  </div>
                )}

                {selectedEntity && (
                  <div className="rounded-2xl border border-white/60 bg-white/90 px-4 py-4 shadow-sm">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Entity Details
                    </p>
                    <p className="mt-2 text-xs text-slate-600">
                      Table: {selectedEntity.database ? `${selectedEntity.database}.` : ""}
                      {selectedEntity.schema ? `${selectedEntity.schema}.` : ""}
                      {selectedEntity.table}
                    </p>
                    <p className="mt-2 text-xs text-slate-600">
                      Join keys: {selectedEntity.join_keys?.length ? selectedEntity.join_keys.join(", ") : "None"}
                    </p>
                    <div className="mt-4 space-y-3">
                      <input
                        value={selectedEntity.database || ""}
                        onChange={(event) => updateSelectedEntity({ database: event.target.value })}
                        placeholder="Database"
                        className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      <input
                        value={selectedEntity.schema || ""}
                        onChange={(event) => updateSelectedEntity({ schema: event.target.value })}
                        placeholder="Schema"
                        className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      <input
                        value={selectedEntity.table || ""}
                        onChange={(event) => updateSelectedEntity({ table: event.target.value })}
                        placeholder="Table"
                        className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      <input
                        value={selectedEntity.join_keys?.join(", ") || ""}
                        onChange={(event) =>
                          updateSelectedEntity({
                            join_keys: event.target.value
                              .split(",")
                              .map((entry) => entry.trim())
                              .filter(Boolean),
                          })
                        }
                        placeholder="Join keys (comma separated)"
                        className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      <input
                        value={selectedEntity.entity_type || ""}
                        onChange={(event) => updateSelectedEntity({ entity_type: event.target.value })}
                        placeholder="Entity type"
                        className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                      <input
                        value={selectedEntity.default_metric || ""}
                        onChange={(event) => updateSelectedEntity({ default_metric: event.target.value })}
                        placeholder="Default metric"
                        className="w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                      />
                    </div>
                    <div className="mt-5 rounded-2xl border border-slate-100 bg-slate-50/60 p-3">
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                          Fields
                        </p>
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={() => importFieldsForEntity(selectedEntity.name)}
                            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600"
                          >
                            Import
                          </button>
                          <button
                            type="button"
                            onClick={() => addEntityField(selectedEntity.name)}
                            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600"
                          >
                            + Field
                          </button>
                        </div>
                      </div>
                      <div className="mt-3 flex gap-2">
                        {(["base", "derived", "all"] as const).map((mode) => (
                          <button
                            key={mode}
                            type="button"
                            onClick={() => setFieldView(mode)}
                            className={`rounded-full px-3 py-1 text-[11px] font-semibold ${
                              fieldView === mode ? "bg-slate-900 text-white" : "bg-white text-slate-500"
                            }`}
                          >
                            {mode}
                          </button>
                        ))}
                      </div>
                      <div className="mt-3 max-h-64 space-y-2 overflow-auto pr-1">
                        {(selectedEntity.fields || [])
                          .filter((field) => {
                            if (fieldView === "all") return true;
                            if (fieldView === "derived") return field.derived;
                            return !field.derived;
                          })
                          .map((field) => (
                            <div
                              key={field.name}
                              className="rounded-2xl border border-slate-200 bg-white p-2"
                            >
                              <div className="grid grid-cols-2 gap-2">
                                <input
                                  value={field.name}
                                  onChange={(event) =>
                                    updateEntityField(selectedEntity.name, field.name, {
                                      name: event.target.value.toLowerCase(),
                                    })
                                  }
                                  placeholder="name"
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                />
                                <div className="flex items-center gap-2">
                                  <input
                                    list="dtype-presets"
                                    value={field.dtype}
                                    onChange={(event) =>
                                      updateEntityField(selectedEntity.name, field.name, {
                                        dtype: event.target.value,
                                      })
                                    }
                                    placeholder="dtype"
                                    className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                  />
                                  <select
                                    value=""
                                    onChange={(event) =>
                                      updateEntityField(selectedEntity.name, field.name, {
                                        dtype: event.target.value,
                                      })
                                    }
                                    className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                  >
                                    <option value="">preset</option>
                                    {DTYPE_PRESETS.map((preset) => (
                                      <option key={preset.value} value={preset.value}>
                                        {preset.label}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                                <select
                                  value={field.role}
                                  onChange={(event) => {
                                    const nextRole = event.target.value;
                                    updateEntityField(selectedEntity.name, field.name, {
                                      role: nextRole,
                                      default_agg:
                                        nextRole === "metric"
                                          ? field.default_agg || inferDefaultAgg(field.name, field.dtype)
                                          : "",
                                    });
                                  }}
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                >
                                  <option value="dimension">dimension</option>
                                  <option value="metric">metric</option>
                                </select>
                                <select
                                  value={field.default_agg || ""}
                                  onChange={(event) =>
                                    updateEntityField(selectedEntity.name, field.name, {
                                      default_agg: event.target.value,
                                    })
                                  }
                                  disabled={field.role !== "metric"}
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600 disabled:cursor-not-allowed disabled:bg-slate-100"
                                >
                                  <option value="">default agg</option>
                                  {AGG_PRESETS.filter(Boolean).map((agg) => (
                                    <option key={agg} value={agg}>
                                      {agg}
                                    </option>
                                  ))}
                                </select>
                                <input
                                  value={field.expr}
                                  onChange={(event) =>
                                    updateEntityField(selectedEntity.name, field.name, {
                                      expr: event.target.value,
                                    })
                                  }
                                  placeholder="expr"
                                  className="col-span-2 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                />
                              </div>
                              <div className="mt-2 flex items-center justify-between">
                                <span className="text-[11px] text-slate-400">
                                  {field.derived ? "derived" : "base"}
                                  {field.default_agg ? ` - ${field.default_agg}` : ""}
                                </span>
                                <button
                                  type="button"
                                  onClick={() => deleteEntityField(selectedEntity.name, field.name)}
                                  className="rounded-full border border-rose-200 bg-rose-50 px-2 py-1 text-[11px] text-rose-600"
                                >
                                  Delete
                                </button>
                              </div>
                              {field.default_agg === "last" && (
                                <div className="mt-2 rounded-xl border border-slate-100 bg-slate-50 px-3 py-2">
                                  <div className="flex items-center justify-between">
                                    <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                      Order by
                                    </p>
                                    <button
                                      type="button"
                                      onClick={() =>
                                        updateFieldOrderBy(selectedEntity.name, field.name, [
                                          ...(field.order_by || []),
                                          { field: "", direction: "asc" },
                                        ])
                                      }
                                      className="rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600"
                                    >
                                      + Add
                                    </button>
                                  </div>
                                  <div className="mt-2 space-y-2">
                                    {(field.order_by || []).length === 0 && (
                                      <p className="text-xs text-slate-500">No order fields set.</p>
                                    )}
                                    {(field.order_by || []).map((entry, index) => (
                                      <div key={`${entry.field}-${index}`} className="grid grid-cols-3 gap-2">
                                        <select
                                          value={entry.field}
                                          onChange={(event) => {
                                            const next = (field.order_by || []).map((item, idx) =>
                                              idx === index ? { ...item, field: event.target.value } : item
                                            );
                                            updateFieldOrderBy(selectedEntity.name, field.name, next);
                                          }}
                                          className="col-span-2 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                        >
                                          <option value="">field</option>
                                          {(selectedEntity.fields || []).map((option) => (
                                            <option key={option.name} value={option.name}>
                                              {option.name}
                                            </option>
                                          ))}
                                        </select>
                                        <div className="flex gap-2">
                                          <select
                                            value={entry.direction || "asc"}
                                            onChange={(event) => {
                                              const next = (field.order_by || []).map((item, idx) =>
                                                idx === index
                                                  ? { ...item, direction: event.target.value as "asc" | "desc" }
                                                  : item
                                              );
                                              updateFieldOrderBy(selectedEntity.name, field.name, next);
                                            }}
                                            className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                          >
                                            <option value="asc">asc</option>
                                            <option value="desc">desc</option>
                                          </select>
                                          <button
                                            type="button"
                                            onClick={() => {
                                              const next = (field.order_by || []).filter((_, idx) => idx !== index);
                                              updateFieldOrderBy(selectedEntity.name, field.name, next);
                                            }}
                                            className="rounded-xl border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-600"
                                          >
                                            Remove
                                          </button>
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ))}
                        {!selectedEntity.fields?.length && (
                          <p className="text-xs text-slate-500">No fields yet.</p>
                        )}
                      </div>
                      <div className="mt-3 rounded-2xl border border-slate-100 bg-slate-50/80 p-3">
                        <div className="flex items-center justify-between">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Prev field metrics
                          </p>
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() =>
                                setPrevMetrics(
                                  selectedEntity.name,
                                  prevCandidates.map((field) => field.name)
                                )
                              }
                              className="rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600"
                            >
                              Select all
                            </button>
                            <button
                              type="button"
                              onClick={() => setPrevMetrics(selectedEntity.name, [])}
                              className="rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600"
                            >
                              Clear
                            </button>
                          </div>
                        </div>
                        <div className="mt-2 max-h-32 space-y-1 overflow-auto pr-1">
                          {!prevCandidates.length && (
                            <p className="text-xs text-slate-500">No base metrics available.</p>
                          )}
                          {prevCandidates.map((field) => (
                            <label key={field.name} className="flex items-center gap-2 text-xs text-slate-600">
                              <input
                                type="checkbox"
                                checked={selectedPrevMetrics.includes(field.name)}
                                onChange={() => togglePrevMetric(selectedEntity.name, field.name)}
                                className="h-3.5 w-3.5 rounded border-slate-300"
                              />
                              {field.name}
                            </label>
                          ))}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => generatePrevFields(selectedEntity.name, selectedPrevMetrics)}
                        className="mt-3 w-full rounded-2xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                      >
                        Generate prev_ fields
                      </button>
                    </div>
                  </div>
                )}

                {selectedNode.kind !== "entity" && (
                  <div className="space-y-3 rounded-2xl border border-white/60 bg-white/90 px-4 py-4 shadow-sm">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Logic
                    </p>
                    {selectedNode.kind === "derived" ? (
                      <>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="flex items-center gap-2">
                            <input
                              list="dtype-presets"
                              value={
                                registryLookup.entityMap
                                  .get(selectedNode.ref.entityName || "")
                                  ?.fields.find((field) => field.name === selectedNode.ref.fieldName)?.dtype || ""
                              }
                              onChange={(event) => updateSelectedDerived({ dtype: event.target.value })}
                              placeholder="dtype"
                              className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                            />
                            <select
                              value=""
                              onChange={(event) => updateSelectedDerived({ dtype: event.target.value })}
                              className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                            >
                              <option value="">preset</option>
                              {DTYPE_PRESETS.map((preset) => (
                                <option key={preset.value} value={preset.value}>
                                  {preset.label}
                                </option>
                              ))}
                            </select>
                          </div>
                          <select
                            value={
                              registryLookup.entityMap
                                .get(selectedNode.ref.entityName || "")
                                ?.fields.find((field) => field.name === selectedNode.ref.fieldName)?.role || "metric"
                            }
                            onChange={(event) => updateSelectedDerived({ role: event.target.value })}
                            className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                          >
                            <option value="metric">metric</option>
                            <option value="dimension">dimension</option>
                          </select>
                          <select
                            value={
                              registryLookup.entityMap
                                .get(selectedNode.ref.entityName || "")
                                ?.fields.find((field) => field.name === selectedNode.ref.fieldName)?.default_agg ||
                              ""
                            }
                            onChange={(event) => updateSelectedDerived({ default_agg: event.target.value })}
                            className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                          >
                            {AGG_PRESETS.map((agg) => (
                              <option key={agg || "none"} value={agg}>
                                {agg || "none"}
                              </option>
                            ))}
                          </select>
                        </div>
                        {(() => {
                          const entity = registryLookup.entityMap.get(selectedNode.ref.entityName || "");
                          const field = entity?.fields.find((f) => f.name === selectedNode.ref.fieldName);
                          if (!field || field.default_agg !== "last") return null;
                          const orderBy = field.order_by || [];
                          const fieldOptions = (entity?.fields || []).map((f) => f.name);
                          return (
                            <div className="rounded-2xl border border-slate-100 bg-slate-50 px-3 py-2">
                              <div className="flex items-center justify-between">
                                <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                  Order by
                                </p>
                                <button
                                  type="button"
                                  onClick={() =>
                                    updateFieldOrderBy(selectedNode.ref.entityName!, selectedNode.ref.fieldName!, [
                                      ...orderBy,
                                      { field: "", direction: "asc" },
                                    ])
                                  }
                                  className="rounded-full border border-slate-200 bg-white px-2 py-1 text-[11px] text-slate-600"
                                >
                                  + Add
                                </button>
                              </div>
                              <div className="mt-2 space-y-2">
                                {orderBy.length === 0 && (
                                  <p className="text-xs text-slate-500">No order fields set.</p>
                                )}
                                {orderBy.map((entry, index) => (
                                  <div key={`${entry.field}-${index}`} className="grid grid-cols-3 gap-2">
                                    <select
                                      value={entry.field}
                                      onChange={(event) => {
                                        const next = orderBy.map((item, idx) =>
                                          idx === index ? { ...item, field: event.target.value } : item
                                        );
                                        updateFieldOrderBy(
                                          selectedNode.ref.entityName!,
                                          selectedNode.ref.fieldName!,
                                          next
                                        );
                                      }}
                                      className="col-span-2 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                    >
                                      <option value="">field</option>
                                      {fieldOptions.map((name) => (
                                        <option key={name} value={name}>
                                          {name}
                                        </option>
                                      ))}
                                    </select>
                                    <div className="flex gap-2">
                                      <select
                                        value={entry.direction || "asc"}
                                        onChange={(event) => {
                                          const next = orderBy.map((item, idx) =>
                                            idx === index
                                              ? { ...item, direction: event.target.value as "asc" | "desc" }
                                              : item
                                          );
                                          updateFieldOrderBy(
                                            selectedNode.ref.entityName!,
                                            selectedNode.ref.fieldName!,
                                            next
                                          );
                                        }}
                                        className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                      >
                                        <option value="asc">asc</option>
                                        <option value="desc">desc</option>
                                      </select>
                                      <button
                                        type="button"
                                        onClick={() => {
                                          const next = orderBy.filter((_, idx) => idx !== index);
                                          updateFieldOrderBy(
                                            selectedNode.ref.entityName!,
                                            selectedNode.ref.fieldName!,
                                            next
                                          );
                                        }}
                                        className="rounded-xl border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-600"
                                      >
                                        Remove
                                      </button>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })()}
                        <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                          Expression
                        </label>
                        <textarea
                          value={logicDraft}
                          onChange={(event) => setLogicDraft(event.target.value)}
                          placeholder="Define formula or rule..."
                          className="h-24 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
                        />
                    <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                      Depends On
                    </label>
                        <input
                          value={logicDependsOn}
                          readOnly
                          placeholder="Auto from expression"
                          className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-600"
                        />
                        {(() => {
                          const entity = registryLookup.entityMap.get(selectedNode.ref.entityName || "");
                          const fieldNames = new Set((entity?.fields || []).map((field) => field.name));
                          const missing = logicDependsOn
                            .split(",")
                            .map((entry) => entry.trim())
                            .filter((entry) => entry && !fieldNames.has(entry));
                          if (!missing.length) return null;
                          return (
                            <p className="text-xs text-rose-500">
                              Missing fields: {missing.join(", ")}
                            </p>
                          );
                        })()}
                        <button
                          type="button"
                          onClick={applyLogic}
                          className="w-full rounded-2xl bg-slate-900 px-3 py-2 text-sm font-semibold text-white shadow"
                        >
                          Apply rule
                        </button>
                        <div className="mt-4 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                            LLM helper
                          </p>
                          <textarea
                            value={ruleInstruction}
                            onChange={(event) => setRuleInstruction(event.target.value)}
                            placeholder="Describe the logic in natural language..."
                            className="mt-2 h-20 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600"
                          />
                          <div className="mt-2 flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={generateRuleWithLLM}
                              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                            >
                              Generate rule
                            </button>
                            <button
                              type="button"
                              onClick={validateRuleJson}
                              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                            >
                              Validate rule
                            </button>
                            <button
                              type="button"
                              onClick={syncRuleJsonFromExpression}
                              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                            >
                              Sync from expression
                            </button>
                          </div>
                          {ruleHelperStatus && (
                            <p className="mt-2 text-xs text-slate-500">{ruleHelperStatus}</p>
                          )}
                        </div>
                        <div className="mt-4 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Rule JSON
                          </p>
                          <textarea
                            value={ruleJsonDraft}
                            onChange={(event) => setRuleJsonDraft(event.target.value)}
                            placeholder='{"entity":"...","field":"...","rules":[{"expr":"..."}]}'
                            className="mt-2 h-40 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                          />
                          <div className="mt-2 flex gap-2">
                            <button
                              type="button"
                              onClick={applyRuleJson}
                              className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                            >
                              Apply rule JSON
                            </button>
                          </div>
                          {ruleValidationIssues.length > 0 && (
                            <div className="mt-2 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-600">
                              {ruleValidationIssues.map((issue, idx) => (
                                <p key={`${issue}-${idx}`}>{issue}</p>
                              ))}
                            </div>
                          )}
                        </div>
                        <div className="mt-4 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Snowflake SQL
                          </p>
                          <textarea
                            value={ruleSqlDraft}
                            readOnly
                            placeholder="Generate SQL from the rule JSON..."
                            className="mt-2 h-40 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                          />
                          <div className="mt-2 flex gap-2">
                            <button
                              type="button"
                              onClick={generateRuleSql}
                              disabled={busy}
                              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                            >
                              Generate SQL
                            </button>
                          </div>
                          {ruleSqlStatus && (
                            <p className="mt-2 text-xs text-slate-500">{ruleSqlStatus}</p>
                          )}
                        </div>
                      </>
                    ) : (
                      <div className="space-y-3">
                        {selectedReasonerIsGraph && (
                          <>
                            <div className="rounded-2xl border border-indigo-100 bg-indigo-50 px-3 py-2 text-xs text-indigo-700">
                              Graph reasoners execute inside RAI using the selected graph topology. Edit the graph node
                              to change traversal and aggregation.
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <input
                                value={
                                  registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.entity_type || ""
                                }
                                onChange={(event) => updateSelectedReasoner({ entity_type: event.target.value })}
                                placeholder="Entity type"
                                className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                              />
                              <input
                                value={registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.type || ""}
                                onChange={(event) => updateSelectedReasoner({ type: event.target.value })}
                                placeholder="Type"
                                className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                              />
                            </div>
                            <div className="mt-3">
                              <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                                Graph binding
                              </label>
                              <input
                                list="graph-id-options"
                                value={selectedReasonerGraphId}
                                onChange={(event) =>
                                  updateSelectedReasoner({ graph_id: event.target.value.trim() })
                                }
                                placeholder="graph id (required)"
                                className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                              />
                              <datalist id="graph-id-options">
                                {(registryLookup.kgGraphs || []).map((graph) => (
                                  <option key={graph.id} value={graph.id} />
                                ))}
                              </datalist>
                              {(() => {
                                if (!selectedReasonerGraphId) {
                                  return (
                                    <p className="mt-2 text-xs text-amber-600">
                                      Missing graph id. Graph reasoners must bind to a graph.
                                    </p>
                                  );
                                }
                                const exists = (registryLookup.kgGraphs || []).some(
                                  (graph) => graph.id === selectedReasonerGraphId
                                );
                                if (!exists) {
                                  return (
                                    <p className="mt-2 text-xs text-rose-600">
                                      Graph id not found in KG graphs.
                                    </p>
                                  );
                                }
                                return (
                                  <p className="mt-2 text-xs text-emerald-600">
                                    Bound to graph {selectedReasonerGraphId}.
                                  </p>
                                );
                              })()}
                              <div className="mt-2 flex gap-2">
                                <button
                                  type="button"
                                  onClick={() => {
                                    if (!selectedReasonerGraphId) return;
                                    setSelectedId(graphNodeId(selectedReasonerGraphId));
                                  }}
                                  className="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold text-slate-600"
                                >
                                  Open graph
                                </button>
                              </div>
                            </div>
                          </>
                        )}
                        {!selectedReasonerIsGraph && (
                          <div className="space-y-3">
                        <div className="grid grid-cols-2 gap-2">
                          <input
                            value={
                              registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.entity_type || ""
                            }
                            onChange={(event) => updateSelectedReasoner({ entity_type: event.target.value })}
                            placeholder="Entity type"
                            className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                          />
                          <input
                            value={registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.type || ""}
                            onChange={(event) => updateSelectedReasoner({ type: event.target.value })}
                            placeholder="Type"
                            className="rounded-xl border border-slate-200 px-2 py-1 text-xs text-slate-600"
                          />
                        </div>
                        <div className="mt-3">
                          <label className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Graph binding
                          </label>
                          <input
                            list="graph-id-options"
                            value={
                              registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.graph_id || ""
                            }
                            onChange={(event) =>
                              updateSelectedReasoner({ graph_id: event.target.value.trim() })
                            }
                            placeholder="graph id (optional)"
                            className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                          />
                          <datalist id="graph-id-options">
                            {(registryLookup.kgGraphs || []).map((graph) => (
                              <option key={graph.id} value={graph.id} />
                            ))}
                          </datalist>
                          {(() => {
                            const graphId =
                              registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.graph_id || "";
                            if (!graphId) {
                              return (
                                <p className="mt-2 text-xs text-amber-600">
                                  Unbound to graph (legacy reasoner).
                                </p>
                              );
                            }
                            const exists = (registryLookup.kgGraphs || []).some((graph) => graph.id === graphId);
                            if (!exists) {
                              return (
                                <p className="mt-2 text-xs text-rose-600">
                                  Graph id not found in KG graphs.
                                </p>
                              );
                            }
                            return (
                              <p className="mt-2 text-xs text-emerald-600">Bound to graph {graphId}.</p>
                            );
                          })()}
                        </div>
                        {reasonerOutputMeta && (
                          <datalist id={`reasoner-output-${selectedReasoner?.id || "unknown"}`}>
                            {reasonerOutputMeta.options.map((option) => (
                              <option key={option} value={option} />
                            ))}
                          </datalist>
                        )}
                        <div className="flex items-center justify-between">
                          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Outputs
                          </p>
                          <button
                            type="button"
                            onClick={addReasonerOutput}
                            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                          >
                            + Output
                          </button>
                        </div>
                        {reasonerOutputMeta?.entityNames?.length ? (
                          <p className="mt-2 text-xs text-slate-500">
                            Entity type: {selectedReasoner?.entity_type || "generic"} - Entities:{" "}
                            {reasonerOutputMeta.entityNames.join(", ")}
                          </p>
                        ) : (
                          <p className="mt-2 text-xs text-rose-500">
                            No entities found for entity_type {selectedReasoner?.entity_type || "generic"}.
                          </p>
                        )}
                        <div className="space-y-2">
                          {(registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.outputs || []).length ===
                            0 && <p className="text-xs text-slate-500">No outputs yet.</p>}
                          {(registryLookup.reasonerMap.get(selectedNode.ref.reasonerId || "")?.outputs || []).map(
                            (output, index) => (
                              <div key={`${output}-${index}`} className="rounded-2xl border border-slate-200 p-3">
                                <div className="flex items-center gap-2">
                                  <input
                                    value={output}
                                    list={`reasoner-output-${selectedReasoner?.id || "unknown"}`}
                                    onChange={(event) => updateReasonerOutput(index, event.target.value)}
                                    placeholder="output_field"
                                    className="flex-1 rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                  />
                                  <button
                                    type="button"
                                    onClick={() => removeReasonerOutput(index)}
                                    className="rounded-xl border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-600"
                                  >
                                    Remove
                                  </button>
                                </div>
                                {(() => {
                                  const info = output ? reasonerOutputMeta?.fieldMap.get(output) : null;
                                  if (!output) {
                                    return (
                                      <p className="mt-2 text-xs text-slate-400">Select an output field.</p>
                                    );
                                  }
                                  if (!info) {
                                    return (
                                      <p className="mt-2 text-xs text-rose-500">
                                        Output not found on entity type {selectedReasoner?.entity_type || "generic"}.
                                      </p>
                                    );
                                  }
                                  return (
                                    <div className="mt-2 text-xs text-slate-500">
                                      <p>
                                        {info.field.derived ? "derived" : "base"} - {info.entity.name}
                                      </p>
                                      <p className="mt-1 text-slate-600">
                                        {fieldExpressionPreview(registry, info.entity.name, info.field)}
                                      </p>
                                      {info.field.depends_on?.length ? (
                                        <p className="mt-1 text-slate-400">
                                          depends_on: {info.field.depends_on.join(", ")}
                                        </p>
                                      ) : null}
                                    </div>
                                  );
                                })()}
                              </div>
                            )
                          )}
                        </div>
                        <div className="mt-3 flex items-center justify-between">
                          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Signals
                          </p>
                          <button
                            type="button"
                            onClick={addReasonerSignal}
                            className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-600"
                          >
                            + Signal
                          </button>
                        </div>
                        <div className="space-y-3">
                          {reasonerSignalsDraft.length === 0 && (
                            <p className="text-xs text-slate-500">No signals yet.</p>
                          )}
                          {reasonerSignalsDraft.map((signal, index) => (
                            <div key={`${signal.name}-${index}`} className="rounded-2xl border border-slate-200 p-3">
                              <div className="grid grid-cols-2 gap-2">
                                <input
                                  value={signal.name}
                                  onChange={(event) => updateReasonerSignal(index, { name: event.target.value })}
                                  placeholder="name"
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                />
                                <input
                                  value={signal.metric_field}
                                  onChange={(event) =>
                                    updateReasonerSignal(index, { metric_field: event.target.value })
                                  }
                                  placeholder="metric_field"
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                />
                                <input
                                  value={signal.threshold}
                                  type="number"
                                  onChange={(event) =>
                                    updateReasonerSignal(index, { threshold: Number(event.target.value) || 0 })
                                  }
                                  placeholder="threshold"
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                />
                                <select
                                  value={signal.direction}
                                  onChange={(event) =>
                                    updateReasonerSignal(index, { direction: event.target.value })
                                  }
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                >
                                  <option value="below">below</option>
                                  <option value="above">above</option>
                                  <option value="equals">equals</option>
                                </select>
                                <input
                                  value={signal.weight ?? ""}
                                  type="number"
                                  onChange={(event) =>
                                    updateReasonerSignal(index, {
                                      weight: event.target.value === "" ? undefined : Number(event.target.value),
                                    })
                                  }
                                  placeholder="weight"
                                  className="rounded-xl border border-slate-200 px-2 py-1 text-xs"
                                />
                                <button
                                  type="button"
                                  onClick={() => removeReasonerSignal(index)}
                                  className="rounded-xl border border-rose-200 bg-rose-50 px-2 py-1 text-xs text-rose-600"
                                >
                                  Remove
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-3 rounded-2xl border border-slate-100 bg-slate-50 px-3 py-3">
                          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                            Drilldown plan
                          </p>
                          <textarea
                            value={reasonerPlanDraft}
                            onChange={(event) => setReasonerPlanDraft(event.target.value)}
                            placeholder='{"steps":[...]}'
                            className="mt-2 h-40 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                          />
                          <div className="mt-2 flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={applyReasonerPlanJson}
                              className="rounded-xl bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow"
                            >
                              Apply plan JSON
                            </button>
                            <button
                              type="button"
                              onClick={generateReasonerPlanWithLLM}
                              disabled={busy}
                              className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                            >
                              Generate plan
                            </button>
                          </div>
                          <textarea
                            value={reasonerPlanInstruction}
                            onChange={(event) => setReasonerPlanInstruction(event.target.value)}
                            placeholder="Describe the drilldown steps in natural language..."
                            className="mt-2 h-16 w-full rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600"
                          />
                          {reasonerPlanStatus && (
                            <p className="mt-2 text-xs text-slate-500">{reasonerPlanStatus}</p>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={applyLogic}
                          className="w-full rounded-2xl bg-slate-900 px-3 py-2 text-sm font-semibold text-white shadow"
                        >
                          Apply reasoner rules
                      </button>
                      </div>
                        )}
                          </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <p className="mt-4 text-sm text-slate-500">Select a node to edit details.</p>
            )}
          </div>
        </motion.aside>
      </main>
    </div>
  );
}
