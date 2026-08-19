const INACTIVE_NODE_MODES = new Set([2, 4]); // LiteGraph NEVER and BYPASS

function asString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function modelKey(model) {
  return `${model.directory}\u0000${model.name}\u0000${model.url}`;
}

function getGraphNodes(graph) {
  if (Array.isArray(graph?.nodes)) return graph.nodes;
  if (Array.isArray(graph?._nodes)) return graph._nodes;
  return [];
}

function nodeLabel(node) {
  const title = asString(node?.title) || asString(node?.type) || "Node";
  const id = node?.id === undefined || node?.id === null ? "?" : String(node.id);
  return `${title} (#${id})`;
}

function activeWorkflowModels(app) {
  const models = app?.extensionManager?.workflow?.activeWorkflow?.activeState?.models;
  return Array.isArray(models) ? models : [];
}

function modelMetadataKey(name, directory) {
  return `${asString(directory)}\u0000${asString(name)}`;
}

function nodeStringWidgetValues(node) {
  const values = new Set();
  const addValue = (value) => {
    if (typeof value === "string" && value.trim()) {
      values.add(value.trim().replaceAll("\\", "/"));
    } else if (Array.isArray(value)) {
      value.forEach(addValue);
    }
  };
  if (Array.isArray(node?.widgets_values)) node.widgets_values.forEach(addValue);
  if (Array.isArray(node?.widgets)) {
    for (const widget of node.widgets) addValue(widget?.value);
  }
  return values;
}

function metadataMatchesSelectedWidget(metadata, selectedValues) {
  const name = asString(metadata?.name).replaceAll("\\", "/");
  if (!name) return false;
  for (const value of selectedValues) {
    if (value === name || value.endsWith(`/${name}`)) return true;
  }
  return false;
}

/**
 * Return only model metadata referenced by active, currently selected widgets.
 * Root workflow metadata enriches live node selections; it is never downloaded
 * merely because it exists in the serialized workflow.
 */
export function scanCurrentGraphModels(app) {
  const rootGraph = app?.rootGraph ?? app?.graph;
  if (!rootGraph) return [];

  const byKey = new Map();
  const visitedGraphs = new WeakSet();
  const workflowMetadata = new Map();
  const workflowMetadataByName = new Map();
  for (const metadata of activeWorkflowModels(app)) {
    const name = asString(metadata?.name);
    if (!name) continue;
    workflowMetadata.set(modelMetadataKey(name, metadata?.directory), metadata);
    if (!workflowMetadataByName.has(name)) workflowMetadataByName.set(name, metadata);
  }

  function visitGraph(graph, parentPath) {
    if (!graph || (typeof graph !== "object" && typeof graph !== "function")) return;
    if (visitedGraphs.has(graph)) return;
    visitedGraphs.add(graph);

    for (const node of getGraphNodes(graph)) {
      if (INACTIVE_NODE_MODES.has(Number(node?.mode))) continue;
      const currentPath = [...parentPath, nodeLabel(node)];
      const embeddedModels = node?.properties?.models;
      const selectedValues = nodeStringWidgetValues(node);
      const selectedMetadata = new Map();

      for (const metadata of workflowMetadata.values()) {
        if (metadataMatchesSelectedWidget(metadata, selectedValues)) {
          selectedMetadata.set(
            modelMetadataKey(metadata?.name, metadata?.directory),
            metadata,
          );
        }
      }
      if (Array.isArray(embeddedModels)) {
        for (const metadata of embeddedModels) {
          if (!metadataMatchesSelectedWidget(metadata, selectedValues)) continue;
          const embeddedName = asString(metadata?.name);
          const enriched = {
            ...(workflowMetadataByName.get(embeddedName) ?? {}),
            ...(workflowMetadata.get(modelMetadataKey(embeddedName, metadata?.directory)) ?? {}),
            ...metadata,
          };
          selectedMetadata.set(
            modelMetadataKey(enriched?.name, enriched?.directory),
            enriched,
          );
        }
      }

      for (const metadata of selectedMetadata.values()) {
        const name = asString(metadata?.name);
        const url = asString(metadata?.url);
        const directory = asString(metadata?.directory);
        if (!name || !url || !directory) continue;
        const candidate = {
          name,
          url,
          directory,
          sources: [currentPath.join(" → ")],
        };
        const key = modelKey(candidate);
        const existing = byKey.get(key);
        if (existing) {
          for (const source of candidate.sources) {
            if (!existing.sources.includes(source)) existing.sources.push(source);
          }
        } else {
          byKey.set(key, candidate);
        }
      }

      if (node?.subgraph) visitGraph(node.subgraph, currentPath);
    }
  }

  visitGraph(rootGraph, []);
  return [...byKey.values()].sort(
    (left, right) =>
      left.directory.localeCompare(right.directory) || left.name.localeCompare(right.name),
  );
}
