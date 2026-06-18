(function () {
  const graphScript = document.getElementById("ai-control-graph-data");
  const svg = document.getElementById("ai-control-graph");
  const search = document.getElementById("ai-control-search");
  const count = document.getElementById("ai-control-node-count");
  const selection = document.getElementById("ai-control-selection");
  const pipelineScopeSelect = document.getElementById("ai-control-pipeline-scope");
  const depthSelect = document.getElementById("ai-control-depth");
  const clearFocus = document.getElementById("ai-control-clear-focus");
  const isolateToggle = document.getElementById("ai-control-isolate");
  if (!graphScript || !svg) {
    return;
  }

  const graph = JSON.parse(graphScript.textContent);
  const defaultKindLabels = {
    provider: "Provider",
    model: "Model",
    knowledge: "Knowledge",
    tool: "Tool",
    agent: "Agent",
    pipeline: "Pipeline",
    step: "Step",
  };
  const defaultKindColors = {
    provider: "#0f766e",
    model: "#2563eb",
    knowledge: "#7c3aed",
    tool: "#c2410c",
    agent: "#be123c",
    pipeline: "#047857",
    step: "#4b5563",
  };
  const defaultRelationColors = {
    serves: "#38bdf8",
    runs: "#f43f5e",
    informs: "#a78bfa",
    enables: "#fb923c",
    contains: "#34d399",
    calls: "#f472b6",
    next: "#94a3b8",
    fallback: "#f59e0b",
  };
  const kindOrder = graph.kindOrder || ["provider", "model", "knowledge", "tool", "agent", "pipeline", "step"];
  const kindLabels = { ...defaultKindLabels, ...(graph.kindLabels || {}) };
  const kindColors = { ...defaultKindColors, ...(graph.kindColors || {}) };
  const relationColors = { ...defaultRelationColors, ...(graph.relationColors || {}) };
  const compactGraph = kindOrder.length <= 6 && graph.nodes.length <= 12;
  const nodeWidth = compactGraph ? 164 : 176;
  const nodeHeight = 72;
  const columnGap = compactGraph ? 38 : 62;
  const rowGap = 24;
  const topOffset = 48;
  const leftOffset = 34;
  let activeKind = "all";
  let query = "";
  let selectedId = "";
  let focusDepth = 1;
  let isolateFocus = true;
  let pipelineScope = "all";

  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const pipelineScopes = new Map((graph.pipelineScopes || []).map((scope) => [scope.id, new Set(scope.node_ids || [])]));
  const incomingByNode = new Map();
  const outgoingByNode = new Map();
  const positions = new Map();
  let tooltip = null;

  graph.edges.forEach((edge) => {
    if (!outgoingByNode.has(edge.source)) outgoingByNode.set(edge.source, []);
    if (!incomingByNode.has(edge.target)) incomingByNode.set(edge.target, []);
    outgoingByNode.get(edge.source).push(edge);
    incomingByNode.get(edge.target).push(edge);
  });

  function statusLabel(status) {
    if (status === "ok") return "OK";
    if (status === "warning") return "WARN";
    if (status === "error") return "ERROR";
    if (status === "inactive") return "OFF";
    return "INFO";
  }

  function shorten(text, limit) {
    const value = String(text || "");
    if (value.length <= limit) return value;
    return `${value.slice(0, limit - 1)}...`;
  }

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function createSvg(name, attrs) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attrs || {}).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function tooltipElement() {
    if (tooltip) {
      return tooltip;
    }
    tooltip = document.createElement("div");
    tooltip.className = "ai-graph-tooltip";
    document.body.appendChild(tooltip);
    return tooltip;
  }

  function showTooltip(event, html) {
    const element = tooltipElement();
    element.innerHTML = html;
    element.style.left = `${event.clientX + 14}px`;
    element.style.top = `${event.clientY + 14}px`;
    element.classList.add("is-visible");
  }

  function hideTooltip() {
    if (tooltip) {
      tooltip.classList.remove("is-visible");
    }
  }

  function traceList(title, rows) {
    if (!rows || !rows.length) {
      return "";
    }
    const items = rows
      .map((row) => {
        const dot = `<span class="ai-legend-dot ai-legend-dot--${escapeHtml(row.kind)}"></span>`;
        return `<li>${dot}<strong>${escapeHtml(row.node)}</strong><small>${escapeHtml(row.label)} / ${escapeHtml(row.status)}</small></li>`;
      })
      .join("");
    return `<h3>${escapeHtml(title)}</h3><ul class="ai-trace-list">${items}</ul>`;
  }

  function selectedNeighborhood(nodeId, depth) {
    if (!nodeId) {
      return { nodes: new Set(), edges: new Set(), incoming: new Set(), outgoing: new Set() };
    }
    const nodes = new Set([nodeId]);
    const edges = new Set();
    const incoming = new Set();
    const outgoing = new Set();
    let frontier = new Set([nodeId]);

    for (let level = 0; level < depth; level += 1) {
      const next = new Set();
      frontier.forEach((currentId) => {
        (incomingByNode.get(currentId) || []).forEach((edge) => {
          const edgeKey = `${edge.source}->${edge.target}:${edge.label}`;
          edges.add(edgeKey);
          incoming.add(edgeKey);
          nodes.add(edge.source);
          next.add(edge.source);
        });
        (outgoingByNode.get(currentId) || []).forEach((edge) => {
          const edgeKey = `${edge.source}->${edge.target}:${edge.label}`;
          edges.add(edgeKey);
          outgoing.add(edgeKey);
          nodes.add(edge.target);
          next.add(edge.target);
        });
      });
      frontier = next;
    }
    return { nodes, edges, incoming, outgoing };
  }

  function edgeKeyFromElement(element) {
    return `${element.dataset.source}->${element.dataset.target}:${element.dataset.label || ""}`;
  }

  function selectionAnalytics(node) {
    const incoming = (node.trace && node.trace.incoming) || [];
    const outgoing = (node.trace && node.trace.outgoing) || [];
    const affectedKinds = new Set([...incoming, ...outgoing].map((item) => item.kind));
    const warningLinks = [...incoming, ...outgoing].filter((item) => item.status !== "ok").length;
    return `
      <div class="ai-selection-metrics">
        <article><span>Incoming</span><strong>${incoming.length}</strong></article>
        <article><span>Outgoing</span><strong>${outgoing.length}</strong></article>
        <article><span>Types touched</span><strong>${affectedKinds.size}</strong></article>
        <article><span>Risk links</span><strong>${warningLinks}</strong></article>
      </div>
    `;
  }

  function emptySelectionHtml() {
    return `
      <h2>Selected node</h2>
      <p>Select a node in the graph to inspect its role and open the admin record.</p>
    `;
  }

  function layoutNodes() {
    kindOrder.forEach((kind, columnIndex) => {
      const nodes = graph.nodes.filter((node) => node.kind === kind);
      nodes.forEach((node, rowIndex) => {
        positions.set(node.id, {
          x: leftOffset + columnIndex * (nodeWidth + columnGap),
          y: topOffset + rowIndex * (nodeHeight + rowGap),
        });
      });
    });
  }

  function drawHeaders() {
    kindOrder.forEach((kind, columnIndex) => {
      const x = leftOffset + columnIndex * (nodeWidth + columnGap);
      const text = createSvg("text", {
        x,
        y: 24,
        class: "ai-column__label",
      });
      if (kindColors[kind]) {
        text.style.fill = kindColors[kind];
      }
      text.textContent = kindLabels[kind];
      svg.appendChild(text);
    });
  }

  function drawArrowMarkers() {
    const defs = createSvg("defs");
    Object.entries(relationColors).forEach(([label, color]) => {
      const markerId = label.replace(/[^a-zA-Z0-9_-]/g, "-");
      const marker = createSvg("marker", {
        id: `ai-arrow-${markerId}`,
        markerWidth: 10,
        markerHeight: 10,
        refX: 8,
        refY: 3,
        orient: "auto",
        markerUnits: "strokeWidth",
      });
      const arrow = createSvg("path", {
        d: "M 0 0 L 8 3 L 0 6 Z",
        fill: color,
      });
      marker.appendChild(arrow);
      defs.appendChild(marker);
    });
    const fallbackMarker = createSvg("marker", {
      id: "ai-arrow-default",
      markerWidth: 10,
      markerHeight: 10,
      refX: 8,
      refY: 3,
      orient: "auto",
      markerUnits: "strokeWidth",
    });
    fallbackMarker.appendChild(createSvg("path", { d: "M 0 0 L 8 3 L 0 6 Z", fill: "#9aa3af" }));
    defs.appendChild(fallbackMarker);
    svg.appendChild(defs);
  }

  function edgeGeometry(source, target) {
    const sourceCenterX = source.x + nodeWidth / 2;
    const sourceCenterY = source.y + nodeHeight / 2;
    const targetCenterX = target.x + nodeWidth / 2;
    const targetCenterY = target.y + nodeHeight / 2;
    const sameColumn = Math.abs(source.x - target.x) < 12;
    const backwards = target.x < source.x;
    const mostlyVertical = Math.abs(sourceCenterX - targetCenterX) < nodeWidth;

    if (sameColumn || mostlyVertical) {
      const direction = targetCenterY >= sourceCenterY ? 1 : -1;
      const side = source.x + nodeWidth + 22;
      const startX = source.x + nodeWidth;
      const startY = sourceCenterY + direction * 12;
      const endX = target.x + nodeWidth;
      const endY = targetCenterY - direction * 12;
      return {
        d: `M ${startX} ${startY} C ${side} ${startY}, ${side} ${endY}, ${endX} ${endY}`,
        labelX: side + 6,
        labelY: (startY + endY) / 2,
      };
    }

    if (backwards) {
      const direction = targetCenterY >= sourceCenterY ? 1 : -1;
      const lane = Math.max(source.x, target.x) + nodeWidth + 28;
      const startX = source.x;
      const startY = sourceCenterY + direction * 10;
      const endX = target.x + nodeWidth;
      const endY = targetCenterY - direction * 10;
      return {
        d: `M ${startX} ${startY} C ${lane} ${startY}, ${lane} ${endY}, ${endX} ${endY}`,
        labelX: lane + 6,
        labelY: (startY + endY) / 2,
      };
    }

    const startX = source.x + nodeWidth;
    const startY = sourceCenterY;
    const endX = target.x + 2;
    const endY = targetCenterY;
    const distanceX = Math.max(endX - startX, 1);
    const lift = Math.min(Math.max(Math.abs(endY - startY) * 0.35, 28), 140);
    const direction = endY >= startY ? -1 : 1;
    const controlOffset = Math.max(distanceX * 0.42, 44);
    return {
      d: `M ${startX} ${startY} C ${startX + controlOffset} ${startY + direction * lift}, ${endX - controlOffset} ${endY + direction * lift}, ${endX} ${endY}`,
      labelX: (startX + endX) / 2,
      labelY: (startY + endY) / 2 + direction * lift * 0.65,
    };
  }

  function drawEdges() {
    const edgeLayer = createSvg("g", { class: "ai-edge-layer" });
    graph.edges.forEach((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return;

      const geometry = edgeGeometry(source, target);
      const group = createSvg("g", {
        class: "ai-edge",
        "data-source": edge.source,
        "data-target": edge.target,
        "data-label": edge.label || "",
        "data-status": edge.status || "ok",
      });
      const path = createSvg("path", {
        d: geometry.d,
      });
      const label = createSvg("text", {
        x: geometry.labelX,
        y: geometry.labelY,
      });
      label.textContent = edge.label || "";
      const sourceNode = byId.get(edge.source);
      const targetNode = byId.get(edge.target);
      const safeMarkerLabel = relationColors[edge.label] ? edge.label.replace(/[^a-zA-Z0-9_-]/g, "-") : "default";
      const edgeColor = relationColors[edge.label] || (targetNode && kindColors[targetNode.kind]) || "#9aa3af";
      path.style.stroke = edgeColor;
      path.setAttribute("marker-end", `url(#ai-arrow-${safeMarkerLabel})`);
      group.appendChild(path);
      group.appendChild(label);
      group.addEventListener("mousemove", (event) => {
        showTooltip(
          event,
          `
            <strong>${escapeHtml(edge.label || "connection")}</strong>
            <span>${escapeHtml(sourceNode ? sourceNode.label : edge.source)} -> ${escapeHtml(targetNode ? targetNode.label : edge.target)}</span>
            <small>Status: ${escapeHtml(edge.status || "ok")}</small>
          `
        );
      });
      group.addEventListener("mouseleave", hideTooltip);
      edgeLayer.appendChild(group);
    });
    svg.appendChild(edgeLayer);
  }

  function drawNodes() {
    const nodeLayer = createSvg("g", { class: "ai-node-layer" });
    graph.nodes.forEach((node) => {
      const pos = positions.get(node.id);
      if (!pos) return;
      const group = createSvg("g", {
        class: "ai-node",
        transform: `translate(${pos.x} ${pos.y})`,
        tabindex: "0",
        role: "button",
        "data-id": node.id,
        "data-kind": node.kind,
        "data-search": `${node.label} ${node.kind} ${node.detail}`.toLowerCase(),
      });

      const body = createSvg("rect", { width: nodeWidth, height: nodeHeight, class: "ai-node__body" });
      if (kindColors[node.kind]) {
        body.style.stroke = kindColors[node.kind];
      }
      group.appendChild(body);

      const kindLabel = kindLabels[node.kind] || node.kind;
      const kindWidth = Math.min(Math.max(kindLabel.length * 6 + 18, 58), 92);
      const kindBadge = createSvg("rect", {
        x: 10,
        y: 8,
        width: kindWidth,
        height: 18,
        rx: 9,
        class: "ai-node__kind-badge",
      });
      if (kindColors[node.kind]) {
        kindBadge.setAttribute("fill", kindColors[node.kind]);
      }
      group.appendChild(kindBadge);

      const kind = createSvg("text", { x: 19, y: 21, class: "ai-node__kind-label" });
      kind.textContent = kindLabel;
      group.appendChild(kind);

      const label = createSvg("text", { x: 12, y: 38, class: "ai-node__label" });
      label.textContent = shorten(node.label, 22);
      group.appendChild(label);

      const detail = createSvg("text", { x: 12, y: 58, class: "ai-node__detail" });
      detail.textContent = shorten(node.detail, 25);
      group.appendChild(detail);

      const badge = createSvg("rect", {
        x: nodeWidth - 54,
        y: 10,
        width: 42,
        height: 18,
        rx: 9,
        class: `ai-status-${node.status || "unknown"}`,
      });
      group.appendChild(badge);

      const badgeText = createSvg("text", {
        x: nodeWidth - 33,
        y: 23,
        "text-anchor": "middle",
        class: "ai-node__status",
      });
      badgeText.textContent = statusLabel(node.status);
      group.appendChild(badgeText);

      group.addEventListener("mousemove", (event) => {
        const trace = node.trace || {};
        showTooltip(
          event,
          `
            <strong>${escapeHtml(node.label)}</strong>
            <span>${escapeHtml(kindLabels[node.kind] || node.kind)} / ${escapeHtml(node.status)}</span>
            <small>${escapeHtml(node.detail || "No detail")}</small>
            <small>${((trace.incoming || []).length)} incoming, ${((trace.outgoing || []).length)} outgoing</small>
          `
        );
      });
      group.addEventListener("mouseleave", hideTooltip);

      group.addEventListener("click", () => selectNode(node.id));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectNode(node.id);
        }
      });
      nodeLayer.appendChild(group);
    });
    svg.appendChild(nodeLayer);
  }

  function selectNode(nodeId) {
    if (selectedId === nodeId) {
      selectedId = "";
      if (selection) {
        selection.innerHTML = emptySelectionHtml();
      }
      applyFilters();
      return;
    }
    selectedId = nodeId;
    const node = byId.get(nodeId);
    applyFilters();
    if (!node || !selection) return;
    const meta = Object.entries(node.meta || {})
      .map(([key, value]) => `<li><strong>${escapeHtml(key)}</strong>: ${escapeHtml(value)}</li>`)
      .join("");
    const safeUrl = node.url || "";
    const trace = node.trace || {};
    selection.innerHTML = `
      <h2>Selected node</h2>
      <p><strong>${escapeHtml(node.label)}</strong></p>
      ${selectionAnalytics(node)}
      <ul class="ai-control__list">
        <li><strong>Type</strong>: ${escapeHtml(kindLabels[node.kind] || node.kind)}</li>
        <li><strong>Status</strong>: ${escapeHtml(node.status)}</li>
        <li><strong>Detail</strong>: ${escapeHtml(node.detail || "-")}</li>
        ${meta}
      </ul>
      ${traceList("Receives from", trace.incoming)}
      ${traceList("Sends to", trace.outgoing)}
      ${safeUrl ? `<a class="ai-control__selection-link button" href="${escapeHtml(safeUrl)}">Open record</a>` : ""}
    `;
  }

  function matchesFilter(nodeElement) {
    const matchesKind = activeKind === "all" || nodeElement.dataset.kind === activeKind;
    const matchesQuery = !query || nodeElement.dataset.search.includes(query);
    const scope = pipelineScope === "all" ? null : pipelineScopes.get(pipelineScope);
    const matchesPipeline = !scope || scope.has(nodeElement.dataset.id);
    return matchesKind && matchesQuery && matchesPipeline;
  }

  function applyFilters() {
    const visible = new Set();
    const focus = selectedNeighborhood(selectedId, focusDepth);
    document.querySelectorAll(".ai-node").forEach((element) => {
      const show = matchesFilter(element);
      const isFocused = !selectedId || focus.nodes.has(element.dataset.id);
      element.classList.toggle("is-hidden", !show || (selectedId && isolateFocus && !isFocused));
      element.classList.toggle("is-selected", element.dataset.id === selectedId);
      element.classList.toggle("is-dimmed", show && selectedId && !isFocused);
      element.classList.toggle("is-neighbor", show && selectedId && isFocused && element.dataset.id !== selectedId);
      if (show && (!selectedId || !isolateFocus || isFocused)) visible.add(element.dataset.id);
    });
    document.querySelectorAll(".ai-edge").forEach((element) => {
      const show = visible.has(element.dataset.source) && visible.has(element.dataset.target);
      const edgeKey = edgeKeyFromElement(element);
      const inFocus = !selectedId || focus.edges.has(edgeKey);
      element.classList.toggle("is-hidden", !show || (selectedId && isolateFocus && !inFocus));
      element.classList.toggle("is-dimmed", show && selectedId && !inFocus);
      element.classList.toggle("is-focused", show && selectedId && inFocus);
      element.classList.toggle("is-incoming", show && selectedId && focus.incoming.has(edgeKey));
      element.classList.toggle("is-outgoing", show && selectedId && focus.outgoing.has(edgeKey));
    });
    if (count) {
      const suffix = selectedId ? ` / ${focus.nodes.size} in focus` : "";
      count.textContent = `${visible.size} visible nodes${suffix}`;
    }
    if (selectedId && !visible.has(selectedId)) {
      selectedId = "";
      document.querySelectorAll(".ai-node").forEach((element) => element.classList.remove("is-selected"));
    }
  }

  function setCanvasSize() {
    const maxRows = Math.max(...kindOrder.map((kind) => graph.nodes.filter((node) => node.kind === kind).length), 1);
    const width = leftOffset * 2 + kindOrder.length * nodeWidth + (kindOrder.length - 1) * columnGap;
    const height = topOffset + maxRows * (nodeHeight + rowGap) + 24;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", width);
    svg.setAttribute("height", Math.max(height, compactGraph ? 360 : 560));
  }

  function bindFilters() {
    if (search) {
      search.addEventListener("input", () => {
        query = search.value.trim().toLowerCase();
        applyFilters();
      });
    }
    document.querySelectorAll(".ai-chip").forEach((button) => {
      button.addEventListener("click", () => {
        if (!button.dataset.kind) return;
        activeKind = button.dataset.kind || "all";
        document.querySelectorAll(".ai-chip").forEach((item) => item.classList.remove("is-active"));
        button.classList.add("is-active");
        applyFilters();
      });
    });
    if (depthSelect) {
      depthSelect.addEventListener("change", () => {
        focusDepth = Number(depthSelect.value || 2);
        applyFilters();
      });
    }
    if (isolateToggle) {
      isolateToggle.addEventListener("change", () => {
        isolateFocus = isolateToggle.checked;
        applyFilters();
      });
    }
    if (pipelineScopeSelect) {
      pipelineScopeSelect.addEventListener("change", () => {
        pipelineScope = pipelineScopeSelect.value || "all";
        if (selectedId && pipelineScope !== "all" && !pipelineScopes.get(pipelineScope)?.has(selectedId)) {
          selectedId = "";
          if (selection) {
            selection.innerHTML = emptySelectionHtml();
          }
        }
        applyFilters();
      });
    }
    if (clearFocus) {
      clearFocus.addEventListener("click", () => {
        selectedId = "";
        if (selection) {
          selection.innerHTML = emptySelectionHtml();
        }
        applyFilters();
      });
    }
  }

  layoutNodes();
  setCanvasSize();
  drawArrowMarkers();
  drawHeaders();
  drawEdges();
  drawNodes();
  bindFilters();
  applyFilters();
})();
