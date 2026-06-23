/* ============================================================
   AI Hub · Mission Deck — graph engine + UI helpers
   Renders the connection graph (control center) and the GAME
   decision graph from the JSON embedded in #ai-control-graph-data.
   Dark "observatory" canvas in both admin themes; HTML node cards
   layered over a <canvas> edge layer. Reads node.trace / node.meta
   / node.url straight from the existing view context — no new
   endpoints. Also wires the tab strip beneath the graph.
   ============================================================ */
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    initTabs();
    initAttention();
    var root = document.querySelector("[data-mc-graph]");
    var dataEl = document.getElementById("ai-control-graph-data");
    if (root && dataEl) {
      try {
        initGraph(root, JSON.parse(dataEl.textContent));
      } catch (err) {
        if (window.console) console.error("Mission Deck graph failed to initialise:", err);
      }
    }
  });

  /* ---------- tab strip (progressive disclosure) ---------- */
  function initTabs() {
    document.querySelectorAll("[data-mc-tabs]").forEach(function (group) {
      var tabs = Array.prototype.slice.call(group.querySelectorAll("[data-mc-tab]"));
      if (!tabs.length) return;
      var panels = {};
      tabs.forEach(function (tab) {
        var id = tab.getAttribute("data-mc-tab");
        var panel = document.getElementById(id);
        if (panel) panels[id] = panel;
      });
      function activate(id) {
        tabs.forEach(function (tab) {
          var on = tab.getAttribute("data-mc-tab") === id;
          tab.classList.toggle("is-on", on);
          tab.setAttribute("aria-selected", on ? "true" : "false");
          tab.setAttribute("tabindex", on ? "0" : "-1");
        });
        Object.keys(panels).forEach(function (key) {
          panels[key].hidden = key !== id;
        });
      }
      tabs.forEach(function (tab, index) {
        tab.addEventListener("click", function () { activate(tab.getAttribute("data-mc-tab")); });
        tab.addEventListener("keydown", function (event) {
          if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
          event.preventDefault();
          var dir = event.key === "ArrowRight" ? 1 : -1;
          var next = tabs[(index + dir + tabs.length) % tabs.length];
          next.focus();
          activate(next.getAttribute("data-mc-tab"));
        });
      });
      var initial = tabs.filter(function (t) { return t.classList.contains("is-on"); })[0] || tabs[0];
      activate(initial.getAttribute("data-mc-tab"));
    });
  }

  /* ---------- attention inbox ---------- */
  function initAttention() {
    var box = document.querySelector("[data-mc-attention]");
    if (!box) return;
    var list = box.querySelector("[data-mc-attn-list]");
    if (!list) return;
    var items = Array.prototype.slice.call(list.querySelectorAll("[data-mc-attn-item]"));
    var filter = box.querySelector("[data-mc-attn-filter]");
    var sort = box.querySelector("[data-mc-attn-sort]");
    var empty = box.querySelector("[data-mc-attn-empty]");
    var openCount = box.querySelector("[data-mc-attn-open-count]");
    var archivedCount = box.querySelector("[data-mc-attn-archived-count]");
    var viewButtons = Array.prototype.slice.call(box.querySelectorAll("[data-mc-attn-view]"));
    var view = "open";
    var storeKey = "aiHubMissionDeckAttention:v1";
    if (!items.length) {
      if (openCount) openCount.textContent = "0";
      if (archivedCount) archivedCount.textContent = "0";
      if (empty) empty.hidden = true;
      return;
    }

    function loadState() {
      try {
        return JSON.parse(window.localStorage.getItem(storeKey) || "{}");
      } catch (err) {
        return {};
      }
    }
    function saveState(state) {
      try {
        window.localStorage.setItem(storeKey, JSON.stringify(state));
      } catch (err) {
        // localStorage may be unavailable in private/locked-down contexts.
      }
    }
    function itemState(state, id) {
      return state[id] || {};
    }
    function isArchived(state, id) {
      return itemState(state, id).archived === true;
    }
    function isSnoozed(state, id) {
      var until = itemState(state, id).snoozedUntil || 0;
      return until && until > Date.now();
    }
    function scoreSeverity(value) {
      return value === "error" ? 2 : value === "warning" ? 1 : 0;
    }
    function matchesFilter(item) {
      var value = filter ? filter.value : "all";
      if (value === "all") return true;
      return item.getAttribute("data-severity") === value || item.getAttribute("data-source") === value;
    }
    function sortItems() {
      var mode = sort ? sort.value : "date";
      items.sort(function (a, b) {
        if (mode === "relevance") return Number(b.dataset.relevance || 0) - Number(a.dataset.relevance || 0);
        if (mode === "severity") return scoreSeverity(b.dataset.severity) - scoreSeverity(a.dataset.severity);
        return Date.parse(b.dataset.timestamp || 0) - Date.parse(a.dataset.timestamp || 0);
      });
      items.forEach(function (item) { list.appendChild(item); });
    }
    function render() {
      var state = loadState();
      var visible = 0, open = 0, archived = 0;
      sortItems();
      items.forEach(function (item) {
        var id = item.getAttribute("data-id");
        var archivedItem = isArchived(state, id);
        var snoozedItem = isSnoozed(state, id);
        var show = matchesFilter(item) && (view === "archived" ? archivedItem : !archivedItem && !snoozedItem);
        item.hidden = !show;
        item.classList.toggle("is-archived", archivedItem);
        item.classList.toggle("is-snoozed", snoozedItem);
        var archiveBtn = item.querySelector("[data-mc-attn-archive]");
        var snoozeBtn = item.querySelector("[data-mc-attn-snooze]");
        var restoreBtn = item.querySelector("[data-mc-attn-restore]");
        if (archiveBtn) archiveBtn.hidden = view === "archived";
        if (snoozeBtn) snoozeBtn.hidden = view === "archived";
        if (restoreBtn) restoreBtn.hidden = view !== "archived";
        if (!archivedItem && !snoozedItem) open += 1;
        if (archivedItem) archived += 1;
        if (show) visible += 1;
      });
      viewButtons.forEach(function (button) {
        var on = button.getAttribute("data-mc-attn-view") === view;
        button.classList.toggle("is-on", on);
        button.setAttribute("aria-selected", on ? "true" : "false");
      });
      if (openCount) openCount.textContent = open;
      if (archivedCount) archivedCount.textContent = archived;
      if (empty) empty.hidden = visible !== 0;
    }
    items.forEach(function (item) {
      var id = item.getAttribute("data-id");
      var archiveBtn = item.querySelector("[data-mc-attn-archive]");
      var snoozeBtn = item.querySelector("[data-mc-attn-snooze]");
      var restoreBtn = item.querySelector("[data-mc-attn-restore]");
      if (archiveBtn) archiveBtn.addEventListener("click", function () {
        var state = loadState();
        state[id] = Object.assign({}, state[id], { archived: true, archivedAt: Date.now() });
        saveState(state);
        render();
      });
      if (snoozeBtn) snoozeBtn.addEventListener("click", function () {
        var state = loadState();
        state[id] = Object.assign({}, state[id], { snoozedUntil: Date.now() + 24 * 60 * 60 * 1000 });
        saveState(state);
        render();
      });
      if (restoreBtn) restoreBtn.addEventListener("click", function () {
        var state = loadState();
        state[id] = Object.assign({}, state[id], { archived: false, snoozedUntil: 0 });
        saveState(state);
        render();
      });
    });
    viewButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        view = button.getAttribute("data-mc-attn-view") || "open";
        render();
      });
    });
    if (filter) filter.addEventListener("change", render);
    if (sort) sort.addEventListener("change", render);
    render();
  }

  /* ---------- graph ---------- */
  var KIND_LABELS = {
    provider: "Provider", model: "Model", knowledge: "Knowledge", tool: "Tool",
    agent: "Agent", pipeline: "Pipeline", step: "Step",
    goal: "Goal", decision: "Decision", action: "Action", memory: "Memory", stop: "Stop"
  };
  // bright "lit" colours that read well on the dark canvas
  var LIT = {
    provider: "#2DD4BF", model: "#60A5FA", knowledge: "#A78BFA", tool: "#FB923C",
    agent: "#FB7185", pipeline: "#34D399", step: "#94A3B8",
    goal: "#60A5FA", decision: "#A78BFA", action: "#FB923C", memory: "#2DD4BF", stop: "#34D399"
  };
  var REL_DEFAULT = {
    serves: "#38bdf8", runs: "#fb7185", informs: "#a78bfa", enables: "#fb923c",
    contains: "#34d399", calls: "#f472b6", next: "#94a3b8", fallback: "#f59e0b",
    guides: "#38bdf8", decides: "#fb7185", acts: "#fb923c", observes: "#a78bfa",
    loops: "#94a3b8", finishes: "#34d399", awaits: "#f59e0b"
  };
  var STATUS_COLOR = {
    ok: "#34D399", success: "#34D399", warning: "#FBBF24", running: "#FBBF24",
    waiting_async: "#FBBF24", pending: "#FBBF24", error: "#FB7185", failed: "#FB7185",
    inactive: "#64748B", unknown: "#94A3B8"
  };
  var STATUS_LABEL = {
    ok: "OK", success: "OK", warning: "WARN", running: "RUN", waiting_async: "WAIT",
    pending: "WAIT", error: "ERROR", failed: "FAIL", inactive: "OFF", unknown: "INFO"
  };

  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function shorten(value, limit) {
    var str = String(value == null ? "" : value);
    return str.length <= limit ? str : str.slice(0, limit - 1) + "…";
  }
  function lighten(hex, amount) {
    var h = String(hex || "#94a3b8").replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    var r = parseInt(h.substr(0, 2), 16), g = parseInt(h.substr(2, 2), 16), b = parseInt(h.substr(4, 2), 16);
    r = Math.round(r + (255 - r) * amount);
    g = Math.round(g + (255 - g) * amount);
    b = Math.round(b + (255 - b) * amount);
    return "rgb(" + r + "," + g + "," + b + ")";
  }
  function rgba(hex, alpha) {
    var h = String(hex || "#94a3b8").replace("#", "");
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    if (h.length !== 6) return hex;
    var r = parseInt(h.substr(0, 2), 16), g = parseInt(h.substr(2, 2), 16), b = parseInt(h.substr(4, 2), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function initGraph(root, graph) {
    var nodes = graph.nodes || [];
    var edges = graph.edges || [];
    if (!nodes.length) return;

    var kindLabels = Object.assign({}, KIND_LABELS, graph.kindLabels || {});
    var relColors = Object.assign({}, REL_DEFAULT, graph.relationColors || {});
    var providedKindColors = graph.kindColors || {};
    function litFor(kind) {
      return LIT[kind] || lighten(providedKindColors[kind] || "#64748b", 0.45);
    }

    var order = graph.kindOrder && graph.kindOrder.length
      ? graph.kindOrder
      : ["provider", "model", "knowledge", "tool", "agent", "pipeline", "step"];
    var present = order.filter(function (k) {
      return nodes.some(function (n) { return n.kind === k; });
    });
    // append any kinds not in the declared order
    nodes.forEach(function (n) { if (present.indexOf(n.kind) === -1) present.push(n.kind); });

    var scopes = {};
    (graph.pipelineScopes || []).forEach(function (s) { scopes[s.id] = s.node_ids || []; });

    var byId = {};
    nodes.forEach(function (n) { byId[n.id] = n; });

    var adjOut = {}, adjIn = {};
    nodes.forEach(function (n) { adjOut[n.id] = []; adjIn[n.id] = []; });
    edges.forEach(function (e) {
      if (adjOut[e.source]) adjOut[e.source].push(e);
      if (adjIn[e.target]) adjIn[e.target].push(e);
    });

    // ---- DOM hooks ----
    var stage = root.querySelector("[data-mc-stage]");
    var inner = root.querySelector("[data-mc-stage-inner]");
    var canvas = root.querySelector("[data-mc-edges]");
    var drawer = root.querySelector("[data-mc-drawer]");
    var drawerBody = root.querySelector("[data-mc-drawer-body]");
    var drawerClose = root.querySelector("[data-mc-drawer-close]");
    var chipWrap = root.querySelector("[data-mc-chips]");
    var search = root.querySelector("[data-mc-search]");
    var depthSel = root.querySelector("[data-mc-depth]");
    var isolateChk = root.querySelector("[data-mc-isolate]");
    var scopeSel = root.querySelector("[data-mc-scope]");
    var fullscreenBtn = root.querySelector("[data-mc-fullscreen]");
    var clearBtn = root.querySelector("[data-mc-clear]");
    var countEl = root.querySelector("[data-mc-count]");
    if (!stage || !inner || !canvas) return;
    var ctx = canvas.getContext("2d");
    var popupWasDragged = false;
    var dragState = null;
    var hoverCard = document.createElement("div");
    hoverCard.className = "mc-hovercard";
    hoverCard.setAttribute("role", "status");
    hoverCard.setAttribute("aria-live", "polite");
    stage.appendChild(hoverCard);

    // ---- layout ----
    var NW = 170, NH = 66, COLGAP = 76, ROWGAP = 24, PADX = 30, PADTOP = 46, PADBOT = 26;
    var pos = {};
    function layout() {
      var maxRows = 1;
      present.forEach(function (kind) {
        var count = nodes.filter(function (n) { return n.kind === kind; }).length;
        if (count > maxRows) maxRows = count;
      });
      var fullH = PADTOP + maxRows * (NH + ROWGAP) - ROWGAP + PADBOT;
      present.forEach(function (kind, ci) {
        var list = nodes.filter(function (n) { return n.kind === kind; });
        var colH = list.length * (NH + ROWGAP) - ROWGAP;
        var startY = PADTOP + ((fullH - PADTOP - PADBOT) - colH) / 2;
        list.forEach(function (n, ri) {
          pos[n.id] = { x: PADX + ci * (NW + COLGAP), y: startY + ri * (NH + ROWGAP) };
        });
      });
      var fullW = PADX * 2 + present.length * NW + (present.length - 1) * COLGAP;
      inner.style.width = fullW + "px";
      inner.style.height = fullH + "px";
      return { w: fullW, h: fullH };
    }

    // ---- build node cards + column headers ----
    var nodeEls = {};
    function build() {
      present.forEach(function (kind, ci) {
        var head = document.createElement("div");
        head.className = "mc-colhead";
        head.textContent = kindLabels[kind] || kind;
        head.style.left = (PADX + ci * (NW + COLGAP)) + "px";
        head.style.color = litFor(kind);
        inner.appendChild(head);
      });
      nodes.forEach(function (n) {
        var lit = litFor(n.kind);
        var el = document.createElement("button");
        el.type = "button";
        el.className = "mc-node";
        el.style.setProperty("--kc", lit);
        el.style.left = pos[n.id].x + "px";
        el.style.top = pos[n.id].y + "px";
        el.setAttribute("data-id", n.id);
        var stColor = STATUS_COLOR[n.status] || STATUS_COLOR.unknown;
        var relationCount = (adjIn[n.id] || []).length + (adjOut[n.id] || []).length;
        el.setAttribute(
          "aria-label",
          (kindLabels[n.kind] || n.kind) + ": " + n.label + ". " +
          (STATUS_LABEL[n.status] || "INFO") + ". " + relationCount + " links."
        );
        el.innerHTML =
          '<span class="mc-node__top">' +
            '<span class="mc-node__kind" style="background:' + lit + '">' + esc(kindLabels[n.kind] || n.kind) + '</span>' +
            '<span class="mc-node__st" style="background:' + stColor + '" title="' + esc(STATUS_LABEL[n.status] || "INFO") + '"></span>' +
          '</span>' +
          '<span class="mc-node__label">' + esc(shorten(n.label, 26)) + '</span>' +
          '<span class="mc-node__detail">' + esc(shorten(n.detail, 30)) + '</span>';
        el.addEventListener("mouseenter", function () { showHover(n.id); });
        el.addEventListener("mouseleave", hideHover);
        el.addEventListener("focus", function () { showHover(n.id); });
        el.addEventListener("blur", hideHover);
        el.addEventListener("click", function (ev) { ev.stopPropagation(); selectNode(n.id); });
        inner.appendChild(el);
        nodeEls[n.id] = el;
      });
    }

    // ---- canvas sizing ----
    var dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
    var CW = 0, CH = 0;
    function sizeCanvas(dim) {
      CW = dim.w; CH = dim.h;
      canvas.width = Math.floor(CW * dpr);
      canvas.height = Math.floor(CH * dpr);
      canvas.style.width = CW + "px";
      canvas.style.height = CH + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function edgePath(e) {
      var a = pos[e.source], b = pos[e.target];
      if (!a || !b) return null;
      var sy = a.y + NH / 2, ey = b.y + NH / 2;
      if (b.x <= a.x + 4) { // backward / same column
        var lane = Math.max(a.x, b.x) + NW + 30;
        return { sx: a.x, sy: sy, ex: b.x + NW, ey: ey, c1x: lane, c1y: sy, c2x: lane, c2y: ey, back: true };
      }
      var sx = a.x + NW, ex = b.x;
      var dx = Math.max(ex - sx, 1);
      return { sx: sx, sy: sy, ex: ex, ey: ey, c1x: sx + dx * 0.45, c1y: sy, c2x: ex - dx * 0.45, c2y: ey, back: false };
    }

    function drawArrow(g, color) {
      var ang = Math.atan2(g.ey - g.c2y, g.ex - g.c2x);
      ctx.save();
      ctx.setLineDash([]);
      ctx.translate(g.ex, g.ey);
      ctx.rotate(ang);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-7, -3.8);
      ctx.lineTo(-7, 3.8);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function draw() {
      ctx.clearRect(0, 0, CW, CH);
      edges.forEach(function (e) {
        var a = byId[e.source], b = byId[e.target];
        if (!a || !b || !nodeVisible(a) || !nodeVisible(b)) return;
        var g = edgePath(e);
        if (!g) return;
        var inFocus = !focusSet || (focusSet[e.source] && focusSet[e.target]);
        if (focusSet && isolate && !inFocus) return; // endpoint is hidden while isolating
        var col = relColors[e.label] || litFor(b.kind);

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(g.sx, g.sy);
        ctx.bezierCurveTo(g.c1x, g.c1y, g.c2x, g.c2y, g.ex, g.ey);

        if (focusSet && !inFocus) {
          ctx.strokeStyle = rgba(col, 0.09);
          ctx.lineWidth = 1.2;
          ctx.setLineDash([]);
          ctx.stroke();
        } else {
          if (focusSet && inFocus) {
            ctx.strokeStyle = rgba(col, 0.30);
            ctx.lineWidth = 6;
            ctx.setLineDash([]);
            ctx.stroke();
          }
          ctx.strokeStyle = rgba(col, focusSet && inFocus ? 0.95 : 0.5);
          ctx.lineWidth = focusSet && inFocus ? 2.4 : 1.5;
          if (e.status === "warning" || e.status === "error" || e.status === "inactive") {
            ctx.setLineDash([6, 4]);
          } else if (focusSet && inFocus && !reduced) {
            ctx.setLineDash([7, 6]);
            ctx.lineDashOffset = -dash;
          } else {
            ctx.setLineDash([]);
          }
          ctx.stroke();
          drawArrow(g, rgba(col, focusSet && inFocus ? 1 : 0.6));
        }
        ctx.restore();
      });
    }

    // ---- focus / filters ----
    var focusSet = null, selectedId = null, hoverId = null, queryStr = "", focusDepth = 1, isolate = isolateChk ? isolateChk.checked : false, scope = "all";
    var hiddenKinds = {};
    var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var dash = 0;

    function neighborhood(id, depth) {
      var set = {}; set[id] = true;
      var frontier = [id];
      for (var level = 0; level < depth; level += 1) {
        var next = [];
        frontier.forEach(function (cur) {
          (adjIn[cur] || []).forEach(function (e) { if (!set[e.source]) { set[e.source] = true; next.push(e.source); } });
          (adjOut[cur] || []).forEach(function (e) { if (!set[e.target]) { set[e.target] = true; next.push(e.target); } });
        });
        frontier = next;
      }
      return set;
    }

    function nodeVisible(n) {
      if (hiddenKinds[n.kind]) return false;
      if (queryStr && (n.label + " " + n.kind + " " + (n.detail || "")).toLowerCase().indexOf(queryStr) === -1) return false;
      if (scope !== "all" && scopes[scope] && scopes[scope].indexOf(n.id) === -1) return false;
      return true;
    }

    function setFocus(id) {
      focusSet = id ? neighborhood(id, focusDepth) : null;
      stage.classList.toggle("is-focusing", !!id);
      nodes.forEach(function (n) {
        var el = nodeEls[n.id];
        if (!el) return;
        var lit = !!(focusSet && focusSet[n.id]);
        el.classList.toggle("is-lit", lit);
        el.classList.toggle("is-root", id === n.id);
        // when isolate is on, hide nodes outside the focus neighbourhood
        var hideForIsolate = !!(focusSet && isolate && !lit && nodeVisible(n));
        el.classList.toggle("is-isolated", hideForIsolate);
      });
      draw();
      startAnim();
    }

    function showHover(id) {
      hoverId = id;
      nodes.forEach(function (n) {
        if (nodeEls[n.id]) nodeEls[n.id].classList.toggle("is-hover", n.id === id);
      });
      var n = byId[id];
      if (!n) return;
      var links = (adjIn[id] || []).length + (adjOut[id] || []).length;
      hoverCard.innerHTML =
        '<b>' + esc(shorten(n.label, 38)) + '</b>' +
        '<span>' + esc(kindLabels[n.kind] || n.kind) + ' / ' +
        esc(STATUS_LABEL[n.status] || "INFO") + ' / ' + links + ' links</span>' +
        (n.detail ? '<small>' + esc(shorten(n.detail, 58)) + '</small>' : "");
      positionHoverCard(id);
      hoverCard.classList.add("is-on");
    }

    function positionHoverCard(id) {
      var p = pos[id];
      if (!p) return;
      var maxX = stage.scrollLeft + stage.clientWidth - 270;
      var maxY = stage.scrollTop + stage.clientHeight - 104;
      hoverCard.style.left = Math.max(stage.scrollLeft + 10, Math.min(p.x + 10, maxX)) + "px";
      hoverCard.style.top = Math.max(stage.scrollTop + 10, Math.min(p.y + NH + 10, maxY)) + "px";
    }

    function hideHover() {
      hoverId = null;
      hoverCard.classList.remove("is-on");
      nodes.forEach(function (n) {
        if (nodeEls[n.id]) nodeEls[n.id].classList.remove("is-hover");
      });
    }

    function applyVisibility() {
      var visible = 0;
      nodes.forEach(function (n) {
        var v = nodeVisible(n);
        nodeEls[n.id].classList.toggle("is-hidden", !v);
        if (v) visible++;
      });
      if (selectedId && !nodeVisible(byId[selectedId])) closeDrawer();
      if (selectedId) setFocus(selectedId); else draw();
      if (countEl) {
        countEl.textContent = visible + " / " + nodes.length + " nodes" +
          (selectedId && focusSet ? " · " + Object.keys(focusSet).length + " in focus" : "");
      }
    }

    function selectNode(id) {
      if (selectedId === id) { closeDrawer(); return; }
      hideHover();
      selectedId = id;
      popupWasDragged = false;
      setFocus(id);
      var n = byId[id];
      var trace = n.trace || { incoming: adjIn[id].map(toTrace(true)), outgoing: adjOut[id].map(toTrace(false)) };
      var ins = trace.incoming || [], outs = trace.outgoing || [];
      var metaRows = Object.keys(n.meta || {}).map(function (k) {
        return '<li><span class="d" style="background:var(--mc-accent-fill)"></span><b>' + esc(k) + '</b>: ' + esc(n.meta[k]) + '</li>';
      }).join("");
      var openLink = n.url ? '<a class="mc-drawer__open" href="' + esc(n.url) + '">Open record in admin →</a>' : "";
      drawerBody.innerHTML =
        '<div class="mc-drawer__kind" style="color:' + litFor(n.kind) + '">' + esc(kindLabels[n.kind] || n.kind) + '</div>' +
        '<h4>' + esc(n.label) + '</h4>' +
        '<p class="mc-drawer__detail">' + esc(n.detail || "—") + '</p>' +
        openLink +
        '<div class="mc-drawer__metrics">' +
          '<div><span>Incoming</span><strong>' + ins.length + '</strong></div>' +
          '<div><span>Outgoing</span><strong>' + outs.length + '</strong></div>' +
          '<div><span>Status</span><strong class="mc-st-word">' + esc(STATUS_LABEL[n.status] || "INFO") + '</strong></div>' +
          '<div><span>Links</span><strong>' + (ins.length + outs.length) + '</strong></div>' +
        '</div>' +
        relList("Receives from", ins) +
        relList("Sends to", outs) +
        (metaRows ? '<div class="mc-drawer__rel">Detail</div><ul>' + metaRows + '</ul>' : "");
      drawer.classList.add("is-open");
      positionDrawer(id);
    }
    function toTrace(incoming) {
      return function (e) {
        var other = byId[incoming ? e.source : e.target] || {};
        return { label: e.label, node: other.label, kind: other.kind, status: e.status };
      };
    }
    function relList(title, rows) {
      if (!rows.length) return "";
      return '<div class="mc-drawer__rel">' + esc(title) + '</div><ul>' + rows.map(function (r) {
        return '<li><span class="d" style="background:' + litFor(r.kind) + '"></span>' + esc(r.node) +
          ' <span class="mc-drawer__edge">· ' + esc(r.label) + '</span></li>';
      }).join("") + '</ul>';
    }
    function closeDrawer() {
      selectedId = null;
      popupWasDragged = false;
      drawer.classList.remove("is-open");
      setFocus(null);
    }

    function positionDrawer(id) {
      if (!drawer) return;
      var p = pos[id];
      if (!p) return;
      var width = Math.max(280, Math.min(340, stage.clientWidth - 24));
      var minX = stage.scrollLeft + 12;
      var minY = stage.scrollTop + 12;
      var maxX = Math.max(minX, stage.scrollLeft + stage.clientWidth - width - 12);
      var x = p.x + NW + 16;
      if (x > maxX) x = p.x - width - 16;
      drawer.style.width = Math.floor(width) + "px";
      drawer.style.height = "";
      drawer.style.left = Math.max(minX, Math.min(x, maxX)) + "px";
      drawer.style.top = Math.max(minY, Math.min(p.y, stage.scrollTop + stage.clientHeight - drawer.offsetHeight - 12)) + "px";
    }

    function constrainDrawer(left, top) {
      var minX = stage.scrollLeft + 12;
      var minY = stage.scrollTop + 12;
      var maxX = Math.max(minX, stage.scrollLeft + stage.clientWidth - drawer.offsetWidth - 12);
      var maxY = Math.max(minY, stage.scrollTop + stage.clientHeight - drawer.offsetHeight - 12);
      drawer.style.left = Math.max(minX, Math.min(left, maxX)) + "px";
      drawer.style.top = Math.max(minY, Math.min(top, maxY)) + "px";
    }

    function startDrawerDrag(event) {
      if (!drawer.classList.contains("is-open")) return;
      if (event.target.closest("a, button, input, select, textarea")) return;
      event.preventDefault();
      popupWasDragged = true;
      dragState = {
        startX: event.clientX,
        startY: event.clientY,
        left: parseFloat(drawer.style.left) || 0,
        top: parseFloat(drawer.style.top) || 0
      };
      drawer.classList.add("is-dragging");
      document.addEventListener("pointermove", moveDrawer);
      document.addEventListener("pointerup", stopDrawerDrag, { once: true });
    }

    function moveDrawer(event) {
      if (!dragState) return;
      constrainDrawer(
        dragState.left + event.clientX - dragState.startX,
        dragState.top + event.clientY - dragState.startY
      );
    }

    function stopDrawerDrag() {
      dragState = null;
      drawer.classList.remove("is-dragging");
      document.removeEventListener("pointermove", moveDrawer);
    }

    function setGraphFullscreen(on) {
      root.classList.toggle("is-fullscreen", on);
      document.body.classList.toggle("mc-graph-fullscreen-lock", on);
      if (fullscreenBtn) {
        fullscreenBtn.setAttribute("aria-pressed", on ? "true" : "false");
        fullscreenBtn.textContent = on
          ? (fullscreenBtn.getAttribute("data-exit") || "Exit full screen")
          : (fullscreenBtn.getAttribute("data-enter") || "Full screen");
      }
      setTimeout(function () {
        if (selectedId && !popupWasDragged) positionDrawer(selectedId);
        if (hoverId) positionHoverCard(hoverId);
      }, 0);
    }

    // ---- controls ----
    function buildChips() {
      if (!chipWrap) return;
      var all = document.createElement("button");
      all.type = "button";
      all.className = "mc-gchip is-all";
      all.setAttribute("aria-pressed", "true");
      all.textContent = "All";
      all.addEventListener("click", function () {
        hiddenKinds = {};
        chipWrap.querySelectorAll(".mc-gchip").forEach(function (c) { c.setAttribute("aria-pressed", "true"); });
        applyVisibility();
      });
      chipWrap.appendChild(all);
      present.forEach(function (kind) {
        var c = document.createElement("button");
        c.type = "button";
        c.className = "mc-gchip";
        c.setAttribute("aria-pressed", "true");
        c.innerHTML = '<span class="mc-gchip__sw" style="background:' + litFor(kind) + '"></span>' + esc(kindLabels[kind] || kind);
        c.addEventListener("click", function () {
          hiddenKinds[kind] = !hiddenKinds[kind];
          c.setAttribute("aria-pressed", hiddenKinds[kind] ? "false" : "true");
          var anyHidden = present.some(function (k) { return hiddenKinds[k]; });
          all.setAttribute("aria-pressed", anyHidden ? "false" : "true");
          applyVisibility();
        });
        chipWrap.appendChild(c);
      });
    }

    if (search) search.addEventListener("input", function () { queryStr = search.value.trim().toLowerCase(); applyVisibility(); });
    if (depthSel) depthSel.addEventListener("change", function () { focusDepth = Number(depthSel.value) || 1; applyVisibility(); });
    if (isolateChk) isolateChk.addEventListener("change", function () { isolate = isolateChk.checked; applyVisibility(); });
    if (scopeSel) scopeSel.addEventListener("change", function () { scope = scopeSel.value || "all"; applyVisibility(); });
    if (fullscreenBtn) fullscreenBtn.addEventListener("click", function () { setGraphFullscreen(!root.classList.contains("is-fullscreen")); });
    if (clearBtn) clearBtn.addEventListener("click", function () { closeDrawer(); });
    if (drawerClose) drawerClose.addEventListener("click", closeDrawer);
    if (drawer) drawer.addEventListener("pointerdown", startDrawerDrag);
    stage.addEventListener("click", function (e) {
      if (e.target === stage || e.target === inner || e.target === canvas) closeDrawer();
    });
    stage.addEventListener("scroll", function () {
      if (selectedId && !popupWasDragged) positionDrawer(selectedId);
      if (hoverId) positionHoverCard(hoverId);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key !== "Escape") return;
      if (root.classList.contains("is-fullscreen")) setGraphFullscreen(false);
      else if (selectedId) closeDrawer();
    });

    var rafId = null;
    function tick() {
      if (!reduced && focusSet) { dash += 0.6; draw(); rafId = requestAnimationFrame(tick); }
      else { rafId = null; }
    }
    function startAnim() {
      if (rafId === null && !reduced && focusSet) rafId = requestAnimationFrame(tick);
    }

    // ---- init ----
    var dim = layout();
    sizeCanvas(dim);
    build();
    buildChips();
    applyVisibility();
    draw();

    if (typeof ResizeObserver !== "undefined") {
      var ro = new ResizeObserver(function () {
        if (inner.offsetWidth && inner.offsetHeight && (inner.offsetWidth !== CW || inner.offsetHeight !== CH)) {
          sizeCanvas({ w: inner.offsetWidth, h: inner.offsetHeight });
          draw();
          if (selectedId && !popupWasDragged) positionDrawer(selectedId);
        }
      });
      ro.observe(inner);
    }
    window.addEventListener("resize", function () {
      if (selectedId && !popupWasDragged) positionDrawer(selectedId);
      if (hoverId) positionHoverCard(hoverId);
    });
  }
})();
