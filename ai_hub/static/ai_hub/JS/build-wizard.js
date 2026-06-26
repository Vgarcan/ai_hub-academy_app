/* ============================================================
   Build Wizard engine
   ============================================================ */
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    initWizard();
  });

  function initWizard() {
    var wiz = document.querySelector("[data-mc-wizard]");
    if (!wiz) return;

    var panels = Array.from(wiz.querySelectorAll(".mc-step-panel"));
    var rails  = Array.from(document.querySelectorAll(".mc-wstep"));
    var meterFill  = document.querySelector(".mc-wiz-meter__fill");
    var meterLabel = document.querySelector(".mc-wiz-meter__label");
    var N = panels.length;
    var current = 0;

    /* ---- sub-section expand / collapse ---- */
    document.querySelectorAll(".mc-sub__head").forEach(function (head) {
      head.addEventListener("click", function () {
        head.closest(".mc-sub").classList.toggle("is-open");
      });
    });

    /* ---- toggle buttons (reuse/create, simple/advanced) ---- */
    document.querySelectorAll("[data-toggle-group]").forEach(function (group) {
      var name = group.getAttribute("data-toggle-group");
      var btns = Array.from(group.querySelectorAll("[data-toggle-value]"));
      var hidden = document.querySelector("input[name='" + name + "']");
      btns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var val = btn.getAttribute("data-toggle-value");
          btns.forEach(function (b) { b.classList.toggle("is-on", b === btn); });
          if (hidden) hidden.value = val;
          // show/hide conditional sections
          document.querySelectorAll("[data-show-when]").forEach(function (el) {
            var cond = el.getAttribute("data-show-when"); // "fieldname:value"
            var parts = cond.split(":");
            if (parts[0] === name) {
              el.classList.toggle("is-visible", parts[1] === val);
            }
          });
          updateManifest();
        });
      });
    });

    /* ---- toolbox chips ---- */
    document.querySelectorAll(".mc-tbox").forEach(function (box) {
      var chk = box.querySelector("input[type=checkbox]");
      box.addEventListener("click", function () {
        box.classList.toggle("is-sel");
        if (chk) chk.checked = box.classList.contains("is-sel");
        updateManifest();
      });
    });

    /* ---- knowledge mode ---- */
    document.querySelectorAll("[data-kmode-group]").forEach(function (group) {
      var btns = Array.from(group.querySelectorAll(".mc-kmode-btn"));
      var hidden = group.querySelector("input[name='knowledge_mode']");
      btns.forEach(function (btn) {
        btn.addEventListener("click", function () {
          var val = btn.getAttribute("data-kmode");
          btns.forEach(function (b) { b.classList.toggle("is-on", b === btn); });
          if (hidden) hidden.value = val;
          group.querySelectorAll(".mc-kpane").forEach(function (pane) {
            pane.classList.toggle("is-on", pane.getAttribute("data-kpane") === val);
          });
          updateManifest();
        });
      });
    });

    /* ---- step row expand ---- */
    document.addEventListener("click", function (e) {
      var num = e.target.closest(".mc-srow__num");
      if (num) num.closest(".mc-srow").classList.toggle("is-open");
    });

    /* ---- add step row (orchestrator) ---- */
    var addRowBtn = document.querySelector("[data-add-step]");
    var agentOptsTmpl = document.getElementById("agent-options-tmpl");
    if (addRowBtn && agentOptsTmpl) {
      addRowBtn.addEventListener("click", function () {
        var container = document.querySelector(".mc-srows");
        var idx = container.querySelectorAll(".mc-srow").length + 1;
        var row = document.createElement("div");
        row.className = "mc-srow";
        row.innerHTML =
          '<div class="mc-srow__top">' +
          '  <span class="mc-srow__num" title="Expand mappings">' + idx + '</span>' +
          '  <select name="step_agent_id"><option value="">— Select agent —</option>' +
          agentOptsTmpl.innerHTML + '</select>' +
          '  <select name="step_on_error" style="width:110px"><option value="stop">stop</option><option value="continue">continue</option></select>' +
          '  <button type="button" class="mc-srow__rm" title="Remove">&#x2715;</button>' +
          '</div>' +
          '<div class="mc-srow__details"><div class="mc-srow__map">' +
          '  <div><div class="mc-map-dir">agent key &#8592; from context path</div>' +
          '    <textarea name="step_input_mapping" rows="3" placeholder=\'{"goal":"goal_text"}\'></textarea></div>' +
          '  <div><div class="mc-map-dir">context key &#8592; from response path</div>' +
          '    <textarea name="step_output_mapping" rows="3" placeholder=\'{"final_output":"llm.content"}\'></textarea></div>' +
          '</div></div>';
        row.querySelector(".mc-srow__rm").addEventListener("click", function () {
          row.remove();
          renumberRows();
          updateManifest();
        });
        container.appendChild(row);
        renumberRows();
        updateManifest();
      });
    }

    function renumberRows() {
      document.querySelectorAll(".mc-srow").forEach(function (row, i) {
        var n = row.querySelector(".mc-srow__num");
        if (n) n.textContent = i + 1;
      });
    }

    /* ---- navigation ---- */
    wiz.querySelectorAll("[data-wiz-back]").forEach(function (btn) {
      btn.addEventListener("click", function () { goTo(current - 1); });
    });
    wiz.querySelectorAll("[data-wiz-next]").forEach(function (btn) {
      btn.addEventListener("click", function () { goTo(current + 1); });
    });
    rails.forEach(function (r, i) {
      r.addEventListener("click", function () { goTo(i); });
    });

    function goTo(n) {
      if (n < 0 || n >= N) return;
      current = n;
      panels.forEach(function (p, i) { p.classList.toggle("is-active", i === n); });
      rails.forEach(function (r, i) { r.classList.toggle("is-active", i === n); });
      updateMeter();
      if (n === N - 1) updateManifest();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function updateMeter() {
      var pct = N > 1 ? (current / (N - 1)) * 100 : 100;
      if (meterFill)  meterFill.style.width  = pct + "%";
      if (meterLabel) meterLabel.textContent  = "Step " + (current + 1) + " of " + N;
    }

    /* ---- manifest (review step) ---- */
    function getVal(name) {
      var el = document.querySelector("[name='" + name + "']");
      return el ? (el.value || "").trim() : "";
    }

    function manifestRow(ic, kind, label, sub) {
      return '<div class="mc-mrow">' +
        '<div class="mc-mrow__ic mc-mrow__ic--' + kind + '">' + ic + '</div>' +
        '<div class="mc-mrow__info"><b>' + esc(label) + '</b><small>' + esc(sub) + '</small></div>' +
        '<span class="mc-mrow__badge mc-mrow__badge--' + kind + '">' +
        (kind === "create" ? "&#xFF0B; create" : "&#x21BB; reuse") + '</span>' +
        '</div>';
    }

    function esc(s) {
      return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    function updateManifest() {
      var container = document.getElementById("wiz-manifest");
      if (!container) return;
      var rows = [];

      // Engine
      var engineMode = getVal("engine_mode");
      if (engineMode === "reuse") {
        var mEl = document.querySelector("[name='engine_reuse_model_id']");
        var mLabel = (mEl && mEl.selectedIndex > 0) ? mEl.options[mEl.selectedIndex].text : "Existing model";
        rows.push(manifestRow("&#9881;", "reuse", mLabel, "ModelConfig"));
      } else {
        var pn = getVal("engine_provider_name") || "New provider";
        var mn = getVal("engine_model_name") || "training/starter";
        rows.push(manifestRow("&#9881;", "create", pn + " / " + mn, "ProviderConfig + ModelConfig"));
      }

      // Agent
      var agentMode = getVal("agent_mode");
      if (agentMode === "reuse") {
        var aEl = document.querySelector("[name='agent_reuse_id']");
        var aLabel = (aEl && aEl.selectedIndex > 0) ? aEl.options[aEl.selectedIndex].text : "Existing agent";
        rows.push(manifestRow("&#x1F916;", "reuse", aLabel, "AgentProfile"));
      } else {
        rows.push(manifestRow("&#x1F916;", "create", getVal("agent_name") || "New agent", "AgentProfile"));
      }

      // Toolboxes
      var selBoxes = Array.from(document.querySelectorAll(".mc-tbox.is-sel"));
      if (selBoxes.length) {
        var tbLabels = selBoxes.map(function (b) {
          var bEl = b.querySelector("b");
          return bEl ? bEl.textContent : "?";
        }).join(", ");
        rows.push(manifestRow("&#x1F9F0;", "create", selBoxes.length + " toolbox assignment(s)", tbLabels));
      }

      // Knowledge
      var kmode = getVal("knowledge_mode");
      if (kmode === "reuse") {
        var kEl = document.querySelector("[name='knowledge_collection_id']");
        var kLabel = (kEl && kEl.selectedIndex > 0) ? kEl.options[kEl.selectedIndex].text : "Collection";
        rows.push(manifestRow("&#x1F4DA;", "reuse", kLabel, "KnowledgeCollection"));
      } else if (kmode === "create") {
        var kcn = getVal("knowledge_collection_name") || "New collection";
        var kdt = getVal("knowledge_doc_title") || "New document";
        rows.push(manifestRow("&#x1F4DA;", "create", kcn + " → " + kdt, "KnowledgeCollection + KnowledgeDocument (active)"));
      }

      // Runtime object
      var kind = wiz.getAttribute("data-mc-wizard");
      if (kind === "game") {
        var goalTxt = getVal("goal_text") || "—";
        var preview = goalTxt.length > 60 ? goalTxt.substring(0, 57) + "…" : goalTxt;
        rows.push(manifestRow("&#x1F3AF;", "create", "GAME session (pending)", preview));
        var flavor = getVal("game_flavor") || "simple";
        if (flavor === "advanced") {
          var wsn = getVal("workspace_name") || "New workspace";
          rows.push(manifestRow("&#x1F3E2;", "create", wsn, "GameWorkspace + GameGoal"));
        }
      } else {
        var pname = getVal("pipeline_name") || "New pipeline";
        var sc = document.querySelectorAll(".mc-srow").length;
        rows.push(manifestRow("&#x1F500;", "create", pname, sc + " step(s) · inactive until activated"));
      }

      container.innerHTML = rows.length ? rows.join("") : '<div class="mc-mrow" style="color:var(--mc-muted)">Fill in the previous steps to see the manifest.</div>';
    }

    updateMeter();
    updateManifest();
  }
})();

