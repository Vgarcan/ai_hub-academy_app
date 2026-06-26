/* ============================================================
   Composed Entity Workspace — tab controller (IA Step 4)
   Progressive enhancement: panels render visible (stacked) with
   no JS; this groups them under a tab bar and shows one at a time.
   Markup contract:
     <div class="ai-tabnav" data-ws-tabs [data-tab-default="config"]>
       <button class="ai-tabnav__btn" data-tab-to="overview">…</button>
     </div>
     <div class="ai-tab" data-tab="overview">…</div>
   Panels are resolved within the nearest enclosing <form> (so the
   change form keeps submitting normally) or the document otherwise.
   ============================================================ */
(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".ai-tabnav[data-ws-tabs]").forEach(initEntityTabs);
  });

  function initEntityTabs(nav) {
    var scope = nav.closest("form") || document;
    var panels = Array.prototype.slice.call(scope.querySelectorAll(".ai-tab[data-tab]"));
    var navButtons = Array.prototype.slice.call(nav.querySelectorAll("[data-tab-to]"));
    // Any element with data-tab-to inside the form can switch tabs — including
    // in-panel "manage" / "jump to config" affordances, not just the nav tabs.
    var triggers = Array.prototype.slice.call(scope.querySelectorAll("[data-tab-to]"));
    if (!panels.length || !navButtons.length) return;

    function show(key) {
      panels.forEach(function (p) { p.hidden = (p.getAttribute("data-tab") !== key); });
      navButtons.forEach(function (b) {
        b.setAttribute("aria-selected", b.getAttribute("data-tab-to") === key ? "true" : "false");
      });
    }

    triggers.forEach(function (b) {
      b.addEventListener("click", function (e) {
        e.preventDefault();
        show(b.getAttribute("data-tab-to"));
      });
    });

    // If the form re-rendered with validation errors, land on the editable tab
    // so the user sees them; otherwise honour the default (or first panel).
    var hasErrors = scope.querySelector(".errornote, .errorlist") !== null;
    var fallback = nav.getAttribute("data-tab-default") || panels[0].getAttribute("data-tab");
    var initial = hasErrors ? (nav.getAttribute("data-tab-error") || "config") : fallback;
    // guard: only use initial if a matching panel exists
    if (!panels.some(function (p) { return p.getAttribute("data-tab") === initial; })) {
      initial = panels[0].getAttribute("data-tab");
    }
    show(initial);
  }
})();
