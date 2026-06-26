/* ============================================================
   AI Hub admin · script barrel (ES module)
   The real logic is split under ./JS/ as three independent,
   self-initialising IIFE modules (each already "use strict" and
   gated on DOMContentLoaded). Loaded via <script type="module">.
   Import order is not significant, but kept stable for clarity.
   ============================================================ */
import "./JS/graph.js";
import "./JS/build-wizard.js";
import "./JS/entity-tabs.js";
