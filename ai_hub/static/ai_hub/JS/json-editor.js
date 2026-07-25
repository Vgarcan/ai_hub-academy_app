(function () {
  "use strict";

  const selector = "textarea[data-ai-json-editor='true']";
  let statusSequence = 0;

  function rootMatches(value, expectedRoot) {
    if (!expectedRoot) {
      return true;
    }
    if (expectedRoot === "array") {
      return Array.isArray(value);
    }
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  function setStatus(textarea, status, message) {
    const container = textarea.closest(".ai-json-editor-shell");
    if (!container) {
      return;
    }

    const output = container.querySelector(".ai-json-editor-status");
    output.textContent = message;
    output.dataset.status = status;
    textarea.classList.toggle("ai-json-editor-invalid", status === "invalid");

    if (status === "invalid") {
      textarea.setAttribute("aria-invalid", "true");
    } else {
      textarea.removeAttribute("aria-invalid");
    }
  }

  function parseEditor(textarea, announceValid) {
    const rawValue = textarea.value.trim();
    const expectedRoot = textarea.dataset.jsonRoot || "";

    if (!rawValue) {
      if (textarea.required) {
        setStatus(textarea, "invalid", "This JSON field is required.");
        return {ok: false};
      }
      const emptyValue = expectedRoot === "array" ? "[]" : "{}";
      setStatus(
        textarea,
        "empty",
        `Optional. An empty value will be saved as ${emptyValue}.`
      );
      return {ok: true, value: expectedRoot === "array" ? [] : {}};
    }

    let parsed;
    try {
      parsed = JSON.parse(rawValue);
    } catch (error) {
      setStatus(textarea, "invalid", `Invalid JSON: ${error.message}`);
      return {ok: false};
    }

    if (!rootMatches(parsed, expectedRoot)) {
      setStatus(
        textarea,
        "invalid",
        `The top-level JSON value must be an ${expectedRoot}.`
      );
      return {ok: false};
    }

    setStatus(textarea, "valid", announceValid ? "Valid JSON." : "");
    return {ok: true, value: parsed};
  }

  function makeButton(label, action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button ai-json-editor-button";
    button.textContent = label;
    button.dataset.action = action;
    return button;
  }

  function enhance(textarea) {
    if (textarea.dataset.aiJsonEnhanced === "true") {
      return;
    }
    textarea.dataset.aiJsonEnhanced = "true";

    const shell = document.createElement("div");
    shell.className = "ai-json-editor-shell";
    textarea.parentNode.insertBefore(shell, textarea);
    shell.appendChild(textarea);

    const toolbar = document.createElement("div");
    toolbar.className = "ai-json-editor-toolbar";
    toolbar.appendChild(makeButton("Format JSON", "format"));
    toolbar.appendChild(makeButton("Compact", "compact"));

    const status = document.createElement("span");
    const statusId = `ai-json-editor-status-${statusSequence += 1}`;
    status.id = statusId;
    status.className = "ai-json-editor-status";
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    toolbar.appendChild(status);
    shell.insertBefore(toolbar, textarea);

    const describedBy = textarea.getAttribute("aria-describedby");
    textarea.setAttribute(
      "aria-describedby",
      describedBy ? `${describedBy} ${statusId}` : statusId
    );

    let validationTimer;
    textarea.addEventListener("input", function () {
      window.clearTimeout(validationTimer);
      validationTimer = window.setTimeout(function () {
        parseEditor(textarea, true);
      }, 250);
    });

    toolbar.addEventListener("click", function (event) {
      const button = event.target.closest("[data-action]");
      if (!button) {
        return;
      }

      const result = parseEditor(textarea, true);
      if (!result.ok) {
        textarea.focus();
        return;
      }

      textarea.value = button.dataset.action === "compact"
        ? JSON.stringify(result.value)
        : JSON.stringify(result.value, null, 2);
      setStatus(textarea, "valid", "Valid JSON.");
      textarea.focus();
    });

    parseEditor(textarea, false);
  }

  function enhanceWithin(root) {
    if (root.matches && root.matches(selector)) {
      enhance(root);
    }
    if (root.querySelectorAll) {
      root.querySelectorAll(selector).forEach(enhance);
    }
  }

  function init() {
    enhanceWithin(document);

    const observer = new MutationObserver(function (mutations) {
      mutations.forEach(function (mutation) {
        mutation.addedNodes.forEach(function (node) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            enhanceWithin(node);
          }
        });
      });
    });
    observer.observe(document.body, {childList: true, subtree: true});
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
