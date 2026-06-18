/* AI Hub Academy — Chat Widget
   AJAX floating panel, accessible from the nav button.
   Ctrl+/  toggles open/close.  Enter sends. Shift+Enter new line.
   ──────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  const ASK_URL = '/assistant/ask/';

  // ── DOM ──────────────────────────────────────────────
  const widget    = document.getElementById('chat-widget');
  const msgArea   = document.getElementById('widget-messages');
  const input     = document.getElementById('widget-input');
  const sendBtn   = document.getElementById('widget-send');
  const toggle    = document.getElementById('chat-toggle-btn');
  const closeBtn  = document.getElementById('widget-close');
  const expandBtn = document.getElementById('widget-expand');
  const welcome   = document.getElementById('widget-welcome');

  if (!widget) return;

  // ── State ─────────────────────────────────────────────
  let isOpen    = false;
  let isLoading = false;
  let isExpanded = false;
  let msgCount  = 0;

  // ── CSRF ──────────────────────────────────────────────
  function getCsrf() {
    const el = document.querySelector('[name=csrfmiddlewaretoken]');
    if (el) return el.value;
    const m = document.cookie.match(/csrftoken=([^;]+)/);
    return m ? m[1] : '';
  }

  // ── Open / close / expand ─────────────────────────────
  function open() {
    isOpen = true;
    widget.classList.add('open');
    if (toggle) { toggle.setAttribute('aria-expanded', 'true'); toggle.classList.add('active'); }
    setTimeout(function() { input.focus(); }, 320);
  }
  function close() {
    isOpen = false;
    widget.classList.remove('open');
    if (toggle) { toggle.setAttribute('aria-expanded', 'false'); toggle.classList.remove('active'); }
  }
  function toggleWidget() { isOpen ? close() : open(); }

  function toggleExpand() {
    isExpanded = !isExpanded;
    widget.classList.toggle('expanded', isExpanded);
    if (expandBtn) {
      expandBtn.title = isExpanded ? 'Reduce' : 'Expand';
      expandBtn.innerHTML = isExpanded ? ICON_SHRINK : ICON_EXPAND;
    }
  }

  // SVG icons for expand/shrink
  const ICON_EXPAND = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';
  const ICON_SHRINK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="10" y1="14" x2="3" y2="21"/><line x1="21" y1="3" x2="14" y2="10"/></svg>';

  // ── Helpers ───────────────────────────────────────────
  function scrollEnd() { msgArea.scrollTop = msgArea.scrollHeight; }

  function esc(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // ── Inline formatting (bold, italic, code, links) ─────
  function inlineFmt(s) {
    // already HTML-escaped; apply inline MD
    s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    s = s.replace(/`([^`\n]+)`/g, '<code class="wm-code">$1</code>');
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return s;
  }

  // ── Full Markdown renderer ─────────────────────────────
  function renderMd(raw) {
    if (!raw) return '';

    const lines = raw.split('\n');
    const out   = [];
    let inCode   = false;
    let codeLang = '';
    let codeLines = [];
    let listItems = [];
    let orderedList = false;

    function flushList() {
      if (!listItems.length) return;
      const tag = orderedList ? 'ol' : 'ul';
      out.push('<' + tag + ' class="wm-list">');
      listItems.forEach(function(item) { out.push('<li>' + item + '</li>'); });
      out.push('</' + tag + '>');
      listItems = [];
      orderedList = false;
    }

    function flushCode() {
      const body = codeLines.map(esc).join('\n');
      const langClass = codeLang ? ' class="language-' + esc(codeLang) + '"' : '';
      out.push('<pre class="wm-pre"><code' + langClass + '>' + body + '</code></pre>');
      codeLines = [];
      codeLang  = '';
    }

    lines.forEach(function(line) {
      // ── code fence ────────────────────────────────
      if (/^```/.test(line)) {
        if (inCode) {
          flushCode();
          inCode = false;
        } else {
          flushList();
          codeLang = line.slice(3).trim();
          inCode = true;
        }
        return;
      }
      if (inCode) { codeLines.push(line); return; }

      // ── headings ──────────────────────────────────
      var hMatch = line.match(/^(#{1,4}) (.+)/);
      if (hMatch) {
        flushList();
        var level = Math.min(hMatch[1].length + 2, 6); // h3..h6
        out.push('<h' + level + ' class="wm-heading">' + inlineFmt(esc(hMatch[2])) + '</h' + level + '>');
        return;
      }

      // ── horizontal rule ───────────────────────────
      if (/^-{3,}$/.test(line.trim())) {
        flushList();
        out.push('<hr class="wm-hr">');
        return;
      }

      // ── blockquote ────────────────────────────────
      var bqMatch = line.match(/^> (.+)/);
      if (bqMatch) {
        flushList();
        out.push('<blockquote class="wm-bq">' + inlineFmt(esc(bqMatch[1])) + '</blockquote>');
        return;
      }

      // ── unordered list ────────────────────────────
      var ulMatch = line.match(/^[\-\*\+] (.+)/);
      if (ulMatch) {
        if (orderedList) flushList();
        orderedList = false;
        listItems.push(inlineFmt(esc(ulMatch[1])));
        return;
      }

      // ── ordered list ──────────────────────────────
      var olMatch = line.match(/^\d+\. (.+)/);
      if (olMatch) {
        if (!orderedList && listItems.length) flushList();
        orderedList = true;
        listItems.push(inlineFmt(esc(olMatch[1])));
        return;
      }

      // ── blank line ────────────────────────────────
      if (line.trim() === '') {
        flushList();
        out.push('<div class="wm-gap"></div>');
        return;
      }

      // ── plain text ────────────────────────────────
      flushList();
      out.push('<span class="wm-line">' + inlineFmt(esc(line)) + '</span><br>');
    });

    flushList();
    if (inCode) flushCode(); // unclosed fence
    return out.join('');
  }

  // ── Render a message bubble ───────────────────────────
  function addMsg(role, text, sources) {
    msgCount++;
    if (welcome && msgCount === 1) welcome.style.display = 'none';

    const row = document.createElement('div');
    row.className = 'wm-msg wm-' + role;

    const av = document.createElement('div');
    av.className = 'wm-avatar';
    av.textContent = role === 'assistant' ? 'AI' : 'You';

    const bub = document.createElement('div');
    bub.className = 'wm-bubble';
    bub.innerHTML = renderMd(text);

    if (sources && sources.length) {
      const bar = document.createElement('div');
      bar.className = 'wm-sources';
      const seen = new Set();
      sources.forEach(function(s) {
        const key = s.page_slug || s.page_title;
        if (seen.has(key)) return;
        seen.add(key);
        const chip = document.createElement('a');
        chip.className = 'wm-source-chip';
        chip.textContent = s.page_title || s.heading || 'Source';
        chip.href = '/docs/' + (s.page_slug || '') + '/';
        chip.target = '_blank';
        bar.appendChild(chip);
      });
      if (bar.children.length) bub.appendChild(bar);
    }

    row.appendChild(av);
    row.appendChild(bub);
    msgArea.appendChild(row);
    scrollEnd();
  }

  function showTyping() {
    const row = document.createElement('div');
    row.className = 'wm-msg wm-assistant wm-typing';
    row.id = 'wm-typing';
    const av = document.createElement('div');
    av.className = 'wm-avatar'; av.textContent = 'AI';
    const bub = document.createElement('div');
    bub.className = 'wm-bubble';
    bub.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    row.appendChild(av); row.appendChild(bub);
    msgArea.appendChild(row);
    scrollEnd();
  }
  function hideTyping() { const el = document.getElementById('wm-typing'); if (el) el.remove(); }

  // ── Send ──────────────────────────────────────────────
  function send() {
    const text = input.value.trim();
    if (!text || isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;
    input.value = '';
    input.style.height = 'auto';

    addMsg('user', text, null);
    showTyping();

    const fd = new FormData();
    fd.append('question', text);
    fd.append('csrfmiddlewaretoken', getCsrf());

    fetch(ASK_URL, {
      method: 'POST',
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      body: fd,
    })
      .then(function(r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function(data) {
        hideTyping();
        addMsg('assistant', data.answer || 'No response received.', data.sources || []);
      })
      .catch(function(err) {
        hideTyping();
        addMsg('assistant', 'Something went wrong. Please try again.', null);
        console.error('[chat-widget]', err);
      })
      .finally(function() {
        isLoading = false;
        sendBtn.disabled = false;
        input.focus();
      });
  }

  // ── Events ────────────────────────────────────────────
  if (toggle)    toggle.addEventListener('click', toggleWidget);
  if (closeBtn)  closeBtn.addEventListener('click', close);
  if (expandBtn) expandBtn.addEventListener('click', toggleExpand);
  sendBtn.addEventListener('click', send);

  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener('input', function() {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });

  // Suggestion chips (delegated — they live inside #widget-messages)
  msgArea.addEventListener('click', function(e) {
    if (e.target.classList.contains('wm-suggestion')) {
      input.value = e.target.textContent.trim();
      send();
    }
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === '/') { e.preventDefault(); toggleWidget(); }
    if (e.key === 'Escape' && isOpen) close();
  });

  // ── Scroll-reveal (IntersectionObserver) ─────────────
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(function(entries) {
      entries.forEach(function(en) {
        if (en.isIntersecting) { en.target.classList.add('visible'); io.unobserve(en.target); }
      });
    }, { threshold: 0.10 });
    document.querySelectorAll('.reveal').forEach(function(el) { io.observe(el); });
  } else {
    // fallback for old browsers
    document.querySelectorAll('.reveal').forEach(function(el) { el.classList.add('visible'); });
  }

}());
