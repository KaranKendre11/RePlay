"""Recording what the human did while they held the session.

The brief asks the handoff to "record what the human did". Asking them to write
it down is the version that stops happening in week two, so this listens
instead: while control is with the operator, a capture-phase listener in every
frame reports clicks, field changes and Enter presses back into the run.

It records *what was touched*, not what was typed. An operator handling an
escalation on a bank screen is very often typing exactly the data this system
is supposed never to persist, so field values are described by their control,
never by their content.
"""

from __future__ import annotations

#: Installed per frame while the operator holds control. Capture phase, so it
#: sees the event even if the application stops propagation.
LISTENER_JS = r"""
(bindingName) => {
  if (window.__replayListening) return;
  window.__replayListening = true;

  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim().slice(0, 80);

  const label = (el) => {
    if (!el) return 'unknown';
    const tag = (el.tagName || '').toLowerCase();
    const text = clean(el.textContent);
    if (tag === 'a' || tag === 'button') return text || tag;
    if (el.value && (el.type === 'submit' || el.type === 'button')) return clean(el.value);
    const cell = el.closest && el.closest('td, th');
    if (cell) {
      let prev = cell.previousElementSibling;
      while (prev && !clean(prev.textContent)) prev = prev.previousElementSibling;
      if (prev) return clean(prev.textContent) + ' field';
    }
    return el.getAttribute('name') ? el.getAttribute('name') + ' field' : tag;
  };

  const send = (kind, el) => {
    try {
      window[bindingName]({ kind, label: label(el) });
    } catch (_) { /* the binding is gone; control has been handed back */ }
  };

  document.addEventListener('click', (e) => send('click', e.target), true);
  document.addEventListener('change', (e) => send('change', e.target), true);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') send('press_enter', e.target);
  }, true);
}
"""

BINDING = "__replayRecordHumanAction"
