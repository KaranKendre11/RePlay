"""Recording what the human did while they held the session.

The brief asks the handoff to "record what the human did". Asking them to write
it down is the version that stops happening in week two, so this listens
instead: while control is with the operator, a capture-phase listener in every
frame reports clicks, field changes and Enter presses back into the run.

It records *what was touched*, not what was typed. An operator handling an
escalation on a bank screen is very often typing exactly the data this system
is supposed never to persist, so field values are described by their control,
never by their content.

That has to include the *neighbouring* content. A data grid has no labels, and
the cell beside the one that was clicked is not a substitute for one: it is
another customer's account number and balance. Describing a cell that has no
label of its own by its column and position keeps the promise; describing it by
what sits next to it put a full account number into the evidence and onto the
operator console.
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
    if (tag === 'a' || tag === 'button') return clean(el.textContent) || tag;
    if (el.value && (el.type === 'submit' || el.type === 'button')) return clean(el.value);

    // The control's own name for itself. Metadata, not screen content.
    const named = el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('name'));
    if (named) return clean(named) + ' field';

    // No label of its own, so say where it is. A <th> is a declared column
    // header and safe to name; a <td> is data, whether it sits in the header
    // row or beside the cell that was clicked.
    const cell = el.closest && el.closest('td, th');
    if (cell) {
      const row = cell.parentElement;
      const index = row ? [].indexOf.call(row.children, cell) : -1;
      const table = cell.closest('table');
      const head = table && table.rows ? table.rows[0] : null;
      const header = head && index >= 0 ? head.children[index] : null;
      const named_th = header && (header.tagName || '').toLowerCase() === 'th'
        ? clean(header.textContent) : '';
      return named_th ? named_th + ' cell' : 'cell ' + (index + 1) + ' in a row';
    }
    return tag;
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
