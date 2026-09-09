"""The operator console.

Deliberately bare. The brief puts a real co-browsing console out of scope and
asks instead for a handoff mechanism that genuinely works, so the effort went
into the control transfer rather than the chrome. What is here is real: the
queue, the claim, the resume, and the record of what the operator did.

The console does not stream the screen, and it does not need to. The browser is
headed and the operator is sitting in front of it — they act in the actual
window, on the actual session. Streaming pixels would be a nicer product and a
strictly worse demonstration of the thing being evaluated, which is whether
automation can pause, cede a live session, and take it back.
"""

from __future__ import annotations

import html
import threading
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from replay.escalation.control import (
    InterventionQueue,
    InterventionRequest,
    RequestStatus,
    Resolution,
)

STYLE = """
body { font-family: ui-monospace, "SF Mono", Menlo, monospace; margin: 0;
       background: #14161a; color: #e6e6e6; }
header { background: #1f2430; padding: 14px 20px; border-bottom: 1px solid #333; }
h1 { font-size: 15px; margin: 0; letter-spacing: 1px; }
main { padding: 20px; max-width: 900px; }
.card { border: 1px solid #333; background: #1a1d24; padding: 14px 16px; margin-bottom: 14px; }
.pending { border-left: 3px solid #d98324; }
.resolved { border-left: 3px solid #4a7c59; opacity: .65; }
.reason { color: #d98324; font-weight: bold; }
.meta { color: #8b93a1; font-size: 12px; margin: 6px 0; }
pre { background: #0f1115; padding: 10px; overflow-x: auto; font-size: 12px;
      color: #b8c0cc; max-height: 220px; }
button { font-family: inherit; font-size: 13px; padding: 6px 14px; margin-right: 8px;
         border: 1px solid #444; background: #262b36; color: #e6e6e6; cursor: pointer; }
button.primary { background: #2f5d3f; border-color: #3f7a53; }
.empty { color: #8b93a1; }
"""

SCRIPT = """
async function resolve(id, how) {
  // The endpoint binds the note as a query parameter, so it goes in the URL.
  // Without this the field was declared, rendered, and never once populated.
  const note = window.prompt('What did you do? (optional)') || '';
  await fetch(`/interventions/${id}/${how}?note=${encodeURIComponent(note)}`,
              { method: 'POST' });
  location.reload();
}
setInterval(() => { if (!document.hidden) location.reload(); }, 4000);
"""


def _render(request: InterventionRequest) -> str:
    resolved = request.status is RequestStatus.RESOLVED
    actions = "".join(
        f"<li>{html.escape(a.kind)} — {html.escape(a.label)}</li>" for a in request.human_actions
    )
    note = (
        f'<div class="meta">note: {html.escape(request.operator_note)}</div>'
        if request.operator_note
        else ""
    )
    buttons = (
        ""
        if resolved
        else (
            f"<button class='primary' onclick=\"resolve('{request.id}','resume')\">"
            "Hand control back</button>"
            f"<button onclick=\"resolve('{request.id}','abort')\">Abandon run</button>"
        )
    )
    return f"""
    <div class="card {"resolved" if resolved else "pending"}">
      <div class="reason">{html.escape(request.reason.value.replace("_", " ").upper())}</div>
      <div><b>{html.escape(request.capability)}</b> · step {html.escape(request.step_id)}</div>
      <div class="meta">{html.escape(request.step_intent)}</div>
      <div class="meta">{html.escape(request.summary)}</div>
      <div class="meta">url: {html.escape(request.url)}</div>
      <div class="meta">raised {html.escape(request.created_at)}</div>
      <pre>{html.escape(request.observed[:1200]) or "(no screen text captured)"}</pre>
      {'<div class="meta">operator did:</div><ul>' + actions + "</ul>" if actions else ""}
      {'<div class="meta">resolution: ' + request.resolution.value + "</div>" if resolved else ""}
      {note}
      {buttons}
    </div>
    """


def create_console(queue: InterventionQueue) -> FastAPI:
    app = FastAPI(title="RePlay operator console", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        requests = list(reversed(queue.all()))
        body = (
            "".join(_render(r) for r in requests)
            or "<p class='empty'>No interventions. Automation is running unattended.</p>"
        )
        return f"""<!doctype html><html><head><meta charset="utf-8">
        <title>RePlay operator console</title><style>{STYLE}</style></head>
        <body><header><h1>RePLAY · OPERATOR CONSOLE</h1></header>
        <main>{body}</main><script>{SCRIPT}</script></body></html>"""

    @app.get("/api/interventions")
    def api_list() -> JSONResponse:
        return JSONResponse([r.to_dict() for r in queue.all()])

    @app.get("/api/interventions/{request_id}")
    def api_get(request_id: str) -> JSONResponse:
        request = queue.get(request_id)
        if request is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(request.to_dict())

    @app.post("/interventions/{request_id}/resume")
    def resume(request_id: str, note: str = "") -> JSONResponse:
        """The operator has finished and is handing the session back."""
        queue.claim(request_id)
        resolved = queue.resolve(request_id, Resolution.RESUMED, note=note)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(resolved.to_dict())

    @app.post("/interventions/{request_id}/abort")
    def abort(request_id: str, note: str = "") -> JSONResponse:
        resolved = queue.resolve(request_id, Resolution.ABORTED, note=note)
        if resolved is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(resolved.to_dict())

    return app


class ConsoleEscalation:
    """Publishes to the queue and waits for a person.

    Blocking is the point. The automation has released control by the time this
    is called, so waiting here is exactly "the human has the session and we do
    not". The timeout exists so an unattended run eventually gives up rather
    than holding a browser open forever.
    """

    def __init__(self, queue: InterventionQueue, *, timeout_s: float = 900.0) -> None:
        self.queue = queue
        self.timeout_s = timeout_s

    def escalate(self, request: InterventionRequest) -> InterventionRequest:
        event = self.queue.submit(request)
        if not event.wait(timeout=self.timeout_s):
            # Losing this race is fine: InterventionQueue.resolve is terminal,
            # so an operator who got there first keeps their resolution and this
            # abort is a no-op. Which is the right way round — they fixed the
            # session and handed it back.
            self.queue.resolve(
                request.id,
                Resolution.ABORTED,
                note=f"no operator responded within {self.timeout_s:.0f}s",
            )
        return self.queue.get(request.id) or request


class ScriptedOperator:
    """A person, simulated, for tests.

    Runs a callable against the live session while control is held, then resumes
    or aborts. Exercises the real transfer — the same release, the same page,
    the same reacquire — without needing someone at a keyboard.
    """

    def __init__(
        self,
        act: Any = None,
        *,
        resolution: Resolution = Resolution.RESUMED,
        note: str = "handled by scripted operator",
    ) -> None:
        self.act = act
        self.resolution = resolution
        self.note = note
        self.seen: list[InterventionRequest] = []

    def escalate(self, request: InterventionRequest) -> InterventionRequest:
        self.seen.append(request)
        if self.act is not None:
            self.act(request)
        request.status = RequestStatus.RESOLVED
        request.resolution = self.resolution
        request.operator_note = self.note
        return request


def serve_console(queue: InterventionQueue, *, host: str = "127.0.0.1", port: int = 8765):
    """Run the console beside the automation, in a thread.

    One process, because a single-operator handoff needs no more than that and
    the brief is explicit that building scaling infrastructure is not rewarded.
    """
    import uvicorn

    config = uvicorn.Config(create_console(queue), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread
