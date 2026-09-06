"""Shared fixtures.

The MERIDIAN CORE server is session-scoped: it is stateless between requests
apart from the Flask session cookie, and starting one server for the whole run
keeps the browser tests quick enough to stay in the default suite.
"""

import socket
import threading

import pytest
from werkzeug.serving import make_server

from targets.meridian.app import create_app


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="session")
def meridian_server() -> str:
    """Serve MERIDIAN CORE for the duration of the test session."""
    port = _free_port()
    server = make_server("127.0.0.1", port, create_app(), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
