"""MERIDIAN CORE — a deliberately hostile stand-in for a legacy teller console.

Every unpleasant property here is intentional. Real back-office banking
applications are server-rendered, frame-based, table-laid-out, and carry no
test IDs. Automating a clean modern app would prove nothing about the problem
this project exists to solve.

The specific traits, and the reason each one is present, are documented in
the templates themselves and pinned by the hostility tests in
``tests/test_meridian.py``.
"""

from targets.meridian.app import create_app

__all__ = ["create_app"]
