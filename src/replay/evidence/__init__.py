"""Run evidence.

``REDACTED`` is re-exported from :mod:`replay.policy.redaction`, where the
redaction rules live: evidence is where masking is applied, not where it is
decided.
"""

from replay.evidence.recorder import EvidenceRecorder, new_run_id
from replay.policy.redaction import REDACTED

__all__ = ["REDACTED", "EvidenceRecorder", "new_run_id"]
