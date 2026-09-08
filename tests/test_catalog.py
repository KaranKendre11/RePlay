"""Capability catalog tests.

The point of this surface is that an agent can find out what a capability
needs, what it returns, and how it can legitimately not-succeed — all *before*
calling it. So most of these check the contract is published, not just that the
endpoint responds.

:func:`test_a_capability_is_invoked_by_name_over_http` is the one the brief
asks for: "show one being invoked".
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from replay.api import create_api, summarise
from replay.artifact import ArtifactStore
from replay.policy import Allowlist

CAPABILITY = "lookup_balance"

#: The test server binds to an ephemeral port, which the shipped policy rightly
#: does not allow. Widening policy.toml to suit the tests would be the wrong way
#: round, so the allowlist is supplied here instead.
TEST_ALLOWLIST = Allowlist.permissive("127.0.0.1:*", "localhost:*")


@pytest.fixture
def client(tmp_path):
    return TestClient(create_api(evidence_dir=tmp_path, allowlist=TEST_ALLOWLIST))


# ---------- discovery ----------


def test_the_catalog_is_the_artifacts_directory(client):
    """No registry to keep in sync. A registry that can disagree eventually does."""
    listed = {c["ref"] for c in client.get("/capabilities").json()}
    on_disk = {a.ref for a in ArtifactStore("artifacts").list_all()}
    assert listed == on_disk


def test_a_capability_publishes_the_arguments_it_takes(client):
    """An agent that must invoke something to learn what it takes has no contract."""
    schema = client.get(f"/capabilities/{CAPABILITY}").json()["arguments"]

    assert schema["required"] == ["member_id"]
    assert schema["properties"]["member_id"]["pattern"]
    assert schema["additionalProperties"] is False


def test_a_published_type_is_a_type_the_call_is_held_to(client):
    """A contract published and never applied is not a contract.

    ``arguments`` is ``dict[str, Any]`` and ``bind_parameters`` checked only
    ``required`` and ``pattern`` before ``str()``-ing the value, so a list bound
    to the literal string ``"['S0', '2']"`` and was typed into the bank
    application.
    """
    response = client.post(
        f"/capabilities/{CAPABILITY}:invoke",
        json={"arguments": {"member_id": ["1", "2"]}},
    )

    assert response.status_code == 422
    assert "declared 'string'" in response.json()["error"]


def test_a_capability_publishes_which_arguments_are_regulated(tmp_path):
    """Otherwise an agent reading the catalogue cannot tell, and will log it."""
    from replay.artifact import invocation_schema
    from replay.artifact.schema import ParamSpec

    artifact = ArtifactStore("artifacts").load(CAPABILITY)
    with_secret = artifact.model_copy(
        update={
            "inputs": [
                *artifact.inputs,
                ParamSpec(name="ssn", description="Member SSN.", sensitive=True),
            ]
        }
    )
    published = invocation_schema(with_secret)["properties"]

    assert published["ssn"]["sensitive"] is True
    assert "sensitive" not in published["member_id"]


def test_a_capability_publishes_what_it_returns(client):
    returns = client.get(f"/capabilities/{CAPABILITY}").json()["returns"]
    assert [r["name"] for r in returns] == ["current_savings_balance"]
    assert returns[0]["type"] == "money"


def test_a_capability_publishes_how_it_can_legitimately_not_succeed(client):
    """The full result space, before invoking rather than after."""
    outcomes = client.get(f"/capabilities/{CAPABILITY}").json()["outcomes"]
    assert {o["code"] for o in outcomes} >= {"MEMBER_NOT_FOUND", "PERMISSION_DENIED"}
    assert all(o["message"] for o in outcomes)


def test_risk_and_approval_are_visible_before_calling(client):
    """So an agent can tell a read-only lookup from something that moves money."""
    read_only = client.get(f"/capabilities/{CAPABILITY}").json()
    write = client.get("/capabilities/open_subaccount").json()

    assert read_only["risk"] == "safe"
    assert write["risk"] == "irreversible"
    assert write["requires_approval"] is True
    assert write["approval"] == "draft"


def test_a_specific_version_can_be_pinned(client):
    pinned = client.get(f"/capabilities/{CAPABILITY}", params={"version": "1.0.0"}).json()
    assert pinned["ref"] == f"{CAPABILITY}@1.0.0"


def test_the_whole_artifact_is_available_for_review(client):
    """A capability is a reviewable document, so it has to be readable in full."""
    artifact = client.get(f"/capabilities/{CAPABILITY}/artifact").json()
    assert artifact["steps"]
    assert artifact["steps"][1]["target"]["rationale"], "the robustness reasoning is there"


def test_a_file_that_does_not_parse_is_an_answer_not_a_traceback(tmp_path):
    """``get_artifact`` and ``invoke`` caught only ``ArtifactNotFound``.

    That is raised solely from a ``path.exists()`` check, so a path that exists
    and does not parse came out of the handler as a raw ``ValidationError`` —
    an HTTP 500 with a traceback, and a 404-versus-500 oracle telling an
    unauthenticated caller which paths exist on the host.
    """
    (tmp_path / "lookup_balance@1.0.0.json").write_text('{"schema_version": "1.0", "na')
    broken = TestClient(
        create_api(
            artifacts_dir=tmp_path,
            evidence_dir=tmp_path / "runs",
            allowlist=TEST_ALLOWLIST,
        )
    )

    unreadable = broken.get(f"/capabilities/{CAPABILITY}/artifact?version=1.0.0")
    assert unreadable.status_code == 500
    assert "not a readable capability artifact" in unreadable.json()["error"]

    # And the endpoint answers identically whether or not the traversal target
    # happens to exist, so it says nothing about the filesystem.
    assert (
        broken.get(f"/capabilities/{CAPABILITY}/artifact?version=../../policy").status_code == 404
    )
    assert broken.get(f"/capabilities/{CAPABILITY}/artifact?version=../../nope").status_code == 404


def test_an_unparseable_override_is_a_refusal_not_a_500(tmp_path):
    """The other unhandled parse on the invoke path."""
    (tmp_path / "overrides" / "northgate").mkdir(parents=True)
    (tmp_path / "overrides" / "northgate" / f"{CAPABILITY}.json").write_text("{ not json")
    client = TestClient(
        create_api(
            evidence_dir=tmp_path / "runs",
            overrides_dir=tmp_path / "overrides",
            allowlist=TEST_ALLOWLIST,
        )
    )

    response = client.post(
        f"/capabilities/{CAPABILITY}:invoke",
        json={"arguments": {"member_id": "12345"}, "tenant": "northgate"},
    )

    assert response.status_code == 409
    assert "not a readable override" in response.json()["error"]


def test_an_unknown_capability_is_a_404(client):
    assert client.get("/capabilities/nope").status_code == 404
    assert client.post("/capabilities/nope:invoke", json={"arguments": {}}).status_code == 404


# ---------- invocation ----------


def test_a_capability_is_invoked_by_name_over_http(client, meridian_server):
    """The brief's stretch goal: show one being invoked.

    This is the whole through-line arriving — the model discovered it once, the
    artifact captured it, and an agent now calls it by name with typed args and
    no model anywhere in the path.
    """
    response = client.post(
        f"/capabilities/{CAPABILITY}:invoke",
        json={"arguments": {"member_id": "12345"}, "target": meridian_server},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["outputs"] == {"current_savings_balance": "4,211.03"}
    assert body["locator_tiers"] == {"s2": 3, "s3": 1, "s4": 4}


def test_a_business_outcome_is_not_an_http_error(client, meridian_server):
    """ "No such member" is the capability working. 200, with a code to branch on."""
    response = client.post(
        f"/capabilities/{CAPABILITY}:invoke",
        json={"arguments": {"member_id": "99999"}, "target": meridian_server},
    )

    assert response.status_code == 200
    assert response.json()["outcome"]["code"] == "MEMBER_NOT_FOUND"


def test_a_malfunction_is_an_http_error(client, meridian_server):
    response = client.post(
        f"/capabilities/{CAPABILITY}:invoke",
        json={"arguments": {"member_id": "oops"}, "target": meridian_server},
    )

    assert response.status_code == 422
    assert response.json()["failure"]["class"] == "invalid_input"


def test_served_evidence_lands_where_the_server_was_told(tmp_path, monkeypatch, meridian_server):
    """And by default that is `runs/`, never the committed `evidence/`.

    `evidence/` is a curated deliverable and it is what the approval tally is
    read from, so a `curl` against the catalog must not dirty it or move the
    numbers that decide whether a capability runs unattended.
    """
    artifacts = Path("artifacts").resolve()
    monkeypatch.chdir(tmp_path)
    served = TestClient(create_api(artifacts_dir=artifacts, allowlist=TEST_ALLOWLIST))

    served.post(
        f"/capabilities/{CAPABILITY}:invoke",
        json={"arguments": {"member_id": "12345"}, "target": meridian_server},
    )

    assert list((tmp_path / "runs").glob("invoke-*/run.jsonl"))
    assert not (tmp_path / "evidence").exists()


# ---------- the guardrails still apply ----------


def test_the_api_cannot_permit_more_than_the_command_line(client, meridian_server):
    """An endpoint that quietly outranks the guardrails makes them decorative."""
    response = client.post(
        "/capabilities/open_subaccount:invoke",
        json={
            "arguments": {
                "member_id": "12345",
                "product_code": "S02",
                "opening_deposit": "50.00",
            },
            "target": meridian_server,
        },
    )

    assert response.status_code == 422
    assert response.json()["failure"]["class"] == "policy_refused"
    assert response.json()["steps"] == [], "nothing was executed"


def test_escalation_is_off_by_default(client, meridian_server):
    """An agent calling an API expects an answer, not a fifteen-minute hold.

    So a blocked capability is refused immediately unless the caller explicitly
    asked for a human to be found.
    """
    response = client.post(
        "/capabilities/open_subaccount:invoke",
        json={
            "arguments": {
                "member_id": "12345",
                "product_code": "S02",
                "opening_deposit": "50.00",
            },
            "target": meridian_server,
        },
    )
    assert response.json()["escalation"] is None


def test_an_unlisted_target_is_refused_by_the_allowlist(tmp_path):
    strict = TestClient(
        create_api(evidence_dir=tmp_path, allowlist=Allowlist(domains=("127.0.0.1:8080",)))
    )
    response = strict.post(
        f"/capabilities/{CAPABILITY}:invoke",
        json={"arguments": {"member_id": "12345"}, "target": "https://evil.example.com"},
    )
    assert response.status_code == 422
    assert response.json()["failure"]["class"] == "policy_refused"


# ---------- the operator console rides along ----------


def test_the_operator_console_is_mounted(client):
    """One process serves both, because a single-operator handoff needs no more."""
    assert client.get("/operator/").status_code == 200
    assert "OPERATOR CONSOLE" in client.get("/operator/").text


# ---------- the CLI view matches ----------


def test_the_cli_and_the_api_show_the_same_thing():
    """One summarise(), so the two surfaces cannot drift apart."""
    artifact = ArtifactStore("artifacts").load(CAPABILITY)
    described = summarise(artifact)
    assert described["ref"] == artifact.ref
    assert described["arguments"]["required"] == [p.name for p in artifact.inputs if p.required]
