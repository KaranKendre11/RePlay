"""MERIDIAN CORE — Flask application.

Structure notes, since the unpleasantness is deliberate:

* Every screen renders inside a real ``<frameset>``. Nothing is a single-page
  document, so the surface layer must carry a frame path on every node.
* No element carries ``data-testid``, ``id``, or an ARIA attribute. Buttons get
  an accessible name from their visible text; text inputs get none at all,
  because their visible label is only an adjacent table cell.
* Failure injection is request-scoped and explicit. See :mod:`.inject`.
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

from flask import (
    Flask,
    make_response,
    render_template,
    request,
    session,
    url_for,
)

from targets.meridian import inject as inj
from targets.meridian.data import find_member

SESSION_LIFETIME = timedelta(minutes=30)


@dataclass(frozen=True)
class Variant:
    """A tenant-specific skin of the same vendor product.

    Present from the start because M11 replays one artifact against two
    variants. Keeping the seam here costs nothing now and avoids retrofitting
    a second app later.
    """

    key: str
    brand: str
    brand_color: str
    teller: str


VARIANTS: dict[str, Variant] = {
    "base": Variant("base", "MERIDIAN CORE", "#1a3a6b", "TLR0431"),
    "northgate": Variant("northgate", "NORTHGATE FCU / MERIDIAN", "#5b1a1a", "TLR8802"),
}


def create_app(variant: str = "base", secret_key: str = "meridian-dev-only") -> Flask:
    app = Flask(__name__)
    app.secret_key = secret_key
    app.permanent_session_lifetime = SESSION_LIFETIME
    skin = VARIANTS[variant]

    # ---------- helpers ----------

    def chrome(screen_code: str, **extra: object) -> dict[str, object]:
        """Template variables every screen needs."""
        return {
            "brand": skin.brand,
            "brand_color": skin.brand_color,
            "teller": skin.teller,
            "screen_code": screen_code,
            "title": f"{skin.brand} {screen_code}",
            "session_short": session.get("sid", "-")[:8],
            **extra,
        }

    def current_injection() -> inj.Injection | None:
        return inj.parse(request.values.get("inject"))

    def inject_qs(injection: inj.Injection | None) -> str:
        return f"?{urlencode({'inject': injection.value})}" if injection else ""

    def notice(heading: str, code: str, detail: str, status: int = 200, back: str | None = None):
        body = render_template(
            "notice.html",
            **chrome("MSG0001", heading=heading, code=code, detail=detail, back_url=back),
        )
        return make_response(body, status)

    def session_is_live() -> bool:
        expires = session.get("expires_at")
        return bool(expires) and datetime.now(UTC).timestamp() < expires

    def start_session() -> None:
        session.permanent = True
        session["sid"] = hashlib.sha1(str(time.time()).encode()).hexdigest()
        session["expires_at"] = (datetime.now(UTC) + SESSION_LIFETIME).timestamp()
        session["seen_interstitial"] = False

    @app.before_request
    def _guard():
        """Apply request-scoped injections before any view runs.

        Order matters. ``error500`` and ``timeout`` short-circuit; ``slow``
        merely delays; ``dialog`` is handled per-view because it needs a
        continue URL.
        """
        if "sid" not in session:
            start_session()

        injection = current_injection()

        if injection is inj.Injection.ERROR500:
            return notice(
                "APPLICATION ERROR",
                "SYS-500",
                "Unhandled exception in transaction processor. Contact support.",
                status=500,
            )

        if injection is inj.Injection.TIMEOUT:
            session["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).timestamp()

        if not session_is_live() and request.endpoint not in {"frameset", "nav", "root_reset"}:
            return notice(
                "SESSION EXPIRED",
                "SEC-0031",
                "Your session has timed out. Sign in again to continue.",
                status=440,
            )

        if injection is inj.Injection.SLOW:
            time.sleep(inj.SLOW_DELAY_SECONDS)

        return None

    def maybe_interstitial(injection: inj.Injection | None, continue_url: str):
        """Show the one-shot system notice for the ``dialog`` injection."""
        if injection is not inj.Injection.DIALOG or session.get("seen_interstitial"):
            return None
        session["seen_interstitial"] = True
        return render_template("interstitial.html", **chrome("MSG0009", continue_url=continue_url))

    # ---------- screens ----------

    @app.route("/")
    def frameset():
        injection = current_injection()
        return render_template(
            "frameset.html",
            start_url=url_for("search") + inject_qs(injection),
        )

    @app.route("/reset")
    def root_reset():
        """Clear session state. Used by tests to make injections repeatable."""
        session.clear()
        start_session()
        return notice("SESSION RESET", "SYS-0000", "New session established.")

    @app.route("/nav")
    def nav():
        return render_template("nav.html", **chrome("NAV0001"))

    @app.route("/search")
    def search():
        injection = current_injection()
        return render_template(
            "search.html",
            **chrome("INQ0100", inject=injection.value if injection else "", message=None),
        )

    @app.route("/member")
    def member_lookup():
        """Search submit target. Renders detail, or a business-outcome notice."""
        injection = current_injection()
        member_id = (request.args.get("f7") or "").strip()

        if not member_id:
            return render_template(
                "search.html",
                **chrome(
                    "INQ0100",
                    inject=injection.value if injection else "",
                    message="MEMBER ID IS REQUIRED.",
                ),
            )

        # The injection hides an otherwise real member, so the not-found path is
        # reachable without depending on a particular ID being absent.
        member = None if injection is inj.Injection.NOT_FOUND else find_member(member_id)

        if member is None:
            return notice(
                "NO RECORD FOUND",
                inj.OUTCOME_CODES[inj.Injection.NOT_FOUND],
                f"No member on file for ID {member_id}.",
                back=url_for("search"),
            )

        if injection is inj.Injection.DENIED or member.status == "RESTRICTED":
            return notice(
                "ACCESS DENIED",
                inj.OUTCOME_CODES[inj.Injection.DENIED],
                "Teller authority insufficient for this member record.",
                back=url_for("search"),
            )

        held = maybe_interstitial(injection, url_for("member_detail", member_id=member.member_id))
        if held:
            return held

        return render_template(
            "member.html",
            **chrome("INQ0200", member=member, inject_qs=inject_qs(injection)),
        )

    @app.route("/member/<member_id>")
    def member_detail(member_id: str):
        injection = current_injection()
        member = find_member(member_id)
        if member is None:
            return notice(
                "NO RECORD FOUND",
                inj.OUTCOME_CODES[inj.Injection.NOT_FOUND],
                f"No member on file for ID {member_id}.",
                back=url_for("search"),
            )
        return render_template(
            "member.html",
            **chrome("INQ0200", member=member, inject_qs=inject_qs(injection)),
        )

    @app.route("/member/<member_id>/subaccount/new")
    def subaccount_new(member_id: str):
        injection = current_injection()
        member = find_member(member_id)
        if member is None:
            return notice(
                "NO RECORD FOUND",
                inj.OUTCOME_CODES[inj.Injection.NOT_FOUND],
                f"No member on file for ID {member_id}.",
                back=url_for("search"),
            )
        return render_template(
            "subaccount_new.html",
            **chrome(
                "ACS0400",
                member=member,
                inject=injection.value if injection else "",
                message=None,
            ),
        )

    @app.route("/member/<member_id>/subaccount", methods=["POST"])
    def subaccount_create(member_id: str):
        injection = current_injection()
        member = find_member(member_id)
        if member is None:
            return notice(
                "NO RECORD FOUND",
                inj.OUTCOME_CODES[inj.Injection.NOT_FOUND],
                f"No member on file for ID {member_id}.",
                back=url_for("search"),
            )

        product = (request.form.get("f12") or "").strip()
        deposit = (request.form.get("f13") or "").strip()
        nickname = (request.form.get("f14") or "").strip()

        reason = _validation_error(injection, product, deposit)
        if reason:
            return render_template(
                "subaccount_new.html",
                **chrome(
                    "ACS0400",
                    member=member,
                    inject=injection.value if injection else "",
                    message=f"{inj.OUTCOME_CODES[inj.Injection.VALIDATION]}: {reason}",
                ),
            )

        if injection is inj.Injection.DENIED:
            return notice(
                "ACCESS DENIED",
                inj.OUTCOME_CODES[inj.Injection.DENIED],
                "Teller authority insufficient to open accounts.",
                back=url_for("member_detail", member_id=member_id),
            )

        suffix = f"{(int(member_id) % 97) + 20:03d}"
        return render_template(
            "subaccount_confirm.html",
            **chrome(
                "ACS0450",
                member=member,
                account_number=f"{member_id}{suffix}",
                product=product,
                deposit=deposit or "0.00",
                reference=f"REF-{member_id}-{suffix}",
                nickname=nickname,
            ),
        )

    return app


def _validation_error(injection: inj.Injection | None, product: str, deposit: str) -> str | None:
    """Server-side validation. The injection forces a rejection on valid input."""
    if injection is inj.Injection.VALIDATION:
        return "Opening deposit below product minimum of 25.00."
    if not product:
        return "Product code is required."
    if deposit:
        try:
            if float(deposit.replace(",", "")) < 0:
                return "Opening deposit may not be negative."
        except ValueError:
            return "Opening deposit is not a valid amount."
    return None
