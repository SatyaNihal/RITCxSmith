"""HTTP client for the RIT DMA REST API.

This is the only module that talks to the RIT server. Credentials live only in
the requests session: they are never logged, printed or written to disk. Every
write (orders, cancels, tender responses) is refused unless the client was
created with ``execute=True``.

API reference: https://rit.306w.ca/RIT-DMA-API/1.0.5/
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from getpass import getpass
from typing import Any, Callable

import requests

DEFAULT_HOST = "flserver.rotman.utoronto.ca"
# Practice ports for the browser / Mac app / DMA API. Rotman announces the
# competition port on the day; pass it with --port.
PRACTICE_PORTS = {"etf": 16635, "volatility": 16625}

DEFAULT_TIMEOUT_S = 5.0
MAX_RATE_LIMIT_RETRIES = 4
MIN_WAIT_S = 0.05
MAX_WAIT_S = 5.0
DEFAULT_WAIT_S = 1.0


class RitError(Exception):
    """Any failure while talking to RIT."""


class AuthError(RitError):
    """HTTP 401: wrong Trader ID or password for this case."""


class RateLimited(RitError):
    """Still HTTP 429 after the allowed number of retries."""

    def __init__(self, what: str, wait_s: float):
        super().__init__(
            f"{what} was still rate-limited after {MAX_RATE_LIMIT_RETRIES} retries "
            f"(server asked for {wait_s:.3f}s)"
        )
        self.wait_s = wait_s


class RequestTimeout(RitError):
    """The server did not answer within the timeout."""


class ConnectionFailed(RitError):
    """The request failed at the network level."""


class RequestRejected(RitError):
    """The server answered with a non-2xx status other than 401/429."""

    def __init__(self, what: str, status_code: int, body: str):
        super().__init__(f"{what} returned HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


class WriteBlocked(RitError):
    """A write was attempted in dry-run mode."""


class SubmissionUncertain(RitError):
    """A write may or may not have reached RIT. Reconcile before retrying."""


@dataclass(frozen=True)
class Credentials:
    trader_id: str
    password: str = field(repr=False)


def prompt_credentials(
    input_fn: Callable[[str], str] = input,
    getpass_fn: Callable[[str], str] = getpass,
) -> Credentials:
    """Ask for the Trader ID and password at runtime (never from argv or files)."""
    trader_id = input_fn("Trader ID: ").strip()
    password = getpass_fn("RIT password: ")
    if not trader_id or not password:
        raise SystemExit("A Trader ID and password are required.")
    return Credentials(trader_id, password)


def _retry_after_seconds(response: Any) -> float:
    """Seconds to wait from a 429 response (Retry-After header, else JSON `wait`)."""
    wait: Any = None
    headers = getattr(response, "headers", None) or {}
    if headers.get("Retry-After") is not None:
        wait = headers.get("Retry-After")
    else:
        try:
            wait = response.json().get("wait")
        except (ValueError, AttributeError):
            wait = None
    try:
        seconds = float(wait) if wait is not None else DEFAULT_WAIT_S
    except (TypeError, ValueError):
        seconds = DEFAULT_WAIT_S
    return min(max(seconds, MIN_WAIT_S), MAX_WAIT_S)


def _short(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _number(value: float) -> float | int:
    value = float(value)
    return int(value) if value.is_integer() else value


class RitClient:
    """Thin wrapper over the RIT DMA REST API with timeouts and 429 back-off."""

    def __init__(
        self,
        credentials: Credentials,
        port: int,
        host: str = DEFAULT_HOST,
        *,
        execute: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = f"http://{host}:{int(port)}/v1"
        self.execute = execute
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._session = session if session is not None else requests.Session()
        self._session.trust_env = False
        self._session.auth = (credentials.trader_id, credentials.password)
        self.stats = {"requests": 0, "rate_limited": 0, "timeouts": 0, "errors": 0, "writes": 0}

    # -- plumbing ---------------------------------------------------------

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "RitClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, resource: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}/{resource}"
        what = f"{method} /{resource}"
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            self.stats["requests"] += 1
            try:
                response = self._session.request(
                    method, url, params=params, timeout=self.timeout_s, allow_redirects=False
                )
            except requests.Timeout as error:
                self.stats["timeouts"] += 1
                raise RequestTimeout(f"{what} timed out after {self.timeout_s:g}s") from error
            except requests.RequestException as error:
                self.stats["errors"] += 1
                raise ConnectionFailed(f"{what} failed: {type(error).__name__}") from error

            if response.status_code == 401:
                raise AuthError("RIT rejected the Trader ID or password (HTTP 401).")
            if response.status_code == 429:
                self.stats["rate_limited"] += 1
                wait_s = _retry_after_seconds(response)
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise RateLimited(what, wait_s)
                self._sleep(wait_s)
                continue
            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except ValueError as error:
                    raise RitError(f"{what} returned a body that is not JSON") from error
            self.stats["errors"] += 1
            raise RequestRejected(what, response.status_code, _short(response.text))
        raise AssertionError("unreachable")

    def get(self, resource: str, **params: Any) -> Any:
        clean = {key: value for key, value in params.items() if value is not None}
        return self._request("GET", resource, clean or None)

    def _write(self, method: str, resource: str, params: dict | None = None) -> Any:
        if not self.execute:
            raise WriteBlocked(f"Dry-run mode refused {method} /{resource}.")
        self.stats["writes"] += 1
        try:
            return self._request(method, resource, params)
        except (RequestTimeout, ConnectionFailed) as error:
            raise SubmissionUncertain(
                f"{method} /{resource} may have reached RIT ({error}). "
                "Reconcile orders and positions before sending anything else."
            ) from error

    # -- reads ------------------------------------------------------------

    def case(self) -> dict:
        return self.get("case")

    def trader(self) -> dict:
        return self.get("trader")

    def securities(self, ticker: str | None = None) -> list[dict]:
        return self.get("securities", ticker=ticker)

    def book(self, ticker: str, limit: int | None = None) -> dict:
        return self.get("securities/book", ticker=ticker, limit=limit)

    def limits(self) -> list[dict]:
        return self.get("limits")

    def tenders(self) -> list[dict]:
        return self.get("tenders")

    def orders(self, status: str = "OPEN") -> list[dict]:
        return self.get("orders", status=status)

    def order(self, order_id: int) -> dict:
        return self.get(f"orders/{int(order_id)}")

    def transactions(self, after: int | None = None, limit: int | None = None) -> list[dict]:
        return self.get("transactions", after=after, limit=limit)

    def news(self, after: int | None = None, limit: int | None = None) -> list[dict]:
        return self.get("news", after=after, limit=limit)

    # -- writes (execute mode only) --------------------------------------

    def submit_order(
        self,
        ticker: str,
        action: str,
        quantity: float,
        order_type: str = "MARKET",
        price: float | None = None,
    ) -> dict:
        action = action.upper()
        order_type = order_type.upper()
        if action not in ("BUY", "SELL"):
            raise ValueError(f"action must be BUY or SELL, not {action!r}")
        if order_type not in ("MARKET", "LIMIT"):
            raise ValueError(f"order_type must be MARKET or LIMIT, not {order_type!r}")
        if not quantity or quantity <= 0:
            raise ValueError(f"quantity must be positive, not {quantity!r}")
        params: dict[str, Any] = {
            "ticker": ticker,
            "type": order_type,
            "quantity": _number(quantity),
            "action": action,
        }
        if order_type == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders need a price")
            params["price"] = price
        return self._write("POST", "orders", params)

    def cancel_order(self, order_id: int) -> dict:
        return self._write("DELETE", f"orders/{int(order_id)}")

    def cancel_all(self) -> dict:
        return self._write("POST", "commands/cancel", {"all": 1})

    def cancel_ticker(self, ticker: str) -> dict:
        return self._write("POST", "commands/cancel", {"ticker": ticker})

    def accept_tender(self, tender_id: int, price: float | None = None) -> dict:
        params = {"price": price} if price is not None else None
        return self._write("POST", f"tenders/{int(tender_id)}", params)

    def decline_tender(self, tender_id: int) -> dict:
        return self._write("DELETE", f"tenders/{int(tender_id)}")
