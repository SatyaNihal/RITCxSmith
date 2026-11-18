"""RitClient tests with a fake HTTP session (no network)."""

import unittest

import requests

from rit_client import (
    AuthError,
    Credentials,
    RateLimited,
    RequestRejected,
    RequestTimeout,
    RitClient,
    SubmissionUncertain,
    WriteBlocked,
)


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None, text=""):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


class FakeSession:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True
        self.auth = None

    def request(self, method, url, params=None, timeout=None, allow_redirects=True):
        self.calls.append({"method": method, "url": url, "params": params, "timeout": timeout, "redirects": allow_redirects})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        pass


def client(responses=(), execute=False):
    session = FakeSession(responses)
    sleeps = []
    rit = RitClient(Credentials("trader", "secret"), 16635, execute=execute, session=session, sleep=sleeps.append)
    return rit, session, sleeps


class RitClientTests(unittest.TestCase):
    def test_session_setup(self):
        rit, session, _ = client([FakeResponse(200, {"tick": 1})])
        rit.case()
        self.assertFalse(session.trust_env)
        self.assertEqual(session.auth, ("trader", "secret"))
        call = session.calls[0]
        self.assertEqual(call["url"], "http://flserver.rotman.utoronto.ca:16635/v1/case")
        self.assertEqual(call["timeout"], 5.0)
        self.assertFalse(call["redirects"])

    def test_dry_run_refuses_every_write_without_sending(self):
        rit, session, _ = client()
        for write in (
            lambda: rit.submit_order("RITC", "BUY", 100),
            lambda: rit.cancel_order(1),
            lambda: rit.cancel_all(),
            lambda: rit.accept_tender(1, 25.0),
            lambda: rit.decline_tender(1),
        ):
            with self.assertRaises(WriteBlocked):
                write()
        self.assertEqual(session.calls, [])

    def test_rate_limit_waits_retry_after_then_succeeds(self):
        rit, session, sleeps = client([FakeResponse(429, headers={"Retry-After": "0.2"}), FakeResponse(200, {"tick": 5})])
        self.assertEqual(rit.case(), {"tick": 5})
        self.assertEqual(sleeps, [0.2])
        self.assertEqual(rit.stats["rate_limited"], 1)

    def test_rate_limit_wait_from_body(self):
        rit, _, sleeps = client([FakeResponse(429, {"code": "RATE_LIMIT_EXCEEDED", "wait": 0.3}), FakeResponse(200, [])])
        rit.tenders()
        self.assertEqual(sleeps, [0.3])

    def test_rate_limit_gives_up(self):
        rit, _, _ = client([FakeResponse(429, {"wait": 0.1})] * 5)
        with self.assertRaises(RateLimited):
            rit.case()

    def test_read_timeout(self):
        rit, _, _ = client([requests.ReadTimeout()])
        with self.assertRaises(RequestTimeout):
            rit.securities()

    def test_write_timeout_is_uncertain(self):
        rit, _, _ = client([requests.ReadTimeout()], execute=True)
        with self.assertRaises(SubmissionUncertain):
            rit.submit_order("RITC", "SELL", 100)

    def test_rejected_order(self):
        rit, _, _ = client([FakeResponse(400, text="Order exceeds limits")], execute=True)
        with self.assertRaises(RequestRejected) as caught:
            rit.submit_order("RITC", "SELL", 100)
        self.assertEqual(caught.exception.status_code, 400)

    def test_auth_error(self):
        rit, _, _ = client([FakeResponse(401, {"code": "UNAUTHORIZED"})])
        with self.assertRaises(AuthError):
            rit.case()

    def test_order_params(self):
        rit, session, _ = client([FakeResponse(200, {"order_id": 1})], execute=True)
        rit.submit_order("RITC", "buy", 100.0, "limit", 25.01)
        self.assertEqual(
            session.calls[0]["params"],
            {"ticker": "RITC", "type": "LIMIT", "quantity": 100, "action": "BUY", "price": 25.01},
        )

    def test_password_never_in_repr(self):
        self.assertNotIn("secret", repr(Credentials("trader", "secret")))


if __name__ == "__main__":
    unittest.main()
