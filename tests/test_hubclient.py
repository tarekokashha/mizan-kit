import json
import pytest
from ledger.hubclient import HubClient, RateLimited


class FakeResp:
    def __init__(self, body, headers=None, status=200):
        self._b = json.dumps(body).encode(); self.headers = headers or {}; self.status = status
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}
    import urllib.error

    def opener(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {"Retry-After": "0"}, None)
        return FakeResp({"ok": True})

    c = HubClient(min_interval=0.0)
    monkeypatch.setattr(c, "_open", opener)
    assert c.get_json("https://example/x") == {"ok": True}
    assert calls["n"] == 3


def test_gives_up_and_raises(monkeypatch):
    import urllib.error

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "rate", {"Retry-After": "0"}, None)

    c = HubClient(min_interval=0.0, max_retries=2)
    monkeypatch.setattr(c, "_open", opener)
    with pytest.raises(RateLimited):
        c.get_json("https://example/x")


def test_missing_info_returns_none(monkeypatch):
    import urllib.error

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "nope", {}, None)

    c = HubClient(min_interval=0.0)
    monkeypatch.setattr(c, "_open", opener)
    assert c.get_info("acme/missing") is None


def test_token_is_never_in_repr():
    c = HubClient(token="hf_secretvalue")
    assert "hf_secretvalue" not in repr(c)
    assert "hf_secretvalue" not in str(c.__dict__.get("_headers", {}))


def test_pagination_follows_link_header(monkeypatch):
    pages = [
        (FakeResp([{"id": "a"}], {"Link": '<https://example/p2>; rel="next"'})),
        (FakeResp([{"id": "b"}], {})),
    ]
    seq = iter(pages)
    c = HubClient(min_interval=0.0)
    monkeypatch.setattr(c, "_open", lambda req, timeout=None: next(seq))
    assert [d["id"] for d in c.iter_datasets()] == ["a", "b"]


def test_authorization_header_reaches_request(monkeypatch):
    captured = {}

    def opener(req, timeout=None):
        captured["req"] = req
        return FakeResp({"ok": True})

    c = HubClient(min_interval=0.0, token="hf_testvalue")
    monkeypatch.setattr(c, "_open", opener)
    c.get_json("https://example/x")
    assert captured["req"].get_header("Authorization") == "Bearer hf_testvalue"


def test_no_token_sends_no_authorization_header(monkeypatch):
    # Clear both env vars the client checks, and also stub the
    # huggingface_hub fallback so this does not depend on whether this
    # machine happens to have a cached CLI login or an OIDC token.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    captured = {}

    def opener(req, timeout=None):
        captured["req"] = req
        return FakeResp({"ok": True})

    c = HubClient(min_interval=0.0, token=None)
    monkeypatch.setattr(c, "_open", opener)
    c.get_json("https://example/x")
    assert captured["req"].get_header("Authorization") is None
