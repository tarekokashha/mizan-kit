"""Token aware Hub access with one global rate limit.

Measured on 2026-09-05: anonymous requests hit HTTP 429 with
"maximum queue size reached" within tens of requests, on both the api
and resolve hosts. A census therefore needs a token and needs to behave.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Iterator

API = "https://huggingface.co/api"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json"


class RateLimited(RuntimeError):
    pass


def _discover_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


class HubClient:
    def __init__(self, min_interval: float = 0.2, max_retries: int = 5,
                 timeout: float = 45.0, token: str | None = None,
                 user_agent: str = "mizan-ledger/0.1"):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._headers = {"User-Agent": user_agent}
        tok = token if token is not None else _discover_token()
        self._has_token = bool(tok)
        # The token itself is kept out of self._headers on purpose. That
        # dict is the kind of thing that ends up dumped in a debug log or
        # pasted into an issue, and the Authorization value must never be
        # in it. It is merged in only at request build time, in _raw.
        self._token = tok
        self._last = 0.0

    def __repr__(self) -> str:
        return f"HubClient(authenticated={self._has_token}, min_interval={self.min_interval})"

    @property
    def authenticated(self) -> bool:
        return self._has_token

    def _open(self, req, timeout=None):
        return urllib.request.urlopen(req, timeout=timeout)

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def _request_headers(self) -> dict:
        headers = dict(self._headers)
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _raw(self, url: str):
        req = urllib.request.Request(url, headers=self._request_headers())
        last = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                return self._open(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = float(ra) if ra and str(ra).isdigit() else 2 ** attempt
                    time.sleep(wait + random.uniform(0, 0.4))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(2 ** attempt + random.uniform(0, 0.4))
        raise RateLimited(f"gave up on {url} after {self.max_retries} attempts: {last}")

    def get_json(self, url: str):
        with self._raw(url) as r:
            return json.loads(r.read())

    def get_info(self, repo: str) -> dict | None:
        try:
            return self.get_json(RESOLVE.format(repo=repo))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            raise
        except (RateLimited, json.JSONDecodeError):
            return None

    def iter_datasets(self, filter_tag: str = "LeRobot", page_size: int = 1000,
                      max_pages: int | None = None) -> Iterator[dict]:
        url = f"{API}/datasets?filter={filter_tag}&limit={page_size}"
        pages = 0
        while url and (max_pages is None or pages < max_pages):
            with self._raw(url) as r:
                body = json.loads(r.read())
                link = r.headers.get("Link") if r.headers else None
            yield from body
            pages += 1
            url = None
            if link and 'rel="next"' in link:
                url = link.split(";")[0].strip("<> ")
