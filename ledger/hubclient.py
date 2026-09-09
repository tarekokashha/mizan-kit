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
from collections.abc import Iterator

API = "https://huggingface.co/api"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json"


def _next_link(link: str | None) -> str | None:
    """The URL of the rel="next" relation in a Link header, or None.

    A Link header can carry more than one relation, comma separated,
    each its own "<url>; rel=\"...\"" segment (RFC 8288). Checking
    whether rel="next" appears anywhere in the whole header and then
    always taking the URL from the first ";"-separated segment (the old
    approach) picks whichever relation happens to come first once more
    than one is present, not necessarily "next". Splitting on "," first
    isolates each relation into its own segment, so the right one is
    found regardless of order.
    """
    if not link:
        return None
    for segment in link.split(","):
        if 'rel="next"' in segment:
            return segment.split(";")[0].strip().strip("<> ")
    return None


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


def _extract_revision(response) -> str:
    """Read the X-Repo-Commit header off a Hub response, or "".

    The resolve endpoint carries the exact commit sha of the revision it
    served in this header, at no extra request cost. A response with no
    headers attribute, or one whose headers do not carry this key (an
    older endpoint, or a fake response a test builds without it), yields
    "" rather than raising: the revision is best effort metadata, never
    a requirement for the caller to keep working.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    try:
        return headers.get("X-Repo-Commit", "") or ""
    except AttributeError:
        return ""


class HubClient:
    def __init__(
        self,
        min_interval: float = 0.2,
        max_retries: int = 5,
        timeout: float = 45.0,
        token: str | None = None,
        user_agent: str = "mizan-ledger/0.1",
    ):
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
        # repo -> revision sha, captured as a side effect of get_info().
        # See _extract_revision and get_revision below (Ruling 9).
        self._revisions: dict[str, str] = {}

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
                    wait = float(ra) if ra and str(ra).isdigit() else 2**attempt
                    time.sleep(wait + random.uniform(0, 0.4))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(2**attempt + random.uniform(0, 0.4))
        raise RateLimited(f"gave up on {url} after {self.max_retries} attempts: {last}")

    def get_json(self, url: str):
        with self._raw(url) as r:
            return json.loads(r.read())

    def get_info(self, repo: str) -> dict | None:
        """The parsed meta/info.json for repo, or None if it genuinely
        is not there.

        None means a genuine 401, 403 or 404: gated, private, or missing.
        An exhausted backoff (RateLimited, see _raw) is a different
        situation entirely, not indistinguishable absence, so it is left
        to propagate rather than being caught here and folded into the
        same None a caller cannot tell apart from real absence. A
        malformed JSON body (json.JSONDecodeError) still returns None:
        that is a genuine response the Hub actually sent, not a retry
        exhaustion.
        """
        try:
            with self._raw(RESOLVE.format(repo=repo)) as r:
                self._revisions[repo] = _extract_revision(r)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            raise
        except json.JSONDecodeError:
            return None

    def get_revision(self, repo: str) -> str:
        """The Hub revision sha last captured for repo, or "".

        Populated as a side effect of get_info(repo): see
        _extract_revision. Returns "" for a repo get_info has not been
        called for yet, or whose response carried no X-Repo-Commit
        header. Never raises.
        """
        return self._revisions.get(repo, "")

    def iter_datasets(
        self, filter_tag: str = "LeRobot", page_size: int = 1000, max_pages: int | None = None
    ) -> Iterator[dict]:
        url = f"{API}/datasets?filter={filter_tag}&limit={page_size}"
        pages = 0
        while url and (max_pages is None or pages < max_pages):
            with self._raw(url) as r:
                body = json.loads(r.read())
                link = r.headers.get("Link") if r.headers else None
            yield from body
            pages += 1
            url = _next_link(link)
