"""Fetch the curated Next.js doc paths at one pinned commit.

Only this module opens a socket. The commit is resolved once (a single
``curl.exe`` call, because the GitHub tree API does not survive Python's
``urllib`` on this box) and then pinned in ``sources.json``; later runs reuse
that pin, so a cold cache or ``--rebuild`` refetches bytes without moving the
commit. File bytes come one path at a time over ``httpx`` with three retries
and backoff, and every fetched path is recorded with its sha256.

Spec departure, approved by Daniel after SPEC-corpus.md was written: ``httpx``
replaces stdlib ``urllib.request`` for file fetch (the tree API hangs urllib).
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

REPO = "vercel/next.js"
BRANCH = "canary"
LICENCE = "MIT"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
COMMIT_URL = "https://api.github.com/repos/{repo}/commits?per_page=1&sha={branch}"
TIMEOUT = 30.0
RETRIES = 3
BACKOFF_SECONDS = (1.0, 2.0, 4.0)

CURATED_PATHS: tuple[str, ...] = (
    # app slice: routing, layouts, data fetching, server/client components, route handlers
    "docs/01-app/01-getting-started/03-layouts-and-pages.mdx",
    "docs/01-app/01-getting-started/05-server-and-client-components.mdx",
    "docs/01-app/01-getting-started/06-fetching-data.mdx",
    "docs/01-app/03-api-reference/02-components/image.mdx",
    "docs/01-app/03-api-reference/03-file-conventions/layout.mdx",
    "docs/01-app/03-api-reference/03-file-conventions/middleware.mdx",
    "docs/01-app/03-api-reference/03-file-conventions/page.mdx",
    "docs/01-app/03-api-reference/03-file-conventions/proxy.mdx",
    "docs/01-app/03-api-reference/03-file-conventions/route.mdx",
    "docs/01-app/03-api-reference/04-functions/cookies.mdx",
    "docs/01-app/03-api-reference/04-functions/use-router.mdx",
    # pages slice: routing, getStaticProps, getServerSideProps, getInitialProps, next/router, API routes
    "docs/02-pages/03-building-your-application/01-routing/05-custom-app.mdx",
    "docs/02-pages/03-building-your-application/01-routing/06-custom-document.mdx",
    "docs/02-pages/03-building-your-application/03-data-fetching/01-get-static-props.mdx",
    "docs/02-pages/03-building-your-application/03-data-fetching/03-get-server-side-props.mdx",
    "docs/02-pages/04-api-reference/03-functions/get-initial-props.mdx",
    "docs/02-pages/04-api-reference/03-functions/use-router.mdx",
)


class FetchError(RuntimeError):
    """A curated path could not be fetched; the message names the path."""


def cache_path(cache_dir: Path, commit: str, path: str) -> Path:
    return cache_dir / "raw" / commit / path


def resolve_commit(repo: str = REPO, branch: str = BRANCH) -> str:
    """Pin one commit with one ``curl.exe`` call. Called only on a first build or ``--repin``."""
    url = COMMIT_URL.format(repo=repo, branch=branch)
    result = subprocess.run(
        ["curl.exe", "-sS", "--max-time", "40", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise FetchError(f"curl.exe failed resolving {repo}@{branch}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        return payload[0]["sha"]
    except (ValueError, KeyError, IndexError) as exc:
        raise FetchError(f"unexpected commit payload for {repo}@{branch}: {exc}") from exc


def fetch_text(client: httpx.Client, commit: str, path: str, repo: str = REPO) -> bytes:
    """Fetch one raw file, retrying three times with backoff before giving up."""
    url = RAW_URL.format(repo=repo, commit=commit, path=path)
    last: Exception | None = None
    for attempt, delay in enumerate((0.0, *BACKOFF_SECONDS)):
        if delay:
            time.sleep(delay)
        try:
            response = client.get(url)
            if response.status_code == 200:
                return response.content
            last = FetchError(f"{path}: HTTP {response.status_code}")
        except httpx.HTTPError as exc:  # network error, timeout, redirect loop
            last = exc
        log.warning("fetch attempt %d failed for %s: %s", attempt + 1, path, last)
    raise FetchError(f"{path}: giving up after {RETRIES} retries ({last})")


def read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_all(
    cache_dir: Path,
    sources_path: Path,
    *,
    repin: bool = False,
    rebuild: bool = False,
    paths: Iterable[str] = CURATED_PATHS,
    repo: str = REPO,
    branch: str = BRANCH,
    client: httpx.Client | None = None,
) -> dict:
    """Fetch every curated path and return the ``sources.json`` manifest."""
    manifest = read_manifest(sources_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if manifest is None or repin:
        commit = resolve_commit(repo=repo, branch=branch)
        log.info("pinned %s@%s", repo, commit)
    else:
        commit = manifest["source_commit"]
        log.info("reusing pin %s@%s", repo, commit)

    owned = client is None
    http_client = client or httpx.Client(timeout=TIMEOUT, follow_redirects=True)
    files: list[dict] = []
    try:
        for path in sorted(dict.fromkeys(paths)):
            target = cache_path(cache_dir, commit, path)
            if rebuild or not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(fetch_text(http_client, commit, path, repo=repo))
            data = target.read_bytes()
            files.append(
                {
                    "path": path,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    finally:
        if owned:
            http_client.close()

    return {
        "repo": repo,
        "branch": branch,
        "licence": LICENCE,
        "source_commit": commit,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }


def read_cached(cache_dir: Path, manifest: dict, path: str) -> str:
    """Read one fetched markdown file; a missing file is an error, never an empty chunk."""
    commit = manifest["source_commit"]
    target = cache_path(cache_dir, commit, path)
    if not target.exists():
        raise FetchError(f"missing cached file for {path} at {commit}; run with --rebuild")
    return target.read_text(encoding="utf-8")
