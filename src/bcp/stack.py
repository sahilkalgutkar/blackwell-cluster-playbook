"""Keep `matrix/stack.yaml` current: compare pinned versions with upstream releases."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

UNSTABLE = re.compile(r"(?i)(rc\d*|alpha|beta|dev|preview|nightly)")


def version_key(tag: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", tag))


def is_stable(tag: str) -> bool:
    return not UNSTABLE.search(tag)


class Http(Protocol):
    def get_json(self, url: str, headers: dict[str, str] | None = None): ...


class UrllibHttp:
    def __init__(self, token: str | None = None, timeout: int = 20) -> None:
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.timeout = timeout

    def get_json(self, url: str, headers: dict[str, str] | None = None):
        headers = {"User-Agent": "bcp-stack-check", "Accept": "application/json", **(headers or {})}
        if self.token and urllib.parse.urlparse(url).hostname == "api.github.com":
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return json.loads(resp.read())


def latest_github_release(
    http: Http, repo: str, pattern: str | None = None, order: str = "version"
) -> str | None:
    """Newest stable release. `pattern` keeps only matching tags (some repos publish several
    projects, e.g. NVIDIA/nccl also releases nccl4py). `order: date` picks the most recently
    published release instead of the highest number, for repos whose numbering was reset."""
    releases = http.get_json(f"https://api.github.com/repos/{repo}/releases?per_page=30")
    candidates = [
        r
        for r in releases
        if not r.get("draft")
        and not r.get("prerelease")
        and is_stable(r["tag_name"])
        and (pattern is None or re.search(pattern, r["tag_name"]))
    ]
    if not candidates:
        return None
    if order == "date":
        return max(candidates, key=lambda r: r.get("published_at") or "")["tag_name"]
    return max((r["tag_name"] for r in candidates), key=version_key)


def latest_github_tag(http: Http, repo: str, pattern: str | None = None) -> str | None:
    tags = [
        t["name"] for t in http.get_json(f"https://api.github.com/repos/{repo}/tags?per_page=100")
    ]
    if pattern:
        tags = [t for t in tags if re.search(pattern, t)]
    tags = [t for t in tags if is_stable(t)]
    return max(tags, key=version_key) if tags else None


def latest_dockerhub_tag(http: Http, repo: str, pattern: str, name_filter: str = "") -> str | None:
    query = urllib.parse.urlencode(
        {"page_size": 100, "ordering": "last_updated", "name": name_filter}
    )
    data = http.get_json(f"https://hub.docker.com/v2/repositories/{repo}/tags?{query}")
    versions = []
    for tag in data.get("results", []):
        m = re.fullmatch(pattern, tag["name"])
        if m:
            version = m.group(1) if m.groups() else tag["name"]
            if is_stable(version):
                versions.append(version)
    return max(versions, key=version_key) if versions else None


def latest_ngc_tag(http: Http, repo: str, pattern: str) -> str | None:
    """Newest matching tag of an NGC (nvcr.io) image. Public images need only an anonymous
    pull token, which the registry hands out without an account."""
    auth = http.get_json(f"https://nvcr.io/proxy_auth?scope=repository:{repo}:pull")
    token = (auth or {}).get("token")
    data = http.get_json(
        f"https://nvcr.io/v2/{repo}/tags/list?n=5000",
        headers={"Authorization": f"Bearer {token}"} if token else None,
    )
    versions = []
    for tag in data.get("tags") or []:
        m = re.fullmatch(pattern, tag)
        if m:
            version = m.group(1) if m.groups() else tag
            if is_stable(version):
                versions.append(version)
    return max(versions, key=version_key) if versions else None


@dataclass
class Status:
    name: str
    source: str
    pinned: str | None
    latest: str | None
    state: str  # current | update | manual | error
    detail: str = ""


def check(components: list[dict], http: Http) -> list[Status]:
    statuses = []
    for comp in components:
        name, source = comp["name"], comp.get("source", "manual")
        pinned = None if comp.get("pinned") is None else str(comp["pinned"])
        if source == "manual":
            statuses.append(Status(name, source, pinned, None, "manual", comp.get("notes", "")))
            continue
        try:
            if source == "github-release":
                latest = latest_github_release(
                    http, comp["repo"], comp.get("tag_pattern"), comp.get("order", "version")
                )
            elif source == "github-tag":
                latest = latest_github_tag(http, comp["repo"], comp.get("tag_pattern"))
            elif source == "dockerhub":
                latest = latest_dockerhub_tag(
                    http, comp["repo"], comp["tag_pattern"], comp.get("tag_filter", "")
                )
            elif source == "ngc":
                latest = latest_ngc_tag(http, comp["repo"], comp["tag_pattern"])
            else:
                raise ValueError(f"unknown source {source!r}")
        except Exception as exc:  # network errors, rate limits, schema drift
            statuses.append(Status(name, source, pinned, None, "error", str(exc)[:200]))
            continue
        newer = (
            latest != pinned
            if comp.get("order") == "date"
            else version_key(latest or "") > version_key(pinned or "")
        )
        if latest is None:
            statuses.append(Status(name, source, pinned, None, "error", "no stable version found"))
        elif pinned is None or newer:
            statuses.append(Status(name, source, pinned, latest, "update"))
        else:
            statuses.append(Status(name, source, pinned, latest, "current"))
    return statuses


def load(path: Path) -> list[dict]:
    return (yaml.safe_load(path.read_text()) or {}).get("components", [])


def rewrite_pinned(text: str, updates: dict[str, str]) -> str:
    """Change `pinned:` values in place so comments and layout in stack.yaml survive."""
    out, current = [], None
    for line in text.splitlines(keepends=True):
        name = re.match(r"^\s*-\s+name:\s*['\"]?([\w.\-]+)", line)
        if name:
            current = name.group(1)
        pinned = re.match(r"^(\s*pinned:\s*)(.*?)(\s*#.*)?(\r?\n)?$", line)
        if pinned and current in updates:
            prefix, comment, newline = pinned.group(1), pinned.group(3), pinned.group(4)
            line = f'{prefix}"{updates[current]}"{comment or ""}{newline or ""}'
        out.append(line)
    return "".join(out)


def render_table(statuses: list[Status]) -> str:
    lines = [f"{'COMPONENT':<22} {'PINNED':<16} {'LATEST':<16} STATUS"]
    for s in statuses:
        lines.append(
            f"{s.name:<22} {s.pinned or '–':<16} {s.latest or '–':<16} {s.state}"
            + (f"  ({s.detail})" if s.detail and s.state == "error" else "")
        )
    return "\n".join(lines)


def render_markdown(statuses: list[Status]) -> str:
    updates = [s for s in statuses if s.state == "update"]
    lines = ["## Upstream version bumps", ""]
    if not updates:
        lines.append("Everything tracked is current.")
    else:
        lines += ["| Component | Pinned | Latest |", "|---|---|---|"]
        lines += [f"| {s.name} | {s.pinned or '–'} | **{s.latest}** |" for s in updates]
        lines += [
            "",
            "Before merging: rebuild the containers that use these versions and rerun "
            "`make demo`. Record any real-hardware rerun in `results/`.",
        ]
    errors = [s for s in statuses if s.state == "error"]
    if errors:
        lines += ["", "Could not check: " + ", ".join(f"{s.name} ({s.detail})" for s in errors)]
    return "\n".join(lines) + "\n"
