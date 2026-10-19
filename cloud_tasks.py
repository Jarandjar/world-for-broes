"""Read-only listing of GitHub Copilot cloud tasks (GitHub is the source of truth)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_URL = "https://api.github.com"
API_VERSION = "2026-03-10"


class CloudTaskError(Exception):
    """Raised with an operator-friendly message when cloud tasks cannot be listed."""


def _http_error_message(error: urllib.error.HTTPError) -> str:
    detail = error.read().decode("utf-8", errors="replace").strip()[:200]
    if error.code == 401:
        return "GitHub rejected the token (HTTP 401). Check that GITHUB_TOKEN is valid."
    if error.code in (403, 404):
        return (f"GitHub denied access to cloud tasks (HTTP {error.code}). The token needs "
                "access to the repository and Copilot cloud task permissions, and "
                f"Copilot cloud tasks must be enabled. {detail}").strip()
    return f"GitHub API request failed with HTTP {error.code}: {detail or error.reason}"


def fetch_cloud_tasks(repository: str | None = None, token: str | None = None,
                      base_url: str = API_URL) -> list[dict[str, Any]]:
    repository = repository or os.environ.get("GITHUB_REPOSITORY", "")
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise CloudTaskError("Set GITHUB_REPOSITORY (or --repo) to the owner/repository name.")
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise CloudTaskError("Set GITHUB_TOKEN (or GH_TOKEN) to list GitHub cloud tasks.")
    owner, repo = (urllib.parse.quote(part, safe="") for part in parts)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/agents/repos/{owner}/{repo}/tasks",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"******",
            "X-GitHub-Api-Version": API_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        raise CloudTaskError(_http_error_message(error)) from error
    except urllib.error.URLError as error:
        raise CloudTaskError(f"Could not reach the GitHub API: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise CloudTaskError("GitHub API returned an invalid JSON response.") from error
    tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(tasks, list):
        raise CloudTaskError("GitHub API returned an unexpected cloud task response.")
    return [task for task in tasks if isinstance(task, dict)]


def _links(task: dict[str, Any]) -> list[str]:
    links: list[str] = []
    for artifact in task.get("artifacts") or []:
        data = artifact.get("data") if isinstance(artifact, dict) else None
        if not isinstance(data, dict):
            continue
        kind = artifact.get("type")
        if kind in ("pull", "pull_request"):
            links.append(f"PR {data.get('html_url') or '#' + str(data.get('number') or data.get('id', '?'))}")
        elif kind == "branch":
            links.append(f"branch {data.get('head_ref') or data.get('name') or data.get('ref') or '?'}")
    return links


def format_cloud_tasks(tasks: list[dict[str, Any]]) -> str:
    lines = ["☁️  GitHub Copilot cloud tasks (source of truth: GitHub, not the local swarm queue):", ""]
    if not tasks:
        lines.append("  (no cloud tasks found)")
    for task in tasks:
        links = ", ".join(_links(task)) or "no linked PR/branch"
        lines.append(f"  {str(task.get('state', 'unknown')):12s} | {str(task.get('id', '?')):38s} | "
                     f"updated {task.get('updated_at') or 'unknown'} | {links}")
    return "\n".join(lines)
