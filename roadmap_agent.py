#!/usr/bin/env python3
"""Promote ready roadmap items to GitHub issues and Copilot cloud tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROADMAP_PATH = Path(__file__).with_name("ROADMAP.md")
ITEM_HEADING = re.compile(r"(?m)^### `(?P<id>[a-z0-9][a-z0-9-]*)` (?P<title>\S.*)$")
SECTION_HEADING = re.compile(r"(?m)^#{2,3}\s")
VALID_STATUSES = {"proposed", "ready", "in-progress", "done"}
PENDING_MARKER = "<!-- cloud-agent-launch:pending -->"
TASK_MARKER = re.compile(r"<!-- cloud-agent-task:\s*([A-Za-z0-9_-]+)\s*-->")


class RoadmapError(Exception):
    """An expected configuration, roadmap, or GitHub API error."""


@dataclass(frozen=True)
class RoadmapItem:
    item_id: str
    title: str
    status: str
    why: str
    outcome: str
    acceptance_criteria: tuple[str, ...]

    @property
    def marker(self) -> str:
        return f"<!-- roadmap-item: {self.item_id} -->"

    def issue_body(self) -> str:
        criteria = "\n".join(f"- [ ] {criterion}" for criterion in self.acceptance_criteria)
        return (
            f"Roadmap item: `{self.item_id}`\n\n"
            f"## Why\n{self.why}\n\n"
            f"## Expected outcome\n{self.outcome}\n\n"
            f"## Acceptance criteria\n{criteria}\n\n"
            f"{self.marker}"
        )

    def agent_prompt(self, issue_url: str) -> str:
        criteria = "\n".join(f"- {criterion}" for criterion in self.acceptance_criteria)
        return (
            f"Implement only roadmap item `{self.item_id}`: {self.title}.\n\n"
            f"Issue: {issue_url}\n\n"
            f"Why: {self.why}\n\n"
            f"Expected outcome: {self.outcome}\n\n"
            f"Acceptance criteria:\n{criteria}\n\n"
            "Follow the repository instructions and run the most relevant tests. "
            "Keep changes limited to this issue, report any blocked acceptance criteria, "
            "and create a pull request for human review. Do not merge the pull request, "
            "modify unrelated roadmap items, or expose or copy secrets."
        )


def _parse_item(item_id: str, title: str, block: str) -> RoadmapItem:
    fields: dict[str, str] = {}
    criteria: list[str] = []
    reading_criteria = False

    for line in block.splitlines():
        field_match = re.match(r"^- (Status|Why|Outcome|Acceptance criteria):\s*(.*)$", line)
        if field_match:
            name, value = field_match.groups()
            if name in fields:
                raise RoadmapError(f"Roadmap item '{item_id}' repeats the {name} field.")
            fields[name] = value.strip()
            reading_criteria = name == "Acceptance criteria"
            continue

        if reading_criteria and re.match(r"^\s+-\s+\S", line):
            criteria.append(re.sub(r"^\s+-\s+", "", line).strip())
        elif line.startswith("- "):
            reading_criteria = False

    missing = [field for field in ("Status", "Why", "Outcome", "Acceptance criteria") if field not in fields]
    if missing:
        raise RoadmapError(f"Roadmap item '{item_id}' is missing: {', '.join(missing)}.")

    status = fields["Status"].lower()
    if status not in VALID_STATUSES:
        allowed = ", ".join(sorted(VALID_STATUSES))
        raise RoadmapError(f"Roadmap item '{item_id}' has invalid status '{status}'; use {allowed}.")
    if not fields["Why"] or not fields["Outcome"] or not criteria:
        raise RoadmapError(
            f"Roadmap item '{item_id}' needs non-empty Why, Outcome, and acceptance criteria."
        )

    return RoadmapItem(
        item_id=item_id,
        title=title.strip(),
        status=status,
        why=fields["Why"],
        outcome=fields["Outcome"],
        acceptance_criteria=tuple(criteria),
    )


def parse_roadmap(text: str) -> list[RoadmapItem]:
    matches = list(ITEM_HEADING.finditer(text))
    if not matches:
        raise RoadmapError("No roadmap items found; use headings formatted as ### `item-id` Title.")

    items: list[RoadmapItem] = []
    seen_ids: set[str] = set()
    for match in matches:
        item_id = match.group("id")
        if item_id in seen_ids:
            raise RoadmapError(f"Roadmap item ID '{item_id}' appears more than once.")
        seen_ids.add(item_id)
        next_section = SECTION_HEADING.search(text, match.end())
        end = next_section.start() if next_section else len(text)
        items.append(_parse_item(item_id, match.group("title"), text[match.end():end]))
    return items


def _api_error_message(error: urllib.error.HTTPError) -> str:
    detail = error.read().decode("utf-8", errors="replace").strip()
    return f"GitHub API request failed with HTTP {error.code}: {detail or error.reason}"


class GitHubApi:
    def __init__(self, token: str, base_url: str = "https://api.github.com") -> None:
        if not token:
            raise RoadmapError("A GitHub API token is required.")
        self.token = token
        self.base_url = base_url.rstrip("/")

    def request(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        body = json.dumps(data).encode("utf-8") if data is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            raise RoadmapError(_api_error_message(error)) from error
        except urllib.error.URLError as error:
            raise RoadmapError(f"Could not reach the GitHub API: {error.reason}") from error
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise RoadmapError("GitHub API returned an invalid JSON response.") from error


def _repository_parts() -> tuple[str, str]:
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise RoadmapError("Set GITHUB_REPOSITORY to the owner/repository name.")
    return parts[0], parts[1]


def _issues(api: GitHubApi, owner: str, repo: str) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"state": "all", "per_page": 100, "page": page})
        batch = api.request("GET", f"/repos/{owner}/{repo}/issues?{query}")
        if not isinstance(batch, list):
            raise RoadmapError("GitHub returned an unexpected issue-list response.")
        issues.extend(issue for issue in batch if "pull_request" not in issue)
        if len(batch) < 100:
            return issues
        page += 1


def _find_issue(api: GitHubApi, owner: str, repo: str, item: RoadmapItem) -> dict[str, Any] | None:
    for issue in _issues(api, owner, repo):
        if item.marker in (issue.get("body") or ""):
            return issue
    return None


def _issue_comments(api: GitHubApi, owner: str, repo: str, issue_number: int) -> list[dict[str, Any]]:
    comments: list[dict[str, Any]] = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        batch = api.request(
            "GET",
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments?{query}",
        )
        if not isinstance(batch, list):
            raise RoadmapError("GitHub returned an unexpected issue-comments response.")
        comments.extend(batch)
        if len(batch) < 100:
            return comments
        page += 1


def _post_comment(api: GitHubApi, owner: str, repo: str, issue_number: int, body: str) -> None:
    api.request(
        "POST",
        f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
        {"body": body},
    )


def _dispatch_item(
    github_api: GitHubApi,
    agent_api: GitHubApi,
    owner: str,
    repo: str,
    base_ref: str,
    item: RoadmapItem,
) -> None:
    issue = _find_issue(github_api, owner, repo, item)
    created = issue is None
    if issue is None:
        issue = github_api.request(
            "POST",
            f"/repos/{owner}/{repo}/issues",
            {
                "title": f"[Roadmap] {item.title}",
                "body": item.issue_body(),
            },
        )
    if not isinstance(issue, dict):
        raise RoadmapError(f"GitHub returned an invalid issue response for '{item.item_id}'.")

    issue_number = issue.get("number")
    issue_url = issue.get("html_url")
    if not isinstance(issue_number, int) or not isinstance(issue_url, str) or not issue_url:
        raise RoadmapError(f"GitHub returned an issue without a number or URL for '{item.item_id}'.")
    issue_state = issue.get("state")
    if issue_state not in {"open", "closed"}:
        raise RoadmapError(f"GitHub returned an issue with an invalid state for '{item.item_id}'.")
    if issue_state == "closed":
        print(f"Skipping {item.item_id}: linked issue #{issue_number} is closed.")
        return
    if created:
        print(f"Created issue #{issue_number} for {item.item_id}.")
    comments = _issue_comments(github_api, owner, repo, issue_number)
    pending_launch = False
    for comment in comments:
        comment_body = comment.get("body") or ""
        task_match = TASK_MARKER.search(comment_body)
        if task_match:
            print(f"Skipping {item.item_id}: already linked to cloud task {task_match.group(1)}.")
            return
        if PENDING_MARKER in comment_body:
            pending_launch = True
    if pending_launch:
        raise RoadmapError(
            f"Issue #{issue_number} has an unfinished launch marker. Check cloud task history "
            "before removing the marker and retrying."
        )

    _post_comment(
        github_api,
        owner,
        repo,
        issue_number,
        f"{PENDING_MARKER}\nLaunching the Copilot cloud task. If the workflow fails, inspect "
        "cloud task history before clearing this marker to avoid duplicate runs.",
    )
    task = agent_api.request(
        "POST",
        f"/agents/repos/{owner}/{repo}/tasks",
        {
            "prompt": item.agent_prompt(issue_url),
            "base_ref": base_ref,
            "create_pull_request": True,
        },
    )
    task_id = task.get("id") if isinstance(task, dict) else None
    if not isinstance(task_id, str) or not task_id:
        raise RoadmapError(f"Copilot cloud agent returned no task ID for roadmap item '{item.item_id}'.")
    task_url = task.get("html_url")
    message = f"<!-- cloud-agent-task: {task_id} -->\nStarted Copilot cloud task `{task_id}`."
    if task_url:
        message += f" [View task]({task_url})."
    _post_comment(github_api, owner, repo, issue_number, message)
    print(f"Started cloud task {task_id} for issue #{issue_number} ({item.item_id}).")


def dispatch(roadmap_path: Path, item_id: str | None) -> int:
    items = parse_roadmap(roadmap_path.read_text(encoding="utf-8"))
    if item_id:
        matches = [item for item in items if item.item_id == item_id]
        if not matches:
            raise RoadmapError(f"Roadmap item '{item_id}' was not found.")
        if matches[0].status != "ready":
            raise RoadmapError(f"Roadmap item '{item_id}' is '{matches[0].status}', not 'ready'.")
        selected = matches
    else:
        selected = [item for item in items if item.status == "ready"]

    if not selected:
        print("No roadmap items are marked ready; nothing to dispatch.")
        return 0

    agent_token = os.environ.get("COPILOT_AGENT_TOKEN", "")
    if not agent_token:
        raise RoadmapError(
            "COPILOT_AGENT_TOKEN is missing. Add a user-to-server token with Agent tasks "
            "read/write permission before dispatching."
        )
    github_token = os.environ.get("GITHUB_TOKEN") or agent_token
    owner, repo = _repository_parts()
    base_ref = os.environ.get("BASE_REF", "main")
    github_api = GitHubApi(github_token)
    agent_api = GitHubApi(agent_token)

    for item in selected:
        _dispatch_item(github_api, agent_api, owner, repo, base_ref, item)
    return 0


def list_tasks(state: str | None) -> int:
    token = os.environ.get("COPILOT_AGENT_TOKEN", "")
    owner, repo = _repository_parts()
    api = GitHubApi(token)
    tasks: list[dict[str, Any]] = []
    page = 1
    while True:
        params = {"per_page": "100", "page": str(page)}
        if state:
            params["state"] = state
        query = urllib.parse.urlencode(params)
        response = api.request("GET", f"/agents/repos/{owner}/{repo}/tasks?{query}")
        batch = response.get("tasks") if isinstance(response, dict) else None
        if not isinstance(batch, list):
            raise RoadmapError("GitHub returned an unexpected cloud-task list response.")
        tasks.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    if not tasks:
        print("No Copilot cloud tasks found.")
        return 0

    for task in tasks:
        artifacts = task.get("artifacts") or []
        artifact = next((entry for entry in artifacts if entry.get("type") == "pull"), None)
        if artifact:
            pull_number = (artifact.get("data") or {}).get("id")
            artifact_text = f"PR #{pull_number}" if pull_number else "PR"
        else:
            branch = next((entry for entry in artifacts if entry.get("type") == "branch"), None)
            branch_data = (branch or {}).get("data") or {}
            artifact_text = branch_data.get("head_ref", "—")
        print(
            f"{task.get('state', 'unknown'):18} "
            f"{task.get('id', 'unknown'):24} "
            f"{artifact_text:24} "
            f"{task.get('updated_at', '')} "
            f"{task.get('html_url') or task.get('url') or ''}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roadmap", type=Path, default=ROADMAP_PATH, help="Roadmap markdown path")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Validate roadmap item format and statuses")
    dispatch_parser = commands.add_parser("dispatch", help="Create issues and start ready cloud tasks")
    dispatch_parser.add_argument("--item-id", help="Dispatch only this ready roadmap item")
    tasks_parser = commands.add_parser("tasks", help="List Copilot cloud tasks for this repository")
    tasks_parser.add_argument("--state", help="Filter by cloud-task state")
    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            items = parse_roadmap(args.roadmap.read_text(encoding="utf-8"))
            for item in items:
                print(f"{item.item_id}: {item.status} — {item.title}")
            return 0
        if args.command == "dispatch":
            return dispatch(args.roadmap, args.item_id)
        if args.command == "tasks":
            return list_tasks(args.state)
    except (OSError, RoadmapError) as error:
        print(f"roadmap_agent: {error}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
