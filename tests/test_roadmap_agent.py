import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from roadmap_agent import (
    ROADMAP_PATH,
    RoadmapError,
    _dispatch_item,
    dispatch,
    parse_roadmap,
)


ROADMAP = """\
# Coding roadmap

### `ready-idea` Ready idea
- Status: ready
- Why: Make the workflow reliable.
- Outcome: A validated cloud task is created.
- Acceptance criteria:
  - Create a scoped issue.
  - Open a pull request for review.

### `future-idea` Future idea
- Status: proposed
- Why: Improve visibility.
- Outcome: Operators can inspect task state.
- Acceptance criteria:
  - Display the task state.
"""


class ParseRoadmapTests(unittest.TestCase):
    def test_parses_items_and_acceptance_criteria(self):
        items = parse_roadmap(ROADMAP)

        self.assertEqual([item.item_id for item in items], ["ready-idea", "future-idea"])
        self.assertEqual(items[0].status, "ready")
        self.assertEqual(
            items[0].acceptance_criteria,
            ("Create a scoped issue.", "Open a pull request for review."),
        )

    def test_issue_body_contains_stable_duplicate_marker(self):
        item = parse_roadmap(ROADMAP)[0]

        self.assertIn("<!-- roadmap-item: ready-idea -->", item.issue_body())

    def test_agent_prompt_requires_pull_request_and_forbids_merge(self):
        item = parse_roadmap(ROADMAP)[0]
        prompt = item.agent_prompt("https://github.com/example/project/issues/10")

        self.assertIn("create a pull request for human review", prompt)
        self.assertIn("Do not merge", prompt)
        self.assertIn("only roadmap item `ready-idea`", prompt)

    def test_rejects_duplicate_ids(self):
        duplicate = ROADMAP.replace("future-idea", "ready-idea")

        with self.assertRaisesRegex(RoadmapError, "appears more than once"):
            parse_roadmap(duplicate)

    def test_rejects_invalid_status(self):
        invalid = ROADMAP.replace("Status: proposed", "Status: someday")

        with self.assertRaisesRegex(RoadmapError, "invalid status"):
            parse_roadmap(invalid)

    def test_rejects_missing_acceptance_criteria(self):
        invalid = ROADMAP.replace("  - Display the task state.\n", "")

        with self.assertRaisesRegex(RoadmapError, "acceptance criteria"):
            parse_roadmap(invalid)

    def test_repository_roadmap_does_not_parse_template_as_an_item(self):
        items = parse_roadmap(ROADMAP_PATH.read_text(encoding="utf-8"))

        self.assertEqual(
            [item.item_id for item in items],
            [
                "cloud-task-dashboard",
                "persistent-swarm-job-history",
                "retrieval-provenance",
                "token-budget-enforcement",
            ],
        )

    def test_dispatch_rejects_unready_idea_without_api_credentials(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ROADMAP.md"
            path.write_text(ROADMAP, encoding="utf-8")

            with self.assertRaisesRegex(RoadmapError, "not 'ready'"):
                dispatch(path, "future-idea")

    def test_default_roadmap_is_inert_without_credentials(self):
        output = StringIO()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ROADMAP.md"
            path.write_text(ROADMAP.replace("Status: ready", "Status: proposed"), encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True), redirect_stdout(output):
                result = dispatch(path, None)

        self.assertEqual(result, 0)
        self.assertIn("No roadmap items are marked ready", output.getvalue())


class FakeApi:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, path, data=None):
        self.calls.append((method, path, data))
        if not self.responses:
            raise AssertionError(f"Unexpected API call: {method} {path}")
        return self.responses.pop(0)


class DispatchTests(unittest.TestCase):
    def test_creates_issue_then_launches_scoped_cloud_task(self):
        item = parse_roadmap(ROADMAP)[0]
        issue_api = FakeApi(
            [
                [],
                {
                    "number": 12,
                    "html_url": "https://github.com/example/project/issues/12",
                    "state": "open",
                },
                [],
                None,
                None,
            ]
        )
        agent_api = FakeApi([{"id": "task_123", "html_url": "https://github.com/copilot/tasks/123"}])

        _dispatch_item(issue_api, agent_api, "example", "project", "main", item)

        task_request = agent_api.calls[0][2]
        self.assertEqual(task_request["base_ref"], "main")
        self.assertIs(task_request["create_pull_request"], True)
        self.assertIn("only roadmap item `ready-idea`", task_request["prompt"])
        self.assertIn("<!-- cloud-agent-task: task_123 -->", issue_api.calls[-1][2]["body"])

    def test_existing_task_marker_prevents_duplicate_cloud_launch(self):
        item = parse_roadmap(ROADMAP)[0]
        issue_api = FakeApi(
            [
                [
                    {
                        "number": 12,
                        "html_url": "https://github.com/example/project/issues/12",
                        "body": item.issue_body(),
                        "state": "open",
                    }
                ],
                [{"body": "<!-- cloud-agent-task: task_123 -->"}],
            ]
        )
        agent_api = FakeApi([])

        _dispatch_item(issue_api, agent_api, "example", "project", "main", item)

        self.assertEqual(agent_api.calls, [])

    def test_completed_task_marker_takes_precedence_over_old_pending_marker(self):
        item = parse_roadmap(ROADMAP)[0]
        issue_api = FakeApi(
            [
                [
                    {
                        "number": 12,
                        "html_url": "https://github.com/example/project/issues/12",
                        "body": item.issue_body(),
                        "state": "open",
                    }
                ],
                [
                    {"body": "<!-- cloud-agent-launch:pending -->"},
                    {"body": "<!-- cloud-agent-task: task_123 -->"},
                ],
            ]
        )
        agent_api = FakeApi([])

        _dispatch_item(issue_api, agent_api, "example", "project", "main", item)

        self.assertEqual(agent_api.calls, [])


if __name__ == "__main__":
    unittest.main()
