import io
import unittest
import urllib.error
from unittest.mock import patch

from cloud_tasks import CloudTaskError, fetch_cloud_tasks, format_cloud_tasks


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class CloudTasksTest(unittest.TestCase):
    def test_format_lists_state_id_time_and_links(self):
        out = format_cloud_tasks([{
            "id": "t1", "state": "completed", "updated_at": "2026-01-01T00:00:00Z",
            "artifacts": [{"type": "pull", "data": {"html_url": "https://github.com/o/r/pull/3"}},
                          {"type": "branch", "data": {"head_ref": "copilot/x"}}],
        }, {"id": "t2", "state": "queued"}])
        self.assertIn("not the local swarm queue", out)
        for text in ("completed", "t1", "2026-01-01T00:00:00Z", "pull/3", "branch copilot/x",
                     "no linked PR/branch"):
            self.assertIn(text, out)

    def test_fetch_parses_tasks(self):
        with patch("urllib.request.urlopen", return_value=FakeResponse(b'{"tasks":[{"id":"a"}]}')):
            self.assertEqual(fetch_cloud_tasks("o/r", "tok"), [{"id": "a"}])

    def test_missing_token_and_repo(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(CloudTaskError, "GITHUB_REPOSITORY"):
                fetch_cloud_tasks()
            with self.assertRaisesRegex(CloudTaskError, "GITHUB_TOKEN"):
                fetch_cloud_tasks("o/r")

    def test_permission_and_network_errors(self):
        denied = urllib.error.HTTPError("u", 403, "Forbidden", {}, io.BytesIO(b"nope"))
        with patch("urllib.request.urlopen", side_effect=denied):
            with self.assertRaisesRegex(CloudTaskError, "denied access"):
                fetch_cloud_tasks("o/r", "tok")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("offline")):
            with self.assertRaisesRegex(CloudTaskError, "offline"):
                fetch_cloud_tasks("o/r", "tok")


if __name__ == "__main__":
    unittest.main()
