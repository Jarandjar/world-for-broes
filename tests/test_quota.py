import io
import unittest
import urllib.error

import roadmap_agent
from roadmap_agent import QuotaExhausted, RoadmapError, _api_error, _launch_budget


def _http_error(code, body):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(body.encode()))


class QuotaTests(unittest.TestCase):
    def test_rate_limit_codes_are_quota(self):
        self.assertIsInstance(_api_error(_http_error(429, "slow down")), QuotaExhausted)
        self.assertIsInstance(_api_error(_http_error(403, "Premium request quota exceeded")), QuotaExhausted)

    def test_other_errors_are_plain(self):
        error = _api_error(_http_error(404, "not found"))
        self.assertNotIsInstance(error, QuotaExhausted)
        self.assertIsInstance(error, RoadmapError)

    def test_launch_budget_respects_caps(self):
        class Api:
            def request(self, method, path, data=None):
                return {"tasks": [{}]} if "queued" in path else {"tasks": []}

        self.assertEqual(_launch_budget(Api(), "o", "r"), 2)
        with unittest.mock.patch.dict("os.environ", {"MAX_ACTIVE_TASKS": "1"}):
            self.assertEqual(_launch_budget(Api(), "o", "r"), 0)


if __name__ == "__main__":
    unittest.main()
