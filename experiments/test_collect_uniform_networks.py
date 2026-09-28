"""Offline checks for bounded uniform-road collection behavior."""

import unittest
from unittest.mock import patch

import requests

import collect_uniform_networks as collector


class CollectionRetryTests(unittest.TestCase):
    def setUp(self):
        self.item = {"city": "Test City, Test Country", "overpass_url": "https://example.org/api"}

    def test_transient_failures_retry_then_succeed(self):
        downloaded = {"city": self.item["city"], "status": "downloaded", "nodes": 2, "edges": 1}
        with (patch.object(collector, "fetch", side_effect=[
                requests.ConnectionError("reset"), requests.Timeout("timeout"), downloaded]) as fetch,
              patch.object(collector, "record_attempt") as record,
              patch.object(collector.time, "sleep") as sleep,
              patch.object(collector.time, "monotonic", return_value=0)):
            result, _, transient = collector.collect_item(self.item, False, 0, None)
        self.assertEqual(result, downloaded)
        self.assertFalse(transient)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(record.call_count, 3)
        self.assertIn(30, [call.args[0] for call in sleep.call_args_list])
        self.assertIn(60, [call.args[0] for call in sleep.call_args_list])

    def test_nontransient_failure_does_not_retry(self):
        with (patch.object(collector, "fetch", side_effect=ValueError("no major roads")) as fetch,
              patch.object(collector, "record_attempt"),
              patch.object(collector.time, "sleep") as sleep):
            result, _, transient = collector.collect_item(self.item, False, 0, None)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(transient)
        self.assertEqual(fetch.call_count, 1)
        sleep.assert_not_called()

    def test_http_status_classification(self):
        for status, expected in ((429, True), (503, True), (400, False)):
            with self.subTest(status=status):
                response = requests.Response()
                response.status_code = status
                self.assertEqual(collector.transient_failure(
                    requests.HTTPError(response=response)), expected)

    def test_socket_permission_error_is_permanent(self):
        error = requests.ConnectionError("[WinError 10013] socket access forbidden")
        self.assertFalse(collector.transient_failure(error))


if __name__ == "__main__":
    unittest.main()
