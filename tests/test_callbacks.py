import json
import os
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bedrock.bases import BaseCallbackHandler
from bedrock.callbacks import PrintCallback, EMFMetricsCallback
from bedrock.converse import ConverseResponse, TokenUsage, ConverseMetrics, ConverseOutput, Message, MessageContent


def _make_mock_response():
    resp = MagicMock(spec=ConverseResponse)
    resp.model_id = "anthropic.claude-3-5-sonnet"
    resp.metrics = MagicMock()
    resp.metrics.latency_ms = 500
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    resp.usage.total_tokens = 150
    resp.cost = MagicMock()
    resp.cost.input_cost = 0.0003
    resp.cost.output_cost = 0.00075
    resp.cost.total_cost = 0.00105
    return resp


class TestBaseCallbackHandler(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            BaseCallbackHandler()

    def test_subclass_must_implement(self):
        class Incomplete(BaseCallbackHandler):
            def on_converse_start(self, converse):
                pass
        with self.assertRaises(TypeError):
            Incomplete()

    def test_complete_subclass(self):
        class Complete(BaseCallbackHandler):
            def on_converse_start(self, converse):
                pass
            def on_converse_end(self, response):
                pass
        cb = Complete()
        self.assertIsInstance(cb, BaseCallbackHandler)


class TestPrintCallback(unittest.TestCase):
    def test_on_converse_start(self):
        cb = PrintCallback()
        cb.on_converse_start(MagicMock())  # Should not raise

    def test_on_converse_end(self):
        cb = PrintCallback()
        resp = _make_mock_response()
        with patch('sys.stdout', new_callable=StringIO) as mock_out:
            cb.on_converse_end(resp)
            output = mock_out.getvalue()
        self.assertIn("500", output)  # latency


class TestEMFMetricsCallback(unittest.TestCase):
    def test_on_converse_start(self):
        cb = EMFMetricsCallback()
        cb.on_converse_start(MagicMock())  # Should not raise

    def test_on_converse_end_json_output(self):
        cb = EMFMetricsCallback()
        resp = _make_mock_response()
        with patch('sys.stdout', new_callable=StringIO) as mock_out:
            cb.on_converse_end(resp)
            output = mock_out.getvalue()
        data = json.loads(output)
        self.assertIn("_aws", data)
        self.assertIn("CloudWatchMetrics", data["_aws"])
        self.assertEqual(data["PromptTokens"], 100)
        self.assertEqual(data["CompletionTokens"], 50)
        self.assertEqual(data["TotalCost"], 0.00105)

    def test_emf_namespace(self):
        cb = EMFMetricsCallback()
        resp = _make_mock_response()
        with patch('sys.stdout', new_callable=StringIO) as mock_out:
            cb.on_converse_end(resp)
            data = json.loads(mock_out.getvalue())
        self.assertEqual(data["_aws"]["CloudWatchMetrics"][0]["Namespace"], "Talos/TokenUsage")


if __name__ == '__main__':
    unittest.main()
