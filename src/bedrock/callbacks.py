import json
import os
import time

from .bases import BaseCallbackHandler
from .converse import ConverseResponse, Converse


class PrintCallback(BaseCallbackHandler):
    def on_converse_start(self, converse: Converse):
        pass

    def on_converse_end(self, response: ConverseResponse):
        print(f"{response.model_id} - Latency {response.metrics.latency_ms}ms\n{response.usage}\n{response.cost}")


class EMFMetricsCallback(BaseCallbackHandler):
    def on_converse_start(self, converse):
        pass

    def on_converse_end(self, response: ConverseResponse):
        print(json.dumps({
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [{
                    "Namespace": 'Talos/TokenUsage',
                    "Dimensions": [["Service", "Function", "DocumentClass"]],
                    "Metrics": [
                        {"Name": "PromptTokens", "Unit": "Count"},
                        {"Name": "CompletionTokens", "Unit": "Count"},
                        {"Name": "TotalTokens", "Unit": "Count"},
                        {"Name": "Duration", "Unit": "Seconds"},
                        {"Name": "PromptCost", "Unit": "USD"},
                        {"Name": "CompletionCost", "Unit": "USD"},
                        {"Name": "TotalCost", "Unit": "USD"},
                    ]
                }]
            },
            "Service": 'talos',
            "Function": os.getenv('AWS_LAMBDA_FUNCTION_NAME', 'Unknown Lambda Function'),
            "PromptTokens": response.usage.input_tokens,
            "CompletionTokens": response.usage.output_tokens,
            "TotalTokens": response.usage.total_tokens,
            "PromptCost": response.cost.input_cost,
            "CompletionCost": response.cost.output_cost,
            "TotalCost": response.cost.total_cost,
            "ModelId": response.model_id,
            "DocumentClass": os.environ.get('TalosDocumentClass', 'N/A'),
            "Duration": response.metrics.latency_ms
        }))
