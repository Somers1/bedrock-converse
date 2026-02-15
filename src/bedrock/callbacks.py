from .bases import BaseCallbackHandler
from .converse import ConverseResponse, Converse


class PrintCallback(BaseCallbackHandler):
    def on_converse_start(self, converse: Converse):
        pass

    def on_converse_end(self, response: ConverseResponse):
        print(f"{response.model_id} - Latency {response.metrics.latency_ms}ms\n{response.usage}\n{response.cost}")
