from abc import abstractmethod, ABC


class BaseCallbackHandler(ABC):
    @abstractmethod
    def on_converse_start(self, converse):
        """ Callback when conversation starts. """

    @abstractmethod
    def on_converse_end(self, response):
        """ Callback when conversation ends. """

    def on_tool_start(self, tool_name: str, tool_input: dict, tool_use_id: str):
        """ Callback before a tool is executed. """

    def on_tool_end(self, tool_name: str, tool_input: dict, tool_use_id: str, result, status: str, duration: float):
        """ Callback after a tool finishes. """

    def on_run_start(self, agent):
        """ Callback at the start of agent.run(). """

    def on_run_end(self, agent, result):
        """ Callback at the end of agent.run(). """

    def on_converse_error(self, converse, error: Exception):
        """ Callback when conversation call fails before a response is returned. """
