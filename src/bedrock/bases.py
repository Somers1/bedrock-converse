from abc import abstractmethod, ABC


class BaseCallbackHandler(ABC):
    @abstractmethod
    def on_converse_start(self, converse):
        """ Callback when conversation starts. """

    @abstractmethod
    def on_converse_end(self, response):
        """ Callback when conversation ends. """
