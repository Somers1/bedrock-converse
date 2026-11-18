from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from queue import SimpleQueue
from threading import Event

END = object()


class Arrivals:
    def __init__(self, source):
        self.source = source
        self.arrived = SimpleQueue()
        self.left = Event()
        self.open = True
        self.pool = ThreadPoolExecutor(max_workers=1)

    def __iter__(self):
        reading = self.pool.submit(self.read)
        try:
            while self.open:
                yield self.waiting()
            reading.result()
        finally:
            self.left.set()
            self.pool.shutdown(wait=False)

    def read(self):
        try:
            with self.source:
                for item in self.source:
                    if self.left.is_set():
                        return
                    self.arrived.put(item)
        finally:
            self.arrived.put(END)

    def waiting(self):
        items = [self.arrived.get()]
        while not self.arrived.empty():
            items.append(self.arrived.get())
        self.open = items[-1] is not END
        return items if self.open else items[:-1]


def joined(events):
    for (kind, _index), run in groupby(events, key=lambda event: (event["type"], event.get("index"))):
        run = list(run)
        if kind == "text_delta":
            yield {**run[0], "text": "".join(event["text"] for event in run)}
        elif kind == "reasoning_delta":
            yield {**run[0], "reasoning": {"text": "".join(event["reasoning"]["text"] for event in run)}}
        elif kind == "tool_use_input_delta":
            yield {**run[0], "partial_json": "".join(event["partial_json"] for event in run)}
        else:
            yield from run
