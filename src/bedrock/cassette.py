import json
import os
from pathlib import Path


class Cassette:
    MODE = 'BEDROCK_CASSETTE_MODE'
    DIRECTORY = 'BEDROCK_CASSETTE_DIR'
    RECORD = 'record'

    def __init__(self, caller, mode, directory):
        self.caller = caller
        self.mode = mode
        self.directory = Path(directory)

    @classmethod
    def wrap(cls, caller):
        if mode := os.environ.get(cls.MODE):
            return cls(caller, mode, os.environ[cls.DIRECTORY])
        return caller.bedrock_client

    @property
    def path(self):
        return self.directory / self.caller.cassette_scope / f'{self.caller.cassette_key}.json'

    def converse(self, **payload):
        return self.record(payload) if self.mode == self.RECORD else json.loads(self.path.read_text())['response']

    def record(self, payload):
        response = self.caller.bedrock_client.converse(**payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({'model_id': self.caller.model_id, 'response': response}, indent=2, default=str))
        return response
