import json


class Storage:

    def __init__(self, path: str, encoding: str = 'utf-8'):
        self.path = path
        self.encoding = encoding
        self._data = None

    def __enter__(self):
        return self.read()

    def __exit__(self, *args, **kwargs):
        self._data = None

    def read(self) -> 'Storage':
        with open(self.path, encoding=self.encoding) as json_file:
            self._data = json.load(json_file)
        return self

    def write(self, data: dict = None) -> 'Storage':
        with open(self.path, 'w', encoding=self.encoding) as json_file:
            json.dump(data if data else self._data, json_file)
        return self

    @property
    def data(self) -> dict:
        return self._data

