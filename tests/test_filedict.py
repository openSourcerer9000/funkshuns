import os
import json
import pytest
from pathlib import Path
from funkshuns.funkshuns import *
mc = Path(os.path.dirname(os.path.abspath(__file__)))/'mockdata'

class Testfiledict:

    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmpdir):
        self.jsonfile = Path(tmpdir) / "filedict.json"
        self.filedict = filedict(self.jsonfile)
        # Teardown: reset the filedict after each test
        yield
        if self.jsonfile.exists():
            self.jsonfile.unlink()

    def test_read_write(self):
        assert self.filedict.data == {}
        self.filedict["test"] = "data"
        assert self.filedict.data == {"test": "data"}
        with open(self.jsonfile) as f:
            assert json.load(f) == {"test": "data"}

    def test_delitem(self):
        self.filedict["to_be_deleted"] = "data"
        del self.filedict["to_be_deleted"]
        assert "to_be_deleted" not in self.filedict
        with open(self.jsonfile) as f:
            assert json.load(f) == {}

    def test_update(self):
        self.filedict.update({"key": "value"})
        assert self.filedict["key"] == "value"
        with open(self.jsonfile) as f:
            assert json.load(f) == {"key": "value"}

    def test_pop(self):
        self.filedict["to_be_popped"] = "data"
        value = self.filedict.pop("to_be_popped")
        assert value == "data"
        assert "to_be_popped" not in self.filedict
        with open(self.jsonfile) as f:
            assert json.load(f) == {}

    def test_popitem(self):
        self.filedict["to_be_popped"] = "data"
        key, value = self.filedict.popitem()
        assert key == "to_be_popped"
        assert value == "data"
        assert "to_be_popped" not in self.filedict
        with open(self.jsonfile) as f:
            assert json.load(f) == {}

    def test_clear(self):
        self.filedict["test"] = "data"
        self.filedict.clear()
        assert len(self.filedict) == 0
        with open(self.jsonfile) as f:
            assert json.load(f) == {}

    def test_persistence(self):
        fd1 = filedict(self.jsonfile)
        fd1["test"] = "data"
        fd2 = filedict(self.jsonfile)
        assert "test" in fd2
        assert fd2["test"] == "data"
        with open(self.jsonfile) as f:
            assert json.load(f) == {"test": "data"}