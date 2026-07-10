"""`dfine predict --json` stdout contract: stdout carries ONLY the JSON result;
every diagnostic (resolver messages, build chatter) goes to stderr — so
`dfine predict --json | jq .` works even on a cold cache. Regression for the
v0.3.0 behavior where the first run corrupted machine-readable output."""

from __future__ import annotations

import json

import pytest

from dfine import cli

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


class StubBox:
    def as_tuple(self):
        return (1.0, 2.0, 3.0, 4.0)


class StubDetection:
    def __init__(self, score: float):
        self.score = score
        self.class_name = "thing"
        self.box = StubBox()

    def as_dict(self):
        return {"score": self.score, "class_id": 0}


class StubDetector:
    """Stands in for dfine.detector.Detector — no GPU, no libdfine."""

    def __init__(self, engine, threshold=0.5):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def detect(self, arr, threshold=None):
        return [StubDetection(0.9), StubDetection(0.7)]


@pytest.fixture()
def env(monkeypatch, tmp_path):
    import dfine.detector

    monkeypatch.setattr(dfine.detector, "Detector", StubDetector)
    monkeypatch.setattr(cli, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_repo_root", lambda: None)
    monkeypatch.setattr(cli, "_gpu_arch", lambda: "89")
    monkeypatch.setattr(cli, "_trt_version", lambda: "10.13")
    img = tmp_path / "img.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img)
    return tmp_path, str(img)


def test_json_stdout_is_pure_even_with_resolver_diagnostics(env, capsys):
    tmp_path, img = env
    # An orphaned cache entry makes the resolver emit its provenance warning —
    # exactly the class of diagnostic that used to land in stdout.
    cli._cache_engine_path("m", "fp16", "aaaaaaaaaaaa", 1, 8).write_bytes(b"e")

    assert cli.main(["predict", "--model", "m", "--image", img, "--json"]) == 0
    out, err = capsys.readouterr()
    dets = json.loads(out)  # must parse: no [dfine] chatter allowed here
    assert [d["score"] for d in dets] == [0.9, 0.7]
    assert "provenance" in err  # ... the diagnostic still reached the user


def test_human_mode_keeps_results_on_stdout(env, capsys):
    tmp_path, img = env
    cli._cache_engine_path("m", "fp16", "aaaaaaaaaaaa", 1, 8).write_bytes(b"e")
    assert cli.main(["predict", "--model", "m", "--image", img]) == 0
    out, _ = capsys.readouterr()
    assert "2 detection(s)" in out
