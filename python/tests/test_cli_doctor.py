"""`dfine doctor` is the one command a bug report needs: it must never crash
(every probe degrades to a printed fact), and its exit code answers the only
binary question — does libdfine load here?"""

from __future__ import annotations

import pytest

from dfine import _ffi, cli


def run_doctor(capsys):
    rc = cli.main(["doctor"])
    return rc, capsys.readouterr().out


def test_doctor_reports_and_fails_when_library_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(_ffi, "_candidate_paths",
                        lambda: [tmp_path / "libdfine_missing.so"])
    monkeypatch.setattr(_ffi, "get_lib",
                        lambda: (_ for _ in ()).throw(RuntimeError("no library here")))
    monkeypatch.setattr(cli, "_cache_dir", lambda: tmp_path)
    rc, out = run_doctor(capsys)
    assert rc == 1
    assert "libdfine      : FAILED" in out and "no library here" in out
    assert "- " + str(tmp_path / "libdfine_missing.so") in out
    assert "engine cache" in out and "(0 engine(s))" in out


def test_doctor_passes_when_library_loads(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(_ffi, "_candidate_paths", lambda: [])
    monkeypatch.setattr(_ffi, "get_lib", lambda: object())
    monkeypatch.setattr(cli, "_cache_dir", lambda: tmp_path)
    (tmp_path / "a.engine").write_bytes(b"e")
    rc, out = run_doctor(capsys)
    assert rc == 0
    assert "loads OK" in out and "(1 engine(s))" in out


def test_doctor_never_needs_a_gpu(monkeypatch, tmp_path, capsys):
    # nvidia-smi absent must degrade to a printed fact, not an exception.
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(_ffi, "get_lib", lambda: object())
    monkeypatch.setattr(cli, "_cache_dir", lambda: tmp_path)
    rc, out = run_doctor(capsys)
    assert rc == 0
    assert "nvidia-smi unavailable" in out or "gpu           :" in out


def test_doctor_registered_in_parser():
    with pytest.raises(SystemExit) as e:
        cli.build_parser().parse_args(["doctor", "--bogus"])
    assert e.value.code == 2
