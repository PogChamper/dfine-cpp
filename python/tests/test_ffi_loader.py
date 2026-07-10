"""Loader diagnostics: an explicit DFINE_LIBRARY is never silently substituted,
and when nothing loads, the error carries both the searched paths and the
system loader's own words (dlerror) — "not found" and "found but missing a
dependency" need different fixes."""

from __future__ import annotations

import pytest

from dfine import _ffi


def test_explicit_library_path_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("DFINE_LIBRARY", str(tmp_path / "no_such_libdfine.so"))
    with pytest.raises(RuntimeError, match="missing file"):
        _ffi._load_library()


def test_load_failure_reports_paths_and_dlerror(monkeypatch, tmp_path):
    monkeypatch.delenv("DFINE_LIBRARY", raising=False)
    # No candidate exists and the loader name is bogus, so every stage fails.
    monkeypatch.setattr(_ffi, "_LIBNAMES", ("libdfine_test_nonexistent.so",))
    monkeypatch.setattr(_ffi, "_candidate_paths",
                        lambda: [tmp_path / "libdfine_test_nonexistent.so"])
    with pytest.raises(RuntimeError) as e:
        _ffi._load_library()
    msg = str(e.value)
    assert "Searched:" in msg and str(tmp_path) in msg
    assert "System loader said:" in msg and "libdfine_test_nonexistent.so:" in msg
