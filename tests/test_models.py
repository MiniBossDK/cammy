import hashlib
import zipfile

import pytest

from cammy.models import ensure_from_zip, sha256_of


def make_zip(tmp_path, files):
    path = tmp_path / "pkg.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_extracts_only_the_named_files_under_their_plain_names(tmp_path):
    archive = make_zip(tmp_path, {"models/a.bin": b"AAAA", "models/b.param": b"BBBB", "evil": b"#!/bin/sh", "models/other.bin": b"zz"})
    out = tmp_path / "out"
    paths = ensure_from_zip(archive.as_uri(), {"models/a.bin": digest(b"AAAA"), "models/b.param": digest(b"BBBB")}, out)
    assert sorted(p.name for p in out.iterdir()) == ["a.bin", "b.param"]       # nothing else, no archive left behind
    assert paths["a.bin"].read_bytes() == b"AAAA"


def test_a_file_with_the_wrong_hash_is_refused_and_not_kept(tmp_path):
    archive = make_zip(tmp_path, {"models/a.bin": b"tampered"})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_from_zip(archive.as_uri(), {"models/a.bin": digest(b"original")}, out)
    assert list(out.iterdir()) == []


def test_existing_correct_files_are_not_downloaded_again(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.bin").write_bytes(b"AAAA")
    paths = ensure_from_zip("file:///does/not/exist.zip", {"models/a.bin": digest(b"AAAA")}, out)    # would fail if it tried
    assert paths["a.bin"].exists()


def test_a_corrupted_existing_file_is_replaced(tmp_path):
    archive = make_zip(tmp_path, {"models/a.bin": b"AAAA"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.bin").write_bytes(b"damaged")
    ensure_from_zip(archive.as_uri(), {"models/a.bin": digest(b"AAAA")}, out)
    assert (out / "a.bin").read_bytes() == b"AAAA"


def test_sha256_of_matches_hashlib(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"hello" * 1000)
    assert sha256_of(f) == digest(b"hello" * 1000)
