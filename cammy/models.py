"""Model file download helpers."""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def ensure_model(name, url):
    """Return models/<name>, downloading it from `url` on first use."""
    path = MODELS_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_from_zip(url, members, directory=None):
    """Make sure each wanted file exists in `directory`, taking it from the zip at `url`.

    `members` maps a path inside the archive to the SHA-256 the file must have.
    Files already present with the right hash are left alone. Otherwise the
    archive is downloaded once, only the named members are read from it
    (nothing else is extracted, nothing is executed), each is checked against
    its hash, and it is written under its plain file name. A file that does not
    match raises ValueError and nothing is kept. Returns {file name: Path}.
    """
    directory = Path(directory) if directory else MODELS_DIR
    wanted = {Path(member).name: (member, digest) for member, digest in members.items()}
    paths = {name: directory / name for name in wanted}
    if all(p.exists() and sha256_of(p) == wanted[n][1] for n, p in paths.items()):
        return paths

    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "download.zip.part"
    try:
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as z:
            for name, (member, digest) in wanted.items():
                data = z.read(member)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError(f"{member} from {url} does not match the expected SHA-256; refusing it")
                tmp = paths[name].with_suffix(paths[name].suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(paths[name])
    finally:
        archive.unlink(missing_ok=True)
    return paths
