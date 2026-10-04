"""再現性のための出所記録（manifest）を集める共通処理.

データセットを作った「入力・コード・環境・パラメータ・出力」を 1 つの JSON にまとめる。
manifest には書籍本文を含めない（ハッシュとメタデータのみ）。
"""

import datetime
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]   # transformers リポジトリのルート


def sha256_file(path, chunk=1 << 20):
    """ファイルの sha256 を返す.

    Args:
        path (str | pathlib.Path): 対象ファイル。
        chunk (int): 読み込み単位（バイト）。

    Returns:
        str: 16 進の sha256。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path, root=None):
    """ファイルのパス・サイズ・sha256（JSONL なら行数も）をまとめる.

    Args:
        path (pathlib.Path): 対象ファイル。
        root (pathlib.Path | None): 相対パスの基準。``None`` なら絶対パスを記録する。

    Returns:
        dict: ``path`` / ``bytes`` / ``sha256``（``rows``）。
    """
    path = Path(path)
    rec = {"path": str(path.relative_to(root)) if root else str(path), "bytes": path.stat().st_size,
           "sha256": sha256_file(path)}
    if path.suffix == ".jsonl":
        with open(path, "rb") as f:
            rec["rows"] = sum(1 for _ in f)
    return rec


def git_info(repo=REPO_ROOT, paths=None):
    """コードの git commit・ブランチ・未コミット変更の有無を返す.

    Args:
        repo (pathlib.Path): リポジトリのルート。
        paths (list[str] | None): 未コミット変更を調べる範囲（``None`` ならリポジトリ全体）。

    Returns:
        dict: ``commit`` / ``branch`` / ``dirty``（git が使えなければ ``error``）。
    """
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    try:
        status = run("status", "--porcelain", "--", *(paths or ["."]))
        return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(status), "dirty_files": status.splitlines()[:20]}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def package_versions(names):
    """Python と指定パッケージのバージョンを返す.

    Args:
        names (list[str]): パッケージ名。

    Returns:
        dict[str, str]: パッケージ名→バージョン（未導入なら ``"not installed"``）。
    """
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = "not installed"
    return out


def hf_revision(repo_id, repo_type="model"):
    """Hub 上のリポジトリの現在の revision（commit sha）を返す.

    Args:
        repo_id (str): ``<owner>/<name>``。
        repo_type (str): ``model`` / ``dataset``。

    Returns:
        str: commit sha。取得できなければ ``"unavailable: <理由>"``。
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.model_info(repo_id) if repo_type == "model" else api.dataset_info(repo_id)
        return info.sha
    except Exception as e:  # noqa: BLE001 — 記録用途なので失敗は文字列で残す
        return f"unavailable: {type(e).__name__}"


def docker_image_id(image):
    """ローカルの Docker イメージ ID（sha256）を返す（取得できなければ理由を返す）.

    Args:
        image (str): イメージ名。

    Returns:
        str: ``sha256:...`` または ``"unavailable: <理由>"``。
    """
    try:
        return subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return f"unavailable: {type(e).__name__}"


def now_iso():
    """現在時刻を ISO 8601（タイムゾーン付き）で返す.

    Returns:
        str: 例 ``2026-10-04T18:00:00+09:00``。
    """
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
