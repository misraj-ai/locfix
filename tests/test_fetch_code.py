"""data.fetch-code on local git repositories (no network): base commit + setup_patch, the
instance-branch fallback, and the check against the teacher's reads."""
import difflib
import json
import subprocess
from pathlib import Path

import pytest

from locfix.cli import main
from locfix.data.fetch_code import apply_section, file_section


def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


def make_repo(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True)
    for p, t in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(t)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    git(root, "config", "uploadpack.allowFilter", "true")
    git(root, "config", "uploadpack.allowAnySHA1InWant", "true")
    return git(root, "rev-parse", "HEAD")


def test_file_section_and_apply():
    patch = ("diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n a = 1\n-b = 2\n+b = 3\n"
             "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-c\n+d\n")
    sec = file_section(patch, "x.py")
    assert "y.py" not in sec and apply_section("a = 1\nb = 2\n", "x.py", sec) == "a = 1\nb = 3\n"
    assert apply_section("unrelated\n", "x.py", sec) is None


@pytest.fixture
def task(tmp_path, all_trace_paths):
    from test_loc_fix_records import make_task

    p = next(x for x in all_trace_paths if x.name.startswith("arrow"))
    return tmp_path, p, make_task(p)


def setup(tmp_path, p, t, row_extra):
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / p.name).write_bytes(p.read_bytes())
    row = {"instance_id": t.instance_id, "problem_statement": t.problem_statement, "fold": "A",
           "verifier": "swe_smith", "teacher": {"patch": t.patch}, **row_extra}
    (tmp_path / "solved.jsonl").write_text(json.dumps(row) + "\n")


def fetch(tmp_path):
    # "GitHub" = local folders under tmp_path/gh (no network)
    main(["data.fetch-code", str(tmp_path / "traces"), "--solved", str(tmp_path / "solved.jsonl"),
          "--out", str(tmp_path / "code"), "--cache", str(tmp_path / "cache"), "--workers", "2",
          "--url-template", str(tmp_path / "gh" / "{repo}")])
    return json.loads((tmp_path / "fetch_code_report.json").read_text())


def test_base_commit_with_setup_patch(task):
    tmp_path, p, t = task
    path = "arrow/factory.py"
    lines = t.files[path].splitlines(keepends=True)
    # the "clean" upstream file differs from the buggy one in one line; setup_patch re-applies the bug
    k = next(i for i, x in enumerate(lines) if x.strip())
    clean = "".join(lines[:k] + ["# clean upstream line\n"] + lines[k + 1:])
    files = {**t.files, path: clean}
    sha = make_repo(tmp_path / "upstream", files)
    diff = difflib.unified_diff(clean.splitlines(keepends=True), lines, f"a/{path}", f"b/{path}")
    bug = f"diff --git a/{path} b/{path}\n" + "".join(diff)
    setup(tmp_path, p, t, {"repo": str(tmp_path / "upstream"), "base_commit": sha, "setup_patch": bug})
    rep = fetch(tmp_path)
    assert rep["fetch"] == {"fetched": 1} and rep["check_per_trace"] == {"ok": 1}
    got = json.loads((tmp_path / "code" / f"{t.instance_id}.json").read_text())
    assert got[path] == t.files[path]
    assert fetch(tmp_path)["fetch"] == {"already_there": 1}                  # re-run skips


def test_swesmith_mirror_branch(task):
    """SWE-smith rows carry the upstream repo and an 8-character commit; the task's code is on the
    branch <instance_id> of swesmith/<Owner>__<repo>.<hash8>."""
    tmp_path, p, t = task
    assert t.instance_id.startswith("arrow-py__arrow.1d70d009.")
    up = tmp_path / "gh" / "swesmith" / "arrow-py__arrow.1d70d009"
    path = "arrow/factory.py"
    make_repo(up, {**t.files, path: "# upstream, without the bug\n" + t.files[path]})
    git(up, "checkout", "-qb", t.instance_id)
    (up / path).write_text(t.files[path])
    git(up, "commit", "-qam", "bug")
    git(up, "checkout", "-q", "-")
    setup(tmp_path, p, t, {"repo": "arrow-py/arrow", "base_commit": "1d70d009"})
    rep = fetch(tmp_path)
    assert rep["fetch"] == {"fetched_from_instance_branch": 1} and rep["check_per_trace"] == {"ok": 1}


def test_instance_branch_fallback(all_trace_paths, tmp_path):
    from test_loc_fix_records import make_task

    p = next(x for x in all_trace_paths if x.name.startswith("astropy"))
    t = make_task(p)
    path = "astropy/coordinates/angles.py"
    up = tmp_path / "upstream"
    sha = make_repo(up, {**t.files, path: "# not the code the teacher saw\n" + t.files[path]})
    git(up, "checkout", "-qb", t.instance_id)
    (up / path).write_text(t.files[path])
    git(up, "commit", "-qam", "bug")
    git(up, "checkout", "-q", "-")
    # the teacher printed nothing of this file that could be compared, so a commit that cannot be
    # fetched is what sends it to the branch here
    assert sha
    setup(tmp_path, p, t, {"repo": str(up), "base_commit": "b" * 40, "verifier": "swebench"})
    rep = fetch(tmp_path)
    assert rep["fetch"] == {"fetched_from_instance_branch": 1}
    assert set(rep["check_per_trace"]) <= {"ok", "unverified"}


def test_line_check_without_harness(all_trace_paths):
    from locfix.data.code_files import check_code
    from test_loc_fix_records import make_task

    for p in all_trace_paths:
        t = make_task(p)
        r = check_code(t.trace, t.files, t.patch, exact=False)
        assert r["status"] in ("ok", "unverified") and r["reads_ok"] is None, p.name
        if r["lines_checked"]:
            path = next(iter(t.files))
            bad = {**t.files, path: "# shifted\n" + t.files[path]}
            assert check_code(t.trace, bad, t.patch, exact=False)["status"] in ("differs", "ok")


def test_missing_repo_is_reported(task):
    tmp_path, p, t = task
    setup(tmp_path, p, t, {"repo": str(tmp_path / "nothing"), "base_commit": "a" * 40})
    rep = fetch(tmp_path)
    assert len(rep["fetch"]) == 1 and next(iter(rep["fetch"])).startswith("no branch") and not (tmp_path / "code" / f"{t.instance_id}.json").exists()
