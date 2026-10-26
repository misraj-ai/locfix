Four real teacher traces from `cleaned/traces/` (first cleaning pass), used as fixtures:

- `agronholm__exceptiongroup...`: long (78 entries), mostly `run_command`, recall left in (cleaning v2 test case); entry 5 repeats entry 3's `read_file` (its `end_line` is ignored, same output); one entry already rewritten by cleaning v1 (`cleaned_from`).
- `arrow-py__arrow...`: `read_file` continued with `start_line`, one large edit, a pre-existing failing test.
- `astropy__astropy-13803`: short and clean (control).
- `benoitc__gunicorn...`: parallel `read_file` calls in one turn, two edits.

Replace with the cleaning-v2 versions when available; keep these as regression inputs.
Decide before publishing whether fixtures may be released (they contain teacher outputs).

`code/<instance_id>.json` — the files each task needs at the base commit (bug applied), in the
format of the data pipeline's `code/<id>.json`. Built by `tools/build_fixture_code.py` from the
upstream repositories (arrow: Apache-2.0; exceptiongroup, gunicorn: MIT; astropy: BSD-3-Clause)
plus every line the teacher saw; verified by replaying all recorded `read_file`, plain `sed -n` and
`edit_file` calls (edit diffs carry line numbers).

`records/locfix_fixtures.jsonl` — localizer/fixer records (spec 04) for gunicorn and astropy, all
stages; embedded with `s1_fixtures.jsonl` in `notebooks/golden_render_check_standalone.py`.
