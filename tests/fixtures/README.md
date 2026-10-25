Four real teacher traces from `cleaned/traces/` (first cleaning pass), used as fixtures:

- `agronholm__exceptiongroup...`: long (78 entries), mostly `run_command`, recall left in (cleaning v2 test case); entry 5 repeats entry 3's `read_file` (its `end_line` is ignored, same output); one entry already rewritten by cleaning v1 (`cleaned_from`).
- `arrow-py__arrow...`: `read_file` continued with `start_line`, one large edit, a pre-existing failing test.
- `astropy__astropy-13803`: short and clean (control).
- `benoitc__gunicorn...`: parallel `read_file` calls in one turn, two edits.

Replace with the cleaning-v2 versions when available; keep these as regression inputs.
Decide before publishing whether fixtures may be released (they contain teacher outputs).
