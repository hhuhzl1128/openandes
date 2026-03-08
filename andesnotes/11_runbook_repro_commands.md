# Reproducible Runbook Commands

Updated: 2026-03-08 01:44 (Asia/Shanghai)

## Goal
Provide copy-ready command chains and mark each chain as validated or pending.

## Command Chain 1 — Build runnable ANDES environment (validated)
```bash
cd /path/to/andes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
andes --help
```

## Command Chain 2 — Run baseline power flow (validated)
```bash
andes run <case.xlsx> -r pflow --no-preamble
```

## Command Chain 3 — Run short dynamic simulation (validated)
```bash
andes run <case.xlsx> -r pflow tds --tf 0.2 --no-preamble
```

## Common Issues
- Environment not activated
- Incorrect case path
- Missing optional dependencies for notebook workflows

## Rule
Keep logs, output paths, and case versions explicit in every reproducible run.
