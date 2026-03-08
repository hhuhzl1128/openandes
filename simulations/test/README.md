# IL200 Four-Setting Frequency Comparison Test

This folder contains a reproducible simulation workflow for the Illinois 200-bus dynamic case with **four settings (S1–S4)**.

## Objective
Run 50-second dynamic simulations for all four settings, apply a **permanent load increase at 5 seconds**, and compare frequency responses.

## Folder Structure
- `cases/`: input case files (`IL200_dyn_db_opt_S1.xlsx` ... `S4.xlsx`)
- `scripts/`: perturbation script used during TDS
- `runs/`: per-setting raw outputs (`.txt/.lst/.npz/.log`)
- `figures/`: comparison plots
- `summaries/`: machine-readable summary results

## Environment
Use a Python environment with ANDES installed.

Example used in this project:
- Python executable: `/Users/hhuhzl/.openclaw/workspace/.envs/.venv-andes312/bin/python`
- ANDES invocation: `python -m andes`

## Reproduce the Simulation
Run from repository root:

```bash
PYBIN=/Users/hhuhzl/.openclaw/workspace/.envs/.venv-andes312/bin/python

for S in S1 S2 S3 S4; do
  mkdir -p simulations/test/runs/run_${S}
  "$PYBIN" -m andes run \
    "simulations/test/cases/IL200_dyn_db_opt_${S}.xlsx" \
    -r pflow tds --tf 50 --no-preamble \
    --pert "simulations/test/scripts/pert_load_increase_5s_permanent.py" \
    -o "simulations/test/runs/run_${S}" \
    > "simulations/test/runs/run_${S}/run.log" 2>&1
done
```

## Perturbation Definition
`pert_load_increase_5s_permanent.py` does the following:
1. Selects the largest PQ load bus.
2. At `t >= 5.0 s`, applies a permanent load increase (implemented via `Req/Xeq` step).
3. Triggers `system.TDS.custom_event=True`.

## Generate Frequency Comparison Figure
```bash
PYBIN=/Users/hhuhzl/.openclaw/workspace/.envs/.venv-andes312/bin/python
"$PYBIN" - <<'PY'
import numpy as np, matplotlib.pyplot as plt, json
from pathlib import Path

base=Path('simulations/test')
res=[]
plt.figure(figsize=(10,6))

for tag in ['S1','S2','S3','S4']:
    run=base/'runs'/f'run_{tag}'
    data=np.load(run/f'IL200_dyn_db_opt_{tag}_out.npz')['data']

    names=[]
    with open(run/f'IL200_dyn_db_opt_{tag}_out.lst', encoding='utf-8', errors='ignore') as f:
        for line in f:
            ps=line.strip().split(',')
            if len(ps)>=2:
                names.append(ps[1].strip())

    t=data[:,0]
    omega_idx=[i for i,n in enumerate(names) if n.startswith('omega GENROU')]
    f_hz=60.0*np.mean(data[:,omega_idx], axis=1)

    plt.plot(t, f_hz, label=tag, linewidth=1.8)
    res.append({"tag":tag, "f_min":float(f_hz.min()), "drop_mHz":float((60-f_hz.min())*1000)})

plt.axvline(5.0, color='k', ls='--', lw=1, alpha=0.6, label='event @5s')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('IL200 Frequency Comparison (S1-S4, 50s, permanent load increase at 5s)')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

(base/'figures').mkdir(parents=True, exist_ok=True)
(base/'summaries').mkdir(parents=True, exist_ok=True)
plt.savefig(base/'figures'/'frequency_comparison_s1_s4_50s.png', dpi=180)
(base/'summaries'/'frequency_comparison_s1_s4_50s.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
PY
```

## Notes
- If ANDES options differ by version, check:
  ```bash
  python -m andes --help
  python -m andes run --help
  ```
- Keep run outputs in separate folders (`run_S1` ... `run_S4`) to avoid cross-run file contamination.
