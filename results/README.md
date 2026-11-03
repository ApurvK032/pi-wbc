# Results

| Folder | What |
|---|---|
| [`stage0/`](stage0/README.md) | Setup: Isaac Lab smoke test, the MuJoCo sim-to-sim bench, cluster runs, run-to-run spread, reward pilot |
| [`stage1/`](stage1/README.md) | The whole-body controller: final test-seed results first, then every step of the work |
| [`figures/`](figures/) | The figures in the READMEs, light and dark versions, drawn from the result files by `scripts/figures.py` |

Every result file records the commit, package versions, host and GPU it came from
(`pi_wbc.provenance`). Large files are gzip-compressed JSON; `pi_wbc.provenance.load_json`
reads both kinds. Development seeds (0-99) were used for every choice; test seeds (1000-1199)
only for final numbers.
