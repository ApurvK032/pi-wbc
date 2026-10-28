# Running on MSI

Long training runs, motion imitation, rendering and VLA fine-tuning run on the Minnesota
Supercomputing Institute (MSI) cluster. This page is the recipe; the Slurm account,
partition and user name are placeholders here and stay out of the repository.

## How it is set up

- **Code** reaches MSI only through git: a clone of this public repository, updated with
  `git pull`. Every job prints the commit it ran from, and result files record it.
- **Container.** MSI's nodes run Rocky Linux 8 (glibc 2.28). Some of Isaac Lab 3.0's
  dependencies (open3d, usd-exchange, omniverseclient) need glibc 2.34 or newer, so everything
  runs in an Apptainer container built from [`msi/pi-wbc.def`](../msi/pi-wbc.def): Ubuntu 24.04,
  Python 3.12 and uv, the same as the workstation. It builds without admin rights.
- **Python environment.** Not baked into the image: `uv sync --frozen` inside the container
  creates `.venv` in the clone from the same `uv.lock` as the workstation, so package versions
  match exactly.
- **GPUs.** Isaac Lab 3.0 needs NVIDIA driver 580.65.06 or newer; MSI's L40S nodes had
  580.178.04 (checked 2026-10-28). Compute nodes can reach the internet, so Isaac Lab
  downloads robot assets on first use.

## One-time setup

On a login node, from the home directory:

```
conda deactivate                                   # if conda's base env is active
git clone --branch <branch> https://github.com/ApurvK032/pi-wbc.git ~/pi-wbc
mkdir -p ~/containers
apptainer build ~/containers/pi-wbc.sif ~/pi-wbc/msi/pi-wbc.def
cd ~/pi-wbc && apptainer exec ~/containers/pi-wbc.sif uv sync --frozen
```

The environment takes about 8 GB (`.venv` plus uv's cache in `~/.cache/uv`); the image 151 MB.
Rebuild the image only when `msi/pi-wbc.def` changes; re-run `uv sync --frozen` when
`uv.lock` changes.

## Jobs

Job scripts are in [`msi/jobs/`](../msi/jobs/). Submit from the clone's root, with the
account and partition on the command line:

```
cd ~/pi-wbc && git pull
mkdir -p logs/slurm
sbatch -A <account> -p <partition> msi/jobs/train_stage0.sbatch [seed]
sbatch -A <account> -p <partition> msi/jobs/bench_stage0.sbatch
squeue --me                                        # your jobs only
tail -f logs/slurm/piwbc-stage0-<jobid>.out
```

| Script | What it does | Time limit |
|---|---|---|
| `train_stage0.sbatch` | Trains `PiWbc-G1-Flat-Stage0` (4096 envs, 1500 iterations), then exports the policy for the MuJoCo bench | 1.5 h |
| `bench_stage0.sbatch` | Throughput sweep, 512-16384 environments, of the built-in G1 flat task | 45 min |

Both run well inside the 24 h GPU limit, so they have no checkpoint-and-requeue logic yet;
stage 1's long runs will add it.

## Bringing results home

Run on the workstation, in the repository (one login with password and Duo):

```
rsync -av --mkpath <user>@agate.msi.umn.edu:pi-wbc/results/ results/
rsync -av --mkpath --include='*/' --include='exported/***' --include='params/***' --exclude='*' \
    <user>@agate.msi.umn.edu:pi-wbc/logs/rsl_rl/ logs/msi/rsl_rl/
```

The first copies small result files (committed on the workstation afterwards); the second
copies only the exported policies and run settings, not checkpoints, into the ignored `logs/`.

## Working rules

- Commands are run by hand on MSI, one short block at a time, each labelled read-only,
  creates files, overwrites/deletes, or spends GPU time.
- Each session starts with `set +H`, so a `!` in a pasted line is not a history command.
- Use `squeue --me` and similar, so output never lists other users; cluster output that names
  other users is never committed.
