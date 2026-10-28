"""Training throughput benchmark: steps/s, peak VRAM and GPU power per environment count.

Runs a training command once per environment count, reads the steps/s figure that rsl_rl
prints each iteration, samples the GPU with nvidia-smi while it runs, and writes one JSON
result file with a provenance record.

Example (the command is a template; {num_envs} is filled in per run):

    uv run pi-wbc-bench --num-envs 512 1024 2048 4096 --out results/stage0/bench.json -- \
        isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-G1 \
        --num_envs {num_envs} --max_iterations 50
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pi_wbc import provenance

# rsl_rl >= 5 prints "Steps per second: 51234" each iteration; older versions printed
# "Computation: 51234 steps/s (collection: ...)".
_STEPS_RE = re.compile(r"(?:Steps per second:\s*|Computation:\s*)([\d.]+)")
# rsl_rl >= 5 also prints "Collection time: 0.512s" and "Learning time: 0.101s".
_COLLECT_RE = re.compile(r"Collection time:\s*([\d.]+)s")
_LEARN_RE = re.compile(r"Learning time:\s*([\d.]+)s")
# Errors that mean the GPU ran out of memory, as opposed to some other failure.
_OOM_RE = re.compile(r"CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation", re.I)


def _parse(pattern: re.Pattern[str], line: str) -> float | None:
    m = pattern.search(line)
    return float(m.group(1)) if m else None


def parse_steps_per_s(line: str) -> float | None:
    """The steps/s value from one rsl_rl log line, or None if the line has none."""
    return _parse(_STEPS_RE, line)


def _values(pattern: re.Pattern[str], lines: list[str]) -> list[float]:
    return [v for v in (_parse(pattern, line) for line in lines) if v is not None]


def _median_after_warmup(values: list[float], warmup: int) -> float | None:
    timed = values[warmup:] if len(values) > warmup else []
    return statistics.median(timed) if timed else None


def parse_gpu_sample(output: str) -> tuple[float, float] | None:
    """(memory used MiB, power W) for the first GPU in nvidia-smi CSV output."""
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            continue
    return None


def gpu_sample() -> tuple[float, float] | None:
    """One (memory used MiB, power W) reading of GPU 0, or None without nvidia-smi."""
    out = provenance._run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_gpu_sample(out) if out else None


class GpuSampler:
    """Polls nvidia-smi in a background thread; keeps (monotonic time, MiB, W) samples."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = gpu_sample()
            if sample is not None:
                self.samples.append((time.monotonic(), *sample))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


@dataclass
class RunResult:
    num_envs: int
    command: str
    exit_code: int
    out_of_memory: bool
    timed_out: bool
    wall_s: float
    iterations_logged: int
    warmup_iters: int
    steps_per_s_median: float | None
    collection_s_median: float | None = None
    learning_s_median: float | None = None
    steps_per_s_all: list[float] = field(default_factory=list)
    vram_baseline_mib: float | None = None
    vram_peak_mib: float | None = None  # whole run, start-up included
    power_median_w: float | None = None  # timed iterations only (after warm-up)
    power_peak_w: float | None = None  # timed iterations only
    power_samples: int = 0


def _timed_samples(
    samples: list[tuple[float, float, float]],
    log_lines: list[str],
    line_times: list[float] | None,
    warmup_iters: int,
) -> list[tuple[float, float, float]]:
    """GPU samples taken during the timed iterations, i.e. after warm-up.

    Iteration k's steps/s line is printed when iteration k ends, so the timed window runs
    from the line of the last warm-up iteration to the last steps/s line. Without line
    times every sample counts.
    """
    if line_times is None:
        return samples
    step_times = [
        t for t, line in zip(line_times, log_lines, strict=True) if _STEPS_RE.search(line)
    ]
    if len(step_times) <= warmup_iters:
        return []
    start = step_times[warmup_iters - 1] if warmup_iters > 0 else float("-inf")
    return [s for s in samples if start <= s[0] <= step_times[-1]]


def summarize(
    num_envs: int,
    command: str,
    exit_code: int,
    log_lines: list[str],
    gpu_samples: list[tuple[float, float, float]],
    baseline_mib: float | None,
    wall_s: float,
    warmup_iters: int,
    timed_out: bool = False,
    line_times: list[float] | None = None,
) -> RunResult:
    """Turn one run's raw log and GPU samples into a result row.

    `gpu_samples` are (monotonic time, memory MiB, power W); `line_times` are the monotonic
    times at which each log line arrived.
    """
    steps = _values(_STEPS_RE, log_lines)
    mem = [m for _, m, _ in gpu_samples]
    power = [p for _, _, p in _timed_samples(gpu_samples, log_lines, line_times, warmup_iters)]
    return RunResult(
        num_envs=num_envs,
        command=command,
        exit_code=exit_code,
        out_of_memory=any(_OOM_RE.search(line) for line in log_lines),
        timed_out=timed_out,
        wall_s=round(wall_s, 1),
        iterations_logged=len(steps),
        warmup_iters=warmup_iters,
        steps_per_s_median=_median_after_warmup(steps, warmup_iters),
        collection_s_median=_median_after_warmup(_values(_COLLECT_RE, log_lines), warmup_iters),
        learning_s_median=_median_after_warmup(_values(_LEARN_RE, log_lines), warmup_iters),
        steps_per_s_all=steps,
        vram_baseline_mib=baseline_mib,
        vram_peak_mib=max(mem) if mem else None,
        power_median_w=statistics.median(power) if power else None,
        power_peak_w=max(power) if power else None,
        power_samples=len(power),
    )


def run_once(
    template: list[str], num_envs: int, warmup_iters: int, log_dir: Path, timeout_s: float
) -> RunResult:
    """Run the training command for one environment count."""
    cmd = [part.replace("{num_envs}", str(num_envs)) for part in template]
    command = shlex.join(cmd)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"envs_{num_envs}.log"

    before = gpu_sample()
    baseline_mib = before[0] if before else None

    lines: list[str] = []
    line_times: list[float] = []
    start = time.monotonic()
    with GpuSampler() as sampler, log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        # The timer stops a run that hangs without printing anything.
        timed_out = threading.Event()

        def _stop() -> None:
            timed_out.set()
            proc.terminate()

        timer = threading.Timer(timeout_s, _stop)
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()  # so the log can be watched while the run is going
                lines.append(line)
                line_times.append(time.monotonic())
            exit_code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait()
        finally:
            timer.cancel()
    wall_s = time.monotonic() - start
    return summarize(
        num_envs,
        command,
        exit_code,
        lines,
        sampler.samples,
        baseline_mib,
        wall_s,
        warmup_iters,
        timed_out=timed_out.is_set(),
        line_times=line_times,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-envs", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="result JSON path")
    parser.add_argument("--warmup-iters", type=int, default=5, help="iterations left out")
    parser.add_argument("--timeout-s", type=float, default=900, help="per-run time limit")
    parser.add_argument("--label", default="", help="free-text note stored in the result")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    args = parser.parse_args(argv)

    template = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not template or not any("{num_envs}" in part for part in template):
        parser.error("give the training command after --, with {num_envs} in it")

    log_dir = Path("logs") / "bench" / args.out.stem  # raw logs stay out of git
    rows: list[dict[str, Any]] = []
    for n in args.num_envs:
        print(f"[bench] num_envs={n} ...", flush=True)
        row = run_once(template, n, args.warmup_iters, log_dir, args.timeout_s)
        print(
            f"[bench] num_envs={n}: exit={row.exit_code} oom={row.out_of_memory} "
            f"steps/s={row.steps_per_s_median} vram_peak={row.vram_peak_mib} MiB "
            f"power_median={row.power_median_w} W",
            flush=True,
        )
        rows.append(asdict(row))
        if row.out_of_memory:
            print("[bench] out of memory; skipping larger counts", flush=True)
            break

    path = provenance.save_json(args.out, {"label": args.label, "runs": rows})
    print(f"[bench] wrote {path}")


def _fmt(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    """A Markdown table of bench result rows, one line per environment count."""
    out = [
        "| Envs | Steps/s | Sim time / iter | Learn time / iter "
        "| Peak VRAM | Median power | Status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r["out_of_memory"]:
            status = "out of memory"
        elif r["timed_out"]:
            status = "timed out"
        elif r["exit_code"] != 0:
            status = f"exit {r['exit_code']}"
        else:
            status = "ok"
        vram = r["vram_peak_mib"]
        out.append(
            f"| {r['num_envs']} | {_fmt(r['steps_per_s_median'], ',.0f')} "
            f"| {_fmt(r.get('collection_s_median'), '.3f')} s "
            f"| {_fmt(r.get('learning_s_median'), '.3f')} s "
            f"| {_fmt(vram / 1024 if vram is not None else None, '.1f')} GB "
            f"| {_fmt(r['power_median_w'], '.0f')} W | {status} |"
        )
    return "\n".join(out)


def table_main(argv: list[str] | None = None) -> None:
    """Print a bench result file as a Markdown table."""
    parser = argparse.ArgumentParser(description=table_main.__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    print(markdown_table(json.loads(args.result.read_text())["runs"]))


if __name__ == "__main__":
    main()
