import sys
from dataclasses import asdict
from pathlib import Path

from pi_wbc import bench

RSL_LINE = (
    "                       Computation: 51234 steps/s (collection: 0.512s, learning 0.101s)\n"
)


RSL5_BLOCK = """\
                       Total steps: 98304
                  Steps per second: {fps}
                   Collection time: {collect}s
                     Learning time: 0.101s
"""


def test_parse_steps_per_s() -> None:
    assert bench.parse_steps_per_s(RSL_LINE) == 51234.0
    assert bench.parse_steps_per_s("Computation: 812.5 steps/s") == 812.5
    assert bench.parse_steps_per_s("          Steps per second: 40312 ") == 40312.0
    assert bench.parse_steps_per_s("Mean reward: 3.2") is None
    assert bench.parse_steps_per_s("Total steps: 98304") is None


def test_summarize_reads_rsl_rl_5_timings() -> None:
    lines: list[str] = []
    for fps, collect in [(1000, 0.9), (4000, 0.5), (5000, 0.4), (6000, 0.3)]:
        lines += RSL5_BLOCK.format(fps=fps, collect=collect).splitlines(keepends=True)
    row = bench.summarize(64, "cmd", 0, lines, [], None, 1.0, warmup_iters=1)
    assert row.steps_per_s_all == [1000.0, 4000.0, 5000.0, 6000.0]
    assert row.steps_per_s_median == 5000.0
    assert row.collection_s_median == 0.4
    assert row.learning_s_median == 0.101


def test_parse_gpu_sample() -> None:
    assert bench.parse_gpu_sample("5120, 143.27\n") == (5120.0, 143.27)
    assert bench.parse_gpu_sample("[N/A], [N/A]\n") is None
    assert bench.parse_gpu_sample("") is None


def test_summarize_drops_warmup_and_finds_peaks() -> None:
    lines = [f"Computation: {v} steps/s\n" for v in (10, 20, 100, 110, 120)]
    row = bench.summarize(
        num_envs=64,
        command="cmd",
        exit_code=0,
        log_lines=lines,
        gpu_samples=[(0.0, 1000.0, 50.0), (1.0, 3000.0, 150.0), (2.0, 2000.0, 100.0)],
        baseline_mib=15.0,
        wall_s=12.34,
        warmup_iters=2,
    )
    assert row.iterations_logged == 5
    assert row.steps_per_s_median == 110.0
    assert row.vram_peak_mib == 3000.0
    assert row.power_median_w == 100.0  # no line times given: every sample counts
    assert row.power_peak_w == 150.0
    assert row.power_samples == 3
    assert row.out_of_memory is False
    assert row.timed_out is False


def test_summarize_leaves_startup_out_of_power() -> None:
    # Start-up until t=40 (idle GPU), then one iteration line per second from t=41 to t=45.
    lines = ["loading\n"] + [f"Steps per second: {v}\n" for v in (10, 20, 100, 110, 120)]
    line_times = [1.0, 41.0, 42.0, 43.0, 44.0, 45.0]
    samples = [(float(t), 500.0, 30.0) for t in range(0, 41)]  # idle start-up
    samples += [(float(t), 4000.0, 180.0) for t in range(41, 46)]  # training
    row = bench.summarize(
        512, "cmd", 0, lines, samples, None, 46.0, warmup_iters=2, line_times=line_times
    )
    assert row.power_median_w == 180.0
    assert row.power_samples == 4  # t = 42..45: from the last warm-up line to the end
    assert row.vram_peak_mib == 4000.0


def test_summarize_detects_oom_and_short_runs() -> None:
    row = bench.summarize(
        num_envs=8192,
        command="cmd",
        exit_code=1,
        log_lines=["Computation: 5 steps/s\n", "torch.OutOfMemoryError: CUDA out of memory.\n"],
        gpu_samples=[],
        baseline_mib=None,
        wall_s=1.0,
        warmup_iters=5,
    )
    assert row.out_of_memory is True
    assert row.steps_per_s_median is None  # every logged iteration was warm-up
    assert row.vram_peak_mib is None


def test_markdown_table() -> None:
    ok = bench.summarize(512, "cmd", 0, ["Steps per second: 42000\n"] * 3, [], None, 1.0, 1)
    oom = bench.summarize(16384, "cmd", 1, ["CUDA out of memory\n"], [], None, 1.0, 1)
    table = bench.markdown_table([asdict(ok), asdict(oom)])
    lines = table.splitlines()
    assert lines[0].startswith("| Envs | Steps/s |")
    assert lines[2].startswith("| 512 | 42,000 |")
    assert lines[3].endswith("| out of memory |")
    assert "—" in lines[3]


def test_run_once_with_fake_trainer(tmp_path: Path) -> None:
    script = (
        "import sys; n = int(sys.argv[1]); "
        "[print(f'Computation: {n * (i + 1)} steps/s', flush=True) for i in range(4)]"
    )
    row = bench.run_once(
        [sys.executable, "-c", script, "{num_envs}"],
        num_envs=100,
        warmup_iters=1,
        log_dir=tmp_path,
        timeout_s=60,
    )
    assert row.exit_code == 0
    assert row.steps_per_s_all == [100.0, 200.0, 300.0, 400.0]
    assert row.steps_per_s_median == 300.0
    assert (tmp_path / "envs_100.log").read_text().count("Computation") == 4


def test_run_once_stops_a_silent_hang(tmp_path: Path) -> None:
    row = bench.run_once(
        [sys.executable, "-c", "import time; time.sleep(60)", "{num_envs}"],
        num_envs=1,
        warmup_iters=0,
        log_dir=tmp_path,
        timeout_s=1,
    )
    assert row.timed_out is True
    assert row.exit_code != 0
    assert row.wall_s < 30
