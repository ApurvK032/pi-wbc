from pathlib import Path

import pytest

from pi_wbc import train_summary

pytest.importorskip("tensorboard")


def test_summary_averages_the_last_iterations(tmp_path: Path) -> None:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(tmp_path))
    for step in range(10):
        writer.add_scalar("Train/mean_reward", float(step), step)
        writer.add_scalar("Episode_Termination/base_contact", 0.5, step)
    writer.close()

    row = train_summary.summarize_run(tmp_path, last=4)
    assert row["iterations"] == 10
    assert row["means"]["Train/mean_reward"] == pytest.approx((6 + 7 + 8 + 9) / 4)
    table = train_summary.markdown_table([row])
    assert "| 7.5 |" in table and "50.0%" in table and "—" in table  # missing tags show a dash
