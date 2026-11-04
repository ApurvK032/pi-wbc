"""The GR00T / LeRobot v2.1 export (pi_wbc.stage3.export)."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pi_wbc.stage3 import export as E

ROOT = Path(__file__).resolve().parents[1]
RECORDING = ROOT / "logs" / "dataset" / "train-v1" / "10000.npz"
# the controller's arm and hand command order (Isaac Lab's, interleaved left and right)
ARM_HAND = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint", "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint", "left_hand_thumb_2_joint", "right_hand_index_0_joint",
    "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
]  # fmt: skip


def _episode(steps: int = 7) -> tuple[dict, dict]:
    t = np.arange(steps, dtype=np.float32)[:, None]
    ep = {
        "phase": np.zeros(steps, dtype=np.int16),
        "joint_pos": t * 100 + np.arange(43, dtype=np.float32),
        "cmd_arm_hand": t * 100 + np.arange(28, dtype=np.float32),
        "cmd_body": t * 100 + np.array([0.75, 0.1, 0.2, 0.3], dtype=np.float32),
        "cmd_velocity": t * 100 + np.array([0.5, -0.1, 0.2], dtype=np.float32),
    }
    return ep, {"joint_names": E.STATE_NAMES, "arm_hand_names": ARM_HAND}


def test_layout_matches_groot_g1() -> None:
    m = E.modality()
    for kind, groups, total in (("state", E.STATE_GROUPS, 43), ("action", E.ACTION_GROUPS, 35)):
        end = 0
        for key, names in groups.items():
            assert m[kind][key]["start"] == end
            end = m[kind][key]["end"]
            assert end - m[kind][key]["start"] == len(names)
        assert end == total
    # GR00T N1.6's unitree_g1 config reads these keys
    assert set(m["action"]) == {"left_arm", "right_arm", "left_hand", "right_hand", "waist",
                                "base_height_command", "navigate_command"}  # fmt: skip
    assert set(E.STATE_GROUPS) == {"left_leg", "right_leg", "waist", "left_arm", "right_arm",
                                   "left_hand", "right_hand"}  # fmt: skip
    assert m["video"]["ego_view"]["original_key"] == E.VIDEO_KEY
    assert len(set(E.STATE_NAMES)) == 43 and sorted(E.STATE_NAMES[15:]) == sorted(ARM_HAND)


def test_frames_take_every_second_step() -> None:
    assert E.frames_of(7).tolist() == [0, 2, 4, 6]
    assert E.frames_of(6).tolist() == [0, 2, 4]


def test_episode_arrays_pick_the_right_joints() -> None:
    ep, meta = _episode()
    state, action = E.episode_arrays(ep, meta)
    assert state.dtype == action.dtype == np.float32
    assert state.shape == (4, 43) and action.shape == (4, 35)
    assert np.array_equal(state, ep["joint_pos"][[0, 2, 4, 6]])
    m = E.modality()["action"]
    for group in ("left_arm", "right_arm", "left_hand", "right_hand"):
        cols = action[:, m[group]["start"] : m[group]["end"]]
        want = [ARM_HAND.index(n) for n in E.ACTION_GROUPS[group]]
        assert np.array_equal(cols[0], np.array(want, dtype=np.float32))
        assert np.array_equal(cols[1], np.array(want, dtype=np.float32) + 200)  # step 2
    assert np.allclose(action[0, m["waist"]["start"] : m["waist"]["end"]], [0.1, 0.2, 0.3])
    assert np.allclose(action[0, m["base_height_command"]["start"]], 0.75)
    assert np.allclose(action[0, m["navigate_command"]["start"] :], [0.5, -0.1, 0.2])


def test_wrong_joint_order_is_refused() -> None:
    ep, meta = _episode()
    meta["joint_names"] = list(reversed(E.STATE_NAMES))
    with pytest.raises(ValueError):
        E.episode_arrays(ep, meta)


def test_table_columns_match_the_features() -> None:
    pytest.importorskip("pyarrow")
    ep, meta = _episode()
    state, action = E.episode_arrays(ep, meta)
    table = E.episode_table(state, action, episode_index=3, first_index=100, task=2)
    feats = E.features(640, 480)
    assert set(table.column_names) == set(feats) - {E.VIDEO_KEY}
    rows = table.to_pylist()
    assert [r["index"] for r in rows] == [100, 101, 102, 103]
    assert [r["timestamp"] for r in rows] == pytest.approx([0.0, 0.04, 0.08, 0.12])
    assert [r["next.done"] for r in rows] == [False, False, False, True]
    assert {r[E.TASK_COLUMN] for r in rows} == {2} and {r["episode_index"] for r in rows} == {3}
    assert np.allclose(rows[1]["action"], action[1])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_video_stats(tmp_path: Path) -> None:
    video = tmp_path / "red.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=640x480:r=25:d=1", "-pix_fmt", "yuv420p", str(video)],
                   check=True)  # fmt: skip
    s = E.video_stats(video, 25, samples=5)
    assert np.array(s["mean"]).shape == (3, 1, 1)
    assert s["count"] == [5]
    r, g, b = (x[0][0] for x in s["mean"])
    assert r > 0.9 and g < 0.1 and b < 0.1


@pytest.mark.skipif(not RECORDING.exists(), reason="needs the recorded training set")
def test_a_real_recording() -> None:
    d = np.load(RECORDING)
    meta = json.loads(str(d["meta"]))
    ep = {k: d[k] for k in ("phase", "joint_pos", "cmd_body", "cmd_arm_hand", "cmd_velocity")}
    state, action = E.episode_arrays(ep, meta)
    assert len(state) == (len(ep["phase"]) + 1) // 2
    m = E.modality()["action"]
    right_elbow = meta["arm_hand_names"].index("right_elbow_joint")
    col = m["right_arm"]["start"] + E.ACTION_GROUPS["right_arm"].index("right_elbow_joint")
    assert np.allclose(action[:, col], ep["cmd_arm_hand"][::2, right_elbow])
    assert np.allclose(action[:, m["base_height_command"]["start"]], ep["cmd_body"][::2, 0])


def test_fifty_frames_per_second() -> None:
    """D-048: one row per control step, timestamps 0.02 s apart."""
    pytest.importorskip("pyarrow")
    assert E.frames_of(7, frame_step=1).tolist() == list(range(7))
    ep, meta = _episode(steps=5)
    state, action = E.episode_arrays(ep, meta, frame_step=1)
    assert np.array_equal(state, ep["joint_pos"]) and len(action) == 5
    assert np.allclose(action[3, 32:], ep["cmd_velocity"][3])
    rows = E.episode_table(state, action, 0, 0, 0, fps=50).to_pylist()
    assert [r["timestamp"] for r in rows] == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.08])
    assert E.features(640, 480, fps=50)[E.VIDEO_KEY]["info"]["video.fps"] == 50
    assert E.FPS * E.FRAME_STEP == E.CONTROL_HZ == 50
