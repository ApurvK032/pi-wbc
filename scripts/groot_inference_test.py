"""Run a GR00T N1.6 checkpoint on observations from an exported dataset: shapes, speed, memory.

Runs in GR00T's environment (third_party/Isaac-GR00T-n1d6/.venv). Stage 4 needs to know whether
closed-loop evaluation fits on the workstation's 12 GB GPU (GR00T's README asks for 16 GB or
more for inference) and how fast it is, before any fine-tuning:

    third_party/Isaac-GR00T-n1d6/.venv/bin/python scripts/groot_inference_test.py \\
        --model nvidia/GR00T-N1.6-3B --dataset logs/export/pi_wbc_g1_pick_place_train_v1 \\
        --out logs/groot/inference_test.json

With the base model and the `UNITREE_G1` tag the actions are not meant to be any good (it was
never trained on this task); the test checks that our observation layout is accepted, that
every command group comes back with the expected size, and the time and memory per call.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import Gr00tPolicy


def frame(video: Path, index: int) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"select=eq(n\\,{index})",
         "-vsync", "0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3)


def gpu_mib() -> int:
    import os

    query = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
    out = subprocess.run(query, capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, mem = (x.strip() for x in line.split(","))
        if int(pid) == os.getpid():
            return int(mem)
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tag", default="UNITREE_G1")
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    meta = args.dataset / "meta"
    modality = json.loads((meta / "modality.json").read_text())
    info = json.loads((meta / "info.json").read_text())
    tasks = {t["task_index"]: t["task"] for t in map(json.loads, (meta / "tasks.jsonl").open())}
    rows = pd.read_parquet(
        args.dataset / info["data_path"].format(episode_chunk=0, episode_index=0)
    )
    video = args.dataset / info["video_path"].format(
        episode_chunk=0, video_key="observation.images.ego_view", episode_index=0)  # fmt: skip
    k = len(rows) // 3  # a third of the way in: walking to the table
    state_row = np.asarray(rows["observation.state"].iloc[k], dtype=np.float32)
    image = frame(video, k)
    text = tasks[int(rows["task_index"].iloc[k])]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    policy = Gr00tPolicy(model_path=args.model, embodiment_tag=EmbodimentTag[args.tag],
                         device="cuda:0", strict=True)  # fmt: skip
    load_s = time.perf_counter() - t0
    cfg = policy.get_modality_config()
    report = {"model": args.model, "tag": args.tag, "load_s": round(load_s, 1),
              "gpu": torch.cuda.get_device_name(0), "batches": {}}  # fmt: skip
    report["modality_config"] = {m: {"keys": list(c.modality_keys),
                                     "delta_indices": list(c.delta_indices)}
                                 for m, c in cfg.items()}  # fmt: skip
    for b in (int(x) for x in args.batches.split(",")):
        obs = {
            "video": {key: np.repeat(image[None, None], b, axis=0)
                      for key in cfg["video"].modality_keys},
            "state": {key: np.repeat(state_row[None, None, modality["state"][key]["start"]:
                                                modality["state"][key]["end"]], b, axis=0)
                      for key in cfg["state"].modality_keys},
            "language": {cfg["language"].modality_keys[0]: [[text]] * b},
        }  # fmt: skip
        action, _ = policy.get_action(obs)  # warm-up
        torch.cuda.synchronize()
        times = []
        for _ in range(args.calls):
            t = time.perf_counter()
            action, _ = policy.get_action(obs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        shapes = {key: list(np.asarray(v).shape) for key, v in action.items()}
        for key, v in action.items():  # every group comes back with its size
            s = modality["action"][key]
            assert np.asarray(v).shape[-1] == s["end"] - s["start"], (key, np.asarray(v).shape)
            assert np.isfinite(np.asarray(v)).all(), key
        report["batches"][b] = {
            "ms_per_call_median": round(1000 * float(np.median(times)), 1),
            "ms_per_call_max": round(1000 * float(np.max(times)), 1),
            "action_shapes": shapes,
            "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "process_gpu_mib": gpu_mib(),
        }
        print(f"[groot] batch {b}: {report['batches'][b]}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[groot] wrote {args.out}")


if __name__ == "__main__":
    main()
