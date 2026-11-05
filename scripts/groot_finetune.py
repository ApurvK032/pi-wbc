"""Fine-tune GR00T N1.6 on an exported dataset (stage 4), in GR00T's environment.

The same steps and defaults as GR00T's `gr00t/experiment/launch_finetune.py` (branch n1d6),
with our arguments recorded next to the checkpoints (`pi_wbc_launch.json`; GR00T writes its full
config there too). Starting from a checkpoint, GR00T passes only the tune flags, state dropout
and fp32 trainable weights to the model; the rest of the model's settings (how many of the
language model's top layers train: 4; bf16 weights) come from the checkpoint's own config.
Each training log line also carries the peak GPU memory so far (`gpu_mem_peak_gib`), and only
the last `--keep` checkpoints stay (each is about 23 GB with its optimizer state).

    third_party/Isaac-GR00T-n1d6/.venv/bin/python scripts/groot_finetune.py \\
        --base-model nvidia/GR00T-N1.6-3B --dataset logs/export/pi_wbc_g1_pick_place_train_v1 \\
        --tag UNITREE_G1 --output logs/groot/run --max-steps 10000 --batch 64
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tag", default="UNITREE_G1")
    parser.add_argument(
        "--modality-config",
        type=Path,
        default=None,
        help="file registering a NEW_EMBODIMENT config",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64, help="global batch size")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep", type=int, default=2, help="checkpoints kept (the latest)")
    parser.add_argument("--no-tune-diffusion", action="store_true")
    parser.add_argument("--no-tune-projector", action="store_true")
    parser.add_argument("--color-jitter", action="store_true",
                        help="GR00T's example jitter: brightness 0.3, contrast 0.4, saturation "
                        "0.5, hue 0.08")  # fmt: skip
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run
    from gr00t.experiment.trainer import Gr00tTrainer

    log = Gr00tTrainer.log

    def log_with_memory(self, logs, start_time=None):  # the peak GPU memory in every log line
        import torch

        if torch.cuda.is_available():
            logs["gpu_mem_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return log(self, logs, start_time=start_time)

    Gr00tTrainer.log = log_with_memory

    tag = EmbodimentTag[args.tag].value
    if args.modality_config is not None:
        sys.path.append(str(args.modality_config.parent))
        importlib.import_module(args.modality_config.stem)
    data = {"dataset_paths": [str(args.dataset)], "mix_ratio": 1.0, "embodiment_tag": tag}
    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [data]}})
    config.load_config_path = None
    m = config.model  # as GR00T's launcher sets them
    m.tune_llm, m.tune_visual = False, False
    m.tune_projector = not args.no_tune_projector
    m.tune_diffusion_model = not args.no_tune_diffusion
    m.state_dropout_prob = 0.0
    m.random_rotation_angle = None
    jitter = {"brightness": 0.3, "contrast": 0.4, "saturation": 0.5, "hue": 0.08}
    m.color_jitter_params = jitter if args.color_jitter else None
    m.extra_augmentation_config = None
    m.load_bf16 = False
    m.reproject_vision = False
    m.eagle_collator = True
    m.model_name = "nvidia/Eagle-Block2A-2B-v2"
    m.backbone_trainable_params_fp32 = True
    m.use_relative_action = True
    t = config.training
    t.experiment_name = args.output.name
    t.start_from_checkpoint = args.base_model
    t.optim = "adamw_torch"
    t.global_batch_size = args.batch
    t.dataloader_num_workers = args.workers
    t.learning_rate = args.lr
    t.gradient_accumulation_steps = args.grad_accum
    t.output_dir = str(args.output)
    t.save_steps = args.save_steps
    t.save_total_limit = args.keep
    t.num_gpus = 1
    t.use_wandb = args.wandb
    t.max_steps = args.max_steps
    t.weight_decay = 1e-5
    t.warmup_ratio = 0.05
    args.output.mkdir(parents=True, exist_ok=True)
    launch = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "pi_wbc_launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    run(config)


if __name__ == "__main__":
    main()
