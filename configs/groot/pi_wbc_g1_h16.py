"""GR00T N1.6 `NEW_EMBODIMENT` config for the pi-wbc G1 datasets (stage 4 draft).

GR00T's own `unitree_g1` layout (the export's names, D-045) with a 16-step action chunk: 0.64 s
at our 25 frames per second, close to the 0.6 s of NVIDIA's 30 steps at 50. Arms as changes from
the current angles, everything else absolute, as in `unitree_g1`.

Its relative-action statistics have shape (16, 7) while the export's `meta/relative_stats.json`
holds `unitree_g1`'s (30, 7), and GR00T keeps keys already in that file: train this config on a
dataset copy without it (data and videos hard-linked, `meta/` copied without
`relative_stats.json`).

    --modality-config configs/groot/pi_wbc_g1_h16.py --tag NEW_EMBODIMENT
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


def _action(rep: ActionRepresentation) -> ActionConfig:
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


ACTION_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist",
               "base_height_command", "navigate_command"]  # fmt: skip
RELATIVE = {"left_arm", "right_arm"}

config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=ACTION_KEYS,
        action_configs=[
            _action(
                ActionRepresentation.RELATIVE if k in RELATIVE else ActionRepresentation.ABSOLUTE
            )
            for k in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}
register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
