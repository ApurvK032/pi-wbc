"""Compute the G1 crouch poses used by PiWbc-G1-Body's leg actions (g1_body.CROUCH_POSES).

For each knee angle, the ankle bends as far as needed (down to its limit, -0.873 rad), the hip
pitch keeps the pelvis level (hip = -(knee + ankle)), the feet stay flat, and the arms are in the
default pose. Prints the pelvis height above the floor and how far the whole-body centre of mass
sits ahead of the ankle (the feet touch the floor from 5 cm behind to 12 cm ahead of it in
MuJoCo Menagerie's model, 2 cm behind in Newton's).

    uv run python scripts/crouch_table.py
"""

import mujoco

SCENE = "third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml"
ARMS = {
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": 1.28,
    "right_elbow_joint": 1.28,
}
SOLE_BELOW_ANKLE = 0.035  # m: foot spheres sit 3 cm below the ankle-roll origin, radius 5 mm


def main() -> None:
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)

    def adr(name: str) -> int:
        return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]

    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    print("knee   ankle   hip     pelvis height  CoM ahead of ankle")
    for knee in (0.3, 1.0, 1.4, 1.6, 1.8, 2.0):
        ankle = max(-0.87, -0.2 - 0.5 * (knee - 0.3))
        hip = -(knee + ankle)
        mujoco.mj_resetData(m, d)
        d.qpos[2], d.qpos[3:7] = 1.0, [1, 0, 0, 0]
        for side in ("left", "right"):
            d.qpos[adr(f"{side}_knee_joint")] = knee
            d.qpos[adr(f"{side}_ankle_pitch_joint")] = ankle
            d.qpos[adr(f"{side}_hip_pitch_joint")] = hip
        for name, value in ARMS.items():
            d.qpos[adr(name)] = value
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        height = d.qpos[2] - (d.xpos[foot][2] - SOLE_BELOW_ANKLE)
        com_ahead = d.subtree_com[0][0] - d.xpos[foot][0]
        print(f"{knee:4.2f}  {ankle:6.3f}  {hip:6.3f}   {height:6.3f} m       {com_ahead:+.3f} m")


if __name__ == "__main__":
    main()
