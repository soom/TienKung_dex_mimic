import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# All bodies in IsaacLab traversal order (merge_fixed_joints=True, 30 bodies).
# Isaac Lab uses breadth-first traversal with children sorted alphabetically
# at each depth level. Fixed joints (imu, camera, head, RGB, radar) are merged away.
#
# Pelvis children (alpha): hip_pitch_l, hip_pitch_r, waist_yaw
#   L1 children (alpha): hip_roll_l, hip_roll_r, waist_roll
#     L2 children (alpha): hip_yaw_l, hip_yaw_r, waist_pitch
#       L3 children (alpha): knee_pitch_l, knee_pitch_r, shoulder_pitch_l, shoulder_pitch_r
#         L4 children (alpha): ankle_pitch_l, ankle_pitch_r, shoulder_roll_l, shoulder_roll_r
#           L5 children (alpha): ankle_roll_l, ankle_roll_r, shoulder_yaw_l, shoulder_yaw_r
#             L6 children (alpha): elbow_pitch_l, elbow_pitch_r
#               L7 children (alpha): elbow_yaw_l, elbow_yaw_r
#                 L8 children (alpha): wrist_pitch_l, wrist_pitch_r
#                   L9 children (alpha): wrist_roll_l, wrist_roll_r
DEX_EVT_ISAACLAB_JOINTS = [
    "pelvis",
    "hip_pitch_l_link",         #  1
    "hip_pitch_r_link",         #  2
    "waist_yaw_link",           #  3
    "hip_roll_l_link",          #  4
    "hip_roll_r_link",          #  5
    "waist_roll_link",          #  6
    "hip_yaw_l_link",           #  7
    "hip_yaw_r_link",           #  8
    "waist_pitch_link",         #  9
    "knee_pitch_l_link",        # 10
    "knee_pitch_r_link",        # 11
    "shoulder_pitch_l_link",    # 12
    "shoulder_pitch_r_link",    # 13
    "ankle_pitch_l_link",       # 14
    "ankle_pitch_r_link",       # 15
    "shoulder_roll_l_link",     # 16
    "shoulder_roll_r_link",     # 17
    "ankle_roll_l_link",        # 18
    "ankle_roll_r_link",        # 19
    "shoulder_yaw_l_link",      # 20
    "shoulder_yaw_r_link",      # 21
    "elbow_pitch_l_link",       # 22
    "elbow_pitch_r_link",       # 23
    "elbow_yaw_l_link",         # 24
    "elbow_yaw_r_link",         # 25
    "wrist_pitch_l_link",       # 26
    "wrist_pitch_r_link",       # 27
    "wrist_roll_l_link",        # 28
    "wrist_roll_r_link",        # 29
]

# MuJoCo DOF order (XML depth-first, 29 joints):
#   0-5:   left leg  (hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll)
#   6-11:  right leg
#  12-14:  waist (yaw, roll, pitch)
#  15-21:  left arm (shoulder_pitch, shoulder_roll, shoulder_yaw,
#                   elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll)
#  22-28:  right arm
#
# IsaacLab DOF order (breadth-first, children alpha-sorted per level):
#   0: pelvis    1: hip_pit_l     2: hip_pit_r     3: waist_yaw
#   4: hip_rol_l 5: hip_rol_r     6: waist_rol     7: hip_yaw_l
#   8: hip_yaw_r 9: waist_pit    10: knee_pit_l   11: knee_pit_r
#  12: shld_pit_l 13: shld_pit_r 14: ank_pit_l    15: ank_pit_r
#  16: shld_rol_l 17: shld_rol_r 18: ank_rol_l    19: ank_rol_r
#  20: shld_yaw_l 21: shld_yaw_r 22: elb_pit_l    23: elb_pit_r
#  24: elb_yaw_l  25: elb_yaw_r  26: wri_pit_l    27: wri_pit_r
#  28: wri_rol_l  29: wri_rol_r

DEX_EVT_ISAACLAB_TO_MUJOCO_DOF = [
    0,      #  1: hip_pitch_l
    6,      #  2: hip_pitch_r
    12,     #  3: waist_yaw
    1,      #  4: hip_roll_l
    7,      #  5: hip_roll_r
    13,     #  6: waist_roll
    2,      #  7: hip_yaw_l
    8,      #  8: hip_yaw_r
    14,     #  9: waist_pitch
    3,      # 10: knee_pitch_l
    9,      # 11: knee_pitch_r
    15,     # 12: shoulder_pitch_l
    22,     # 13: shoulder_pitch_r
    4,      # 14: ankle_pitch_l
    10,     # 15: ankle_pitch_r
    16,     # 16: shoulder_roll_l
    23,     # 17: shoulder_roll_r
    5,      # 18: ankle_roll_l
    11,     # 19: ankle_roll_r
    17,     # 20: shoulder_yaw_l
    24,     # 21: shoulder_yaw_r
    18,     # 22: elbow_pitch_l
    25,     # 23: elbow_pitch_r
    19,     # 24: elbow_yaw_l
    26,     # 25: elbow_yaw_r
    20,     # 26: wrist_pitch_l
    27,     # 27: wrist_pitch_r
    21,     # 28: wrist_roll_l
    28,     # 29: wrist_roll_r
]

DEX_EVT_MUJOCO_TO_ISAACLAB_DOF = [
    0,      #  0: hip_pitch_l
    3,      #  1: hip_roll_l
    6,      #  2: hip_yaw_l
    9,      #  3: knee_pitch_l
    13,     #  4: ankle_pitch_l
    17,     #  5: ankle_roll_l
    1,      #  6: hip_pitch_r
    4,      #  7: hip_roll_r
    7,      #  8: hip_yaw_r
    10,     #  9: knee_pitch_r
    14,     # 10: ankle_pitch_r
    18,     # 11: ankle_roll_r
    2,      # 12: waist_yaw
    5,      # 13: waist_roll
    8,      # 14: waist_pitch
    11,     # 15: shoulder_pitch_l
    15,     # 16: shoulder_roll_l
    19,     # 17: shoulder_yaw_l
    21,     # 18: elbow_pitch_l
    23,     # 19: elbow_yaw_l
    25,     # 20: wrist_pitch_l
    27,     # 21: wrist_roll_l
    12,     # 22: shoulder_pitch_r
    16,     # 23: shoulder_roll_r
    20,     # 24: shoulder_yaw_r
    22,     # 25: elbow_pitch_r
    24,     # 26: elbow_yaw_r
    26,     # 27: wrist_pitch_r
    28,     # 28: wrist_roll_r
]

# MuJoCo body order (32 bodies, XML depth-first, includes fixed hand_index links):
#   0: pelvis
#   1-6:   left leg
#   7-12:  right leg
#  13-15:  waist
#  16-22:  left arm (+ left_hand_index at 23, fixed)
#  24-30:  right arm (+ right_hand_index at 31, fixed)
#
# Body mapping maps the 30 "effective" bodies (excluding fixed hand_index links).
# MuJoCo indices after the left_hand_index (at 23) are shifted by +1;
# after right_hand_index (at 31) by +2.

DEX_EVT_ISAACLAB_TO_MUJOCO_BODY = [
    0,      #  0: pelvis
    1,      #  1: hip_pitch_l_link
    7,      #  2: hip_pitch_r_link
    13,     #  3: waist_yaw_link
    2,      #  4: hip_roll_l_link
    8,      #  5: hip_roll_r_link
    14,     #  6: waist_roll_link
    3,      #  7: hip_yaw_l_link
    9,      #  8: hip_yaw_r_link
    15,     #  9: waist_pitch_link
    4,      # 10: knee_pitch_l_link
    10,     # 11: knee_pitch_r_link
    16,     # 12: shoulder_pitch_l_link
    24,     # 13: shoulder_pitch_r_link (shifted +1 past left_hand_index)
    5,      # 14: ankle_pitch_l_link
    11,     # 15: ankle_pitch_r_link
    17,     # 16: shoulder_roll_l_link
    25,     # 17: shoulder_roll_r_link
    6,      # 18: ankle_roll_l_link
    12,     # 19: ankle_roll_r_link
    18,     # 20: shoulder_yaw_l_link
    26,     # 21: shoulder_yaw_r_link
    19,     # 22: elbow_pitch_l_link
    27,     # 23: elbow_pitch_r_link
    20,     # 24: elbow_yaw_l_link
    28,     # 25: elbow_yaw_r_link
    21,     # 26: wrist_pitch_l_link
    29,     # 27: wrist_pitch_r_link
    22,     # 28: wrist_roll_l_link
    30,     # 29: wrist_roll_r_link (shifted +2 past both hand_index)
]

DEX_EVT_MUJOCO_TO_ISAACLAB_BODY = [
    0,      #  0: pelvis
    1,      #  1: hip_pitch_l_link
    4,      #  2: hip_roll_l_link
    7,      #  3: hip_yaw_l_link
    10,     #  4: knee_pitch_l_link
    14,     #  5: ankle_pitch_l_link
    18,     #  6: ankle_roll_l_link
    2,      #  7: hip_pitch_r_link
    5,      #  8: hip_roll_r_link
    8,      #  9: hip_yaw_r_link
    11,     # 10: knee_pitch_r_link
    15,     # 11: ankle_pitch_r_link
    19,     # 12: ankle_roll_r_link
    3,      # 13: waist_yaw_link
    6,      # 14: waist_roll_link
    9,      # 15: waist_pitch_link
    12,     # 16: shoulder_pitch_l_link
    16,     # 17: shoulder_roll_l_link
    20,     # 18: shoulder_yaw_l_link
    22,     # 19: elbow_pitch_l_link
    24,     # 20: elbow_yaw_l_link
    26,     # 21: wrist_pitch_l_link
    28,     # 22: wrist_roll_l_link
    # left_hand_index at MJCF idx 23 — skipped (fixed)
    13,     # 24: shoulder_pitch_r_link
    17,     # 25: shoulder_roll_r_link
    21,     # 26: shoulder_yaw_r_link
    23,     # 27: elbow_pitch_r_link
    25,     # 28: elbow_yaw_r_link
    27,     # 29: wrist_pitch_r_link
    29,     # 30: wrist_roll_r_link
    # right_hand_index at MJCF idx 31 — skipped (fixed)
]

# Leg joints in IsaacLab DOF order (used for lower-body observations)
# hip_pitch_l/r, hip_roll_l/r, hip_yaw_l/r, knee_pitch_l/r, ankle_pitch_l/r, ankle_roll_l/r
DEX_EVT_LOWER_JOINT_ISAACLAB_INDICES = [
    1, 2, 4, 5, 7, 8, 10, 11, 14, 15, 18, 19,
]

DEX_EVT_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": DEX_EVT_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": DEX_EVT_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": DEX_EVT_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": DEX_EVT_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": DEX_EVT_MUJOCO_TO_ISAACLAB_BODY,
    "lower_joint_isaaclab_indices": DEX_EVT_LOWER_JOINT_ISAACLAB_INDICES,
}

DEX_EVT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ASSET_DIR}/dex_evt/urdf/evt2.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 早期训练关闭，仿 Pro 策略，避免策略随机探索阶段误杀
            solver_position_iteration_count=4,  # 8→4, 降低物理求解迭代防止极端位姿卡死
            solver_velocity_iteration_count=2,  # 4→2
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.97),
        joint_pos={
            # Legs - left: 对齐 NPZ 首帧 (~0 rad)，微小屈膝防锁死
            "hip_yaw_l_joint": 0.0,
            "hip_roll_l_joint": 0.0,
            "hip_pitch_l_joint": -0.05,
            "knee_pitch_l_joint": 0.05,
            "ankle_pitch_l_joint": -0.05,
            "ankle_roll_l_joint": 0.0,
            # Legs - right
            "hip_yaw_r_joint": 0.0,
            "hip_roll_r_joint": 0.0,
            "hip_pitch_r_joint": -0.05,
            "knee_pitch_r_joint": 0.05,
            "ankle_pitch_r_joint": -0.05,
            "ankle_roll_r_joint": 0.0,
            # Arms - left：接近伸直，留安全裕度防过伸
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_yaw_l_joint": 0.0,
            "elbow_pitch_l_joint": -0.1,
            "elbow_yaw_l_joint": 0.0,
            "wrist_pitch_l_joint": 0.0,
            "wrist_roll_l_joint": 0.0,
            # Arms - right
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": -0.1,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": -0.1,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,
            # Waist：中立
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_yaw_.*_joint",
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_yaw_.*_joint": 142.0,
                "hip_roll_.*_joint": 200.0,
                "hip_pitch_.*_joint": 200.0,
                "knee_pitch_.*_joint": 330.0,
            },
            velocity_limit_sim={
                "hip_yaw_.*_joint": 13.823,
                "hip_roll_.*_joint": 12.566,
                "hip_pitch_.*_joint": 12.566,
                "knee_pitch_.*_joint": 11.106,
            },
            stiffness={
                "hip_yaw_.*_joint": 350.0,    # 200→350, 接近 Pro 500
                "hip_roll_.*_joint": 600.0,   # 450→600, 接近 Pro 700
                "hip_pitch_.*_joint": 600.0,  # 450→600
                "knee_pitch_.*_joint": 650.0, # 500→650, 接近 Pro 700
            },
            damping={
                "hip_yaw_.*_joint": 10.0,     # 6→10, 维持阻尼比
                "hip_roll_.*_joint": 15.0,    # 12→15
                "hip_pitch_.*_joint": 15.0,   # 12→15
                "knee_pitch_.*_joint": 14.0,  # 12→14
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_.*_joint", "ankle_roll_.*_joint"],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 100.0,
                "ankle_roll_.*_joint": 50.0,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 14.6,
                "ankle_roll_.*_joint": 14.6,
            },
            stiffness={
                "ankle_pitch_.*_joint": 60.0,   # 30→60, 翻倍提升 PD 推力
                "ankle_roll_.*_joint": 35.0,    # 16.8→35, 翻倍
            },
            damping={
                "ankle_pitch_.*_joint": 7.0,    # 5.0→7.0, 维持阻尼比
                "ankle_roll_.*_joint": 3.0,     # 2.0→3.0
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 90.0,
                "shoulder_roll_.*_joint": 60.0,
                "shoulder_yaw_.*_joint": 36.0,
                "elbow_pitch_.*_joint": 60.0,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 15.184,
                "shoulder_roll_.*_joint": 14.137,
                "shoulder_yaw_.*_joint": 9.739,
                "elbow_pitch_.*_joint": 14.137,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 150.0,
                "shoulder_roll_.*_joint": 50.0,
                "shoulder_yaw_.*_joint": 50.0,
                "elbow_pitch_.*_joint": 150.0,
            },
            damping={
                "shoulder_pitch_.*_joint": 5.0,
                "shoulder_roll_.*_joint": 2.5,
                "shoulder_yaw_.*_joint": 2.5,
                "elbow_pitch_.*_joint": 5.0,
            },
        ),
        "arms_distal": ImplicitActuatorCfg(
            joint_names_expr=[
                "elbow_yaw_.*_joint",
                "wrist_pitch_.*_joint",
                "wrist_roll_.*_joint",
            ],
            effort_limit_sim={
                "elbow_yaw_.*_joint": 36.0,
                "wrist_pitch_.*_joint": 36.0,
                "wrist_roll_.*_joint": 36.0,
            },
            velocity_limit_sim={
                "elbow_yaw_.*_joint": 9.739,
                "wrist_pitch_.*_joint": 9.739,
                "wrist_roll_.*_joint": 9.739,
            },
            stiffness={
                "elbow_yaw_.*_joint": 500.0,
                "wrist_pitch_.*_joint": 200.0,
                "wrist_roll_.*_joint": 200.0,
            },
            damping={
                "elbow_yaw_.*_joint": 15.0,   # 5→15: action_scale=0, PD直接跟踪, ζ: 0.112→0.335
                "wrist_pitch_.*_joint": 8.0,  # 2→8:  同上, ζ: 0.071→0.283
                "wrist_roll_.*_joint": 8.0,   # 2→8:  同上
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 91.0,
                "waist_roll_joint": 91.0,
                "waist_pitch_joint": 91.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 9.2153,
                "waist_roll_joint": 9.2153,
                "waist_pitch_joint": 9.2153,
            },
            stiffness={
                "waist_yaw_joint": 400.0,
                "waist_roll_joint": 400.0,
                "waist_pitch_joint": 400.0,
            },
            damping={
                "waist_yaw_joint": 10.0,   # 5→10: 腰偏航振荡通过腿长放大为脚坐标偏差, ζ: 0.125→0.250
                "waist_roll_joint": 10.0,
                "waist_pitch_joint": 10.0,
            },
        ),
    },
)

DEX_EVT_ACTION_SCALE = {}
DEX_EVT_ACTION_SCALE = {}
for actuator in DEX_EVT_CFG.actuators.values():
    effort_limit = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    joint_names = actuator.joint_names_expr
    if not isinstance(effort_limit, dict):
        effort_limit = {name: effort_limit for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    for name in joint_names:
        if name in effort_limit and name in stiffness and stiffness[name]:
            DEX_EVT_ACTION_SCALE[name] = 0.25 * effort_limit[name] / stiffness[name]
