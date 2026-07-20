import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

# Walker C1 — 29 DOF (6×2 legs + 3 waist + 7×2 arms), no head joints.
#
# IsaacLab body order (breadth-first, children alpha-sorted, after merge_fixed_joints):
#   0: base_link
#   1: L_hip_pitch_link         2: R_hip_pitch_link         3: waist_yaw_link
#   4: L_hip_roll_link          5: R_hip_roll_link          6: waist_pitch_link
#   7: L_hip_yaw_link           8: R_hip_yaw_link           9: waist_roll_link
#  10: L_knee_pitch_link       11: R_knee_pitch_link       12: L_shoulder_pitch_link
#  13: R_shoulder_pitch_link   14: L_ankle_pitch_link      15: R_ankle_pitch_link
#  16: L_shoulder_roll_link    17: R_shoulder_roll_link    18: L_ankle_roll_link
#  19: R_ankle_roll_link       20: L_shoulder_yaw_link     21: R_shoulder_yaw_link
#  22: L_elbow_pitch_link      23: R_elbow_pitch_link
#  24: L_elbow_yaw_link        25: R_elbow_yaw_link
#  26: L_wrist_pitch_link      27: R_wrist_pitch_link
#  28: L_wrist_roll_link       29: R_wrist_roll_link

WALKER_C1_ISAACLAB_BODIES = [
    "base_link",                 #  0
    "L_hip_pitch_link",          #  1
    "R_hip_pitch_link",          #  2
    "waist_yaw_link",            #  3
    "L_hip_roll_link",           #  4
    "R_hip_roll_link",           #  5
    "waist_pitch_link",          #  6
    "L_hip_yaw_link",            #  7
    "R_hip_yaw_link",            #  8
    "waist_roll_link",           #  9
    "L_knee_pitch_link",         # 10
    "R_knee_pitch_link",         # 11
    "L_shoulder_pitch_link",     # 12
    "R_shoulder_pitch_link",     # 13
    "L_ankle_pitch_link",        # 14
    "R_ankle_pitch_link",        # 15
    "L_shoulder_roll_link",      # 16
    "R_shoulder_roll_link",      # 17
    "L_ankle_roll_link",         # 18
    "R_ankle_roll_link",         # 19
    "L_shoulder_yaw_link",       # 20
    "R_shoulder_yaw_link",       # 21
    "L_elbow_pitch_link",        # 22
    "R_elbow_pitch_link",        # 23
    "L_elbow_yaw_link",          # 24
    "R_elbow_yaw_link",          # 25
    "L_wrist_pitch_link",        # 26
    "R_wrist_pitch_link",        # 27
    "L_wrist_roll_link",         # 28
    "R_wrist_roll_link",         # 29
]

# Leg joints in IsaacLab DOF order
WALKER_C1_LOWER_JOINT_ISAACLAB_INDICES = [
    0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 18, 19,
]

WALKER_C1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ASSET_DIR}/walker_c1/urdf/walker_astron.urdf",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=2,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.90),
        joint_pos={
            # Left leg
            "L_hip_yaw_joint": 0.0,
            "L_hip_roll_joint": 0.0,
            "L_hip_pitch_joint": -0.05,
            "L_knee_pitch_joint": 0.10,
            "L_ankle_pitch_joint": -0.05,
            "L_ankle_roll_joint": 0.0,
            # Right leg
            "R_hip_yaw_joint": 0.0,
            "R_hip_roll_joint": 0.0,
            "R_hip_pitch_joint": -0.05,
            "R_knee_pitch_joint": 0.10,
            "R_ankle_pitch_joint": -0.05,
            "R_ankle_roll_joint": 0.0,
            # Left arm
            "L_shoulder_pitch_joint": 0.0,
            "L_shoulder_roll_joint": 0.1,
            "L_shoulder_yaw_joint": 0.0,
            "L_elbow_pitch_joint": -0.1,
            "L_elbow_yaw_joint": 0.0,
            "L_wrist_pitch_joint": 0.0,
            "L_wrist_roll_joint": 0.0,
            # Right arm
            "R_shoulder_pitch_joint": 0.0,
            "R_shoulder_roll_joint": -0.1,
            "R_shoulder_yaw_joint": 0.0,
            "R_elbow_pitch_joint": -0.1,
            "R_elbow_yaw_joint": 0.0,
            "R_wrist_pitch_joint": 0.0,
            "R_wrist_roll_joint": 0.0,
            # Waist
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
                "L_hip_yaw_joint", "R_hip_yaw_joint",
                "L_hip_roll_joint", "R_hip_roll_joint",
                "L_hip_pitch_joint", "R_hip_pitch_joint",
                "L_knee_pitch_joint", "R_knee_pitch_joint",
            ],
            effort_limit_sim={
                "L_hip_yaw_joint": 110.0, "R_hip_yaw_joint": 110.0,
                "L_hip_roll_joint": 165.0, "R_hip_roll_joint": 165.0,
                "L_hip_pitch_joint": 210.0, "R_hip_pitch_joint": 210.0,
                "L_knee_pitch_joint": 210.0, "R_knee_pitch_joint": 210.0,
            },
            velocity_limit_sim={
                "L_hip_yaw_joint": 13.823, "R_hip_yaw_joint": 13.823,
                "L_hip_roll_joint": 12.566, "R_hip_roll_joint": 12.566,
                "L_hip_pitch_joint": 12.566, "R_hip_pitch_joint": 12.566,
                "L_knee_pitch_joint": 11.106, "R_knee_pitch_joint": 11.106,
            },
            stiffness={
                "L_hip_yaw_joint": 120.0, "R_hip_yaw_joint": 120.0,
                "L_hip_roll_joint": 180.0, "R_hip_roll_joint": 180.0,
                "L_hip_pitch_joint": 280.0, "R_hip_pitch_joint": 280.0,
                "L_knee_pitch_joint": 280.0, "R_knee_pitch_joint": 280.0,
            },
            damping={
                "L_hip_yaw_joint": 3.0, "R_hip_yaw_joint": 3.0,
                "L_hip_roll_joint": 5.0, "R_hip_roll_joint": 5.0,
                "L_hip_pitch_joint": 5.0, "R_hip_pitch_joint": 5.0,
                "L_knee_pitch_joint": 5.0, "R_knee_pitch_joint": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_ankle_pitch_joint", "R_ankle_pitch_joint",
                "L_ankle_roll_joint", "R_ankle_roll_joint",
            ],
            effort_limit_sim={
                "L_ankle_pitch_joint": 60.0, "R_ankle_pitch_joint": 60.0,
                "L_ankle_roll_joint": 60.0, "R_ankle_roll_joint": 60.0,
            },
            velocity_limit_sim={
                "L_ankle_pitch_joint": 14.6, "R_ankle_pitch_joint": 14.6,
                "L_ankle_roll_joint": 14.6, "R_ankle_roll_joint": 14.6,
            },
            stiffness={
                "L_ankle_pitch_joint": 100.0, "R_ankle_pitch_joint": 100.0,
                "L_ankle_roll_joint": 100.0, "R_ankle_roll_joint": 100.0,
            },
            damping={
                "L_ankle_pitch_joint": 3.0, "R_ankle_pitch_joint": 3.0,
                "L_ankle_roll_joint": 3.0, "R_ankle_roll_joint": 3.0,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_shoulder_pitch_joint", "R_shoulder_pitch_joint",
                "L_shoulder_roll_joint", "R_shoulder_roll_joint",
                "L_shoulder_yaw_joint", "R_shoulder_yaw_joint",
                "L_elbow_pitch_joint", "R_elbow_pitch_joint",
            ],
            effort_limit_sim={
                "L_shoulder_pitch_joint": 60.0, "R_shoulder_pitch_joint": 60.0,
                "L_shoulder_roll_joint": 60.0, "R_shoulder_roll_joint": 60.0,
                "L_shoulder_yaw_joint": 25.0, "R_shoulder_yaw_joint": 25.0,
                "L_elbow_pitch_joint": 25.0, "R_elbow_pitch_joint": 25.0,
            },
            velocity_limit_sim={
                "L_shoulder_pitch_joint": 15.184, "R_shoulder_pitch_joint": 15.184,
                "L_shoulder_roll_joint": 14.137, "R_shoulder_roll_joint": 14.137,
                "L_shoulder_yaw_joint": 9.739, "R_shoulder_yaw_joint": 9.739,
                "L_elbow_pitch_joint": 14.137, "R_elbow_pitch_joint": 14.137,
            },
            stiffness={
                "L_shoulder_pitch_joint": 160.0, "R_shoulder_pitch_joint": 160.0,
                "L_shoulder_roll_joint": 160.0, "R_shoulder_roll_joint": 160.0,
                "L_shoulder_yaw_joint": 45.0, "R_shoulder_yaw_joint": 45.0,
                "L_elbow_pitch_joint": 50.0, "R_elbow_pitch_joint": 50.0,
            },
            damping={
                "L_shoulder_pitch_joint": 4.0, "R_shoulder_pitch_joint": 4.0,
                "L_shoulder_roll_joint": 4.0, "R_shoulder_roll_joint": 4.0,
                "L_shoulder_yaw_joint": 1.2, "R_shoulder_yaw_joint": 1.2,
                "L_elbow_pitch_joint": 2.0, "R_elbow_pitch_joint": 2.0,
            },
        ),
        "arms_distal": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_elbow_yaw_joint", "R_elbow_yaw_joint",
                "L_wrist_pitch_joint", "R_wrist_pitch_joint",
                "L_wrist_roll_joint", "R_wrist_roll_joint",
            ],
            effort_limit_sim={
                "L_elbow_yaw_joint": 25.0, "R_elbow_yaw_joint": 25.0,
                "L_wrist_pitch_joint": 6.0, "R_wrist_pitch_joint": 6.0,
                "L_wrist_roll_joint": 6.0, "R_wrist_roll_joint": 6.0,
            },
            velocity_limit_sim={
                "L_elbow_yaw_joint": 9.739, "R_elbow_yaw_joint": 9.739,
                "L_wrist_pitch_joint": 9.739, "R_wrist_pitch_joint": 9.739,
                "L_wrist_roll_joint": 9.739, "R_wrist_roll_joint": 9.739,
            },
            stiffness={
                "L_elbow_yaw_joint": 80.0, "R_elbow_yaw_joint": 80.0,
                "L_wrist_pitch_joint": 40.0, "R_wrist_pitch_joint": 40.0,
                "L_wrist_roll_joint": 40.0, "R_wrist_roll_joint": 40.0,
            },
            damping={
                "L_elbow_yaw_joint": 7.0, "R_elbow_yaw_joint": 7.0,
                "L_wrist_pitch_joint": 4.0, "R_wrist_pitch_joint": 4.0,
                "L_wrist_roll_joint": 4.0, "R_wrist_roll_joint": 4.0,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 63.0,
                "waist_roll_joint": 110.0,
                "waist_pitch_joint": 165.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 9.2153,
                "waist_roll_joint": 9.2153,
                "waist_pitch_joint": 9.2153,
            },
            stiffness={
                "waist_yaw_joint": 80.0,
                "waist_roll_joint": 150.0,
                "waist_pitch_joint": 180.0,
            },
            damping={
                "waist_yaw_joint": 4.0,
                "waist_roll_joint": 3.0,
                "waist_pitch_joint": 5.0,
            },
        ),
    },
)

WALKER_C1_ACTION_SCALE = {}
for actuator in WALKER_C1_CFG.actuators.values():
    effort_limit = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    joint_names = actuator.joint_names_expr
    if not isinstance(effort_limit, dict):
        effort_limit = {name: effort_limit for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    for name in joint_names:
        if name in effort_limit and name in stiffness and stiffness[name]:
            WALKER_C1_ACTION_SCALE[name] = 0.25 * effort_limit[name] / stiffness[name]
