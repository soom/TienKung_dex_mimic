import os

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self._setup_noise_schedule()

    def _setup_noise_schedule(self):
        schedule = self.cfg.get("noise_std_schedule", None)
        if schedule is None:
            return

        std_init = schedule.get("init_std", 0.5)
        std_final = schedule.get("final_std", 0.05)
        decay_iters = max(schedule.get("decay_iters", 10000), 1)
        entropy_init = schedule.get("entropy_init", None)
        entropy_final = schedule.get("entropy_final", None)
        mode = schedule.get("mode", "exp")

        original_update = self.alg.update
        runner = self

        def _scheduled_update():
            result = original_update()
            # current_learning_iteration is set to `it` after this call returns,
            # so we read it here (off-by-one is negligible over 15k iters)
            it = runner.current_learning_iteration
            t = min(it / decay_iters, 1.0)
            if mode == "linear":
                std = std_init + (std_final - std_init) * t
            else:
                std = std_init * (std_final / std_init) ** t
            with torch.no_grad():
                runner.alg.policy.std.fill_(std)
            if entropy_init is not None and entropy_final is not None:
                runner.alg.entropy_coef = entropy_init + (entropy_final - entropy_init) * t
            return result

        self.alg.update = _scheduled_update

    def load(self, path: str, load_optimizer: bool = True):
        try:
            result = super().load(path, load_optimizer)
            self.current_learning_iteration = 0
            return result
        except (ValueError, RuntimeError) as e:
            msg = str(e)
            # Optimizer param group mismatch (freeze mode changed between runs)
            if "optimizer" in msg.lower() and "parameter group" in msg.lower():
                print("[MotionOnPolicyRunner] Optimizer state mismatch (freeze mode changed?) — "
                      "loading weights only, reinitializing optimizer.")
                result = super().load(path, load_optimizer=False)
                self.current_learning_iteration = 0
                return result
            # Missing keys in state dict (new code adds buffers old checkpoints lack)
            if "Missing key(s) in state_dict" in msg:
                print("[MotionOnPolicyRunner] State dict has missing keys (new code vs old checkpoint?) — "
                      "loading with strict=False.")
                loaded_dict = torch.load(path, weights_only=False, map_location="cpu")
                self.alg.policy.load_state_dict(loaded_dict["model_state_dict"], strict=False)
                if self.empirical_normalization:
                    if "obs_norm_state_dict" in loaded_dict:
                        self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                    if "privileged_obs_norm_state_dict" in loaded_dict:
                        self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                self.current_learning_iteration = 0
                return loaded_dict.get("infos", {})
            raise

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
