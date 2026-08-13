import pytest

pytest.importorskip("hydra")

from omegaconf import OmegaConf

from vapo.utils.utils import load_cfg


def _config(policy_marker, affordance_encoder, affordance_path, parent_folder):
    return OmegaConf.create(
        {
            "paths": {"parent_folder": parent_folder},
            "gripper_cam_aff_path": affordance_path,
            "agent": {
                "net_cfg": {"policy_marker": policy_marker},
                "hyperparameters": {"policy_marker": policy_marker},
            },
            "env_wrapper": {"policy_marker": policy_marker},
            "env": {},
            "eval_env": {},
            "affordance": {
                "gripper_cam": {
                    "model_path": "${gripper_cam_aff_path}",
                    "hyperparameters": {"cfg": {"encoder_type": affordance_encoder}},
                },
                "static_cam": {"model_path": "static.ckpt"},
            },
        }
    )


def test_loading_sac_history_can_preserve_only_current_affordance(tmp_path):
    historical = _config("historical-sac", "resnet18", "old.ckpt", "old-parent")
    current = _config("current", "dinov3", "new.ckpt", "current-parent")
    historical_path = tmp_path / "config.yaml"
    OmegaConf.save(historical, historical_path)

    run_cfg, net_cfg, env_wrapper, agent_cfg = load_cfg(
        str(historical_path),
        current,
        current_sections=("affordance",),
    )

    assert net_cfg.policy_marker == "historical-sac"
    assert env_wrapper.policy_marker == "historical-sac"
    assert agent_cfg.policy_marker == "historical-sac"
    assert run_cfg.affordance.gripper_cam.hyperparameters.cfg.encoder_type == "dinov3"
    assert run_cfg.affordance.gripper_cam.model_path == "new.ckpt"
    assert historical.affordance.gripper_cam.hyperparameters.cfg.encoder_type == "resnet18"


def test_loading_sac_history_remains_fully_historical_by_default(tmp_path):
    historical = _config("historical-sac", "resnet18", "old.ckpt", "old-parent")
    current = _config("current", "dinov3", "new.ckpt", "current-parent")
    historical_path = tmp_path / "config.yaml"
    OmegaConf.save(historical, historical_path)

    run_cfg, _, _, _ = load_cfg(str(historical_path), current)

    assert run_cfg.affordance.gripper_cam.hyperparameters.cfg.encoder_type == "resnet18"
    assert run_cfg.affordance.gripper_cam.model_path == "old.ckpt"
