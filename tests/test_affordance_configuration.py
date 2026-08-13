from pathlib import Path

import pytest

hydra = pytest.importorskip("hydra")
gym = pytest.importorskip("gym")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("segmentation_models_pytorch")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import vapo.wrappers.affordance.aff_wrapper_base as wrapper_module
from vapo.wrappers.affordance.aff_wrapper_base import AffordanceWrapperBase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


class MinimalSimulationEnv(gym.Env):
    save_images = False
    viz = False
    task = "pickup"


def _compose_tabletop(config_name, monkeypatch):
    monkeypatch.setenv("DINOV3_REPO_DIR", str(PROJECT_ROOT / "dinov3"))
    monkeypatch.setenv("DINOV3_WEIGHTS_PATH", str(PROJECT_ROOT / "dinov3.pth"))
    monkeypatch.setenv("VAPO_GRIPPER_DINOV3_CKPT", str(PROJECT_ROOT / "gripper_dinov3.ckpt"))
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name)


def _build_wrapper(cfg, monkeypatch):
    initialized_channels = {}

    def record_init(affordance_cfg, cam_type, in_channels):
        initialized_channels[cam_type] = in_channels
        return None

    monkeypatch.setattr(wrapper_module, "init_aff_net", record_init)
    wrapper = AffordanceWrapperBase(
        MinimalSimulationEnv(),
        max_ts=100,
        affordance_cfg=cfg.affordance,
        **cfg.env_wrapper,
    )
    return wrapper, initialized_channels


def test_tabletop_dinov3_composition_keeps_camera_preprocessing_independent(monkeypatch):
    cfg = _compose_tabletop("cfg_tabletop_dinov3", monkeypatch)

    wrapper, initialized_channels = _build_wrapper(cfg, monkeypatch)

    assert wrapper.gripper_aff_img_size == 128
    assert initialized_channels == {"gripper": 3, "static": 1}
    assert cfg.img_resize == 128
    assert cfg.affordance.gripper_cam.model_path.endswith("gripper_dinov3.ckpt")
    assert cfg.affordance.gripper_cam.hyperparameters.cfg.encoder_type == "dinov3"
    assert cfg.affordance.static_cam.hyperparameters.cfg.encoder_type == "resnet18"


def test_wrapper_rejects_dinov3_with_legacy_grayscale_preprocessing(monkeypatch):
    cfg = _compose_tabletop("cfg_tabletop", monkeypatch)
    cfg.affordance.gripper_cam.hyperparameters = OmegaConf.load(
        CONFIG_DIR / "aff_model" / "gripper_cam_dinov3.yaml"
    )

    with pytest.raises(
        ValueError,
        match=r"DINOv3 gripper preprocessing must produce 3 RGB channels.*dinov3_rgb",
    ):
        _build_wrapper(cfg, monkeypatch)
