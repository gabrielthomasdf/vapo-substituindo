from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gym")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("segmentation_models_pytorch")

from omegaconf import OmegaConf

import vapo.wrappers.affordance.aff_wrapper_base as wrapper_module
from vapo.wrappers.affordance.aff_wrapper_base import AffordanceWrapperBase


class RecordingAffordanceNet:
    def __init__(self, prediction_size):
        self.prediction_size = prediction_size
        self.input_shape = None
        self.hough_mask_shape = None
        self.hough_directions_shape = None

    def __call__(self, obs):
        self.input_shape = tuple(obs.shape)
        size = self.prediction_size
        logits = torch.zeros((1, 2, size, size), dtype=torch.float32)
        probs = torch.zeros_like(logits)
        mask = torch.zeros((1, size, size), dtype=torch.int64)
        directions = torch.zeros((1, 2, size, size), dtype=torch.float32)
        return logits, probs, mask, directions

    def get_centers(self, mask, directions):
        # This is the boundary immediately before the real Hough Voting layer.
        self.hough_mask_shape = tuple(mask.shape)
        self.hough_directions_shape = tuple(directions.shape)
        object_masks = torch.zeros(
            (mask.shape[0], mask.shape[-2], mask.shape[-1]), dtype=torch.int32
        )
        return [], directions, object_masks


def _make_wrapper(monkeypatch, channels, affordance_size):
    wrapper = object.__new__(AffordanceWrapperBase)
    wrapper.img_size = 64
    wrapper.affordance_cfg = OmegaConf.create(
        {
            "gripper_cam": {
                "densify_reward": True,
                "target_in_obs": False,
                "use_distance": False,
            }
        }
    )
    wrapper.aff_transforms = {
        "gripper": lambda image: torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(affordance_size, affordance_size),
            mode="nearest",
        ).squeeze(0)[:channels]
    }
    wrapper.gripper_cam_aff_net = RecordingAffordanceNet(affordance_size)
    wrapper.gripper_cam = object()
    wrapper.save_images = False
    wrapper.viz = False
    wrapper.episode = 0
    wrapper.obs_it = 0
    wrapper._curr_detected_obj = None
    wrapper.env = SimpleNamespace(termination_radius=0.1)

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    depth = np.zeros((64, 64), dtype=np.float32)
    wrapper.get_images = lambda obs_cfg, obs_dict, cam_type: (depth, rgb)

    monkeypatch.setattr(wrapper_module, "tt", lambda value: torch.from_numpy(value).float())
    monkeypatch.setattr(wrapper_module, "viz_aff_centers_preds", lambda *args, **kwargs: {})
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    return wrapper


def _run_gripper_observation(wrapper):
    observation_cfg = SimpleNamespace(use_img=True, use_depth=True)
    affordance_cfg = SimpleNamespace(use=True)
    obs, _ = wrapper.get_cam_obs(
        {}, "gripper", wrapper.gripper_cam_aff_net, observation_cfg, affordance_cfg
    )
    return obs


def test_dinov3_gripper_keeps_native_affordance_shapes_until_hough(monkeypatch):
    wrapper = _make_wrapper(monkeypatch, channels=3, affordance_size=128)

    obs = _run_gripper_observation(wrapper)

    net = wrapper.gripper_cam_aff_net
    assert net.input_shape == (1, 3, 128, 128)
    assert net.hough_mask_shape == (1, 128, 128)
    assert net.hough_directions_shape == (1, 2, 128, 128)
    assert obs["gripper_aff"].shape == (1, 64, 64)
    assert obs["gripper_img_obs"].shape == (3, 64, 64)
    assert obs["gripper_depth_obs"].shape == (1, 64, 64)


def test_resnet18_gripper_preserves_official_64_shape_path(monkeypatch):
    wrapper = _make_wrapper(monkeypatch, channels=1, affordance_size=64)

    obs = _run_gripper_observation(wrapper)

    net = wrapper.gripper_cam_aff_net
    assert net.input_shape == (1, 1, 64, 64)
    assert net.hough_mask_shape == (1, 64, 64)
    assert net.hough_directions_shape == (1, 2, 64, 64)
    assert obs["gripper_aff"].shape == (1, 64, 64)
    assert obs["gripper_img_obs"].shape == (3, 64, 64)
    assert obs["gripper_depth_obs"].shape == (1, 64, 64)


def test_gripper_affordance_resolution_follows_encoder_configuration():
    dino_cfg = OmegaConf.create(
        {"hyperparameters": {"cfg": {"encoder_type": "dinov3"}}}
    )
    legacy_dino_cfg = OmegaConf.create(
        {"hyperparameters": {"cfg": {"dinov3_cfg": {"model_name": "unused"}}}}
    )
    resnet_cfg = OmegaConf.create(
        {"hyperparameters": {"cfg": {"encoder_type": "resnet18"}}}
    )

    assert AffordanceWrapperBase._get_gripper_aff_img_size(dino_cfg, 64) == 128
    assert AffordanceWrapperBase._get_gripper_aff_img_size(legacy_dino_cfg, 64) == 128
    assert AffordanceWrapperBase._get_gripper_aff_img_size(resnet_cfg, 64) == 64


def test_sac_mask_resize_uses_nearest_neighbor():
    mask = np.zeros((1, 128, 128), dtype=np.int64)
    mask[:, :64, :64] = 1

    resized = AffordanceWrapperBase._resize_gripper_aff_for_policy(mask, 64)

    assert resized.shape == (1, 64, 64)
    assert resized.dtype == mask.dtype
    assert set(np.unique(resized)) == {0, 1}


def test_find_target_center_maps_prediction_center_to_original_depth_resolution(monkeypatch):
    wrapper = _make_wrapper(monkeypatch, channels=3, affordance_size=128)
    recorded = {}
    object_masks = torch.zeros((1, 128, 128), dtype=torch.int32)
    object_masks[0, 64, 32] = 2
    wrapper.gripper_cam_aff_net.get_centers = lambda mask, directions: (
        [torch.tensor([64, 32])],
        directions,
        object_masks,
    )
    wrapper.get_world_pt = lambda cam, pixel, depth, orig_shape: recorded.setdefault(
        "pixel", pixel
    )
    predictions = {
        "gripper_aff": torch.ones((1, 128, 128), dtype=torch.int64),
        "gripper_aff_probs": torch.ones((1, 2, 128, 128), dtype=torch.float32),
        "gripper_center_dir": torch.zeros((1, 2, 128, 128), dtype=torch.float32),
    }

    wrapper.find_target_center(
        wrapper.gripper_cam,
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640), dtype=np.float32),
        predictions,
    )

    np.testing.assert_array_equal(recorded["pixel"], np.array([240, 160]))
