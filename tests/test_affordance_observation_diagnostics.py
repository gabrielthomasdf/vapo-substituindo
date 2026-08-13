from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vapo.wrappers.affordance.observation_diagnostics import (
    build_offline_runtime_comparison,
    build_report,
    collect_replayed_inference,
    collect_single_observation,
)
from vapo.wrappers.affordance.simulation_diagnostics import (
    collect_positioned_observation,
)


class FakeEncoder(torch.nn.Module):
    def _prepare_input(self, value):
        return value + 1


class FakeHoughVoting(torch.nn.Module):
    def forward(self, mask, directions):
        return mask, torch.ones(1, dtype=torch.int32), directions


class FakeAffordanceNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = SimpleNamespace(encoder=FakeEncoder())
        self.hough_voting_layer = FakeHoughVoting()

    def forward(self, value):
        self.unet.encoder._prepare_input(value)
        mask = torch.ones((1, 128, 128), dtype=torch.int64)
        directions = torch.full((1, 2, 128, 128), 0.5)
        probabilities = torch.ones((1, 2, 128, 128))
        return probabilities, probabilities, mask, directions

    def get_centers(self, mask, directions):
        self.hough_voting_layer(mask.int(), directions)
        return [torch.tensor([64, 32])], directions, mask


class FakeWrapper:
    def __init__(self):
        self.gripper_cam_aff_net = FakeAffordanceNet()
        self.curr_detected_obj = None
        self.aff_transforms = {
            "gripper": lambda image: torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(128, 128), mode="nearest"
            ).squeeze(0)
        }

    def get_world_pt(self, cam, pixel, depth, orig_shape):
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    def observation(self, raw_observation):
        image = torch.zeros((1, 3, 128, 128))
        _, _, mask, directions = self.gripper_cam_aff_net(image)
        centers, _, _ = self.gripper_cam_aff_net.get_centers(mask, directions)
        if centers:
            self.get_world_pt(None, centers[0].numpy(), np.ones((64, 64)), (64, 64))
        return {
            "gripper_aff": np.ones((1, 64, 64), dtype=np.int64),
            "gripper_img_obs": np.zeros((3, 200, 200), dtype=np.uint8),
            "gripper_depth_obs": np.ones((1, 64, 64), dtype=np.float32),
            "target_distance": np.array([0.08], dtype=np.float32),
        }

    def transform_obs(self, observation, split="validation"):
        observation["gripper_img_obs"] = torch.nn.functional.interpolate(
            observation["gripper_img_obs"].unsqueeze(0), size=(64, 64)
        ).squeeze(0)
        return observation


class FakeSimulationEnv:
    def __init__(self):
        self.events = []
        self.robot_position = np.array([0.1, 0.2, 0.39], dtype=np.float32)

    def reset(self, eval=False):
        self.events.append(("reset", eval))

    def get_target_pos(self):
        self.events.append(("get_target_pos",))
        return np.array([0.1, 0.2, 0.3], dtype=np.float32), False

    def move_to_target(self, target_pos):
        self.events.append(("move_to_target", np.asarray(target_pos).copy()))

    def get_obs(self):
        self.events.append(("get_obs",))
        return {"robot_obs": np.array([*self.robot_position, 0, 0, 0, 1])}


def test_collects_single_observation_without_changing_pipeline_callables():
    wrapper = FakeWrapper()
    original_get_centers = wrapper.gripper_cam_aff_net.get_centers
    original_get_world_pt = wrapper.get_world_pt
    original_prepare_input = wrapper.gripper_cam_aff_net.unet.encoder._prepare_input

    captured = collect_single_observation(wrapper, {})

    assert captured["dinov3_model_input"].shape == (1, 3, 128, 128)
    assert captured["affordance_logits"].shape == (1, 2, 128, 128)
    assert captured["affordance_probabilities"].shape == (1, 2, 128, 128)
    assert captured["dinov3_mask_128"].shape == (1, 128, 128)
    assert captured["center_directions_128"].shape == (1, 2, 128, 128)
    assert captured["hough_centers_2d"].tolist() == [[64, 32]]
    assert captured["world_points_3d"][0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert wrapper.gripper_cam_aff_net.get_centers == original_get_centers
    assert wrapper.get_world_pt == original_get_world_pt
    assert wrapper.gripper_cam_aff_net.unet.encoder._prepare_input == original_prepare_input


def test_replays_preprocessing_and_inference_from_the_exact_runtime_frame():
    wrapper = FakeWrapper()
    captured = collect_single_observation(wrapper, {})
    original_prepare_input = wrapper.gripper_cam_aff_net.unet.encoder._prepare_input

    replayed = collect_replayed_inference(wrapper, captured["gripper_img_obs"])
    comparison = build_offline_runtime_comparison(captured, replayed)

    assert comparison["checks"] == {
        "missing_values": {},
        "preprocessing_matches": True,
        "inference_matches": True,
        "runtime_foreground_pixel_count": 128 * 128,
        "offline_replay_foreground_pixel_count": 128 * 128,
    }
    assert wrapper.gripper_cam_aff_net.unet.encoder._prepare_input == original_prepare_input


def test_offline_runtime_comparison_identifies_inference_divergence():
    wrapper = FakeWrapper()
    captured = collect_single_observation(wrapper, {})
    replayed = collect_replayed_inference(wrapper, captured["gripper_img_obs"])
    replayed["affordance_logits"][0, 0, 0, 0] += 1

    comparison = build_offline_runtime_comparison(captured, replayed)

    assert comparison["checks"]["preprocessing_matches"]
    assert not comparison["checks"]["inference_matches"]
    assert not comparison["values"]["affordance_logits"]["within_tolerance"]
    assert comparison["values"]["affordance_logits"]["max_abs_diff"] == 1


def test_report_checks_dinov3_shapes_and_non_finite_values():
    captured = collect_single_observation(FakeWrapper(), {})

    report = build_report(captured, encoder_type="dinov3", policy_img_size=64)

    assert report["checks"]["shapes_match"]
    assert report["checks"]["all_finite"]
    assert report["checks"]["hough_center_count"] == 1
    assert report["checks"]["valid_world_point_count"] == 1
    assert report["checks"]["center_found"]
    assert report["checks"]["world_conversion_succeeded"]
    assert report["center_conversions"] == [
        {
            "hough_center_2d": [64, 32],
            "depth_pixel": [64, 32],
            "world_point_3d": pytest.approx([0.1, 0.2, 0.3]),
        }
    ]


def test_report_identifies_nan_and_infinity():
    captured = collect_single_observation(FakeWrapper(), {})
    captured["target_distance"] = np.array([np.nan], dtype=np.float32)
    captured["world_points_3d"] = [
        np.array([0.1, np.inf, 0.3], dtype=np.float32)
    ]

    report = build_report(captured, encoder_type="dinov3", policy_img_size=64)

    assert not report["checks"]["all_finite"]
    assert report["checks"]["non_finite_values"]["target_distance"] == 1
    assert report["checks"]["non_finite_values"]["world_points_3d"] == 1


def test_positioned_simulation_observation_moves_before_collecting_report():
    env = FakeSimulationEnv()
    wrapper = FakeWrapper()

    captured, report = collect_positioned_observation(
        wrapper,
        env,
        encoder_type="dinov3",
        policy_img_size=64,
    )

    assert [event[0] for event in env.events] == [
        "reset",
        "get_target_pos",
        "move_to_target",
        "get_obs",
    ]
    np.testing.assert_allclose(env.events[2][1], [0.1, 0.2, 0.3])
    assert captured["dinov3_model_input"].shape == (1, 3, 128, 128)
    assert report["positioning"] == {
        "target_pos": pytest.approx([0.1, 0.2, 0.3]),
        "robot_position_after_move_to_target": pytest.approx([0.1, 0.2, 0.39]),
        "robot_target_distance_after_move_to_target": pytest.approx(0.09),
        "curr_detected_obj_before": None,
        "curr_detected_obj_after_seed": pytest.approx([0.1, 0.2, 0.3]),
        "curr_detected_obj_after_observation": pytest.approx([0.1, 0.2, 0.3]),
    }
    assert report["checks"]["center_found"]
    assert report["checks"]["world_conversion_succeeded"]


def test_positioned_simulation_observation_reports_missing_center_without_crash():
    env = FakeSimulationEnv()
    wrapper = FakeWrapper()

    def no_centers(mask, directions):
        wrapper.gripper_cam_aff_net.hough_voting_layer(mask.int(), directions)
        return [], directions, mask

    wrapper.gripper_cam_aff_net.get_centers = no_centers

    _, report = collect_positioned_observation(
        wrapper,
        env,
        encoder_type="dinov3",
        policy_img_size=64,
    )

    assert report["checks"]["all_finite"]
    assert report["checks"]["shapes_match"]
    assert report["checks"]["hough_center_count"] == 0
    assert report["checks"]["valid_world_point_count"] == 0
    assert not report["checks"]["center_found"]
    assert not report["checks"]["world_conversion_succeeded"]


def test_positioned_simulation_observation_can_retry_without_hiding_failed_frames():
    env = FakeSimulationEnv()
    wrapper = FakeWrapper()
    original_get_centers = wrapper.gripper_cam_aff_net.get_centers
    calls = 0

    def centers_after_first_frame(mask, directions):
        nonlocal calls
        calls += 1
        if calls == 1:
            wrapper.gripper_cam_aff_net.hough_voting_layer(mask.int(), directions)
            return [], directions, mask
        return original_get_centers(mask, directions)

    wrapper.gripper_cam_aff_net.get_centers = centers_after_first_frame

    _, report = collect_positioned_observation(
        wrapper,
        env,
        encoder_type="dinov3",
        policy_img_size=64,
        observation_attempts=3,
    )

    assert report["observation_attempts"]["performed"] == 2
    assert report["observation_attempts"]["selected"] == 2
    assert not report["observation_attempts"]["results"][0]["checks"]["center_found"]
    assert report["observation_attempts"]["results"][1]["checks"]["center_found"]
