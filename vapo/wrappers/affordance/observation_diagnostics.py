from contextlib import contextmanager

import numpy as np
import torch


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _stack_points(points, width):
    if not points:
        return np.empty((0, width), dtype=np.float32)
    return np.stack([_snapshot(point) for point in points])


def _tensor_observation(observation):
    return {
        key: torch.from_numpy(value).float()
        if isinstance(value, np.ndarray)
        else value
        for key, value in observation.items()
    }


def _instance_override(instance, name):
    return instance.__dict__.get(name, None), name in instance.__dict__


def _restore_instance_override(instance, name, value, existed):
    if existed:
        setattr(instance, name, value)
    else:
        instance.__dict__.pop(name, None)


@contextmanager
def _record_wrapper_pipeline(wrapper, captured):
    """Attach read-only probes and restore every patched callable afterwards."""
    aff_net = wrapper.gripper_cam_aff_net
    if aff_net is None:
        raise RuntimeError("The gripper affordance model is disabled.")

    handles = []
    original_get_centers = aff_net.get_centers
    original_get_world_pt = wrapper.get_world_pt
    encoder = getattr(getattr(aff_net, "unet", None), "encoder", None)
    original_prepare_input = getattr(encoder, "_prepare_input", None)
    get_centers_override = _instance_override(aff_net, "get_centers")
    get_world_pt_override = _instance_override(wrapper, "get_world_pt")
    prepare_input_override = (
        _instance_override(encoder, "_prepare_input") if encoder is not None else (None, False)
    )

    def record_model_input(module, inputs):
        captured["affordance_model_input"] = _snapshot(inputs[0])

    def record_model_output(module, inputs, output):
        captured["affordance_logits"] = _snapshot(output[0])
        captured["affordance_probabilities"] = _snapshot(output[1])
        captured["dinov3_mask_128"] = _snapshot(output[2])
        captured["model_center_directions"] = _snapshot(output[3])

    handles.append(aff_net.register_forward_pre_hook(record_model_input))
    handles.append(aff_net.register_forward_hook(record_model_output))

    hough_layer = getattr(aff_net, "hough_voting_layer", None)
    if hough_layer is not None:
        def record_hough_input(module, inputs):
            captured["hough_mask"] = _snapshot(inputs[0])
            captured["center_directions_128"] = _snapshot(inputs[1])

        handles.append(hough_layer.register_forward_pre_hook(record_hough_input))

    if original_prepare_input is not None:
        def record_dinov3_input(value):
            prepared = original_prepare_input(value)
            captured["dinov3_model_input"] = _snapshot(prepared)
            return prepared

        encoder._prepare_input = record_dinov3_input

    def record_get_centers(mask, directions):
        result = original_get_centers(mask, directions)
        captured["hough_centers_2d"] = _stack_points(result[0], 2)
        return result

    def record_get_world_pt(cam, pixel, depth, orig_shape):
        world_point = original_get_world_pt(cam, pixel, depth, orig_shape)
        captured.setdefault("center_projection_pixels", []).append(_snapshot(pixel))
        captured.setdefault("world_points_3d", []).append(
            None if world_point is None else _snapshot(world_point)
        )
        return world_point

    aff_net.get_centers = record_get_centers
    wrapper.get_world_pt = record_get_world_pt
    try:
        yield
    finally:
        _restore_instance_override(aff_net, "get_centers", *get_centers_override)
        _restore_instance_override(wrapper, "get_world_pt", *get_world_pt_override)
        if original_prepare_input is not None:
            _restore_instance_override(encoder, "_prepare_input", *prepare_input_override)
        for handle in handles:
            handle.remove()


def collect_single_observation(wrapper, raw_observation):
    """Run the wrapper once and return full, detached diagnostic values."""
    captured = {}
    with _record_wrapper_pipeline(wrapper, captured):
        wrapper_observation = wrapper.observation(raw_observation)

    sac_observation = wrapper.transform_obs(
        _tensor_observation(wrapper_observation), split="validation"
    )
    captured["sac_gripper_aff"] = _snapshot(wrapper_observation["gripper_aff"])
    captured["gripper_img_obs"] = _snapshot(wrapper_observation["gripper_img_obs"])
    captured["gripper_img_obs_sac"] = _snapshot(sac_observation["gripper_img_obs"])
    captured["gripper_depth_obs"] = _snapshot(wrapper_observation["gripper_depth_obs"])
    captured["gripper_depth_obs_sac"] = _snapshot(sac_observation["gripper_depth_obs"])
    captured["target_distance"] = _snapshot(wrapper_observation["target_distance"])

    captured.setdefault("hough_centers_2d", np.empty((0, 2), dtype=np.float32))
    captured.setdefault("center_projection_pixels", [])
    captured.setdefault("world_points_3d", [])
    return captured


def collect_replayed_inference(wrapper, gripper_img_obs):
    """Replay preprocessing and model inference from an exact captured CHW frame."""
    aff_net = wrapper.gripper_cam_aff_net
    if aff_net is None:
        raise RuntimeError("The gripper affordance model is disabled.")

    encoder = getattr(getattr(aff_net, "unet", None), "encoder", None)
    original_prepare_input = getattr(encoder, "_prepare_input", None)
    prepare_input_override = (
        _instance_override(encoder, "_prepare_input")
        if encoder is not None
        else (None, False)
    )
    captured = {"gripper_img_obs": _snapshot(gripper_img_obs)}

    if original_prepare_input is not None:
        def record_dinov3_input(value):
            prepared = original_prepare_input(value)
            captured["dinov3_model_input"] = _snapshot(prepared)
            return prepared

        encoder._prepare_input = record_dinov3_input

    try:
        image = torch.from_numpy(np.asarray(gripper_img_obs)).float()
        processed = wrapper.aff_transforms["gripper"](image)
        model_input = processed.unsqueeze(0)
        first_parameter = next(aff_net.parameters(), None)
        if first_parameter is not None:
            model_input = model_input.to(first_parameter.device)
        captured["affordance_model_input"] = _snapshot(model_input)

        with torch.no_grad():
            logits, probabilities, mask, directions = aff_net(model_input)
        captured["affordance_logits"] = _snapshot(logits)
        captured["affordance_probabilities"] = _snapshot(probabilities)
        captured["dinov3_mask_128"] = _snapshot(mask)
        captured["model_center_directions"] = _snapshot(directions)
        return captured
    finally:
        if original_prepare_input is not None:
            _restore_instance_override(
                encoder, "_prepare_input", *prepare_input_override
            )


def collect_same_tensor_inference(wrapper, affordance_model_input):
    """Repeat model inference without applying preprocessing again."""
    aff_net = wrapper.gripper_cam_aff_net
    if aff_net is None:
        raise RuntimeError("The gripper affordance model is disabled.")

    encoder = getattr(getattr(aff_net, "unet", None), "encoder", None)
    original_prepare_input = getattr(encoder, "_prepare_input", None)
    prepare_input_override = (
        _instance_override(encoder, "_prepare_input")
        if encoder is not None
        else (None, False)
    )
    model_input = torch.from_numpy(np.asarray(affordance_model_input)).float()
    first_parameter = next(aff_net.parameters(), None)
    if first_parameter is not None:
        model_input = model_input.to(first_parameter.device)
    captured = {"affordance_model_input": _snapshot(model_input)}

    if original_prepare_input is not None:
        def record_dinov3_input(value):
            prepared = original_prepare_input(value)
            captured["dinov3_model_input"] = _snapshot(prepared)
            return prepared

        encoder._prepare_input = record_dinov3_input

    try:
        with torch.no_grad():
            logits, probabilities, mask, directions = aff_net(model_input)
        captured["affordance_logits"] = _snapshot(logits)
        captured["affordance_probabilities"] = _snapshot(probabilities)
        captured["dinov3_mask_128"] = _snapshot(mask)
        captured["model_center_directions"] = _snapshot(directions)
        return captured
    finally:
        if original_prepare_input is not None:
            _restore_instance_override(
                encoder, "_prepare_input", *prepare_input_override
            )


def _array_comparison(runtime_value, replayed_value, atol, rtol):
    runtime_array = _snapshot(runtime_value)
    replayed_array = _snapshot(replayed_value)
    same_shape = runtime_array.shape == replayed_array.shape
    supports_nan = np.issubdtype(runtime_array.dtype, np.inexact) and np.issubdtype(
        replayed_array.dtype, np.inexact
    )
    exactly_equal = same_shape and (
        np.array_equal(runtime_array, replayed_array, equal_nan=True)
        if supports_nan
        else np.array_equal(runtime_array, replayed_array)
    )
    result = {
        "runtime": array_summary(runtime_array),
        "offline_replay": array_summary(replayed_array),
        "same_shape": same_shape,
        "exactly_equal": bool(exactly_equal),
        "within_tolerance": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
    }
    if not same_shape:
        return result

    if np.issubdtype(runtime_array.dtype, np.number) and np.issubdtype(
        replayed_array.dtype, np.number
    ):
        result["within_tolerance"] = bool(
            np.allclose(
                runtime_array,
                replayed_array,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
        )
        difference = np.abs(
            runtime_array.astype(np.float64) - replayed_array.astype(np.float64)
        )
        finite_difference = difference[np.isfinite(difference)]
        if finite_difference.size:
            result["max_abs_diff"] = float(finite_difference.max())
            result["mean_abs_diff"] = float(finite_difference.mean())
    else:
        result["within_tolerance"] = result["exactly_equal"]
    return result


def _missing_values(reference, candidate, names):
    return {
        name: {
            "runtime": name not in reference,
            "candidate": name not in candidate,
        }
        for name in names
        if name not in reference or name not in candidate
    }


def _compare_values(reference, candidate, names, atol, rtol):
    return {
        name: _array_comparison(
            reference[name], candidate[name], atol=atol, rtol=rtol
        )
        for name in names
        if name in reference and name in candidate
    }


def _all_within_tolerance(comparisons, names):
    return all(
        name in comparisons and comparisons[name]["within_tolerance"]
        for name in names
    )


def _all_exactly_equal(comparisons, names):
    return all(
        name in comparisons and comparisons[name]["exactly_equal"]
        for name in names
    )


def _mask_metrics(runtime_mask, replayed_mask):
    runtime_array = _snapshot(runtime_mask)
    replayed_array = _snapshot(replayed_mask)
    if runtime_array.shape != replayed_array.shape:
        return {
            "differing_pixel_count": None,
            "differing_fraction": None,
            "iou": None,
            "dice": None,
            "runtime_foreground_pixel_count": int(np.count_nonzero(runtime_array)),
            "offline_replay_foreground_pixel_count": int(
                np.count_nonzero(replayed_array)
            ),
        }

    differing_pixel_count = int(np.count_nonzero(runtime_array != replayed_array))
    runtime_foreground = runtime_array != 0
    replayed_foreground = replayed_array != 0
    intersection = int(np.count_nonzero(runtime_foreground & replayed_foreground))
    union = int(np.count_nonzero(runtime_foreground | replayed_foreground))
    foreground_total = int(runtime_foreground.sum() + replayed_foreground.sum())
    return {
        "differing_pixel_count": differing_pixel_count,
        "differing_fraction": (
            float(differing_pixel_count / runtime_array.size)
            if runtime_array.size
            else 0.0
        ),
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / foreground_total)
        if foreground_total
        else 1.0,
        "runtime_foreground_pixel_count": int(runtime_foreground.sum()),
        "offline_replay_foreground_pixel_count": int(replayed_foreground.sum()),
    }


def build_offline_runtime_comparison(
    runtime_captured,
    replayed_captured,
    same_tensor_captured,
    atol=1e-6,
    rtol=1e-5,
):
    """Separate preprocessing, same-input inference, and semantic replay checks."""
    preprocessing_names = (
        "gripper_img_obs",
        "affordance_model_input",
        "dinov3_model_input",
    )
    inference_names = (
        "affordance_logits",
        "affordance_probabilities",
        "dinov3_mask_128",
        "model_center_directions",
    )
    same_tensor_inference_names = ("dinov3_model_input",) + inference_names
    same_tensor_names = ("affordance_model_input",) + same_tensor_inference_names
    missing = {
        group: values
        for group, values in {
            "preprocessing": _missing_values(
                runtime_captured, replayed_captured, preprocessing_names
            ),
            "same_tensor_inference": _missing_values(
                runtime_captured, same_tensor_captured, same_tensor_names
            ),
            "end_to_end_replay": _missing_values(
                runtime_captured, replayed_captured, inference_names
            ),
        }.items()
        if values
    }
    preprocessing = _compare_values(
        runtime_captured, replayed_captured, preprocessing_names, atol, rtol
    )
    same_tensor = _compare_values(
        runtime_captured, same_tensor_captured, same_tensor_names, atol, rtol
    )
    end_to_end = _compare_values(
        runtime_captured, replayed_captured, inference_names, atol, rtol
    )
    mask_metrics = _mask_metrics(
        runtime_captured.get("dinov3_mask_128", np.empty(0)),
        replayed_captured.get("dinov3_mask_128", np.empty(0)),
    )
    preprocessing_matches = (
        "preprocessing" not in missing
        and _all_within_tolerance(preprocessing, preprocessing_names)
    )
    same_tensor_reproducible = (
        "same_tensor_inference" not in missing
        and _all_exactly_equal(same_tensor, same_tensor_names)
    )
    semantic_replay_equivalent = (
        "end_to_end_replay" not in missing
        and mask_metrics["differing_pixel_count"] == 0
    )
    return {
        "frame_source": "gripper_img_obs captured from the selected runtime observation",
        "atol": float(atol),
        "rtol": float(rtol),
        "preprocessing": {
            "values": preprocessing,
            "equivalent": preprocessing_matches,
        },
        "same_tensor_inference": {
            "input_source": "captured runtime affordance_model_input",
            "reproducibility_definition": "bitwise equality for input and outputs",
            "values": same_tensor,
            "reproducible": same_tensor_reproducible,
            "within_numerical_tolerance": _all_within_tolerance(
                same_tensor, same_tensor_names
            ),
            "semantically_reproducible": (
                "dinov3_mask_128" in same_tensor
                and same_tensor["dinov3_mask_128"]["exactly_equal"]
            ),
        },
        "end_to_end_replay": {
            "values": end_to_end,
            "mask_metrics": mask_metrics,
            "mask_metric_definition": "foreground is any nonzero class label",
            "numerically_equivalent": _all_within_tolerance(
                end_to_end, inference_names
            ),
            "semantically_equivalent": semantic_replay_equivalent,
        },
        "checks": {
            "missing_values": missing,
            "preprocessing_matches": preprocessing_matches,
            "same_tensor_inference_reproducible": same_tensor_reproducible,
            "semantic_replay_equivalent": semantic_replay_equivalent,
            "runtime_foreground_pixel_count": mask_metrics[
                "runtime_foreground_pixel_count"
            ],
            "offline_replay_foreground_pixel_count": mask_metrics[
                "offline_replay_foreground_pixel_count"
            ],
        },
    }


def array_summary(value):
    array = _snapshot(value)
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nan_count": 0,
        "posinf_count": 0,
        "neginf_count": 0,
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        nan_count = int(np.isnan(array).sum())
        posinf_count = int(np.isposinf(array).sum())
        neginf_count = int(np.isneginf(array).sum())
        finite_values = array[np.isfinite(array)]
        summary.update(
            {
                "min": float(finite_values.min()) if finite_values.size else None,
                "max": float(finite_values.max()) if finite_values.size else None,
                "mean": float(finite_values.mean()) if finite_values.size else None,
                "nan_count": nan_count,
                "posinf_count": posinf_count,
                "neginf_count": neginf_count,
            }
        )
        unique = np.unique(array)
        if unique.size <= 10 and not (nan_count or posinf_count or neginf_count):
            summary["unique_values"] = unique.tolist()
    return summary


def build_report(captured, encoder_type, policy_img_size):
    tensor_names = (
        "affordance_model_input",
        "dinov3_model_input",
        "affordance_logits",
        "affordance_probabilities",
        "dinov3_mask_128",
        "hough_mask",
        "model_center_directions",
        "center_directions_128",
        "sac_gripper_aff",
        "gripper_img_obs",
        "gripper_img_obs_sac",
        "gripper_depth_obs",
        "gripper_depth_obs_sac",
        "target_distance",
    )
    values = {
        name: array_summary(captured[name])
        for name in tensor_names
        if name in captured
    }

    centers = captured["hough_centers_2d"]
    pixels = captured["center_projection_pixels"]
    world_points = captured["world_points_3d"]
    values["hough_centers_2d"] = array_summary(centers)
    values["center_projection_pixels"] = array_summary(_stack_points(pixels, 2))
    values["world_points_3d"] = array_summary(
        _stack_points([point for point in world_points if point is not None], 3)
    )
    conversions = []
    for index, center in enumerate(centers):
        conversions.append(
            {
                "hough_center_2d": _snapshot(center).tolist(),
                "depth_pixel": pixels[index].tolist() if index < len(pixels) else None,
                "world_point_3d": (
                    world_points[index].tolist()
                    if index < len(world_points) and world_points[index] is not None
                    else None
                ),
            }
        )

    non_finite = {
        name: stats["nan_count"] + stats["posinf_count"] + stats["neginf_count"]
        for name, stats in values.items()
    }
    non_finite = {name: count for name, count in non_finite.items() if count}

    expected_affordance_size = 128 if encoder_type == "dinov3" else policy_img_size
    expected_shapes = {
        "affordance_model_input": [
            1,
            3 if encoder_type == "dinov3" else 1,
            expected_affordance_size,
            expected_affordance_size,
        ],
        "dinov3_mask_128": [1, expected_affordance_size, expected_affordance_size],
        "hough_mask": [1, expected_affordance_size, expected_affordance_size],
        "center_directions_128": [1, 2, expected_affordance_size, expected_affordance_size],
        "sac_gripper_aff": [1, policy_img_size, policy_img_size],
        "gripper_img_obs_sac": [3, policy_img_size, policy_img_size],
        "gripper_depth_obs_sac": [1, policy_img_size, policy_img_size],
        "target_distance": [1],
    }
    if encoder_type == "dinov3":
        expected_shapes["dinov3_model_input"] = [1, 3, 128, 128]
    shape_errors = {}
    for name, expected in expected_shapes.items():
        actual = values.get(name, {}).get("shape")
        if actual != expected:
            shape_errors[name] = {"expected": expected, "actual": actual}

    return {
        "encoder_type": encoder_type,
        "policy_img_size": policy_img_size,
        "coordinate_convention": {
            "hough_center_2d": "(y, x), as returned by Hough Voting",
            "depth_pixel": "pixel array passed unchanged to get_world_pt",
            "world_point_3d": "(x, y, z) in the robot/world frame",
        },
        "values": values,
        "center_conversions": conversions,
        "checks": {
            "all_finite": not non_finite,
            "non_finite_values": non_finite,
            "shapes_match": not shape_errors,
            "shape_errors": shape_errors,
            "hough_center_count": int(len(centers)),
            "valid_world_point_count": int(sum(point is not None for point in world_points)),
            "center_found": bool(len(centers)),
            "world_conversion_succeeded": bool(any(point is not None for point in world_points)),
        },
    }
