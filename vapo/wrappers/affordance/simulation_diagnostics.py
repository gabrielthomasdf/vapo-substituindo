import numpy as np

from vapo.wrappers.affordance.observation_diagnostics import (
    build_report,
    collect_single_observation,
)


def _optional_list(value):
    if value is None:
        return None
    return np.asarray(value).copy().tolist()


def collect_positioned_observation(
    wrapper,
    env,
    encoder_type,
    policy_img_size,
    observation_attempts=1,
):
    """Move the simulated robot to its target and diagnose a fresh observation."""
    if observation_attempts < 1:
        raise ValueError("observation_attempts must be at least 1.")

    curr_detected_obj_before = _optional_list(wrapper.curr_detected_obj)
    env.reset(eval=True)
    target_pos, _ = env.get_target_pos()
    target_pos = np.asarray(target_pos).copy()

    wrapper.curr_detected_obj = target_pos.copy()
    curr_detected_obj_after_seed = _optional_list(wrapper.curr_detected_obj)
    env.move_to_target(target_pos)

    attempts = []
    selected_captured = None
    selected_report = None
    selected_raw_observation = None
    for attempt_index in range(observation_attempts):
        raw_observation = env.get_obs()
        captured = collect_single_observation(wrapper, raw_observation)
        report = build_report(captured, encoder_type, policy_img_size)
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "checks": report["checks"].copy(),
            }
        )
        selected_captured = captured
        selected_report = report
        selected_raw_observation = raw_observation
        if report["checks"]["center_found"] and report["checks"]["world_conversion_succeeded"]:
            break

    robot_position = np.asarray(selected_raw_observation["robot_obs"][:3]).copy()
    selected_report["positioning"] = {
        "target_pos": target_pos.tolist(),
        "robot_position_after_move_to_target": robot_position.tolist(),
        "robot_target_distance_after_move_to_target": float(
            np.linalg.norm(robot_position - target_pos)
        ),
        "curr_detected_obj_before": curr_detected_obj_before,
        "curr_detected_obj_after_seed": curr_detected_obj_after_seed,
        "curr_detected_obj_after_observation": _optional_list(wrapper.curr_detected_obj),
    }
    selected_report["observation_attempts"] = {
        "requested": int(observation_attempts),
        "performed": len(attempts),
        "selected": len(attempts),
        "results": attempts,
    }
    return selected_captured, selected_report
