import json
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig

from vapo.affordance.affordance_model import AffordanceModel
from vapo.wrappers.affordance.aff_wrapper_real_world import AffordanceWrapperRealWorld
from vapo.wrappers.affordance.observation_diagnostics import (
    build_report,
    collect_single_observation,
)
from vapo.wrappers.real_world.panda_tabletop_wrapper import PandaEnvWrapper


def _arrays_for_npz(captured):
    arrays = {}
    for name, value in captured.items():
        if isinstance(value, np.ndarray):
            arrays[name] = value
    for index, pixel in enumerate(captured["center_projection_pixels"]):
        arrays["center_projection_pixel_%02d" % index] = pixel
    for index, point in enumerate(captured["world_points_3d"]):
        if point is not None:
            arrays["world_point_3d_%02d" % index] = point
    return arrays


@hydra.main(
    config_path="../../config",
    config_name="validate_affordance_observation",
)
def main(cfg):
    robot = hydra.utils.instantiate(cfg.robot)
    raw_env = hydra.utils.instantiate(cfg.robot_env, robot=robot)
    panda_env = PandaEnvWrapper(**cfg.panda_env_wrapper, env=raw_env)
    wrapper = None
    try:
        wrapper_args = cfg.env_wrapper.copy()
        # Older real-world configs may already carry this constructor flag.
        wrapper_args.pop("real_world", None)
        wrapper = AffordanceWrapperRealWorld(
            panda_env,
            max_ts=cfg.agent.learn_config.max_episode_length,
            train=False,
            affordance_cfg=cfg.affordance,
            save_images=False,
            real_world=True,
            **wrapper_args,
        )

        # get_obs() only acquires sensor/robot state. It does not reset, move,
        # open the gripper, execute an action, or instantiate SAC.
        raw_observation = panda_env.get_obs()
        captured = collect_single_observation(wrapper, raw_observation)
        encoder_type = AffordanceModel._resolve_encoder_type(
            cfg.affordance.gripper_cam.hyperparameters.cfg
        )
        report = build_report(captured, encoder_type, int(cfg.env_wrapper.img_size))

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "observation_report.json"
        arrays_path = output_dir / "observation_values.npz"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(str(arrays_path), **_arrays_for_npz(captured))

        print(json.dumps(report, indent=2))
        print("Full values: %s" % arrays_path)
        print("Summary: %s" % report_path)

        checks = report["checks"]
        passed = (
            checks["all_finite"]
            and checks["shapes_match"]
            and checks["center_found"]
            and checks["world_conversion_succeeded"]
        )
        if not passed:
            raise RuntimeError("Single-observation wrapper validation failed; see observation_report.json.")
    finally:
        if wrapper is not None:
            wrapper.close()
        else:
            raw_env.close()


if __name__ == "__main__":
    main()
