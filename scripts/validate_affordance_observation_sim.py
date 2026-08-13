import json
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig

from vapo.affordance.affordance_model import AffordanceModel
from vapo.wrappers.affordance.aff_wrapper_sim import AffordanceWrapperSim
from vapo.wrappers.affordance.simulation_diagnostics import (
    collect_positioned_observation,
)
from vapo.wrappers.play_table_rl import PlayTableRL


def _arrays_for_npz(captured):
    arrays = {
        name: value
        for name, value in captured.items()
        if isinstance(value, np.ndarray)
    }
    for index, pixel in enumerate(captured["center_projection_pixels"]):
        arrays["center_projection_pixel_%02d" % index] = pixel
    for index, point in enumerate(captured["world_points_3d"]):
        if point is not None:
            arrays["world_point_3d_%02d" % index] = point
    return arrays


@hydra.main(config_path="../config", config_name="validate_affordance_observation_sim")
def main(cfg):
    env = PlayTableRL(viz=cfg.viz_obs, save_images=cfg.save_images, **cfg.env)
    wrapper = None
    try:
        wrapper = AffordanceWrapperSim(
            env,
            max_ts=cfg.agent.learn_config.max_episode_length,
            train=False,
            affordance_cfg=cfg.affordance,
            **cfg.env_wrapper,
        )
        encoder_type = AffordanceModel._resolve_encoder_type(
            cfg.affordance.gripper_cam.hyperparameters.cfg
        )
        captured, report = collect_positioned_observation(
            wrapper,
            env,
            encoder_type=encoder_type,
            policy_img_size=int(cfg.env_wrapper.img_size),
            observation_attempts=int(cfg.observation_attempts),
        )

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
            raise RuntimeError(
                "Positioned simulation observation validation failed; "
                "see observation_report.json."
            )
    finally:
        if wrapper is not None:
            wrapper.close()
        else:
            env.close()


if __name__ == "__main__":
    main()
