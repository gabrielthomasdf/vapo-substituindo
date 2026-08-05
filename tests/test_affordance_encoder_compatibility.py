import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("segmentation_models_pytorch")

from omegaconf import OmegaConf

from vapo.affordance.affordance_model import AffordanceModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRIPPER_ORIGINAL_CONFIG = PROJECT_ROOT / "config" / "aff_model" / "gripper_cam.yaml"
STATIC_ORIGINAL_CONFIG = PROJECT_ROOT / "config" / "aff_model" / "static_cam.yaml"
DINO_CONFIG = PROJECT_ROOT / "config" / "aff_model" / "default.yaml"


def _assert_forward_shapes(outputs, batch_size, n_classes, height, width):
    affordance_logits, affordance_probs, affordance_mask, center_directions = outputs
    assert affordance_logits.shape == (batch_size, n_classes, height, width)
    assert affordance_probs.shape == (batch_size, n_classes, height, width)
    assert affordance_mask.shape == (batch_size, height, width)
    assert center_directions.shape == (batch_size, 2, height, width)


def _load_camera_config(config_path):
    camera_config = OmegaConf.load(config_path)
    return camera_config.cfg, camera_config.n_classes


def _load_dinov3_config():
    repo_dir = os.environ.get("DINOV3_REPO_DIR")
    weights_path = os.environ.get("DINOV3_WEIGHTS_PATH")
    if not repo_dir or not weights_path:
        pytest.skip("Set DINOV3_REPO_DIR and DINOV3_WEIGHTS_PATH to run the DINOv3 tests.")
    cfg = OmegaConf.load(DINO_CONFIG)
    # cfg_affordance.yaml supplies this value through Hydra composition.
    cfg.n_classes = 2
    return cfg


def _checkpoint_from_env(variable_name):
    checkpoint = os.environ.get(variable_name)
    if not checkpoint:
        pytest.skip(f"Set {variable_name} to a real checkpoint path to run this compatibility test.")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        pytest.fail(f"{variable_name} does not point to a file: {checkpoint_path}")
    return checkpoint_path


def _forward(model, input_shape):
    model.eval()
    device = next(model.parameters()).device
    input_tensor = torch.zeros(input_shape, dtype=torch.float32, device=device)
    with torch.no_grad():
        return model(input_tensor)


def test_legacy_config_without_encoder_fields_defaults_to_resnet18():
    cfg = OmegaConf.create({})
    assert AffordanceModel._resolve_encoder_type(cfg) == "resnet18"


def test_legacy_dinov3_config_without_encoder_type_defaults_to_dinov3():
    cfg = OmegaConf.create({"dinov3_cfg": {"model_name": "unused"}})
    assert AffordanceModel._resolve_encoder_type(cfg) == "dinov3"


def test_unknown_encoder_has_clear_error():
    cfg = OmegaConf.create({"encoder_type": "unknown"})
    with pytest.raises(ValueError, match="Unknown affordance encoder_type 'unknown'"):
        AffordanceModel._resolve_encoder_type(cfg)


def test_missing_dinov3_config_has_clear_error():
    with pytest.raises(ValueError, match="requires a dinov3_cfg section"):
        AffordanceModel._validate_dinov3_cfg(None)


@pytest.mark.parametrize(
    "config_path,input_shape",
    [
        (GRIPPER_ORIGINAL_CONFIG, (1, 1, 64, 64)),
        (STATIC_ORIGINAL_CONFIG, (1, 1, 200, 200)),
    ],
)
def test_original_model_structure_and_forward(config_path, input_shape):
    cfg, n_classes = _load_camera_config(config_path)
    model = AffordanceModel(cfg=cfg, input_channels=1, n_classes=n_classes)
    assert model.encoder_type == "resnet18"
    assert any(key.startswith("unet.encoder.") for key in model.state_dict())
    assert "center_direction_net.weight" in model.state_dict()

    outputs = _forward(model, input_shape)
    _assert_forward_shapes(outputs, input_shape[0], n_classes, input_shape[2], input_shape[3])


def test_dinov3_gripper_structure_and_forward():
    cfg = _load_dinov3_config()
    model = AffordanceModel(cfg=cfg, input_channels=3, n_classes=cfg.n_classes)
    assert model.encoder_type == "dinov3"
    assert any(key.startswith("unet.encoder.backbone.") for key in model.state_dict())
    assert any(key.startswith("unet.encoder.projections.") for key in model.state_dict())
    assert "center_direction_net.weight" in model.state_dict()

    input_shape = (1, 3, 128, 128)
    outputs = _forward(model, input_shape)
    _assert_forward_shapes(outputs, input_shape[0], cfg.n_classes, input_shape[2], input_shape[3])


def test_original_gripper_official_checkpoint_forward():
    checkpoint = _checkpoint_from_env("VAPO_GRIPPER_ORIGINAL_CKPT")
    cfg, n_classes = _load_camera_config(GRIPPER_ORIGINAL_CONFIG)
    model = AffordanceModel.load_from_checkpoint(
        str(checkpoint), cfg=cfg, input_channels=1, n_classes=n_classes, map_location="cpu"
    )
    outputs = _forward(model, (1, 1, 64, 64))
    _assert_forward_shapes(outputs, 1, n_classes, 64, 64)


def test_original_static_official_checkpoint_forward():
    checkpoint = _checkpoint_from_env("VAPO_STATIC_ORIGINAL_CKPT")
    cfg, n_classes = _load_camera_config(STATIC_ORIGINAL_CONFIG)
    model = AffordanceModel.load_from_checkpoint(
        str(checkpoint), cfg=cfg, input_channels=1, n_classes=n_classes, map_location="cpu"
    )
    outputs = _forward(model, (1, 1, 200, 200))
    _assert_forward_shapes(outputs, 1, n_classes, 200, 200)


def test_dinov3_gripper_checkpoint_forward():
    checkpoint = _checkpoint_from_env("VAPO_GRIPPER_DINOV3_CKPT")
    cfg = _load_dinov3_config()
    model = AffordanceModel.load_from_checkpoint(
        str(checkpoint), cfg=cfg, input_channels=3, n_classes=cfg.n_classes, map_location="cpu"
    )
    outputs = _forward(model, (1, 3, 128, 128))
    _assert_forward_shapes(outputs, 1, cfg.n_classes, 128, 128)
