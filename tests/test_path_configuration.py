import os
from pathlib import Path

import pytest

pytest.importorskip("omegaconf")

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERAL_PATHS = PROJECT_ROOT / "config" / "paths" / "general_paths.yaml"


def _load_paths():
    return OmegaConf.create({"paths": OmegaConf.load(GENERAL_PATHS)})


def test_general_paths_preserve_legacy_layout_without_vapo_root(monkeypatch):
    monkeypatch.delenv("VAPO_ROOT", raising=False)

    cfg = _load_paths()

    assert os.path.normpath(cfg.paths.vapo_path) == os.path.normpath("../vapo/")
    assert os.path.normpath(cfg.paths.vr_data) == os.path.normpath("../vapo/VREnv/data/")
    assert os.path.normpath(cfg.paths.trained_models) == os.path.normpath("../vapo/trained_models/")
    assert os.path.normpath(cfg.paths.datasets) == os.path.normpath("../vapo/datasets/")


def test_general_paths_use_optional_vapo_root_for_all_derived_paths(monkeypatch):
    monkeypatch.setenv("VAPO_ROOT", "/srv/vapo-substituindo")

    cfg = _load_paths()

    assert os.path.normpath(cfg.paths.vapo_path) == os.path.normpath("/srv/vapo-substituindo")
    assert os.path.normpath(cfg.paths.vr_data) == os.path.normpath("/srv/vapo-substituindo/VREnv/data/")
    assert os.path.normpath(cfg.paths.trained_models) == os.path.normpath(
        "/srv/vapo-substituindo/trained_models/"
    )
    assert os.path.normpath(cfg.paths.datasets) == os.path.normpath("/srv/vapo-substituindo/datasets/")
