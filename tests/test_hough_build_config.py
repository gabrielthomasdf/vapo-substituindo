import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOUGH_SETUP = PROJECT_ROOT / "vapo" / "affordance" / "hough_voting" / "setup.py"


def _capture_extension(monkeypatch):
    captured = {}

    setuptools_module = ModuleType("setuptools")

    def record_setup(**kwargs):
        captured.update(kwargs)

    setuptools_module.setup = record_setup

    cpp_extension_module = ModuleType("torch.utils.cpp_extension")
    cpp_extension_module.BuildExtension = object()
    cpp_extension_module.CUDAExtension = lambda **kwargs: SimpleNamespace(**kwargs)
    torch_module = ModuleType("torch")
    torch_utils_module = ModuleType("torch.utils")
    torch_module.utils = torch_utils_module
    torch_utils_module.cpp_extension = cpp_extension_module

    monkeypatch.setitem(sys.modules, "setuptools", setuptools_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_module)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension_module)
    runpy.run_path(str(HOUGH_SETUP), run_name="__main__")
    return captured["ext_modules"][0]


@pytest.mark.parametrize(
    "configured_path,expected_path",
    [
        (None, "/usr/local/include/eigen3"),
        ("/opt/eigen3", "/opt/eigen3"),
    ],
)
def test_hough_build_uses_configured_eigen_path_with_original_fallback(
    monkeypatch, configured_path, expected_path
):
    if configured_path is None:
        monkeypatch.delenv("EIGEN3_INCLUDE_DIR", raising=False)
    else:
        monkeypatch.setenv("EIGEN3_INCLUDE_DIR", configured_path)

    extension = _capture_extension(monkeypatch)

    assert extension.include_dirs == [expected_path]
