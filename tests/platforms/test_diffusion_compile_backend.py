# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend selection tests; no Ascend hardware or MindIE-SD installation needed."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def npu_platform(monkeypatch):
    # Load the real Omni NPU class without importing vllm-ascend's device runtime.
    ascend = ModuleType("vllm_ascend.platform")
    ascend.NPUPlatform = type("NPUPlatform", (), {})
    monkeypatch.setitem(sys.modules, "vllm_ascend.platform", ascend)
    path = Path(__file__).resolve().parents[2] / "vllm_omni/platforms/npu/platform.py"
    spec = importlib.util.spec_from_file_location("_test_npu_compile_platform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NPUOmniPlatform


@pytest.fixture
def compile_config():
    return SimpleNamespace(
        diffusion_compile_backend="mindiesd",
        model_class_name="QwenImagePipeline",
        diffusion_compile_granularity="regional",
        diffusion_compile_dynamic=False,
        dtype=torch.bfloat16,
        quantization_config=None,
        parallel_config=SimpleNamespace(
            world_size=1,
            vae_patch_parallel_size=1,
            text_encoder_tp_size=1,
            use_hsdp=False,
            enable_expert_parallel=False,
        ),
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
        cache_backend="none",
        lora_path=None,
        enable_sleep_mode=False,
    )


@pytest.fixture
def mindiesd(monkeypatch):
    module = ModuleType("mindiesd.compilation")
    module.CompilationConfig = SimpleNamespace(aclgraph_only=False, aclgraph_with_compile=False)
    module.MindieSDBackend = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", module)
    return module


@pytest.mark.parametrize(
    "platform", [OmniPlatformEnum.CUDA, OmniPlatformEnum.ROCM, OmniPlatformEnum.XPU, OmniPlatformEnum.MUSA]
)
@pytest.mark.parametrize("requested", ["auto", "inductor"])
def test_inductor_platforms_keep_backend(monkeypatch, platform, requested):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", platform, raising=False)
    monkeypatch.setattr(OmniPlatform, "supports_torch_inductor", lambda: True)
    assert (
        OmniPlatform.get_diffusion_compile_backend(SimpleNamespace(diffusion_compile_backend=requested)) == "inductor"
    )


def test_platform_rejects_foreign_backend(monkeypatch):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", OmniPlatformEnum.CUDA, raising=False)
    with pytest.raises(ValueError, match="not supported"):
        OmniPlatform.get_diffusion_compile_backend(SimpleNamespace(diffusion_compile_backend="mindiesd"))


def test_unavailable_inductor_is_not_passed_as_none(monkeypatch):
    monkeypatch.setattr(OmniPlatform, "_omni_enum", OmniPlatformEnum.UNSPECIFIED, raising=False)
    monkeypatch.setattr(OmniPlatform, "supports_torch_inductor", lambda: False)
    assert OmniPlatform.get_diffusion_compile_backend(SimpleNamespace(diffusion_compile_backend="auto")) is None
    with pytest.raises(ValueError, match="not supported"):
        OmniPlatform.get_diffusion_compile_backend(SimpleNamespace(diffusion_compile_backend="inductor"))


def test_npu_auto_does_not_import_mindiesd(npu_platform, compile_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", None)
    compile_config.diffusion_compile_backend = "auto"
    assert npu_platform.get_diffusion_compile_backend(compile_config) is None


def test_npu_resolves_backend_without_inductor(npu_platform, compile_config, mindiesd):
    assert npu_platform.supports_torch_inductor() is False
    assert npu_platform.get_diffusion_compile_backend(compile_config) is mindiesd.MindieSDBackend.return_value
    mindiesd.MindieSDBackend.assert_called_once_with()
    assert compile_config.diffusion_compile_backend == "mindiesd"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("diffusion_compile_backend", "inductor", "requires"),
        ("model_class_name", "QwenImageEditPipeline", "QwenImagePipeline only"),
        ("diffusion_compile_granularity", "full", "regional"),
        ("diffusion_compile_dynamic", True, "diffusion_compile_dynamic=False"),
        ("dtype", torch.float16, "BF16"),
        ("quantization_config", object(), "unquantized"),
        ("enable_cpu_offload", True, "offload"),
        ("enable_layerwise_offload", True, "offload"),
        ("enable_distributed_layerwise_offload", True, "offload"),
        ("cache_backend", "cache_dit", "cache_backend"),
        ("lora_path", "adapter", "LoRA"),
        ("enable_sleep_mode", True, "sleep"),
    ],
)
def test_npu_rejects_unsupported_config(npu_platform, compile_config, mindiesd, field, value, message):
    setattr(compile_config, field, value)
    with pytest.raises(ValueError, match=message):
        npu_platform.get_diffusion_compile_backend(compile_config)
    mindiesd.MindieSDBackend.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("world_size", 2),
        ("vae_patch_parallel_size", 2),
        ("text_encoder_tp_size", 2),
        ("use_hsdp", True),
        ("enable_expert_parallel", True),
    ],
)
def test_npu_rejects_parallelism(npu_platform, compile_config, field, value):
    setattr(compile_config.parallel_config, field, value)
    with pytest.raises(ValueError, match="single NPU"):
        npu_platform.get_diffusion_compile_backend(compile_config)


def test_npu_missing_dependency_has_actionable_error(npu_platform, compile_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "mindiesd.compilation", None)
    with pytest.raises(RuntimeError, match="could not be imported") as error:
        npu_platform.get_diffusion_compile_backend(compile_config)
    assert isinstance(error.value.__cause__, ImportError)


@pytest.mark.parametrize("flag", ["aclgraph_only", "aclgraph_with_compile"])
def test_npu_rejects_aclgraph_without_overriding_global_config(npu_platform, compile_config, mindiesd, flag):
    setattr(mindiesd.CompilationConfig, flag, True)
    with pytest.raises(ValueError, match=flag):
        npu_platform.get_diffusion_compile_backend(compile_config)
    assert getattr(mindiesd.CompilationConfig, flag) is True
    mindiesd.MindieSDBackend.assert_not_called()


def test_npu_rejects_incompatible_config_api(npu_platform, compile_config, mindiesd):
    del mindiesd.CompilationConfig.aclgraph_only
    with pytest.raises(RuntimeError, match="compilation API"):
        npu_platform.get_diffusion_compile_backend(compile_config)


def test_npu_backend_init_failure_preserves_cause(npu_platform, compile_config, mindiesd):
    mindiesd.MindieSDBackend.side_effect = TypeError("incompatible constructor")
    with pytest.raises(RuntimeError, match="Failed to initialize") as error:
        npu_platform.get_diffusion_compile_backend(compile_config)
    assert isinstance(error.value.__cause__, TypeError)
