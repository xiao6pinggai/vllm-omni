# Regional Compilation

Regional compilation applies `torch.compile` to the repeated transformer blocks
declared by a diffusion model. It is the default compilation scope when
diffusion inference runs without `--enforce-eager`.

## Configuration

Dynamic compilation is enabled by default so the compiled regions can handle
mixed resolutions. For a fixed-shape workload, disable it explicitly:

```bash
vllm serve <model> --omni --no-diffusion-compile-dynamic
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    diffusion_compile_granularity: regional
    diffusion_compile_dynamic: false
```

For an experimental whole-transformer compile scope, set
`--diffusion-compile-granularity full` or use
`diffusion_compile_granularity: full` in the stage configuration. Full scope may
still contain graph breaks; it does not force one graph. It is rejected when
HSDP, sequence parallelism, CPU offload, or layerwise offload is enabled. Use
regional scope with those features.

These settings control the generic model-runner compilation path. Pipelines
that provide their own `setup_compile()` implementation manage their compilation
policy independently. Compilation is lazy, so backend or graph errors can first
surface on the initial request.

Use `--enforce-eager` to disable the model runner's generic compile setup.
Pipelines that compile internally define their own eager-mode behavior.

## Experimental Ascend NPU backend

`diffusion_compile_backend` defaults to `auto`: existing Inductor platforms
keep using Inductor, and NPU keeps running eagerly. To opt into MindIE-SD
compilation on NPU:

```bash
vllm serve Qwen/Qwen-Image --omni --dtype bfloat16 \
    --diffusion-compile-backend mindiesd \
    --diffusion-compile-granularity regional \
    --no-diffusion-compile-dynamic --cache-backend none
```

The same fields are accepted by the Python API and per-stage deploy config:

```yaml
stages:
  - stage_id: 0
    enforce_eager: false
    diffusion_compile_backend: mindiesd
    diffusion_compile_granularity: regional
    diffusion_compile_dynamic: false
    cache_backend: none
```

This first integration is limited to the unquantized BF16 `QwenImagePipeline`,
one NPU, regional compilation and static shapes. CPU/layerwise offload,
HSDP/EP, diffusion cache backends, LoRA and sleep mode are rejected.
Use one request at a time with batch size 1 and fixed resolution for initial
validation. Input shapes also depend on prompt length and guidance settings;
`dynamic=False` does not guarantee that different requests reuse one graph.
The full model and its compilation workspace must fit on the device.

MindIE-SD is an optional NPU dependency, not installed for other platforms.
Use a build compatible with the target PyTorch, torch_npu and CANN versions.
The build must export `MindieSDBackend` and `CompilationConfig` from
`mindiesd.compilation`. This integration requires both
`CompilationConfig.aclgraph_only` and `CompilationConfig.aclgraph_with_compile`
to exist and be false. It checks these process-global settings rather than
overriding them. See the
[MindIE-SD compilation documentation](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docs/zh/features/compilation.md).
An NPU-tested dependency version matrix is not yet established.

Backend construction happens inside the worker, before loading weights so
MindIE-SD can register its CANN custom operators early. Explicit backend
selection fails clearly on missing dependencies, unsupported configuration or
synchronous compile setup errors. It does not silently fall back to eager.
`--enforce-eager` bypasses backend resolution; `auto` retains the previous
setup-failure fallback policy. Pipelines with their own `setup_compile()`
continue to own their policy under `auto` and reject explicit backend overrides.

### NPU validation checklist

CPU tests cover backend selection, argument forwarding and lazy Dynamo backend
invocation, not the correctness or performance of MindIE-SD on Qwen-Image.
Before treating this experimental path as validated:

1. Record the model revision, Omni/vLLM/vLLM-Ascend revisions, MindIE-SD build,
   PyTorch/torch_npu/CANN versions, device model and available memory.
2. Run eager and compiled servers separately, using the same model, prompt,
   seed, inference steps, resolution and guidance. For eager, add
   `--enforce-eager` to the command above. Do not enable offload or cache.
3. Send identical requests sequentially, separating first-request latency from
   multiple warm requests. The
   [text-to-image example](../examples/online_serving/text_to_image.md)
   describes requests. Capture `TORCH_LOGS=graph_breaks,recompiles` server logs
   and MindIE-SD backend debug logs to confirm actual graph processing.
   A "selected backend" or "configured for lazy compile" message is not proof.
4. Compare intermediate denoising outputs/latents and final images against eager
   with identical initial noise; record numerical error as well as visual quality.
5. Record warm end-to-end latency, synchronized DiT step latency and peak NPU
   memory. Inspect graph breaks and recompilations before claiming a speedup.
   Server startup warmup may already consume some compilation cost.
6. Test changes in prompt length and resolution separately. Runtime backend
   failures propagate; the runner does not retry a partially executed request
   eagerly.

ACLGraph, full compilation, other pipelines and offload/cache combinations
require separate NPU validation and are outside this first integration.
