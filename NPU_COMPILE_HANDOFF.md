# NPU diffusion compile：本地 / 服务器 agent 交接

本文件是本次实验性 MindIE-SD 接入的交接入口及异步交流记录。
请先读正文，再读末尾最新记录；不要把“代码已接入”理解为“NPU 已验证”。

## 1. 当前状态与代码基线

- 本地已实现接入，CPU 隔离测试通过；尚无 NPU 正确性、显存、性能结论。
- Omni 基线：`2ce15ab7a023d4fb800629e36c06bbe75a4324e2`。
- 本地 vLLM：`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`。
- 本地 vLLM-Ascend：`875cb553ad536c3bed13f2718fe8e4dd067bab0d`。
- 本次只修改 Omni；以上三者是源码基线，不是已通过 NPU 验收的兼容版本矩阵。
- 交接远端：`origin` = `https://github.com/xiao6pinggai/vllm-omni.git`，分支 `main`。
  不向 `upstream` 推送，不强推。服务器 remote 名称可以不同，应核对 URL。
- 本文档与初版代码一起提交；用 `git log --oneline -- NPU_COMPILE_HANDOFF.md`
  查看文档记录所属提交，不在提交自身内部硬编码其最终 hash。

## 2. 改动入口

| 文件 | 职责 |
| --- | --- |
| `vllm_omni/platforms/interface.py` | 新增 diffusion 专用 `get_diffusion_compile_backend()`，不覆盖 vLLM 的 `get_compile_backend()` |
| `vllm_omni/platforms/npu/platform.py` | 实验配置校验、可选依赖导入、ACLGraph 状态检查、worker 内创建 MindIE-SD backend |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | 加载权重前解析 backend；向 regional/full 透传；区分 auto 与显式选择的错误处理 |
| `vllm_omni/diffusion/data.py` | `diffusion_compile_backend` 配置及合法值校验 |
| `vllm_omni/config/{omni_config,stage_config}.py` | 结构化配置、部署配置透传 |
| `vllm_omni/engine/{arg_utils,async_omni_engine}.py` | engine args 和默认 diffusion stage 透传 |
| `vllm_omni/entrypoints/cli/serve.py` | `--diffusion-compile-backend` CLI |
| `tests/platforms/test_diffusion_compile_backend.py` | 新增平台选择、NPU 限制、依赖和错误路径测试 |
| `tests/diffusion/{test_compile,test_diffusion_model_runner,test_diffusion_engine}.py` | backend 透传、真实 CPU Dynamo、runner 和配置测试 |
| `tests/config/test_omni_config.py`、`tests/entrypoints/test_async_omni_diffusion_config.py` | 配置与 CLI 路由测试 |
| `docs/user_guide/diffusion/regional_compilation.md` | 使用方法和 NPU 验收说明 |

没有改动 `diffusion/compile.py` 的编译算法：它原本就透传 `**compile_kwargs`。
没有改动 Qwen-Image forward、attention、scheduler、VAE、text encoder 或底层 compiler。

调用关系：

```text
load_model() [enforce_eager=False]
  -> platform.get_diffusion_compile_backend(od_config)
  -> worker-local MindieSDBackend()，先于权重加载导入 CANN 自定义算子
  -> 加载 pipeline
  -> _compile_model(backend)
  -> _compile_transformer(..., backend=backend)
  -> regionally_compile(..., backend=backend, dynamic=False)
  -> torch.compile(QwenImageTransformerBlock.forward, ...)
  -> 首次 forward 才真正交给 backend 编译
```

## 3. 行为约定与实验边界

- `diffusion_compile_backend=auto` 是默认值。原 Inductor 平台继续使用 Inductor；
  NPU 默认仍 eager，不因安装 MindIE-SD 自动启用 compile。
- 只有显式选择 `mindiesd` 才进入本次 NPU 路径。
- 第一版校验：`QwenImagePipeline`、未量化 BF16、regional、`dynamic=False`、
  单 NPU；拒绝 HSDP/EP、VAE/文本编码器并行、CPU/layerwise offload、
  diffusion cache backend、LoRA、sleep mode。
- 单请求、batch=1、固定分辨率是第一轮验收基线，不是对所有请求形状的运行时限制。
  prompt 长度、CFG 分支等仍可能触发新图；`dynamic=False` 不代表不会重编译。
- `mindiesd.compilation` 必须导出 `MindieSDBackend` 和 `CompilationConfig`；
  后者必须有 `aclgraph_only`、`aclgraph_with_compile`，且两者为 false。
  代码检查这些全局设置，但不覆盖它们；当前未锁定真机验证过的 MindIE-SD wheel。
- 显式选择 backend 时，配置、依赖和同步 setup 错误明确失败；不会静默 eager。
  auto 保留既有同步 setup 失败回退。首次 forward 的懒编译错误直接传播，
  不通过捕获任意异常来重新执行已部分运行的请求。
- `--enforce-eager` 跳过本次 compile backend 解析；attention 自身仍可能使用 MindIE-SD。
- 自定义 `pipeline.setup_compile()` 在 auto 下保持原有行为，拒绝显式 backend 覆盖。
- 不要直接把 `supports_torch_inductor()` 改成 true；不要把 `backend=None` 传给
  `torch.compile` 当默认 Inductor；不要替换语言模型的 AscendCompiler。
- 如需放宽限制，先记录具体阻碍和验证证据，再做独立、可复核的修改。

## 4. 本地验证：哪些通过，哪些没有

- Python 3.12.13、PyTorch 2.10.0+cpu、pytest 9.1.1。
- 53 项隔离 CPU 测试通过：平台与 regional 逻辑从源码加载，NPU 依赖使用 mock；
  runner 选取原始方法和测试的 AST 执行，隔离 Windows 上无关运行依赖。
  包含真实 `torch.compile` 首次调用 backend、同形状复用及懒编译失败传播。
- Ruff 0.14.10 lint/format 通过；15 个 Python 文件语法检查通过；
  `git diff --check` 和新增测试文件 marker 检查通过。
- 标准仓库 pytest 已尝试，测试收集因缺少 `aenum` 等完整运行依赖失败。
  CLI / 完整配置集成测试没有完成实际运行，未运行全仓库测试或完整 type check。
- 没有执行 NPU、GPU、Qwen-Image 端到端验证，没有性能或精度提升结论。
- 本地隔离 harness 在本地工作区，不属于本仓库；服务器应运行下面的标准测试，
  不应将本地隔离结果作为真实软件栈已兼容的证明。

## 5. 服务器 agent 接手顺序

### 5.1 同步与环境检查

先检查本地改动并核对远端。工作区不干净时，不要 reset/覆盖未提交工作；
同步有分歧时停止并记录情况，不强推。确认当前分支为 main 且 origin URL 正确后：

```bash
git status --short
git remote -v
git branch --show-current
git pull --ff-only origin main
git rev-parse HEAD
```

确认推理进程实际加载的是这个 checkout，而不是旧 wheel 或别处安装的源码：

```bash
python -c "import vllm_omni; print(vllm_omni.__file__)"
vllm serve --help | grep diffusion-compile
npu-smi info
```

记录 Python、torch、torch_npu、CANN、MindIE-SD、vLLM、Ascend、Omni 的版本/提交，
设备型号和可用显存。不要为通过 import 就盲目升级整套依赖。
可以在独立进程做 API 检查（通过只说明可导入，不说明真实图可执行）：

```bash
python - <<'PY'
import inspect
from mindiesd.compilation import CompilationConfig, MindieSDBackend
print('backend file:', inspect.getfile(MindieSDBackend))
print('constructor:', inspect.signature(MindieSDBackend))
for flag in ('aclgraph_only', 'aclgraph_with_compile'):
    print(flag, getattr(CompilationConfig, flag, '<missing>'))
PY
```

### 5.2 标准测试

在已具备项目运行/测试依赖的环境内：

```bash
python -m pytest -q \
  tests/platforms/test_diffusion_compile_backend.py \
  tests/diffusion/test_compile.py \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/diffusion/test_diffusion_engine.py \
  tests/config/test_omni_config.py \
  tests/entrypoints/test_async_omni_diffusion_config.py \
  -k compile
```

若失败，区分测试收集/依赖问题、接入代码问题和实际 backend 问题，保留首个完整堆栈。

### 5.3 NPU 最小对照实验

先单独启动 eager 服务，确认同一模型在当前环境能正常出图，再停止它启动编译服务。
不要同时占用同一 NPU。编译服务示例（设备编号和模型路径按现场调整）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 TORCH_LOGS=graph_breaks,recompiles \
vllm serve Qwen/Qwen-Image --omni --dtype bfloat16 --port 8091 \
  --diffusion-compile-backend mindiesd \
  --diffusion-compile-granularity regional \
  --no-diffusion-compile-dynamic --cache-backend none
```

eager 对照使用同样命令增加 `--enforce-eager`。确保启动/部署配置没有启用
offload、并行、LoRA、sleep 或缓存。模型及编译工作区需要能放入单卡显存；
如 OOM，记录显存和失败阶段，不把开启 offload 后的结果算作该基线验收通过。

请求格式参考 [text-to-image 示例](examples/online_serving/text_to_image/README.md)。
固定 prompt、seed、分辨率、steps、guidance，单请求 batch=1；先测一次，
再顺序重复多次。区分服务启动预热、首次请求和稳定态开销。

必须回答：

1. 真实 backend 是否处理了 FX 图？选中 backend / lazy wrapper 的 INFO 日志不足以证明。
   记录 MindIE-SD backend 调试日志或等价可观测证据，注明是否实际触发融合。
2. 是否出现 graph break、持续重编译、意外 eager？分别记录原因，不能只看“不报错”。
3. 相同初始 noise 下的中间 denoising 输出/latents 和最终图片是否与 eager 一致？
   给出使用的数值指标、误差和视觉检查；仅相同 seed 不替代初始状态核查。
4. 首次耗时、稳定态端到端耗时、同步计时的 DiT step 耗时、峰值 NPU 显存是多少？
   重复次数和同步方式一起记录，不用首次请求声称性能提升。
5. 固定基线通过后，再单独改变 prompt 长度/分辨率检查重编译和正确性。

## 6. 双向交流约定

- 每次完成修改、测试或发现阻塞，都在末尾追加记录，不覆盖其他 agent 的历史记录。
- 时间写完整日期、时分秒、时区；署名至少包括 agent 名称与角色（本地开发 / NPU 验证）。
- 每条记录包含：输入代码提交、环境、执行命令、结果/日志路径、修改文件、待对方处理的问题。
- 明确区分“观察事实”“推测”“尚未验证”，不要用编译包装成功代替 backend 执行成功。
- 大日志、权重、图片、二进制缓存不提交 Git；文档写脱敏摘要和可定位路径。
  不写密钥、token、登录凭据或敏感请求数据。
- 只提交本任务相关文件；记录代码修改对应提交 hash（当前提交自身可在后续记录中补充）。
- 推送前确认远端无新提交；只正常快进同步。如存在冲突，人工核对并保留双方记录。
- 文档不会自动同步或唤醒另一侧 agent；提交/push 后需由用户通知另一侧 pull 并阅读最新记录。

推荐追加模板：

```text
### YYYY-MM-DD HH:mm:ss +08:00 | Agent 名称（角色）
- 输入代码提交：
- 环境与设备：
- 本次目标：
- 执行命令：
- 观察结果与日志路径：
- 推测 / 尚未验证：
- 本次修改文件及提交：
- 需要另一侧处理：
- 下一步：
- 署名：Agent 名称（角色）
```

## 7. 追加记录

### 2026-08-31 14:27:21 +08:00 | Codex（本地开发 agent）

- 输入代码提交：Omni `2ce15ab7a023d4fb800629e36c06bbe75a4324e2`，其余基线见第 1 节。
- 环境与设备：Windows；CPU PyTorch 2.10.0；无 NPU。
- 本次目标：按用户确认的方案完成最小实验接入，并准备服务器交接。
- 观察结果：第 4 节列明的 53 项隔离测试及静态检查通过；未完成真实软件栈/NPU 验证。
- 修改文件：第 2 节各文件及本文档，作为同一初版提交；未修改 vLLM/vLLM-Ascend。
- 推送准备：已查询 origin/main，仍为上述 Omni 基线，具备正常快进条件；不使用强推。
- 需要服务器 agent 处理：先检查 MindIE-SD 实际 API/版本，再运行标准测试与 eager/compile 对照，
  尤其确认 backend 的真实调用、数值正确性和稳定态性能。
- 若接口或算子不兼容：追加完整环境、首次失败阶段及堆栈摘要；不要直接删掉配置检查或吞掉异常。
- 下一步：本地提交并正常推送；服务器收到后执行第 5 节，将结果按第 6 节追加并回传。
- 署名：Codex（本地开发 agent，OpenAI）；时间：2026-08-31 14:27:21 +08:00。

### 2026-08-31 22:45:25 +08:00 | opencode（NPU 验证 agent，本机）
[root@vllm-omni-lhg lhg]# export GLOO_SOCKET_IFNAME=lo

ASCEND_RT_VISIBLE_DEVICES=0 \
python3 -u -m vllm_omni.entrypoints.cli.main serve \
  /home/DiffusionWeights/Qwen-Image/ \
  --host 127.0.0.1 \
  --port 33547 \
  --omni \
  --trust-remote-code \
  --disable-log-stats \
  --tensor-parallel-size 1 \
  --diffusion-compile-backend mindiesd \
  --diffusion-compile-granularity regional \
  --no-diffusion-compile-dynamic --cache-backend none
[W831 22:40:35.018378730 FunctionLoader.cpp:48] Warning: LD_PRELOAD detected, FunctionLoader prefers RTLD_DEFAULT for symbol resolution. (function operator())
INFO 08-31 22:40:39 [__init__.py:52] Available plugins for group vllm.platform_plugins:
INFO 08-31 22:40:39 [__init__.py:54] - ascend -> vllm_ascend:register
INFO 08-31 22:40:39 [__init__.py:57] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 08-31 22:40:39 [__init__.py:237] Platform plugin ascend is activated
INFO 08-31 22:40:39 [platform.py:70] Breakable CUDAGraph on Ascend is opt-in; using VLLM_USE_BREAKABLE_CUDAGRAPH=0.
INFO 08-31 22:40:45 [patch.py:252] NVFP4 W4A4 weight_scale NaN-clamp: installed.
INFO 08-31 22:40:45 [patch.py:489] inductor factorable-divisibility patch: skipped (upstream already proves it).
INFO 08-31 22:40:45 [patch.py:568] [cumem-cuda] CuMemAllocator._python_free_callback patched: asleep guard extended to all platforms.
2026-08-31 22:40:48,516 - 2237126 - ServiceProfiler - INFO - VLLM_USE_V1 not set, auto-detected via vLLM 0.27.1+empty: default 1
INFO 08-31 22:40:48 [__init__.py:112] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 08-31 22:40:48 [__init__.py:112] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
INFO 08-31 22:40:48 [serve.py:194] Detected diffusion model: /home/DiffusionWeights/Qwen-Image/
INFO 08-31 22:40:48 [logo.py:52]        █     █     █▄   ▄█       ▄▀▀▀▀▄ █▄   ▄█ █▄    █ ▀█▀ 
INFO 08-31 22:40:48 [logo.py:52]  ▄▄ ▄█ █     █     █ ▀▄▀ █  ▄▄▄  █    █ █ ▀▄▀ █ █ ▀▄  █  █  
INFO 08-31 22:40:48 [logo.py:52]   █▄█▀ █     █     █     █       █    █ █     █ █   ▀▄█  █  
INFO 08-31 22:40:48 [logo.py:52]    ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀        ▀▀▀▀  ▀     ▀ ▀     ▀ ▀▀▀ 
INFO 08-31 22:40:48 [logo.py:52] 
(APIServer pid=2237126) INFO 08-31 22:40:48 [api_utils.py:345] vLLM server version 0.27.1, serving model /home/DiffusionWeights/Qwen-Image/
(APIServer pid=2237126) INFO 08-31 22:40:48 [api_utils.py:273] non-default args: {'model_tag': '/home/DiffusionWeights/Qwen-Image/', 'host': '127.0.0.1', 'port': 33547, 'model': '/home/DiffusionWeights/Qwen-Image/', 'trust_remote_code': True, 'disable_log_stats': True}
(APIServer pid=2237126) INFO 08-31 22:40:48 [omni_base.py:186] [AsyncOmni] Initializing with model /home/DiffusionWeights/Qwen-Image/
(APIServer pid=2237126) INFO 08-31 22:40:48 [async_omni_engine.py:176] [AsyncOmniEngine] Initializing with model /home/DiffusionWeights/Qwen-Image/
(APIServer pid=2237126) INFO 08-31 22:40:48 [async_omni_engine.py:275] [AsyncOmniEngine] Launching Orchestrator thread with 1 stages
(APIServer pid=2237126) INFO 08-31 22:40:48 [stage_utils.py:167] Stage 0 logical-to-physical device mapping: 0->0
(APIServer pid=2237126) WARNING 08-31 22:40:48 [ring_globals.py:73] FlashInfer is unavailable; falling back to other attention backends. Reason: No module named 'flashinfer'
(APIServer pid=2237126) INFO 08-31 22:40:49 [multiproc_executor.py:294] Starting server...
[W831 22:40:50.493774480 FunctionLoader.cpp:48] Warning: LD_PRELOAD detected, FunctionLoader prefers RTLD_DEFAULT for symbol resolution. (function operator())
INFO 08-31 22:40:55 [__init__.py:52] Available plugins for group vllm.platform_plugins:
INFO 08-31 22:40:55 [__init__.py:54] - ascend -> vllm_ascend:register
INFO 08-31 22:40:55 [__init__.py:57] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 08-31 22:40:55 [__init__.py:237] Platform plugin ascend is activated
INFO 08-31 22:40:55 [platform.py:70] Breakable CUDAGraph on Ascend is opt-in; using VLLM_USE_BREAKABLE_CUDAGRAPH=0.
INFO 08-31 22:41:00 [patch.py:252] NVFP4 W4A4 weight_scale NaN-clamp: installed.
INFO 08-31 22:41:00 [patch.py:489] inductor factorable-divisibility patch: skipped (upstream already proves it).
INFO 08-31 22:41:00 [patch.py:568] [cumem-cuda] CuMemAllocator._python_free_callback patched: asleep guard extended to all platforms.
2026-08-31 22:41:03,614 - 2237376 - ServiceProfiler - INFO - VLLM_USE_V1 not set, auto-detected via vLLM 0.27.1+empty: default 1
INFO 08-31 22:41:03 [__init__.py:112] Registered model loader `<class 'vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic'>` with load format `netloader`
INFO 08-31 22:41:03 [__init__.py:112] Registered model loader `<class 'vllm_ascend.model_loader.rfork.rfork_loader.RForkModelLoader'>` with load format `rfork`
INFO 08-31 22:41:03 [diffusion_worker.py:843] Worker 0 created result MessageQueue
INFO 08-31 22:41:13 [scheduler.py:242] Chunked prefill is enabled with max_num_batched_tokens=2048.
INFO 08-31 22:41:13 [kernel.py:306] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
WARNING 08-31 22:41:13 [platform.py:430] Model config is missing. Skipping Ascend-specific config updates.
INFO 08-31 22:41:13 [compilation.py:329] Enabled custom fusions: norm_quant, act_quant
INFO 08-31 22:41:13 [diffusion_worker.py:306] Final IR op priority after setting vLLM-Omni overrides: IrOpPriorityConfig(rms_norm=['native'], fused_add_rms_norm=['native'])
INFO 08-31 22:41:13 [ascend_config.py:158] Dynamic EPLB is False
INFO 08-31 22:41:13 [ascend_config.py:159] The number of redundant experts is 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
INFO 08-31 22:41:13 [diffusion_worker.py:318] Worker 0: Initialized device and distributed environment.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
INFO 08-31 22:41:13 [parallel_state.py:646] Building SP subgroups from explicit sp_group_ranks (sp_size=1, ulysses=1, ring=1, use_ulysses_low=True).
INFO 08-31 22:41:13 [parallel_state.py:688] SP group details for rank 0: sp_group=[0], ulysses_group=[0], ring_group=[0]
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
Process DiffusionWorker-0:
Traceback (most recent call last):
  File "/home/lhg/code/vllm-omni/vllm_omni/platforms/npu/platform.py", line 218, in get_diffusion_compile_backend
    from mindiesd.compilation import CompilationConfig, MindieSDBackend
ModuleNotFoundError: No module named 'mindiesd'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/worker/diffusion_worker.py", line 1238, in worker_main
    worker_proc = WorkerProc(
                  ^^^^^^^^^^^
  File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/worker/diffusion_worker.py", line 848, in __init__
    self.worker = self._create_worker(gpu_id, od_config, worker_extension_cls, custom_pipeline_args)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/worker/diffusion_worker.py", line 876, in _create_worker
    wrapper = WorkerWrapperBase(
              ^^^^^^^^^^^^^^^^^^
  File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/worker/diffusion_worker.py", line 1310, in __init__
    self.worker = worker_class(
                  ^^^^^^^^^^^^^
  File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/worker/diffusion_worker.py", line 259, in __init__
    self.load_model(load_format=self.od_config.diffusion_load_format)
  File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/worker/diffusion_worker.py", line 368, in load_model
    self.model_runner.load_model(
  File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/worker/diffusion_model_runner.py", line 268, in load_model
    else current_omni_platform.get_diffusion_compile_backend(self.od_config)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lhg/code/vllm-omni/vllm_omni/platforms/npu/platform.py", line 220, in get_diffusion_compile_backend
    raise RuntimeError(
RuntimeError: MindIE-SD diffusion compilation was requested, but mindiesd.compilation could not be imported. Install a MindIE-SD build compatible with PyTorch, torch_npu and CANN, or use enforce_eager=True.
(APIServer pid=2237126) ERROR 08-31 22:41:17 [multiproc_executor.py:338] Rank 0 scheduler is dead. Please check if there are relevant logs.
(APIServer pid=2237126) ERROR 08-31 22:41:18 [multiproc_executor.py:340] Exit code: 1
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253] [StageRuntime] Stage initialization failed; shutting down 0 initialized client(s)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253] Traceback (most recent call last):
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 240, in initialize
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     initialized_clients = self._initialize_stage_replicas(stage_plans, self._stage_init_timeout)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 507, in _initialize_stage_replicas
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     raise primary_exc
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 468, in _init_group
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     client = self._initialize_replica(
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]              ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 532, in _initialize_replica
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     return self._initialize_local_diffusion_replica(plan, stage_init_timeout)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 653, in _initialize_local_diffusion_replica
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     client, resources = launch_diffusion_stage_replica(
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_engine_startup.py", line 1546, in launch_diffusion_stage_replica
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     client = initialize_diffusion_stage(
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_init_utils.py", line 1412, in initialize_diffusion_stage
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     return create_diffusion_client(model, od_config, metadata, stage_init_timeout, batch_size, use_inline)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/stage_diffusion_client.py", line 55, in create_diffusion_client
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     return InlineStageDiffusionClient(model, od_config, metadata, batch_size=batch_size)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/inline_stage_diffusion_client.py", line 63, in __init__
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     self._engine = DiffusionEngine.make_engine(self.od_config)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 689, in make_engine
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     engine = engine_class(config, scheduler=scheduler)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 190, in __init__
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     self._init_executor(od_config)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 239, in _init_executor
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     self.executor = executor_class(od_config)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/abstract.py", line 64, in __init__
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     self._init_executor()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 111, in _init_executor
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     processes, result_handles = self._launch_workers(broadcast_handle, self.wake_events)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 336, in _launch_workers
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     data = reader.recv()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]            ^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 250, in recv
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     buf = self._recv_bytes()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]           ^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 430, in _recv_bytes
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     buf = self._recv(4)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]           ^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 399, in _recv
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253]     raise EOFError
(APIServer pid=2237126) ERROR 08-31 22:41:18 [stage_runtime.py:253] EOFError
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449] [AsyncOmniEngine] Orchestrator thread crashed
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449] Traceback (most recent call last):
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 443, in _bootstrap_orchestrator
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     loop.run_until_complete(_run_orchestrator())
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/usr/local/python3.12.13/lib/python3.12/asyncio/base_events.py", line 691, in run_until_complete
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     return future.result()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]            ^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 401, in _run_orchestrator
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     self._initialize_stages(stage_init_timeout)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 351, in _initialize_stages
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     self._runtime.initialize()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 259, in initialize
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     raise exc
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 240, in initialize
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     initialized_clients = self._initialize_stage_replicas(stage_plans, self._stage_init_timeout)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 507, in _initialize_stage_replicas
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     raise primary_exc
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 468, in _init_group
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     client = self._initialize_replica(
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]              ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 532, in _initialize_replica
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     return self._initialize_local_diffusion_replica(plan, stage_init_timeout)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 653, in _initialize_local_diffusion_replica
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     client, resources = launch_diffusion_stage_replica(
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_engine_startup.py", line 1546, in launch_diffusion_stage_replica
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     client = initialize_diffusion_stage(
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_init_utils.py", line 1412, in initialize_diffusion_stage
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     return create_diffusion_client(model, od_config, metadata, stage_init_timeout, batch_size, use_inline)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/stage_diffusion_client.py", line 55, in create_diffusion_client
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     return InlineStageDiffusionClient(model, od_config, metadata, batch_size=batch_size)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/inline_stage_diffusion_client.py", line 63, in __init__
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     self._engine = DiffusionEngine.make_engine(self.od_config)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 689, in make_engine
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     engine = engine_class(config, scheduler=scheduler)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 190, in __init__
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     self._init_executor(od_config)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 239, in _init_executor
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     self.executor = executor_class(od_config)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/abstract.py", line 64, in __init__
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     self._init_executor()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 111, in _init_executor
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     processes, result_handles = self._launch_workers(broadcast_handle, self.wake_events)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 336, in _launch_workers
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     data = reader.recv()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]            ^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 250, in recv
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     buf = self._recv_bytes()
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]           ^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 430, in _recv_bytes
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     buf = self._recv(4)
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]           ^^^^^^^^^^^^^
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 399, in _recv
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449]     raise EOFError
(APIServer pid=2237126) ERROR 08-31 22:41:18 [async_omni_engine.py:449] EOFError
(APIServer pid=2237126) INFO 08-31 22:41:18 [async_omni_engine.py:1859] [AsyncOmniEngine] Shutting down Orchestrator
(APIServer pid=2237126) Exception in thread orchestrator:
(APIServer pid=2237126) Traceback (most recent call last):
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(APIServer pid=2237126)     self.run()
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/threading.py", line 1012, in run
(APIServer pid=2237126)     self._target(*self._args, **self._kwargs)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 443, in _bootstrap_orchestrator
(APIServer pid=2237126)     loop.run_until_complete(_run_orchestrator())
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/asyncio/base_events.py", line 691, in run_until_complete
(APIServer pid=2237126)     return future.result()
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 401, in _run_orchestrator
(APIServer pid=2237126)     self._initialize_stages(stage_init_timeout)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 351, in _initialize_stages
(APIServer pid=2237126)     self._runtime.initialize()
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 259, in initialize
(APIServer pid=2237126)     raise exc
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 240, in initialize
(APIServer pid=2237126)     initialized_clients = self._initialize_stage_replicas(stage_plans, self._stage_init_timeout)
(APIServer pid=2237126)                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 507, in _initialize_stage_replicas
(APIServer pid=2237126)     raise primary_exc
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 468, in _init_group
(APIServer pid=2237126)     client = self._initialize_replica(
(APIServer pid=2237126)              ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 532, in _initialize_replica
(APIServer pid=2237126)     return self._initialize_local_diffusion_replica(plan, stage_init_timeout)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 653, in _initialize_local_diffusion_replica
(APIServer pid=2237126)     client, resources = launch_diffusion_stage_replica(
(APIServer pid=2237126)                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_engine_startup.py", line 1546, in launch_diffusion_stage_replica
(APIServer pid=2237126)     client = initialize_diffusion_stage(
(APIServer pid=2237126)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_init_utils.py", line 1412, in initialize_diffusion_stage
(APIServer pid=2237126)     return create_diffusion_client(model, od_config, metadata, stage_init_timeout, batch_size, use_inline)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/stage_diffusion_client.py", line 55, in create_diffusion_client
(APIServer pid=2237126)     return InlineStageDiffusionClient(model, od_config, metadata, batch_size=batch_size)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/inline_stage_diffusion_client.py", line 63, in __init__
(APIServer pid=2237126)     self._engine = DiffusionEngine.make_engine(self.od_config)
(APIServer pid=2237126)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 689, in make_engine
(APIServer pid=2237126)     engine = engine_class(config, scheduler=scheduler)
(APIServer pid=2237126)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 190, in __init__
(APIServer pid=2237126)     self._init_executor(od_config)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 239, in _init_executor
(APIServer pid=2237126)     self.executor = executor_class(od_config)
(APIServer pid=2237126)                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/abstract.py", line 64, in __init__
(APIServer pid=2237126)     self._init_executor()
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 111, in _init_executor
(APIServer pid=2237126)     processes, result_handles = self._launch_workers(broadcast_handle, self.wake_events)
(APIServer pid=2237126)                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 336, in _launch_workers
(APIServer pid=2237126)     data = reader.recv()
(APIServer pid=2237126)            ^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 250, in recv
(APIServer pid=2237126)     buf = self._recv_bytes()
(APIServer pid=2237126)           ^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 430, in _recv_bytes
(APIServer pid=2237126)     buf = self._recv(4)
(APIServer pid=2237126)           ^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 399, in _recv
(APIServer pid=2237126)     raise EOFError
(APIServer pid=2237126) EOFError
(APIServer pid=2237126) Traceback (most recent call last):
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/threading.py", line 1075, in _bootstrap_inner
(APIServer pid=2237126)     self.run()
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/threading.py", line 1012, in run
(APIServer pid=2237126)     self._target(*self._args, **self._kwargs)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 443, in _bootstrap_orchestrator
(APIServer pid=2237126)     loop.run_until_complete(_run_orchestrator())
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/asyncio/base_events.py", line 691, in run_until_complete
(APIServer pid=2237126)     return future.result()
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 401, in _run_orchestrator
(APIServer pid=2237126)     self._initialize_stages(stage_init_timeout)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 351, in _initialize_stages
(APIServer pid=2237126)     self._runtime.initialize()
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 259, in initialize
(APIServer pid=2237126)     raise exc
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 240, in initialize
(APIServer pid=2237126)     initialized_clients = self._initialize_stage_replicas(stage_plans, self._stage_init_timeout)
(APIServer pid=2237126)                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 507, in _initialize_stage_replicas
(APIServer pid=2237126)     raise primary_exc
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 468, in _init_group
(APIServer pid=2237126)     client = self._initialize_replica(
(APIServer pid=2237126)              ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 532, in _initialize_replica
(APIServer pid=2237126)     return self._initialize_local_diffusion_replica(plan, stage_init_timeout)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_runtime.py", line 653, in _initialize_local_diffusion_replica
(APIServer pid=2237126)     client, resources = launch_diffusion_stage_replica(
(APIServer pid=2237126)                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_engine_startup.py", line 1546, in launch_diffusion_stage_replica
(APIServer pid=2237126)     client = initialize_diffusion_stage(
(APIServer pid=2237126)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/stage_init_utils.py", line 1412, in initialize_diffusion_stage
(APIServer pid=2237126)     return create_diffusion_client(model, od_config, metadata, stage_init_timeout, batch_size, use_inline)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/stage_diffusion_client.py", line 55, in create_diffusion_client
(APIServer pid=2237126)     return InlineStageDiffusionClient(model, od_config, metadata, batch_size=batch_size)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/inline_stage_diffusion_client.py", line 63, in __init__
(APIServer pid=2237126)     self._engine = DiffusionEngine.make_engine(self.od_config)
(APIServer pid=2237126)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 689, in make_engine
(APIServer pid=2237126)     engine = engine_class(config, scheduler=scheduler)
(APIServer pid=2237126)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 190, in __init__
(APIServer pid=2237126)     self._init_executor(od_config)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/diffusion_engine.py", line 239, in _init_executor
(APIServer pid=2237126)     self.executor = executor_class(od_config)
(APIServer pid=2237126)                     ^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/abstract.py", line 64, in __init__
(APIServer pid=2237126)     self._init_executor()
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 111, in _init_executor
(APIServer pid=2237126)     processes, result_handles = self._launch_workers(broadcast_handle, self.wake_events)
(APIServer pid=2237126)                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/diffusion/executor/multiproc_executor.py", line 336, in _launch_workers
(APIServer pid=2237126)     data = reader.recv()
(APIServer pid=2237126)            ^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 250, in recv
(APIServer pid=2237126)     buf = self._recv_bytes()
(APIServer pid=2237126)           ^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 430, in _recv_bytes
(APIServer pid=2237126)     buf = self._recv(4)
(APIServer pid=2237126)           ^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/connection.py", line 399, in _recv
(APIServer pid=2237126)     raise EOFError
(APIServer pid=2237126) EOFError
(APIServer pid=2237126) 
(APIServer pid=2237126) The above exception was the direct cause of the following exception:
(APIServer pid=2237126) 
(APIServer pid=2237126) Traceback (most recent call last):
(APIServer pid=2237126)   File "<frozen runpy>", line 198, in _run_module_as_main
(APIServer pid=2237126)   File "<frozen runpy>", line 88, in _run_code
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/cli/main.py", line 78, in <module>
(APIServer pid=2237126)     main()
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/cli/main.py", line 72, in main
(APIServer pid=2237126)     args.dispatch_function(args)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/cli/serve.py", line 119, in cmd
(APIServer pid=2237126)     uvloop.run(omni_run_server(args))
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/site-packages/uvloop/__init__.py", line 96, in run
(APIServer pid=2237126)     return __asyncio.run(
(APIServer pid=2237126)            ^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/asyncio/runners.py", line 195, in run
(APIServer pid=2237126)     return runner.run(main)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/asyncio/runners.py", line 118, in run
(APIServer pid=2237126)     return self._loop.run_until_complete(task)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/site-packages/uvloop/__init__.py", line 48, in wrapper
(APIServer pid=2237126)     return await main
(APIServer pid=2237126)            ^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/openai/api_server.py", line 472, in omni_run_server
(APIServer pid=2237126)     await omni_run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/openai/api_server.py", line 499, in omni_run_server_worker
(APIServer pid=2237126)     async with build_async_omni(
(APIServer pid=2237126)                ^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=2237126)     return await anext(self.gen)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/openai/api_server.py", line 654, in build_async_omni
(APIServer pid=2237126)     async with build_async_omni_from_stage_config(
(APIServer pid=2237126)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/contextlib.py", line 210, in __aenter__
(APIServer pid=2237126)     return await anext(self.gen)
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/openai/api_server.py", line 720, in build_async_omni_from_stage_config
(APIServer pid=2237126)     async_omni = AsyncOmni(model=model, **kwargs)
(APIServer pid=2237126)                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/async_omni.py", line 147, in __init__
(APIServer pid=2237126)     OmniBase.__init__(self, model=model, **kwargs)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/entrypoints/omni_base.py", line 192, in __init__
(APIServer pid=2237126)     self.engine = AsyncOmniEngine(
(APIServer pid=2237126)                   ^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 290, in __init__
(APIServer pid=2237126)     self._wait_for_orchestrator_init(startup_future, startup_timeout)
(APIServer pid=2237126)   File "/home/lhg/code/vllm-omni/vllm_omni/engine/async_omni_engine.py", line 494, in _wait_for_orchestrator_init
(APIServer pid=2237126)     startup_future.result(
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/concurrent/futures/_base.py", line 456, in result
(APIServer pid=2237126)     return self.__get_result()
(APIServer pid=2237126)            ^^^^^^^^^^^^^^^^^^^
(APIServer pid=2237126)   File "/usr/local/python3.12.13/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(APIServer pid=2237126)     raise self._exception
(APIServer pid=2237126) RuntimeError: Orchestrator initialization failed: 
(APIServer pid=2237126) [ERROR] 2026-08-31-22:41:18 (PID:2237126, Device:-1, RankID:-1) ERR99999 UNKNOWN applicaiton exception
[root@vllm-omni-lhg lhg]# 
