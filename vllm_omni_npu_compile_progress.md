# vLLM-Omni NPU Diffusion Compile 项目进展

## 1. 项目目标

为 vLLM-Omni 的 diffusion 推理增加 Ascend NPU 编译支持，当前目标模型为 Qwen-Image。

当前技术路线：

- CUDA：保留现有编译路径
- NPU：使用 MindIE-SD `MindieSDBackend`
- 编译粒度：regional compile
- `dynamic=False`
- 当前主要测试模型：Qwen-Image
- 当前稳定测试模式：`--step-execution --max-num-seqs 1`

---

## 2. Omni 侧已完成修改

### 2.1 增加 NPU diffusion compile backend

原有 diffusion compile 逻辑主要依赖：

```python
supports_torch_inductor()
```

NPU 返回 `False` 时会跳过编译。

当前已在 NPU platform 中增加：

```python
get_diffusion_compile_backend(...)
```

实现：

```text
CUDA -> 原有 compile backend
NPU  -> MindieSDBackend
```

NPU 通过以下参数启用编译：

```bash
--diffusion-compile-backend mindiesd
--diffusion-compile-granularity regional
--no-diffusion-compile-dynamic
```

### 2.2 Qwen-Image regional compile 已打通

Qwen-Image 已定义 repeated block：

```text
QwenImageTransformerBlock
```

当前实际运行链路：

```text
QwenImageTransformerBlock
-> torch.compile
-> TorchDynamo / FX Graph
-> MindieSDBackend
-> MindIE-SD pattern passes
-> compiled forward
```

因此当前已确认 MindIE-SD backend 实际参与编译执行。

---

## 3. MindIE-SD 集成状态

已打通：

```text
vLLM-Omni
PyTorch
torch_npu
CANN
MindIE-SD
```

源码目录：

```text
vLLM-Omni:
/home/lhg/code/vllm-omni

MindIE-SD:
/home/lhg/code/MindIE-SD
```

---

## 4. Qwen-Image residual gate pattern 兼容问题

### 4.1 问题

MindIE-SD 中存在：

```python
x + y * gate
```

融合：

```text
WanResidualGatePatternGroup
-> mindiesd::residual_gate_add
```

Qwen-Image 实际存在 broadcasting：

```text
x.shape = (1, 1024, 3072)
y.shape = (1, 1, 3072)
```

原始 PyTorch 允许 broadcasting，但当前 MindIE-SD fused op 要求：

```text
x.shape == y.shape
```

因此会报：

```text
x and y must have the same shape
```

### 4.2 当前临时处理

已修改：

```text
/home/lhg/code/MindIE-SD/mindiesd/compilation/compiliation_config.py
```

设置：

```python
enable_wan_residual_gate: bool = False
```

运行时已确认：

```python
CompilationConfig.fusion_patterns.enable_wan_residual_gate == False
```

该配置通过 pattern registry 中的：

```python
getattr(fusion_config, config_key, False)
```

实际控制 pattern 是否注册。

### 4.3 后续正式修复

后续应修改：

```text
mindiesd/compilation/patterns/wan_residual_gate_pattern.py
```

增加 shape guard：

```text
x.shape == y.shape
    -> 允许 residual_gate_add fusion

需要 broadcasting
    -> 不 fusion
    -> 保留原始 PyTorch graph
```

不建议永久全局关闭 `WanResidualGatePatternGroup`。

---

## 5. request-mode 偶发 500 问题

曾出现：

```text
denoise 正常完成
-> 等待约 30 秒
-> _ASYNC_OUTPUT_TIMEOUT
-> HTTP 500
```

已排除：

- `GLOO_SOCKET_IFNAME=lo`
- HTTPX 客户端
- curl keepalive
- compile 本身

使用：

```bash
--step-execution
--max-num-seqs 1
```

后请求稳定正常。

因此当前 compile benchmark 固定使用：

```bash
--step-execution --max-num-seqs 1
```

该问题与 MindIE-SD compile 独立。

---

## 6. TP2 状态

已验证：

```text
Qwen-Image eager TP2               OK
Qwen-Image MindIE-SD compile TP2   OK
```

TP2 compile 基础命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=14,15 \
python3 -u -m vllm_omni.entrypoints.cli.main serve \
  /home/DiffusionWeights/Qwen-Image/ \
  --host 127.0.0.1 \
  --port 33547 \
  --omni \
  --trust-remote-code \
  --disable-log-stats \
  --tensor-parallel-size 2 \
  --step-execution \
  --max-num-seqs 1 \
  --diffusion-compile-backend mindiesd \
  --diffusion-compile-granularity regional \
  --no-diffusion-compile-dynamic \
  --cache-backend none
```

Omni NPU backend 中原先限制：

```python
parallel_config.world_size != 1
```

当前已为 TP2 验证放开。

单卡当前也不存在 OOM 问题，不再作为待排查项。

---

## 7. 当前推理时延指标

当前已观察到的稳定端到端请求时延：

### 512 x 512

```text
Eager TP2:               ~2.8 s
MindIE-SD compile TP2:   ~2.6 s
```

收益：

```text
~0.2 s
~7.1%
```

### 1024 x 1024

```text
Eager TP2:               ~9.0 s
MindIE-SD compile TP2:   ~8.8 s
```

收益：

```text
~0.2 s
~2.2%
```

当前结论：

```text
MindIE-SD compile 已产生稳定性能收益，
但收益较小，约固定减少 0.2 s。
```

---

## 8. 当前 benchmark 请求参数

主要请求参数：

```json
{
  "model": "/home/DiffusionWeights/Qwen-Image/",
  "prompt": "生成一张写实风格的图片：一辆红色跑车停在平静的湖边。",
  "size": "1024x1024",
  "num_inference_steps": 50,
  "true_cfg_scale": 4.0,
  "seed": 42,
  "response_format": "b64_json",
  "output_format": "png"
}
```

Python benchmark 已支持：

- warmup
- concurrency
- total latency
- response headers latency
- first byte latency
- mean / p50 / p95 / p99
- profiling

正式性能统计时，`total_latency_ms` 应在完整 HTTP response body 接收完成后立即记录。

以下客户端操作不计入推理时延：

```text
JSON parse
Base64 decode
PNG write
```

推荐正式 benchmark：

```text
warmup_requests = 4
warmup_concurrency = 1
num_requests = 10
concurrency = 1
```

---

## 9. 当前 compile 性能判断

当前模式主要是：

```text
regional torch.compile
+ MindIE-SD pattern fusion
```

ACLGraph 尚未启用。

同时：

```text
enable_wan_residual_gate = False
```

使一个 fusion 暂时关闭。

当前主要可能优化：

```text
elementwise
Norm
AdaLN
RoPE
部分 kernel 数量
部分 host / dispatcher 开销
```

以下开销仍然存在：

```text
MatMul / Linear
Attention
TP communication
VAE
step execution orchestration
其他未融合算子
```

512 和 1024 都约固定减少 0.2 s，说明当前收益更可能来自小算子融合、调度和 kernel launch，而不是大规模 GEMM / Attention 主计算。

---

# 10. 下一阶段核心：ACLGraph

下一阶段目标：

```text
MindIE-SD Pattern Fusion
+
ACLGraph Capture / Replay
```

当前：

```text
FX Graph
-> MindIE-SD pattern fusion
-> normal compiled execution
```

目标：

```text
FX Graph
-> MindIE-SD pattern fusion
-> ACLGraph capture
-> ACLGraph replay
```

目标是进一步减少：

```text
Python / dispatcher overhead
NPU kernel launch overhead
动态图调度开销
```

---

## 11. 首先检查当前 MindIE-SD ACLGraph API

执行：

```bash
cd /home/lhg/code/MindIE-SD

python3 - <<'PY'
from mindiesd.compilation import CompilationConfig

for name in [
    "aclgraph_only",
    "aclgraph_with_compile",
    "aclgraph_lazy_capture",
    "aclgraph_max_entries",
    "safe_output_mode",
    "enable_freezing",
]:
    print(name, "=", getattr(CompilationConfig, name, "<MISSING>"))
PY
```

重点确认：

```text
aclgraph_with_compile
aclgraph_lazy_capture
aclgraph_max_entries
```

---

## 12. ACLGraph 目标配置

如果当前 API 支持，优先测试：

```python
CompilationConfig.aclgraph_only = False
CompilationConfig.aclgraph_with_compile = True
```

如果存在：

```python
CompilationConfig.aclgraph_lazy_capture = True
```

则同时启用。

目标模式是：

```text
Pattern Fusion + ACLGraph
```

不是：

```text
ACLGraph only
```

---

## 13. Omni NPU backend 下一步修改

当前 NPU backend 中存在 ACLGraph 禁止逻辑，例如：

```python
for flag in ("aclgraph_only", "aclgraph_with_compile"):
    if getattr(CompilationConfig, flag):
        raise ValueError(...)
```

下一阶段需要调整。

目标配置顺序：

```python
CompilationConfig.aclgraph_only = False
CompilationConfig.aclgraph_with_compile = True

if hasattr(CompilationConfig, "aclgraph_lazy_capture"):
    CompilationConfig.aclgraph_lazy_capture = True

backend = MindieSDBackend()
```

配置必须在第一次 backend compile / capture 前完成。

---

## 14. ACLGraph 验证步骤

### Step 1

保持当前稳定配置不变：

```text
TP2
step_execution=True
max_num_seqs=1
regional compile
dynamic=False
cache_backend=none
```

仅新增 ACLGraph。

### Step 2

确认运行日志实际进入：

```text
ACLGraph capture
ACLGraph cache
ACLGraph replay
```

不能只根据配置值判断。

### Step 3

对比三组：

```text
1. Eager
2. MindIE-SD Pattern compile
3. MindIE-SD Pattern compile + ACLGraph
```

固定：

```text
prompt
seed
steps
resolution
TP=2
concurrency=1
```

测试：

```text
512 x 512
1024 x 1024
```

每组：

```text
4 warmup
10 formal requests
```

统计：

```text
mean
p50
p95
```

### Step 4

使用 `torch_npu.profiler` 比较：

```text
kernel 数量
MatMul / Linear 时间
Attention 时间
HCCL 时间
elementwise 时间
Norm / RoPE / AdaLN 时间
Host launch gap
ACLGraph replay 时间
```

重点确认 ACLGraph 是否减少 host / launch overhead。

---

## 15. ACLGraph 正确性验证

固定：

```text
prompt
seed
resolution
steps
CFG
```

比较：

```text
Eager
Pattern compile
Pattern compile + ACLGraph
```

重点检查：

```text
输出图片是否一致
static input 是否正确刷新
是否存在 stale input
第二次及后续 replay 是否正常
不同请求是否错误复用数据
```

---

## 16. 多 shape ACLGraph cache

后续测试：

```text
512x512
1024x1024
其他 resolution
```

如果支持：

```python
CompilationConfig.aclgraph_max_entries
```

需要验证不同 shape 的 graph cache 行为，并限制 cache 数量，防止长期运行时 graph 持续累积。

---

# 17. 当前项目状态

```text
NPU compile backend 接入
    DONE

Qwen-Image regional torch.compile
    DONE

MindIE-SD backend 实际执行
    DONE

MindIE-SD Pattern Fusion
    DONE

Qwen residual_gate broadcasting
    TEMP FIX: enable_wan_residual_gate=False
    TODO: pattern shape guard

Qwen-Image eager TP2
    DONE

Qwen-Image MindIE-SD compile TP2
    DONE

request-mode 30s / HTTP 500
    WORKAROUND: --step-execution --max-num-seqs 1

当前时延：
    512 eager TP2      ~2.8 s
    512 compile TP2    ~2.6 s

    1024 eager TP2     ~9.0 s
    1024 compile TP2   ~8.8 s

ACLGraph
    NEXT

Pattern compile + ACLGraph
    NEXT

ACLGraph correctness / profiler / graph cache
    NEXT
```

---

# 18. Codex 下一步任务

1. 检查当前 MindIE-SD `CompilationConfig` 的 ACLGraph API。
2. 阅读 `MindieSDBackend` 中 `aclgraph_only`、`aclgraph_with_compile` 的真实控制逻辑。
3. 调整 Omni NPU backend 当前禁止 ACLGraph 的限制。
4. 首先启用：
   ```python
   aclgraph_only = False
   aclgraph_with_compile = True
   ```
5. 如果存在：
   ```python
   aclgraph_lazy_capture = True
   ```
   同时启用。
6. 保持 TP2 + step execution，确认 ACLGraph 实际 capture/replay。
7. 对比 Eager / Pattern compile / Pattern compile + ACLGraph。
8. 使用 profiler 分析 ACLGraph 实际节省的时间来源。
9. 验证固定 seed 下 ACLGraph replay 输出正确性。
10. 最后正式修复 `WanResidualGatePattern` broadcasting shape guard。
