# 监督微调（SFT）

[系列首页](../README_zh.md) · [English](training.md) · [推理](inference_zh.md)

以下保留 0.6B SFT 流程。从仓库根目录执行，并显式选择需要微调的版本。

```bash
export MODEL=Edge0/Audio8-TTS-Realistic-0.6b
# Preview 0.6B: export MODEL=Edge0/Audio8-TTS-Preview-0.6b
```

先安装训练依赖：

```bash
pip install -r requirements-train.txt
```

### 1. 创建原始数据 manifest

目标 `audio` 必填；`reference_audio` 和 `reference_text` 可选，但必须同时出现。

```json
{"id":"utt_001","text":"目标音频文本","audio":"audio/target.wav","reference_audio":"audio/reference.wav","reference_text":"参考音频文本"}
{"id":"utt_002","text":"另一条文本","audio":"audio/another.wav"}
```

### 2. 预计算 codec indices

```bash
python audio8_tts_prepare.py \
  --model "${MODEL}" \
  --input-jsonl data/train.jsonl \
  --output-jsonl prepared_data/train.jsonl \
  --batch-size 4
```

生成的 manifest 使用相对路径指向经过校验的 `[10, T]` NumPy 数组。除非传入
`--overwrite`，否则已有的有效数组会被复用。

### 3. 开始训练

单卡：

```bash
MODEL="${MODEL}" \
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=1 \
bash audio8_tts_sft.sh
```

单机八卡：

```bash
MODEL="${MODEL}" \
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=8 \
BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
bash audio8_tts_sft.sh
```

多机训练时，在每个节点设置 `NNODES`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。常用超参
和输出路径可通过 `audio8_tts_sft.sh` 中列出的环境变量覆盖，也可以在命令末尾附加 Transformers
参数。

SFT 同时优化 slow semantic/EOS loss 和 fast codebook teacher-forcing loss。只训练其中一个分支时，
可设置 `FREEZE_SLOW_AR=true` 或 `FREEZE_FAST_AR=true`。导出目录仍可通过标准 `AutoModel` 和
`AutoProcessor` 接口配合 `trust_remote_code=True` 加载。
