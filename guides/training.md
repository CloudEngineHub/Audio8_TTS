# Supervised fine-tuning (SFT)

[All releases](../README.md) · [中文](training_zh.md) · [Inference](inference.md)

These instructions retain the 0.6B SFT workflow. Run from the repository root and explicitly choose the release to fine-tune.

```bash
export MODEL=Edge0/Audio8-TTS-Realistic-0.6b
# Preview 0.6B: export MODEL=Edge0/Audio8-TTS-Preview-0.6b
```

Install the training dependencies first:

```bash
pip install -r requirements-train.txt
```

### 1. Create a raw manifest

The target `audio` field is required. `reference_audio` and `reference_text`
are optional, but must be provided together.

```json
{"id":"utt_001","text":"Target transcript","audio":"audio/target.wav","reference_audio":"audio/reference.wav","reference_text":"Reference transcript"}
{"id":"utt_002","text":"Another transcript","audio":"audio/another.wav"}
```

### 2. Precompute codec indices

```bash
python audio8_tts_prepare.py \
  --model "${MODEL}" \
  --input-jsonl data/train.jsonl \
  --output-jsonl prepared_data/train.jsonl \
  --batch-size 4
```

The prepared manifest points to validated `[10, T]` NumPy arrays using paths
relative to the prepared manifest. Existing valid arrays are reused unless
`--overwrite` is passed.

### 3. Train

Single GPU:

```bash
MODEL="${MODEL}" \
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=1 \
bash audio8_tts_sft.sh
```

Eight GPUs on one node:

```bash
MODEL="${MODEL}" \
TRAIN_JSONL=prepared_data/train.jsonl \
NPROC_PER_NODE=8 \
BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=8 \
bash audio8_tts_sft.sh
```

For multi-node training, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT` on each node. Common hyperparameters and output paths can be
overridden through the environment variables in `audio8_tts_sft.sh`; additional
Transformers arguments may be appended to the command.

SFT optimizes both the slow semantic/EOS objective and the fast codebook
teacher-forcing objective. Set `FREEZE_SLOW_AR=true` or `FREEZE_FAST_AR=true`
when adapting only one branch. The exported directory remains loadable with
standard `AutoModel` and `AutoProcessor` APIs using `trust_remote_code=True`.
