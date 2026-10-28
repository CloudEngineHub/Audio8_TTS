"""CPU regression checks for the two SFT output-label contracts.

These tests use small local modules and a deterministic tokenizer, never model
downloads or checkpoint weights. They exercise the public data builder and
Trainer rather than reproducing their implementation.
"""

import re
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import TrainingArguments

import audio8_tts_sft as sft
from audio8_tts_data import build_sft_example, collate_sft_examples


def config_for(family: str) -> SimpleNamespace:
    if family == "0.1b":
        return SimpleNamespace(
            vocab_size=69633,
            codebook_size=4096,
            num_codebooks=10,
            semantic_begin_id=65537,
            semantic_end_id=69632,
            eos_token_id=228,
        )
    return SimpleNamespace(
        vocab_size=155776,
        codebook_size=4096,
        num_codebooks=10,
        semantic_begin_id=151678,
        semantic_end_id=155773,
        eos_token_id=151645,
    )


def tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=128,
        codebook_size=4,
        num_codebooks=10,
        semantic_begin_id=100,
        semantic_end_id=103,
        eos_token_id=2,
        pad_token_id=0,
        norm_fastlayer_input=False,
    )


def test_compact_targets_map_semantic_and_eos_without_mutating_input():
    config = config_for("0.1b")
    labels = torch.tensor([[-100, 65537, 69632, 228, 709, 17]])
    original = labels.clone()

    targets = sft.slow_loss_targets(labels, config, 4097)

    assert targets.tolist() == [[-100, 0, 4095, 4096, -100, -100]]
    assert torch.equal(labels, original)
    assert torch.isfinite(F.cross_entropy(torch.zeros((6, 4097)), targets.reshape(-1)))


def test_full_vocabulary_targets_preserve_existing_loss_and_gradients():
    config = config_for("0.6b")
    labels = torch.tensor([[-100, 151678, 155773, 151645, 198]])
    generator = torch.Generator().manual_seed(7)
    logits_before = torch.randn((5, config.vocab_size), generator=generator).requires_grad_()
    logits_after = logits_before.detach().clone().requires_grad_()

    targets = sft.slow_loss_targets(labels, config, config.vocab_size)
    loss_before = F.cross_entropy(logits_before, labels.reshape(-1))
    loss_after = F.cross_entropy(logits_after, targets.reshape(-1))
    loss_before.backward()
    loss_after.backward()

    assert torch.equal(targets, labels)
    torch.testing.assert_close(loss_after, loss_before, rtol=0, atol=0)
    torch.testing.assert_close(logits_after.grad, logits_before.grad, rtol=0, atol=0)


def test_unknown_slow_output_contract_is_rejected():
    config = config_for("0.1b")
    with pytest.raises(ValueError):
        sft.slow_loss_targets(torch.tensor([[65537, 228]]), config, 4098)


class PromptTokenizer:
    """Small tokenizer that keeps chat delimiters distinct for builder tests."""

    special = {"<|im_start|>": 1, "<|im_end|>": 2, "<|voice|>": 3, "<|speaker:0|>": 4}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        pieces = re.findall(r"<\|[^|]+\|>|.", text, flags=re.DOTALL)
        return [self.special.get(piece, 7 if piece == "\n" else 5) for piece in pieces]


def codec_frames(semantic):
    semantic = torch.tensor(semantic, dtype=torch.long)
    return torch.stack([(semantic + index) % 4 for index in range(10)])


def test_builder_preserves_reference_masking_shift_eos_and_padding():
    config = tiny_config()
    target = codec_frames([0, 1, 3])
    example = build_sft_example(
        tokenizer=PromptTokenizer(),
        config=config,
        text="Hello.",
        target_codes=target,
        reference_codes=codec_frames([2, 2]),
        reference_text="Reference.",
    )
    labels = example["labels"]
    semantic_positions = labels[0].ge(100) & labels[0].le(103)

    # Three target frames are supervised; the two reference frames are inputs
    # only. The next input column is the target predicted at this position.
    assert labels[0, semantic_positions].tolist() == [100, 101, 103]
    assert torch.equal(labels[1:, semantic_positions], target)
    input_semantic = example["input_ids"][0].ge(100) & example["input_ids"][0].le(103)
    assert int(input_semantic.sum()) == 5
    for position in torch.where(semantic_positions)[0].tolist():
        assert torch.equal(example["input_ids"][:, position + 1], labels[:, position])
    assert labels[0, labels[0].ne(-100)].tolist() == [100, 101, 103, 2, 7]
    assert labels[1:, ~semantic_positions].eq(-100).all()

    shorter = build_sft_example(
        tokenizer=PromptTokenizer(), config=config, text="Hi.", target_codes=codec_frames([1])
    )
    batch = collate_sft_examples([example, shorter], pad_token_id=config.pad_token_id)
    padding = batch["attention_mask"].eq(0)
    assert padding.any()
    assert batch["labels"].permute(0, 2, 1)[padding].eq(-100).all()
    assert batch["input_ids"][:, 0][padding].eq(config.pad_token_id).all()
    compact = sft.slow_loss_targets(batch["labels"][:, 0], config, 5)
    assert compact[0, compact[0].ne(-100)].tolist() == [0, 1, 3, 4]
    assert compact[padding].eq(-100).all()


def training_args(path, accumulation=1):
    return TrainingArguments(
        output_dir=str(path),
        use_cpu=True,
        report_to=[],
        save_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        dataloader_pin_memory=False,
        gradient_accumulation_steps=accumulation,
    )


class TinyDualAR(nn.Module):
    """Minimal model exposing the actual slow/fast Trainer interface."""

    def __init__(self, compact):
        super().__init__()
        self.config = tiny_config()
        classes = 5 if compact else 128
        logits = torch.zeros((1, 6, classes))
        expected = [0, 0, 3, 4, 1, 0] if compact else [0, 100, 103, 2, 7, 0]
        for position, prediction in enumerate(expected):
            logits[0, position, prediction] = 6
        self.slow_logits = nn.Parameter(logits)
        self.slow_hidden = nn.Parameter(torch.ones((1, 6, 4)))
        self.fast_project_in = nn.Identity()
        self.fast_embeddings = nn.Embedding(4, 4)
        self.fast_layers = nn.ModuleList()
        self.fast_norm = nn.Identity()
        self.fast_output = nn.Linear(4, 4, bias=False)
        self.register_buffer("fast_freqs_cis", torch.zeros((10, 4)))

    @staticmethod
    def _causal_mask(attention_mask, positions, length):
        return torch.ones((length, length), dtype=torch.bool).tril()[None, None]

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        return SimpleNamespace(logits=self.slow_logits, hidden_states=self.slow_hidden)


@pytest.mark.parametrize("compact", [False, True], ids=["full-vocabulary", "compact"])
def test_trainer_uses_correct_slow_accuracy_and_keeps_all_fast_targets(
    tmp_path, monkeypatch, compact
):
    model = TinyDualAR(compact)
    trainer = sft.Audio8TTSTrainer(
        model=model,
        args=training_args(tmp_path),
        slow_loss_weight=1.0,
        fast_loss_weight=1.0,
    )
    labels = torch.full((1, 11, 6), -100, dtype=torch.long)
    labels[0, 0] = torch.tensor([-100, 100, 103, 2, 7, -100])
    expected_codebooks = codec_frames([0, 3]).T
    labels[0, 1:, 1:3] = expected_codebooks.T
    original = labels.clone()
    seen = []
    original_fast_logits = sft.fast_codebook_logits

    def capture_fast_inputs(core, hidden, codebooks):
        seen.append(codebooks.detach().clone())
        return original_fast_logits(core, hidden, codebooks)

    monkeypatch.setattr(sft, "fast_codebook_logits", capture_fast_inputs)
    loss, outputs = trainer.compute_loss(
        model,
        {"input_ids": torch.zeros((1, 11, 6), dtype=torch.long), "labels": labels},
        return_outputs=True,
    )
    assert torch.isfinite(loss)
    assert torch.equal(labels, original)
    assert len(seen) == 1
    assert torch.equal(seen[0], expected_codebooks)
    assert outputs["codebook_logits"].shape == (2, 10, 4)
    assert trainer._metric_sums["slow_accuracy"] == pytest.approx(1.0)
    assert trainer._metric_sums["semantic_frames"] == 2
    expected_fast_loss = F.cross_entropy(
        outputs["codebook_logits"].reshape(-1, 4), expected_codebooks.reshape(-1)
    )
    assert trainer._metric_sums["fast_loss"] == pytest.approx(float(expected_fast_loss.detach()))
    loss.backward()
    assert model.slow_logits.grad is not None
    assert model.slow_hidden.grad is not None
    assert model.slow_hidden.grad[:, 1:3].abs().sum() > 0


def test_main_rejects_unsupported_checkpointing_before_dataset_access(tmp_path, monkeypatch):
    model = TinyDualAR(compact=False)
    model.supports_gradient_checkpointing = False
    model.config.max_seq_len = 2048
    monkeypatch.setattr(
        sft.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=PromptTokenizer()),
    )
    monkeypatch.setattr(sft.AutoModel, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audio8_tts_sft.py",
            "--model_name_or_path",
            "local-test-model",
            "--train_jsonl",
            str(tmp_path / "does-not-exist.jsonl"),
            "--output_dir",
            str(tmp_path / "output"),
            "--gradient_checkpointing",
            "true",
            "--use_cpu",
            "true",
            "--report_to",
            "none",
        ],
    )
    with pytest.raises(ValueError, match="does not support gradient checkpointing"):
        sft.main()


class Branches(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family == "0.1b":
            self.slow = nn.Linear(2, 2)
            self.semantic_output = nn.Linear(2, 2)
        else:
            self.embeddings = nn.Embedding(4, 2)
            self.layers = nn.ModuleList([nn.Linear(2, 2)])
            self.norm = nn.LayerNorm(2)
        self.codebook_embeddings = nn.Embedding(4, 2)
        self.fast_project_in = nn.Linear(2, 2)
        self.fast_embeddings = nn.Embedding(4, 2)
        self.fast_layers = nn.ModuleList([nn.Linear(2, 2)])
        self.fast_norm = nn.LayerNorm(2)
        self.fast_output = nn.Linear(2, 2)


@pytest.mark.parametrize("family", ["0.1b", "0.6b"])
@pytest.mark.parametrize("freeze_slow,freeze_fast", [(False, False), (True, False), (False, True)])
def test_freeze_flags_cover_both_architectures(family, freeze_slow, freeze_fast):
    model = Branches(family)
    fast_parameters = {
        id(parameter) for name, parameter in model.named_parameters() if name.startswith("fast_")
    }
    sft.set_trainable_modules(model, freeze_slow=freeze_slow, freeze_fast=freeze_fast)
    for parameter in model.parameters():
        should_freeze = freeze_fast if id(parameter) in fast_parameters else freeze_slow
        assert parameter.requires_grad is not should_freeze


@pytest.mark.parametrize("family", ["0.1b", "0.6b"])
def test_freezing_every_branch_is_rejected(family):
    with pytest.raises(ValueError, match="no trainable parameters"):
        sft.set_trainable_modules(Branches(family), freeze_slow=True, freeze_fast=True)


class ScalarMeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, values, **kwargs):
        # **kwargs makes Transformers infer loss-kwargs support unless the
        # Audio8 Trainer explicitly declares its microbatch-mean interface.
        return self.weight * values


class ScalarMeanTrainer(sft.Audio8TTSTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return model(inputs["values"]).square().mean()


def test_trainer_mean_loss_accumulation_matches_equal_sized_batch(tmp_path):
    gradients = []
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    for accumulation in (1, 4):
        model = ScalarMeanModel()
        trainer = ScalarMeanTrainer(
            model=model,
            args=training_args(tmp_path / str(accumulation), accumulation),
            slow_loss_weight=1.0,
            fast_loss_weight=1.0,
        )
        trainer.current_gradient_accumulation_steps = accumulation
        for microbatch in values.chunk(accumulation):
            trainer.training_step(model, {"values": microbatch}, num_items_in_batch=torch.tensor(4))
        gradients.append(model.weight.grad.detach().clone())

    torch.testing.assert_close(gradients[0], torch.tensor(15.0))
    torch.testing.assert_close(gradients[1], gradients[0])
