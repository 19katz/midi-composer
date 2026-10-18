"""Upweight the tokens that say what a bar is for, not every scale degree.

Chord, rhythm, tempo, phrase, and the melody line are rare next to the figuration,
so a flat loss mostly teaches the model to copy eighth notes. Scale degrees on the
right-hand and left-hand lines stay at weight 1; upweighting those would reward
the repetitive figuration this run is trying to thin out.
"""

import torch
import torch.nn.functional as F

from .vocab import music_tokens


def _ids(tokenizer, pred):
    unk = getattr(tokenizer, "unk_token_id", None)
    out = set()
    for tok in music_tokens():
        if not pred(tok):
            continue
        i = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(i, int) and i >= 0 and i != unk:
            out.add(i)
    return out


def _weights(labels, special, boundary, mel_id, factor):
    """Weight per label position. 0 where the label is masked (-100)."""
    weight = torch.ones(labels.shape, device=labels.device, dtype=torch.float32)
    rows = labels.detach().tolist()
    for b, seq in enumerate(rows):
        in_mel = False
        for t, tok in enumerate(seq):
            if tok == -100:
                weight[b, t] = 0.0
                continue
            if tok == mel_id:
                in_mel = True
                weight[b, t] = factor
            elif tok in boundary:
                in_mel = False
                weight[b, t] = factor if tok in special else 1.0
            elif in_mel or tok in special:
                weight[b, t] = factor
    return weight


def install_weighted_loss(trainer, tokenizer, factor=3.0):
    """Replace the trainer loss after `train_on_responses_only`.

    That helper masks the brief in the labels. This runs on those labels, so the
    extra weight stays on the score. If the model does not return logits, the
    original loss is used and a one-line warning is printed.
    """
    special = _ids(tokenizer, lambda t: (
        t.startswith("<|ch_") or t.startswith("<|q_")
        or t in ("<|rhythm_same|>", "<|rhythm_new|>", "<|tempo_hold|>", "<|phrase|>", "<|mel|>")
    ))
    boundary = _ids(tokenizer, lambda t: t in ("<|rh|>", "<|lh|>", "<|ped|>", "<|bar|>", "<|piece_end|>"))
    mel = _ids(tokenizer, lambda t: t == "<|mel|>")
    mel_id = next(iter(mel), None)
    factor = float(factor)
    original = trainer.compute_loss
    warned = {"on": False}

    def compute_loss(model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            return original(model, inputs, return_outputs=return_outputs, **kwargs)
        fwd = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**fwd)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, tuple):
            logits = outputs[0]
        if logits is None:
            if not warned["on"]:
                print("weighted loss: model did not return logits; using the unweighted loss")
                warned["on"] = True
            return original(model, inputs, return_outputs=return_outputs, **kwargs)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        w = _weights(labels, special, boundary, mel_id, factor)[..., 1:].contiguous().view(-1)
        denom = w.sum().clamp(min=1.0)
        loss = (flat * w).sum() / denom
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss
    return trainer
