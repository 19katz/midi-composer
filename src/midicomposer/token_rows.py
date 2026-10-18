"""Train only the embedding and lm_head rows of the music tokens.

Qwen does not tie those two matrices. Training every row OOMs a T4, and zeroing the
gradient of a full matrix still makes Adam store a state for every row. Each module
keeps a small parameter for the new rows. A forward hook writes that parameter into
the outputs, so the frozen matrix never enters the optimizer.
"""

import torch
import torch.nn as nn


def _targets(model):
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    found = [("embed", embed)]
    if head is not None and head is not embed:
        found.append(("lm_head", head))
    return found


def _hook(n_base, n_new):
    def hook(module, args, output):
        rows = module.new_token_rows
        incoming = args[0] if args else None
        if incoming is None:
            return output
        # Embedding ids are integers. The lm_head receives hidden states.
        if not incoming.dtype.is_floating_point:
            ids = incoming
            flat = False
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
                flat = output.dim() == 2
            local = (ids - n_base).clamp(0, n_new - 1)
            mask = (ids >= n_base) & (ids < n_base + n_new)
            picked = rows[local]
            mixed = torch.where(mask.unsqueeze(-1), picked.to(output.dtype), output)
            return mixed.squeeze(0) if flat else mixed
        extra = incoming.matmul(rows.transpose(0, 1)).to(output.dtype)
        # Concatenate instead of writing into a clone: an in-place write on a
        # tensor that requires grad breaks the backward.
        return torch.cat(
            (output[..., :n_base], extra, output[..., n_base + n_new:]),
            dim=-1,
        )

    return hook


def attach_new_token_training(model, n_base, n_new):
    """Freeze both tables and train `n_new` rows starting at index `n_base`.

    Call this after `add_new_tokens` and `get_peft_model`, and before the Trainer
    is built, so the optimizer sees the small parameters.
    """
    n_base, n_new = int(n_base), int(n_new)
    if n_new <= 0:
        raise ValueError("n_new must be positive")
    for _name, mod in _targets(model):
        if getattr(mod, "new_token_rows", None) is not None:
            continue
        weight = mod.weight
        weight.requires_grad_(False)
        rows = nn.Parameter(weight.detach()[n_base:n_base + n_new].clone())
        mod.register_parameter("new_token_rows", rows)
        mod.register_forward_hook(_hook(n_base, n_new))
    model._music_token_span = (n_base, n_new)
    return model


def save_new_token_rows(model, path):
    """PEFT's save_pretrained does not include this parameter, so it is stored beside the adapter."""
    n_base, n_new = model._music_token_span
    blob = {"n_base": n_base, "n_new": n_new, "rows": {}}
    for name, mod in _targets(model):
        rows = mod.new_token_rows.detach().cpu()
        blob["rows"][name] = rows
        with torch.no_grad():
            mod.weight[n_base:n_base + n_new].copy_(rows.to(mod.weight.device, dtype=mod.weight.dtype))
    torch.save(blob, path)
    return path


def load_new_token_rows(model, path):
    blob = torch.load(path, map_location="cpu")
    n_base, n_new = int(blob["n_base"]), int(blob["n_new"])
    attach_new_token_training(model, n_base, n_new)
    for name, mod in _targets(model):
        rows = blob["rows"][name]
        device, dtype = mod.new_token_rows.device, mod.new_token_rows.dtype
        with torch.no_grad():
            mod.new_token_rows.copy_(rows.to(device=device, dtype=dtype))
            mod.weight[n_base:n_base + n_new].copy_(rows.to(device=mod.weight.device, dtype=mod.weight.dtype))
    model._music_token_span = (n_base, n_new)
    return model
