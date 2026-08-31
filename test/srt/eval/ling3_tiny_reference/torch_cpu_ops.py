"""Pure-Torch CPU replacements for the fused FLA ops used by Ling3.

The official Hugging Face model remains the source of model structure,
parameters, routing, MLA, residuals, and logits. Only GPU/Triton-only FLA
primitives are replaced with their documented PyTorch equations so that the
official remote model can be used as a CPU numerical golden.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_KDA_LAYER_IDS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22)
_CAPTURE_COUNTERS = {"short_conv": 0, "kda": 0, "gated_rmsnorm": 0}


def _capture_targets() -> set[int]:
    value = os.environ.get("LING3_TORCH_KDA_CAPTURE_LAYERS", "0,1,2,4")
    return {int(item) for item in value.split(",") if item.strip()}


def _capture_tensor(layer_id: int, name: str, value: torch.Tensor) -> None:
    capture_dir = os.environ.get("LING3_TORCH_KDA_CAPTURE_DIR")
    if not capture_dir or layer_id not in _capture_targets():
        return
    destination = Path(capture_dir)
    destination.mkdir(parents=True, exist_ok=True)
    array = value.detach().float().cpu().numpy()
    np.save(destination / f"layer{layer_id:03d}_{name}.npy", array)


def _next_capture_layer(component: str) -> int | None:
    index = _CAPTURE_COUNTERS[component]
    _CAPTURE_COUNTERS[component] = index + 1
    if index >= len(_KDA_LAYER_IDS):
        return None
    return _KDA_LAYER_IDS[index]


def _reset_capture_counters() -> None:
    for component in _CAPTURE_COUNTERS:
        _CAPTURE_COUNTERS[component] = 0


def _activate(x: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation is None:
        return x
    if activation in ("silu", "swish"):
        return F.silu(x)
    raise ValueError(f"Unsupported short-convolution activation: {activation}")


def _conv_sequence(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    cache: torch.Tensor | None,
    activation: str | None,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise conv with fp32 accumulation.

    ``x`` is [B, T, D], ``weight`` is [D, 1, W], and the FLA cache is
    [B, D, W]. Only the last W-1 cached tokens participate in the next window.
    """

    batch, _, hidden = x.shape
    kernel_size = weight.shape[-1]
    x_channels = x.transpose(1, 2)
    if cache is None:
        cache = x.new_zeros(batch, hidden, kernel_size)
    history = cache[:, :, -(kernel_size - 1) :] if kernel_size > 1 else cache[:, :, :0]
    padded = torch.cat((history, x_channels), dim=-1)
    windows = padded.unfold(-1, kernel_size, 1)
    y = (windows.float() * weight[:, 0, :].float()[None, :, None, :]).sum(dim=-1)
    if bias is not None:
        y = y + bias.float()[None, :, None]
    y = y.transpose(1, 2)
    if residual is not None:
        y = y + residual.float()
    y = _activate(y, activation).to(x.dtype)

    new_cache = torch.cat((cache, x_channels), dim=-1)[:, :, -kernel_size:].contiguous()
    return y, new_cache


def short_convolution_forward_cpu(
    self,
    x: torch.Tensor,
    residual: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    cache: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CPU forward compatible with ``fla.modules.ShortConvolution``."""

    if x.device.type != "cpu":
        raise ValueError("Ling3 Torch golden replacement is CPU-only")

    capture_layer = None
    capture_stream = None
    if x.ndim == 3 and x.shape[1] > 1:
        call_index = _CAPTURE_COUNTERS["short_conv"]
        _CAPTURE_COUNTERS["short_conv"] = call_index + 1
        layer_index, stream_index = divmod(call_index, 3)
        if layer_index < len(_KDA_LAYER_IDS):
            capture_layer = _KDA_LAYER_IDS[layer_index]
            capture_stream = ("q", "k", "v")[stream_index]
            _capture_tensor(capture_layer, f"{capture_stream}_proj", x)
    if mask is not None:
        x = x * mask.unsqueeze(-1)

    if cu_seqlens is None:
        y, new_cache = _conv_sequence(
            x, self.weight, self.bias, cache, self.activation, residual
        )
    else:
        if x.shape[0] != 1:
            raise ValueError("Packed short convolution expects batch size 1")
        boundaries = cu_seqlens.detach().cpu().tolist()
        outputs = []
        states = []
        for sequence_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            sequence_cache = (
                None if cache is None else cache[sequence_index : sequence_index + 1]
            )
            sequence_residual = None if residual is None else residual[:, start:end]
            output, state = _conv_sequence(
                x[:, start:end],
                self.weight,
                self.bias,
                sequence_cache,
                self.activation,
                sequence_residual,
            )
            outputs.append(output)
            states.append(state)
        y = torch.cat(outputs, dim=1)
        new_cache = torch.cat(states, dim=0)
    if capture_layer is not None:
        _capture_tensor(capture_layer, f"{capture_stream}_post_conv", y)
    return y, new_cache if output_final_state else None


def rms_norm_gated_forward_cpu(
    self,
    x: torch.Tensor,
    g: torch.Tensor,
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
):
    """CPU equation for ``FusedRMSNormGated(activation='sigmoid')``."""

    capture_layer = _next_capture_layer("gated_rmsnorm")
    if capture_layer is not None:
        _capture_tensor(capture_layer, "o_before_norm", x)
        _capture_tensor(capture_layer, "output_gate", g)
    residual_out = x.float()
    if residual is not None:
        residual_out = residual_out + residual.float()
    variance = residual_out.square().mean(dim=-1, keepdim=True)
    normalized = residual_out * torch.rsqrt(variance + self.eps)
    if self.weight is not None:
        normalized = normalized * self.weight.float()
    if self.bias is not None:
        normalized = normalized + self.bias.float()
    if self.activation in ("silu", "swish"):
        normalized = normalized * F.silu(g.float())
    elif self.activation == "sigmoid":
        normalized = normalized * torch.sigmoid(g.float())
    else:
        raise ValueError(f"Unsupported gated RMSNorm activation: {self.activation}")
    output = normalized.to(x.dtype)
    if capture_layer is not None:
        _capture_tensor(capture_layer, "o_after_norm", output)
    if not prenorm:
        return output
    residual_dtype = torch.float32 if residual_in_fp32 else x.dtype
    return output, residual_out.to(residual_dtype)


def _l2_normalize(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().sum(dim=-1, keepdim=True) + epsilon)).to(
        x.dtype
    )


def _activate_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor:
    heads, key_dim = g.shape[-2:]
    gate = g.float()
    if dt_bias is not None:
        gate = gate + dt_bias.float().reshape(heads, key_dim)
    exp_a = 1.0 if A_log is None else A_log.float().exp().reshape(heads, 1)
    if lower_bound is None:
        return -exp_a * F.softplus(gate)
    return lower_bound * torch.sigmoid(exp_a * gate)


def _recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = v.dtype
    batch, sequence_length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if scale is None:
        scale = key_dim**-0.5
    q, k, v, g, beta = (item.float() for item in (q, k, v, g, beta))
    state = torch.zeros(
        batch, heads, key_dim, value_dim, dtype=torch.float32, device=q.device
    )
    if initial_state is not None:
        state = state + initial_state.float()
    outputs = []
    for token_index in range(sequence_length):
        q_i = q[:, token_index] * scale
        k_i = k[:, token_index]
        v_i = v[:, token_index]
        state = state * torch.exp(g[:, token_index])[..., None]
        residual = v_i - (k_i[..., None] * state).sum(dim=-2)
        state = state + torch.einsum(
            "bhk,bhv->bhkv", beta[:, token_index, :, None] * k_i, residual
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_i, state))
    output = (
        torch.stack(outputs, dim=1)
        if outputs
        else v.new_empty(batch, 0, heads, value_dim)
    )
    return output.to(dtype), state


def kda_cpu_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    state_v_first: bool = False,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pure-Torch CPU implementation of the inference KDA recurrence."""

    capture_layer = _next_capture_layer("kda")
    if capture_layer is not None:
        _capture_tensor(capture_layer, "q_post_conv", q)
        _capture_tensor(capture_layer, "k_post_conv", k)
        _capture_tensor(capture_layer, "v_post_conv", v)
        _capture_tensor(capture_layer, "raw_gate", g)
        _capture_tensor(capture_layer, "raw_beta", beta)
    if state_v_first:
        raise NotImplementedError("Ling3 Tiny uses state_v_first=False")
    if use_qk_l2norm_in_kernel:
        q, k = _l2_normalize(q), _l2_normalize(k)
    if use_gate_in_kernel:
        g = _activate_kda_gate(g, A_log, dt_bias, lower_bound)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float()).to(beta.dtype)
        if allow_neg_eigval:
            beta = beta * 2
    if capture_layer is not None:
        _capture_tensor(capture_layer, "q_normalized", q)
        _capture_tensor(capture_layer, "k_normalized", k)
        _capture_tensor(capture_layer, "activated_gate", g)
        _capture_tensor(capture_layer, "beta", beta)

    if cu_seqlens is None:
        output, final_state = _recurrent_kda(q, k, v, g, beta, scale, initial_state)
    else:
        if q.shape[0] != 1:
            raise ValueError("Packed KDA CPU reference expects batch size 1")
        boundaries = cu_seqlens.detach().cpu().tolist()
        outputs = []
        states = []
        for sequence_index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            sequence_state = (
                None
                if initial_state is None
                else initial_state[sequence_index : sequence_index + 1]
            )
            output, state = _recurrent_kda(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                g[:, start:end],
                beta[:, start:end],
                scale,
                sequence_state,
            )
            outputs.append(output)
            states.append(state)
        output = torch.cat(outputs, dim=1)
        final_state = torch.cat(states, dim=0)
    if capture_layer is not None:
        _capture_tensor(capture_layer, "kda_output", output)
        _capture_tensor(capture_layer, "kda_final_state", final_state)
    return output, final_state if output_final_state else None


def install_cpu_reference_ops(model) -> dict[str, str]:
    """Patch only the fused FLA call sites in an official HF model instance."""

    # The golden generator installs the shims once for a reduced self-test and
    # once for the real checkpoint. Reset here so capture layer IDs always refer
    # to the first real prefill after each installation.
    _reset_capture_counters()
    modeling_module = importlib.import_module(model.__class__.__module__)
    modeling_module.ShortConvolution.forward = short_convolution_forward_cpu
    modeling_module.FusedRMSNormGated.forward = rms_norm_gated_forward_cpu
    modeling_module.chunk_kda = kda_cpu_reference
    modeling_module.fused_recurrent_kda = kda_cpu_reference
    return {
        "modeling_module": modeling_module.__name__,
        "short_convolution": "pure_torch_fp32_accumulator",
        "gated_rmsnorm": "pure_torch_fp32_accumulator",
        "kda": "pure_torch_recurrent_reference",
    }
