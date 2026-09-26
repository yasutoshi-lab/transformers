#!/usr/bin/env python
"""M0: fla-core KDA smoke test on Blackwell (sm_120) — Camellia design §7.

Gate: KDA (fla/ops/kda) forward/backward must run on sm_120 in bf16, including
the varlen (cu_seqlens) path used for packed training. If this fails, the
design falls back to a pure-MLA configuration (Notes 決定記録 2).

fla-core 0.5.2 API notes (verified on this host):
  * fused_kda_gate(g, A_log, dt_bias) — A_log/dt_bias must be fp32 on CUDA.
  * ShortConvolution.forward: with cu_seqlens, input must be [1, total, C]
    (packed/flattened convention, B=1).
  * chunk_kda / fused_recurrent_kda: same cu_seqlens convention.

Tests:
  A. Single-sequence chunk fwd/bwd (bf16, CUDA)
  B. Packed varlen (cu_seqlens, flattened [1, B*T]) fwd/bwd
  C. chunk vs fused_recurrent consistency (short sequence)
  D. varlen 1-doc vs plain fwd consistency
  E. 50-step AdamW loop: loss must decrease

Usage:
    CUDA_VISIBLE_DEVICES=0 python m0_kda_smoke.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate


def _check(label: str, cond: bool, info: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label} {info}", flush=True)
    return cond


class KDALayer(nn.Module):
    """Kimi-Linear-style KDA layer with Camellia dims (design §1 KDA 層)."""

    def __init__(self, hidden: int, num_heads: int, head_dim: int, conv_k: int, eps: float = 1e-5):
        super().__init__()
        proj = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, proj, bias=False)
        self.k_proj = nn.Linear(hidden, proj, bias=False)
        self.v_proj = nn.Linear(hidden, proj, bias=False)
        self.q_conv1d = ShortConvolution(proj, kernel_size=conv_k, activation="silu")
        self.k_conv1d = ShortConvolution(proj, kernel_size=conv_k, activation="silu")
        self.v_conv1d = ShortConvolution(proj, kernel_size=conv_k, activation="silu")
        # A_log / dt_bias stay fp32 (Kimi-Linear convention; kernel expects fp32)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.f_a_proj = nn.Linear(hidden, head_dim, bias=False)
        self.f_b_proj = nn.Linear(head_dim, proj, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(proj, dtype=torch.float32))
        self.b_proj = nn.Linear(hidden, num_heads, bias=False)
        self.g_a_proj = nn.Linear(hidden, head_dim, bias=False)
        self.g_b_proj = nn.Linear(head_dim, proj, bias=False)
        self.o_norm = FusedRMSNormGated(head_dim, eps=eps, activation="sigmoid")
        self.o_proj = nn.Linear(proj, hidden, bias=False)

    def _core(self, h: torch.Tensor, cu_seqlens: torch.Tensor | None, mode: str):
        B, T, _ = h.shape
        if cu_seqlens is not None:
            h = h.reshape(1, B * T, -1)  # packed convention: B must be 1
        q = self.q_conv1d(self.q_proj(h), cu_seqlens=cu_seqlens)[0]
        k = self.k_conv1d(self.k_proj(h), cu_seqlens=cu_seqlens)[0]
        v = self.v_conv1d(self.v_proj(h), cu_seqlens=cu_seqlens)[0]
        # 0.5.2 API: g must already be [..., H, K] (head_dim arg removed from the kernel)
        g = self.f_b_proj(self.f_a_proj(h)).view(*h.shape[:-1], self.num_heads, self.head_dim)
        g = fused_kda_gate(g, self.A_log, self.dt_bias)
        beta = self.b_proj(h).float().sigmoid()
        if cu_seqlens is not None:
            # packed convention: single flattened stream
            q = q.view(1, B * T, self.num_heads, self.head_dim)
            k = k.view(1, B * T, self.num_heads, self.head_dim)
            v = v.view(1, B * T, self.num_heads, self.head_dim)
            g = g.view(1, B * T, self.num_heads, self.head_dim)
            beta = beta.view(1, B * T, self.num_heads)
        else:
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)
            g = g.view(B, T, self.num_heads, self.head_dim)
            beta = beta.view(B, T, self.num_heads)
        if mode == "chunk":
            o, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        o = o.reshape(B, T, self.num_heads, self.head_dim)
        h_out = h.reshape(B, T, -1)
        gate = self.g_b_proj(self.g_a_proj(h_out)).view(B, T, self.num_heads, self.head_dim)
        o = self.o_norm(o, gate)
        return self.o_proj(o.reshape(B, T, self.num_heads * self.head_dim))

    def forward(self, h: torch.Tensor, cu_seqlens: torch.Tensor | None = None):
        return self._core(h, cu_seqlens, mode="chunk")


def build_layer(hidden: int, num_heads: int, head_dim: int, conv_k: int, device: str, dtype) -> KDALayer:
    layer = KDALayer(hidden, num_heads, head_dim, conv_k)
    layer = layer.to(device=device)
    for name, p in list(layer.named_parameters()):
        if name not in ("A_log", "dt_bias"):
            p.data = p.data.to(dtype)
    nn.init.normal_(layer.dt_bias, std=0.1)
    return layer


def main() -> int:
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA required"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"device: {name} (sm_{cap[0]}{cap[1]})", flush=True)

    ok = True
    hidden, num_heads, head_dim, conv_k = 1536, 12, 128, 4  # Camellia dims
    layer = build_layer(hidden, num_heads, head_dim, conv_k, device, dtype)
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"KDA layer params: {n_params / 1e6:.2f}M (design estimate ~10.4M)", flush=True)
    ok &= _check("param count near design estimate", 9e6 < n_params < 12e6, f"({n_params / 1e6:.2f}M)")

    # --- A. single sequence fwd/bwd -------------------------------------
    B, T = 2, 1024
    h = torch.randn(B, T, hidden, device=device, dtype=dtype, requires_grad=True)
    out = layer(h)
    out.float().pow(2).mean().backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    ok &= _check("A: single-seq chunk fwd/bwd finite", torch.isfinite(out).all().item() and grads_ok)

    # --- B. packed varlen (cu_seqlens) fwd/bwd ---------------------------
    B, T = 2, 1024
    h = torch.randn(B, T, hidden, device=device, dtype=dtype, requires_grad=True)
    # 2 docs per row: [400, 624] and [512, 512] over flattened B*T = 2048
    cu = torch.tensor([0, 400, 1024, 1536, 2048], device=device, dtype=torch.long)
    out = layer(h, cu_seqlens=cu)
    out.float().pow(2).mean().backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    ok &= _check("B: varlen (cu_seqlens) fwd/bwd finite", torch.isfinite(out).all().item() and grads_ok)

    # --- C. chunk vs fused_recurrent consistency -------------------------
    Tc = 512
    h = torch.randn(1, Tc, hidden, device=device, dtype=dtype)
    o_chunk = layer._core(h, None, mode="chunk")
    o_recur = layer._core(h, None, mode="fused_recurrent")
    diff = (o_chunk.float() - o_recur.float()).abs().max().item()
    rel = diff / max(o_chunk.float().abs().max().item(), 1e-6)
    ok &= _check("C: chunk ~= fused_recurrent", rel < 5e-2, f"(max abs diff={diff:.3e}, rel={rel:.2e})")

    # --- D. varlen 1-doc vs plain fwd consistency ------------------------
    Td = 384
    h = torch.randn(1, Td, hidden, device=device, dtype=dtype)
    ref = layer(h)
    cu1 = torch.tensor([0, Td], device=device, dtype=torch.long)
    got = layer(h, cu_seqlens=cu1)
    diff = (ref.float() - got.float()).abs().max().item()
    rel = diff / max(ref.float().abs().max().item(), 1e-6)
    ok &= _check("D: varlen 1-doc == plain fwd", rel < 5e-2, f"(max abs diff={diff:.3e}, rel={rel:.2e})")

    # --- E. 50-step AdamW loss decrease ----------------------------------
    layer2 = build_layer(hidden, num_heads, head_dim, conv_k, device, dtype)
    opt = torch.optim.AdamW(layer2.parameters(), lr=1e-3)
    x = torch.randn(1, 256, hidden, device=device, dtype=dtype)
    target = torch.randn(1, 256, hidden, device=device, dtype=dtype)
    first = last = None
    for step in range(50):
        opt.zero_grad()
        out = layer2(x)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(layer2.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    ok &= _check(
        "E: 50-step loss decreases",
        last is not None and first is not None and last < first * 0.98,
        f"({first:.4f} -> {last:.4f})",
    )

    print(f"\nM0 result: {'PASS — KDA is viable on sm_120' if ok else 'FAIL — fallback to pure MLA per design'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
