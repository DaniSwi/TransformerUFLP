"""Encoder-decoder con puntero: la arquitectura Glimpse adaptada al UFLP.

Las dos ideas centrales se respetan:

  Pesos compartidos      una sola capa `embed` codifica todas las filas. No hay
                         "pesos de la facilidad 3" ni "de la 17": permutar las
                         filas de la entrada permuta los logits.
  Independencia de n     ningún parámetro tiene una dimensión igual a M. El
                         puntero es un producto punto, no una capa densa de
                         salida, así que el mismo modelo acepta instancias de
                         cualquier tamaño.

Dos diferencias deliberadas con Kool et al. (2019):

  LayerNorm en vez de BatchNorm, porque evaluamos instancia por instancia y en
  tamaños que no vimos entrenando, donde las estadísticas de lote quedarían mal
  calibradas.

  El encoder corre en cada paso, no una vez por instancia: seis de nuestras
  nueve columnas dependen del estado. Congelar el ranking inicial da soluciones
  58 % más caras.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import MODEL, ModelConfig
from .features import C_MASCARA, C_STOP, C_ULTIMA


class EncoderLayer(nn.Module):
    """MHA + residual + LayerNorm, luego feed-forward + residual + LayerNorm."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + self.drop(a))
        h = self.norm2(h + self.drop(self.ff(h)))
        return h


class UFLP_Attention(nn.Module):
    """Encoder de atención + decoder con glimpse y puntero enmascarado.

    Entrada : (B, n, 9)   con n = M + 1 filas (M candidatas + la fila STOP)
    Salida  : (B, n)      un logit por fila; softmax -> probabilidad de acción
    """

    def __init__(self, cfg: ModelConfig = MODEL):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # ---------------- ENCODER ----------------
        self.embed = nn.Linear(cfg.vec_len, d)          # pesos COMPARTIDOS
        self.layers = nn.ModuleList(
            EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers))

        # ---------------- DECODER ----------------
        self.W_ctx = nn.Linear(3 * d, d, bias=False)    # [h_graph; h_ult; h_stop]
        self.glimpse = nn.MultiheadAttention(d, cfg.n_heads, batch_first=True)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        """(B, n, 9) -> H (B, n, d) y h_graph (B, d)."""
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return h, h.mean(dim=1)

    def decode(self, x: torch.Tensor, h: torch.Tensor,
               h_graph: torch.Tensor) -> torch.Tensor:
        """Contexto -> glimpse -> puntero con máscara. Devuelve logits (B, n)."""
        d = self.cfg.d_model
        B = x.size(0)
        ar = torch.arange(B, device=x.device)

        # filas que resumen el estado, localizadas por su flag (nunca por índice)
        i_ultima = x[:, :, C_ULTIMA].argmax(dim=1)
        i_stop = x[:, :, C_STOP].argmax(dim=1)
        contexto = self.W_ctx(
            torch.cat([h_graph, h[ar, i_ultima], h[ar, i_stop]], dim=-1))

        invalida = x[:, :, C_MASCARA] > 0.5                  # (B, n) True = ignorar

        # GLIMPSE: una mirada al conjunto antes de apuntar
        q, _ = self.glimpse(contexto.unsqueeze(1), h, h,
                            key_padding_mask=invalida, need_weights=False)

        # PUNTERO: un escalar por fila
        logits = (self.Wq(q) @ self.Wk(h).transpose(1, 2)).squeeze(1)
        logits = self.cfg.clip_c * torch.tanh(logits / math.sqrt(d))
        return logits.masked_fill(invalida, -1e9)

    def forward(self, x: torch.Tensor,
                return_probabilities: bool = False) -> torch.Tensor:
        if x.dim() == 2:                       # (n, 9) -> (1, n, 9)
            x = x.unsqueeze(0)
        h, h_graph = self.encode(x)
        logits = self.decode(x, h, h_graph)
        if return_probabilities:
            return torch.softmax(logits, dim=-1)
        return logits

    # ------------------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Dónde están los parámetros. Para la tabla de la sección 5.1."""
        def n(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "embedding": n(self.embed),
            f"encoder x{self.cfg.n_encoder_layers}": n(self.layers),
            "contexto (W)": n(self.W_ctx),
            "glimpse": n(self.glimpse),
            "puntero": n(self.Wq) + n(self.Wk),
        }
