"""Export Silero TTS v5_5_ru (torch.package) to LiteRT models + frontend data
for the on-device Android engine, using LiteRT Torch (formerly AI Edge Torch).

The original package mixes TorchScript networks with Python glue code.  The
networks are re-implemented here as plain PyTorch modules (dropping training
only branches and a dead decoder layer whose output is never used), loaded
with the original weights, verified against the TorchScript originals and
converted with ``litert_torch``.  The glue (text processing, stress, duration
post-processing, ISTFT) is re-implemented in Kotlin, see ``core/``.

LiteRT runs fixed shapes best (memory planning, XNNPACK packing, and the
Android ``CompiledModel`` API has no input resizing), so every network is
converted with a few static-size signatures that share one copy of the
weights.  Inputs are zero-padded up to the next size and a mask restores the
exact unpadded result on the valid positions: attention gets a key padding
bias and every convolution input is masked, so padded positions look exactly
like the zero padding of the original convolutions.

Outputs (in --out):
    text.tflite        tokens, type_ids, speaker, mask, pitch_scale, pitch_shift
                         -> log_dur, hidden          signatures L<n>
    decoder.tflite     frames (hidden per mel frame), mask -> mel      T<n>
    vocoder.tflite     mel, mask -> re, im (spectrum for a 2400-point ISTFT)  W<n>
    homosolver.tflite  BERT homograph disambiguation                   S<n>
    accentor.bin       int8 fastText n-gram embeddings + MLP weights
    ngrams.bin         open-addressing hash table: n-gram -> id
    bert_vocab.bin     open-addressing hash table: wordpiece -> id
    bert_emb.bin       int8 BERT word embeddings (looked up by the engine)
    exceptions.tsv     stress exceptions
    homodict.tsv       homograph variants
    config.json        symbols, speakers, constants, signature sizes

Tensors are time-major ([1, T, channels]): the decoder output windows feed the
vocoder without transposes and the Kotlin ISTFT reads contiguous frames.

Usage:
    python export_models.py --package v5_5_ru.pt --out export_dir
"""
import argparse
import json
import math
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

MODEL_URL = 'https://models.silero.ai/models/tts/ru/v5_5_ru.pt'

# Static signature sizes.  Text chunks are at most 250 characters (plus stress
# marks), which needs <= 512 tokens; the engine shortens chunks for slow speech
# so that they stay below the largest decoder size (28.8 s).
TEXT_SIZES = [32, 64, 128, 256, 512]
# multiples of the hourglass shorten factor (3)
DECODER_SIZES = [96, 144, 192, 288, 384, 576, 768, 1152, 1536, 2304]
# vocoder windows: 80 = first window (52 frames + 28 frames of right context),
# 320 = steady state (264 frames + 2 x 28 context), 160 for short remainders
VOCODER_SIZES = [80, 160, 320]
VOCODER_CONTEXT = 28
BERT_SIZES = [32, 64, 128, 256, 512]

# additive attention bias of padded keys; exp(-1e4) == 0 in fp32 and fp16
MASK_BIAS = 1e4


def key_bias(mask):
    """[1, L] float mask (1 = valid) -> additive attention bias [1, 1, 1, L]."""
    return ((mask - 1.0) * MASK_BIAS)[:, None, None, :]


# ----------------------------------------------------------------------------
# Clean re-implementations (with padding masks)
# ----------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.d, self.h = d, heads
        self.hd = d // heads
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)

    def load(self, mha):
        self.in_proj.weight.copy_(mha.in_proj_weight)
        self.in_proj.bias.copy_(mha.in_proj_bias)
        self.out_proj.weight.copy_(mha.out_proj.weight)
        self.out_proj.bias.copy_(mha.out_proj.bias)

    def forward(self, x, bias):
        b, l, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.reshape(b, l, self.h, self.hd).transpose(1, 2)
        k = k.reshape(b, l, self.h, self.hd).transpose(1, 2)
        v = v.reshape(b, l, self.h, self.hd).transpose(1, 2)
        att = torch.softmax(torch.matmul(q * (1.0 / math.sqrt(self.hd)), k.transpose(2, 3)) + bias, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(b, l, self.d)
        return self.out_proj(y)


class FFTBlock(nn.Module):
    """Post-norm transformer block with a convolutional feed-forward."""

    def __init__(self, src):
        super().__init__()
        d = src.norm1.weight.shape[0]
        self.attn = SelfAttention(d, src.self_attn.num_heads)
        self.attn.load(src.self_attn)
        self.conv1 = nn.Conv1d(d, src.conv1.weight.shape[0], src.conv1.weight.shape[2], padding=src.conv1.padding[0])
        self.conv2 = nn.Conv1d(src.conv2.weight.shape[1], d, 1)
        self.norm1 = nn.LayerNorm(d, eps=src.norm1.eps)
        self.norm2 = nn.LayerNorm(d, eps=src.norm2.eps)
        assert not src.use_linear and not src.pitch_pred
        for name in ['conv1', 'conv2', 'norm1', 'norm2']:
            getattr(self, name).load_state_dict(getattr(src, name).state_dict())

    def forward(self, x, m, bias):
        # m [1, L, 1]: zeroing padded positions before the k=9 convolution makes
        # them identical to the convolution's own zero padding
        x = self.norm1(x + self.attn(x, bias))
        y = (x * m).transpose(1, 2)
        y = y + self.conv2(F.relu(self.conv1(y)))
        return self.norm2(y.transpose(1, 2))


def positions(mask):
    """Positions 0, 1, ... of the valid elements, [1, L] float.  Computed at
    runtime so that the converter does not fold a table into every signature
    (padded positions repeat the last one; they are masked anyway)."""
    return torch.cumsum(mask, dim=1) - 1.0


class PosEnc(nn.Module):
    """Sinusoidal positional encoding computed in-graph (the original stores a
    5000 x d table)."""

    def __init__(self, src):
        super().__init__()
        d = src.pe.shape[-1]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        ref = torch.zeros(src.pe.shape[0], d)
        pos = torch.arange(src.pe.shape[0]).unsqueeze(1).float()
        ref[:, 0::2] = torch.sin(pos * div)
        ref[:, 1::2] = torch.cos(pos * div)
        err = (ref - src.pe[:, 0, :]).abs().max().item()
        # the pitch predictor table deviates from the float32 sinusoid by <3e-5
        # for the first 1000 positions (it was generated differently); harmless
        assert err < 1e-3, f'positional encoding is not the standard sinusoid (max err {err})'
        self.register_buffer('div', div)
        self.register_buffer('scale', src.scale.clone())
        self.d = d

    def forward(self, x, mask):
        ang = positions(mask).unsqueeze(-1) * self.div
        pe = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(x.shape[0], x.shape[1], self.d)
        return x + self.scale * pe


class ForwardTransformer(nn.Module):
    def __init__(self, src, n_layers):
        super().__init__()
        self.pos = PosEnc(src.pos_encoder)
        self.layers = nn.ModuleList([FFTBlock(getattr(src.layers, str(i))) for i in range(n_layers)])
        self.norm = nn.LayerNorm(src.norm.weight.shape[0], eps=src.norm.eps)
        self.norm.load_state_dict(src.norm.state_dict())

    def forward(self, x, mask):
        m, bias = mask.unsqueeze(-1), key_bias(mask)
        x = self.pos(x, mask)
        for layer in self.layers:
            x = layer(x, m, bias)
        return self.norm(x)


class HourGlass(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.pos = PosEnc(src.pos_encoder)
        self.pre = FFTBlock(getattr(src.pre_vanilla_layers, '0'))
        # The original runs shorten_layers[0] and discards its output; only
        # shorten_layers[1] contributes.
        self.short = FFTBlock(getattr(src.shorten_layers, '1'))
        self.post = FFTBlock(getattr(src.post_vanilla_layers, '0'))
        self.k = src.shorten_factor
        self.up = nn.Linear(src.upsample.proj.weight.shape[1], src.upsample.proj.weight.shape[0])
        self.up.load_state_dict(src.upsample.proj.state_dict())
        self.dim = src.upsample.dim
        self.norm = nn.LayerNorm(src.norm.weight.shape[0], eps=src.norm.eps)
        self.norm.load_state_dict(src.norm.state_dict())

    def forward(self, x, mask):
        # x [1, T, d] with T a multiple of k, mask [1, T]
        b, t, d = x.shape
        assert t % self.k == 0
        m, bias = mask.unsqueeze(-1), key_bias(mask)
        # a group of k frames is valid if any of its frames is
        mask_s = mask.reshape(b, t // self.k, self.k).amax(-1)
        x = self.pre(self.pos(x, mask), m, bias)
        # the original zero-pads to a multiple of k before pooling
        x = x * m
        down = F.avg_pool1d(x.transpose(1, 2), self.k, self.k).transpose(1, 2)
        y = self.up(self.short(down, mask_s.unsqueeze(-1), key_bias(mask_s))).reshape(b, -1, self.dim)
        return self.norm(self.post(y + x, m, bias))


def copy_emb(src):
    e = nn.Embedding(*src.weight.shape)
    e.weight.copy_(src.weight)
    return e


class TextModel(nn.Module):
    """Token level: duration + pitch predictors, pitch scaling and the encoder.

    One call per sentence.  ``pitch_scale``/``pitch_shift`` implement the
    package's ``update_pitch_coef`` (shift = (coef - 1) * mean_std_coef[speaker],
    computed by the engine), so the predicted pitch never leaves the graph.
    """

    def __init__(self, tts):
        super().__init__()
        dp = tts.dur_predictor.dur_pred
        pp = tts.pitch_predictor.pitch_pred
        assert not dp.utt_type_emb and not dp.focus_emb and pp.utt_type_emb
        self.d_emb, self.d_spk = copy_emb(dp.embedding), copy_emb(dp.speaker_embedding)
        self.d_tr = ForwardTransformer(dp.transformer, 4)
        self.d_lin = nn.Linear(64, 1)
        self.d_lin.load_state_dict(dp.lin.state_dict())
        self.p_emb, self.p_spk, self.p_type = copy_emb(pp.embedding), copy_emb(pp.speaker_embedding), copy_emb(pp.type_embedding)
        self.p_tr = ForwardTransformer(pp.transformer, 4)
        self.p_lin = nn.Linear(64, 1)
        self.p_lin.load_state_dict(pp.lin.state_dict())
        t = tts.tacotron
        assert t.pitch_emb_type == 'default' and t.pitch_strength == 1.0
        self.emb, self.spk = copy_emb(t.embedding), copy_emb(t.speaker_embedding)
        self.enc = ForwardTransformer(t.encoder, 4)
        self.pitch_proj = nn.Conv1d(1, 128, 3, padding=1)
        self.pitch_proj.load_state_dict(t.pitch_proj.state_dict())

    def predict(self, tokens, type_ids, speaker, mask):
        x = self.d_emb(tokens) + self.d_spk(speaker).unsqueeze(1)
        log_dur = self.d_lin(self.d_tr(x, mask)).squeeze(-1)
        x = self.p_emb(tokens) + self.p_spk(speaker).unsqueeze(1) + self.p_type(type_ids)
        pitch = self.p_lin(self.p_tr(x, mask)).squeeze(-1)
        return log_dur, pitch

    def encode(self, tokens, speaker, pitch, mask):
        x = self.enc(self.emb(tokens), mask) + self.spk(speaker).unsqueeze(1)
        return x + self.pitch_proj(pitch.unsqueeze(1)).transpose(1, 2)

    def forward(self, tokens, type_ids, speaker, mask, pitch_scale, pitch_shift):
        # tokens, type_ids [1, L] int32; speaker [1] int32; mask [1, L];
        # pitch_scale, pitch_shift [1]
        log_dur, pitch = self.predict(tokens, type_ids, speaker, mask)
        # update_pitch_coef
        pitch = torch.where(pitch.abs() < 0.001, 0.0, pitch)
        pc = pitch * pitch_scale
        pitch = torch.where(pc == 0, 0.0, pc + pitch_shift) * mask
        hidden = self.encode(tokens, speaker, pitch, mask)
        return {'log_dur': log_dur, 'hidden': hidden}  # [1, L], [1, L, 128]


class DecoderModel(nn.Module):
    """Frame level: hourglass decoder + mel projection."""

    def __init__(self, tts):
        super().__init__()
        t = tts.tacotron
        self.dec = HourGlass(t.decoder)
        self.lin = nn.Linear(128, t.n_mel_channels)
        self.lin.load_state_dict(t.lin.state_dict())

    def forward(self, frames, mask):
        # frames [1, T, 128]: encoder output of the token each frame belongs to
        # (length regulator, done by the engine); mask [1, T]
        return {'mel': self.lin(self.dec(frames, mask))}  # [1, T, 192]


class ConvNeXt(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.dw = nn.Conv1d(512, 512, 7, padding=3, groups=512)
        self.dw.load_state_dict(src.dwconv.state_dict())
        self.norm = nn.LayerNorm(512, eps=src.norm.eps)
        self.norm.load_state_dict(src.norm.state_dict())
        self.pw1 = nn.Linear(512, 1536)
        self.pw1.load_state_dict(src.pwconv1.state_dict())
        self.pw2 = nn.Linear(1536, 512)
        self.pw2.load_state_dict(src.pwconv2.state_dict())
        self.gamma = nn.Parameter(src.gamma.clone())

    def forward(self, x, m):
        # x [1, T, 512] (time-major)
        y = self.norm(self.dw((x * m).transpose(1, 2)).transpose(1, 2))
        return x + self.pw2(F.gelu(self.pw1(y))) * self.gamma


class VocoderModel(nn.Module):
    """Vocos backbone + ISTFT head up to the complex spectrum.

    The inverse FFT / overlap-add runs in Kotlin (n_fft=2400 is not a power of
    two), which also allows streaming: the engine runs the vocoder in windows
    with VOCODER_CONTEXT frames of context, which covers the receptive field of
    the 9 convolutions (27 frames).
    """

    def __init__(self, tts):
        super().__init__()
        bb = tts.vocoder.backbone
        self.embed = nn.Conv1d(192, 512, 7, padding=3)
        self.embed.load_state_dict(bb.embed.state_dict())
        self.norm = nn.LayerNorm(512, eps=bb.norm.eps)
        self.norm.load_state_dict(bb.norm.state_dict())
        self.blocks = nn.ModuleList([ConvNeXt(getattr(bb.convnext, str(i))) for i in range(8)])
        self.final = nn.LayerNorm(512, eps=bb.final_layer_norm.eps)
        self.final.load_state_dict(bb.final_layer_norm.state_dict())
        self.out = nn.Linear(512, 2402)
        self.out.load_state_dict(tts.vocoder.head.out.state_dict())
        self.n_bins = 1201

    def forward(self, mel, mask):
        # mel [1, W, 192], mask [1, W]
        m = mask.unsqueeze(-1)
        x = self.norm(self.embed((mel * m).transpose(1, 2)).transpose(1, 2))
        for b in self.blocks:
            x = b(x, m)
        x = self.out(self.final(x))
        mag = torch.clamp(torch.exp(x[..., :self.n_bins]), max=100.0)
        p = x[..., self.n_bins:]
        return {'re': mag * torch.cos(p), 'im': mag * torch.sin(p)}  # [1, W, 1201] each


class BertLayer(nn.Module):
    def __init__(self, src):
        super().__init__()
        s = src.attention.self
        self.h, self.hd = s.num_attention_heads, s.attention_head_size
        self.q, self.k, self.v = [nn.Linear(312, 312) for _ in range(3)]
        self.q.load_state_dict(s.query.state_dict())
        self.k.load_state_dict(s.key.state_dict())
        self.v.load_state_dict(s.value.state_dict())
        self.o = nn.Linear(312, 312)
        self.o.load_state_dict(src.attention.output.dense.state_dict())
        self.ln1 = nn.LayerNorm(312, eps=src.attention.output.LayerNorm.eps)
        self.ln1.load_state_dict(src.attention.output.LayerNorm.state_dict())
        d_ff = src.intermediate.dense.weight.shape[0]
        self.i = nn.Linear(312, d_ff)
        self.i.load_state_dict(src.intermediate.dense.state_dict())
        self.out = nn.Linear(d_ff, 312)
        self.out.load_state_dict(src.output.dense.state_dict())
        self.ln2 = nn.LayerNorm(312, eps=src.output.LayerNorm.eps)
        self.ln2.load_state_dict(src.output.LayerNorm.state_dict())

    def forward(self, x, bias):
        b, l, d = x.shape
        q = self.q(x).reshape(b, l, self.h, self.hd).transpose(1, 2)
        k = self.k(x).reshape(b, l, self.h, self.hd).transpose(1, 2)
        v = self.v(x).reshape(b, l, self.h, self.hd).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hd) + bias, dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(b, l, d)
        x = self.ln1(self.o(y) + x)
        return self.ln2(self.out(F.gelu(self.i(x))) + x)


FRONTEND_LETTERS = set('абвгдежзийклмнопрстуфхцчшщъыьэюяё')
FRONTEND_PUNCT = set('!+,-.:;?–…')
BERT_SPECIAL = ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[HOMO]', '[/HOMO]')


def prune_vocab(vocab):
    """Keep only wordpieces the frontend can produce: the text reaching BERT is
    lowercased Cyrillic + a few punctuation marks, so tokens with Latin letters,
    digits, capitals etc. are unreachable.  Lossless: greedy wordpiece matching
    never considers a token containing characters absent from the input.
    Returns the new token -> id mapping and the kept old ids (new id order)."""
    kept = []
    for tok, i in sorted(vocab.items(), key=lambda x: x[1]):
        body = tok[2:] if tok.startswith('##') else tok
        if tok in BERT_SPECIAL or (body and all(c in FRONTEND_LETTERS for c in body)) or \
                (len(tok) == 1 and tok in FRONTEND_PUNCT):
            kept.append((tok, i))
    return {tok: n for n, (tok, _) in enumerate(kept)}, [i for _, i in kept]


class HomoSolver(nn.Module):
    """BERT homograph classifier, one homograph per call (the sequences of one
    sentence all have the same length, so the original never pads them).

    The int8 word embedding lookup is done by the engine (bert_emb.bin): the
    gathered rows are de-quantized exactly like the package does and passed
    in as ``embeddings``.
    """

    def __init__(self, hs, max_len):
        super().__init__()
        bert = hs.bert
        self.register_buffer('pos', bert.embeddings.position_embeddings.weight[:max_len].detach().clone())
        self.register_buffer('tok', bert.embeddings.token_type_embeddings.weight[0].detach().clone())
        self.ln = nn.LayerNorm(312, eps=bert.embeddings.LayerNorm.eps)
        self.ln.load_state_dict(bert.embeddings.LayerNorm.state_dict())
        self.layers = nn.ModuleList([BertLayer(getattr(bert.encoder.layer, str(i))) for i in range(bert.num_hidden_layers)])
        clf = hs.homo_clf
        self.c0 = nn.Linear(624, 256)
        self.c0.load_state_dict(getattr(clf, '0').state_dict())
        self.c1 = nn.Linear(256, 1)
        self.c1.load_state_dict(getattr(clf, '3').state_dict())

    def forward(self, embeddings, mask, start, end):
        # embeddings [B, S, 312] (de-quantized word embeddings), mask [B, S],
        # start/end [B] int32 (positions of [HOMO] / [/HOMO])
        s = embeddings.shape[1]
        x = self.ln(embeddings + self.pos[positions(mask).to(torch.int32)] + self.tok)
        bias = key_bias(mask)
        for layer in self.layers:
            x = layer(x, bias)
        pos = torch.arange(s, device=x.device, dtype=start.dtype).unsqueeze(0)
        marker = (pos == start.unsqueeze(1)).float().unsqueeze(-1)
        inside = ((pos > start.unsqueeze(1)) & (pos < end.unsqueeze(1))).float().unsqueeze(-1)
        marker_emb = (x * marker).sum(1)
        word_emb = (x * inside).sum(1) / inside.sum(1)
        h = F.relu(self.c0(torch.cat([marker_emb, word_emb], dim=1)))
        return {'logit': self.c1(h).squeeze(-1)}  # [B]


class BertEmbeddings:
    """int8 word embeddings of the reachable wordpieces (pruned vocabulary)."""

    def __init__(self, hs, kept_ids):
        bert = hs.bert
        wq = bert.embeddings.word_embeddings.weight.detach().clone().to(torch.int8)
        self.wq = wq[torch.tensor(kept_ids)].clone()
        self.scale = float(bert.scale)
        self.zp = float(bert.zero_point)

    def __call__(self, ids):
        return (self.wq[ids].float() - self.zp) * self.scale

    def write(self, path):
        """Layout: 'BEM1', V, D, scale(f32), zp(f32), int8[V*D]."""
        with open(path, 'wb') as f:
            f.write(struct.pack('<4sii', b'BEM1', *self.wq.shape))
            f.write(struct.pack('<ff', self.scale, self.zp))
            f.write(self.wq.numpy().tobytes())


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------

def check(name, a, b, tol):
    err = (a - b).abs().max().item()
    scale = b.abs().max().item()
    print(f'  {name}: max abs err {err:.3e} (max |ref| {scale:.3e})')
    if err > tol * max(1.0, scale):
        raise SystemExit(f'verification failed for {name}')


def i32(x):
    return torch.as_tensor(x).to(torch.int32)


def pad_to(x, size, dim, value=0):
    pad = [0, 0] * (x.dim() - dim - 1) + [0, size - x.shape[dim]]
    return F.pad(x, pad, value=value)


def length_mask(n, size):
    m = torch.zeros(1, size)
    m[:, :n] = 1
    return m


def verify(tts, hs, text_model, decoder, vocoder, homo, bert_emb, symbol_to_id, tokenizer, remap):
    """Re-implementations vs the TorchScript originals (unpadded, then padded)."""
    print('Verifying re-implementations against TorchScript ...')
    text = '|прив+ет! к+ак тво+и дел+а, др+уг? всё хорош+о.~'
    tokens = torch.tensor([[symbol_to_id[c] for c in text]])
    spk = torch.tensor([4])
    L = tokens.shape[1]
    mask = torch.zeros(1, L, dtype=torch.bool)
    type_ids = torch.randint(0, 6, (1, L))
    ref_dur = tts.dur_predictor(tokens, spk, mask, 1.0, None)
    ref_pitch = tts.pitch_predictor(tokens, spk, mask, type_ids, None)
    ones = torch.ones(1, L)
    log_dur, pitch = text_model.predict(i32(tokens), i32(type_ids), i32(spk), ones)
    check('log_dur', log_dur, ref_dur, 1e-4)
    check('pitch', pitch.unsqueeze(1), ref_pitch, 1e-4)

    durs = torch.clamp(torch.round(torch.exp(ref_dur) - 1), min=0)
    durs[0, 0] = min(durs[0, 0].item(), 5)
    durs[0, -2] = 13
    ref_mel = tts.tacotron(tokens, spk, mask, durs.clone(), ref_pitch, None)
    hidden = text_model.encode(i32(tokens), i32(spk), ref_pitch[:, 0], ones)
    frame_idx = torch.repeat_interleave(torch.arange(L), durs[0].long())
    frames = hidden[:, frame_idx]
    T = frames.shape[1]
    Tp = -(-T // 3) * 3
    mel = decoder(pad_to(frames, Tp, 1), length_mask(T, Tp))['mel'][:, :T]
    check('mel', mel.transpose(1, 2), ref_mel, 1e-4)

    ref_audio = tts.vocoder(ref_mel, 48000, 0., True)
    out = vocoder(ref_mel.transpose(1, 2), torch.ones(1, T))
    spec = torch.complex(out['re'], out['im']).transpose(1, 2)
    audio = tts.vocoder.head.istft(spec)
    check('audio', audio, ref_audio, 1e-4)

    print('Verifying padding masks ...')
    for size in [L + 1, L + 37]:
        pm = length_mask(L, size)
        o = text_model(pad_to(i32(tokens), size, 1), pad_to(i32(type_ids), size, 1), i32(spk), pm,
                       torch.tensor([1.1]), torch.tensor([0.3]))
        o_ref = text_model(i32(tokens), i32(type_ids), i32(spk), ones, torch.tensor([1.1]), torch.tensor([0.3]))
        check(f'text L={size} log_dur', o['log_dur'][:, :L], o_ref['log_dur'], 1e-5)
        check(f'text L={size} hidden', o['hidden'][:, :L], o_ref['hidden'], 1e-5)
    for size in [Tp + 3, Tp + 96]:
        m2 = decoder(pad_to(frames, size, 1), length_mask(T, size))['mel'][:, :T]
        check(f'decoder T={size}', m2, mel, 1e-5)
    mel_t = ref_mel.transpose(1, 2)
    ref = vocoder(mel_t, torch.ones(1, T))
    o = vocoder(pad_to(mel_t, T + 50, 1), length_mask(T, T + 50))
    check('vocoder padded re', o['re'][:, :T], ref['re'], 1e-5)
    check('vocoder padded im', o['im'][:, :T], ref['im'], 1e-5)

    sent = 'мой замок на горе, а на двери висит замок.'
    marked = [sent[:4] + ' [HOMO] ' + sent[4:9] + ' [/HOMO] ' + sent[9:],
              sent[:-6] + ' [HOMO] ' + sent[-6:-1] + ' [/HOMO] ' + sent[-1:]]
    for m in marked:
        ids = torch.tensor([tokenizer(m)])
        st = torch.tensor([ids[0].tolist().index(tokenizer.homo_start_id)])
        en = torch.tensor([ids[0].tolist().index(tokenizer.homo_end_id)])
        ref = hs(ids, st, en).reshape(-1)
        S = ids.shape[1]
        emb = bert_emb(remap[ids])
        out = homo(emb, torch.ones(1, S), i32(st), i32(en))['logit']
        check('homo logit', out, ref, 1e-4)
        out = homo(pad_to(emb, S + 20, 1), length_mask(S, S + 20), i32(st), i32(en))['logit']
        check('homo logit padded', out, ref, 1e-4)


# ----------------------------------------------------------------------------
# LiteRT conversion
# ----------------------------------------------------------------------------

def text_inputs(n):
    return {'tokens': torch.zeros(1, n, dtype=torch.int32), 'type_ids': torch.zeros(1, n, dtype=torch.int32),
            'speaker': torch.zeros(1, dtype=torch.int32), 'mask': torch.ones(1, n),
            'pitch_scale': torch.ones(1), 'pitch_shift': torch.zeros(1)}


def decoder_inputs(n):
    return {'frames': torch.zeros(1, n, 128), 'mask': torch.ones(1, n)}


def vocoder_inputs(n):
    return {'mel': torch.zeros(1, n, 192), 'mask': torch.ones(1, n)}


def homo_inputs(n):
    return {'embeddings': torch.zeros(1, n, 312), 'mask': torch.ones(1, n),
            'start': torch.zeros(1, dtype=torch.int32), 'end': torch.ones(1, dtype=torch.int32)}


SIGNATURES = {
    # name: (prefix, sizes, sample input factory)
    'text': ('L', TEXT_SIZES, text_inputs),
    'decoder': ('T', DECODER_SIZES, decoder_inputs),
    'vocoder': ('W', VOCODER_SIZES, vocoder_inputs),
    'homosolver': ('S', BERT_SIZES, homo_inputs),
}


def convert(module, name, out_dir):
    import litert_torch
    prefix, sizes, inputs = SIGNATURES[name]
    builder = litert_torch
    for n in sizes:
        builder = builder.signature(f'{prefix}{n}', module, sample_kwargs=inputs(n))
    edge = builder.convert()
    path = os.path.join(out_dir, name + '.tflite')
    edge.export(path)
    print(f'  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, signatures {prefix}{sizes})')


# ----------------------------------------------------------------------------
# Data files
# ----------------------------------------------------------------------------

def fnv1a(s):
    """32-bit FNV-1a over UTF-16 code units (mirrored in Kotlin)."""
    h = 0x811C9DC5
    data = s.encode('utf-16-le')
    for i in range(0, len(data), 2):
        h ^= data[i] | (data[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def write_hash_table(path, mapping):
    """Open-addressing (linear probing) string -> int table.

    Layout (little endian):
        int32 magic 'SHT1', int32 n_slots (power of 2), int32 n_entries
        int32[n_slots]  slot -> entry index or -1
        int32[n_entries] value
        int32[n_entries + 1] char offsets into the string blob
        uint16[...]      UTF-16 string blob
    """
    keys = list(mapping.keys())
    n = len(keys)
    n_slots = 1
    while n_slots < 2 * n:
        n_slots *= 2
    slots = np.full(n_slots, -1, dtype='<i4')
    for idx, k in enumerate(keys):
        s = fnv1a(k) & (n_slots - 1)
        while slots[s] != -1:
            s = (s + 1) & (n_slots - 1)
        slots[s] = idx
    values = np.array([mapping[k] for k in keys], dtype='<i4')
    offsets = np.zeros(n + 1, dtype='<i4')
    blob = []
    for i, k in enumerate(keys):
        units = np.frombuffer(k.encode('utf-16-le'), dtype='<u2')
        blob.append(units)
        offsets[i + 1] = offsets[i] + len(units)
    blob = np.concatenate(blob) if blob else np.zeros(0, dtype='<u2')
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sii', b'SHT1', n_slots, n))
        f.write(slots.tobytes())
        f.write(values.tobytes())
        f.write(offsets.tobytes())
        f.write(blob.astype('<u2').tobytes())


def write_accentor(path, acc_model):
    """int8 embedding [V, 16] + float32 MLP weights.

    Layout: 'ACC1', V, D, scale(f32), zp(f32), int8[V*D] (padded to 4 bytes),
    then for stress_clf and yo_clf: n_layers, per layer (out, in, W[out*in], b[out]).
    """
    emb = acc_model.embedding.weight
    scale, zp = float(acc_model.scale), float(acc_model.zero_point)
    q = torch.round(emb / scale + zp)
    assert (scale * (q - zp) - emb).abs().max() == 0, 'accentor embedding is not int8-representable'
    q = q.to(torch.int8).numpy()
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sii', b'ACC1', q.shape[0], q.shape[1]))
        f.write(struct.pack('<ff', scale, zp))
        raw = q.tobytes()
        raw += b'\0' * ((-len(raw)) % 4)
        f.write(raw)
        for clf in [acc_model.stress_clf, acc_model.yo_clf]:
            layers = [getattr(clf, str(i)) for i in (0, 2, 4, 6)]
            f.write(struct.pack('<i', len(layers)))
            for lin in layers:
                w = lin.weight.detach().numpy().astype('<f4')
                b = lin.bias.detach().numpy().astype('<f4')
                f.write(struct.pack('<ii', w.shape[0], w.shape[1]))
                f.write(w.tobytes())
                f.write(b.tobytes())
    # make sure the MLPs are Linear-ReLU-Linear-ReLU-Linear-ReLU-Linear
    for clf in [acc_model.stress_clf, acc_model.yo_clf]:
        names = [getattr(clf, str(i)).original_name for i in range(7)]
        assert names == ['Linear', 'ReLU'] * 3 + ['Linear'], names


def load_package(path):
    if not os.path.isfile(path):
        print(f'Downloading {MODEL_URL}')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.hub.download_url_to_file(MODEL_URL, path)
    imp = torch.package.PackageImporter(path)
    model = imp.load_pickle('tts_models', 'model')
    assert len(model.packages) == 1
    pk = model.packages[0]
    assert len(pk.models) == 1 and not pk.phons
    return pk


def build_modules(pk):
    """Returns (modules by name, extra objects needed for verification / data)."""
    tts = pk.models[0]
    hs_wrapper = pk.accentor.homosolver
    tokenizer = hs_wrapper.tokenizer
    # unpack the int8 BERT word embeddings the way the package does at runtime
    hs_ts = hs_wrapper.model
    bert_q = hs_ts.bert.embeddings.word_embeddings.weight.data.clone()
    vocab, kept_ids = prune_vocab(tokenizer.vocab)
    remap = torch.full((len(tokenizer.vocab),), vocab['[UNK]'], dtype=torch.long)
    remap[torch.tensor(kept_ids)] = torch.arange(len(kept_ids))
    bert_emb = BertEmbeddings(hs_ts, kept_ids)
    homo = HomoSolver(hs_ts, max(BERT_SIZES)).eval()
    hs_ts.bert.embeddings.word_embeddings.weight.data = hs_ts.bert.scale * (bert_q.float() - hs_ts.bert.zero_point)
    modules = {'text': TextModel(tts).eval(), 'decoder': DecoderModel(tts).eval(),
               'vocoder': VocoderModel(tts).eval(), 'homosolver': homo}
    return modules, dict(tts=tts, hs=hs_ts, tokenizer=tokenizer, vocab=vocab, remap=remap, bert_emb=bert_emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', default='v5_5_ru.pt', help='path to v5_5_ru.pt (downloaded if missing)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='+', choices=list(SIGNATURES), help='convert only these networks')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pk = load_package(args.package)
    modules, x = build_modules(pk)
    print(f'BERT vocabulary: {len(x["tokenizer"].vocab)} -> {len(x["vocab"])} reachable wordpieces')
    verify(x['tts'], x['hs'], modules['text'], modules['decoder'], modules['vocoder'], modules['homosolver'],
           x['bert_emb'], pk.symbol_to_id, x['tokenizer'], x['remap'])

    print('Converting with LiteRT Torch ...')
    for name, module in modules.items():
        if args.models is None or name in args.models:
            convert(module, name, args.out)

    print('Writing frontend data ...')
    stress = pk.accentor
    acc = stress.accentor
    vocab = x['vocab']
    write_accentor(os.path.join(args.out, 'accentor.bin'), acc.model)
    write_hash_table(os.path.join(args.out, 'ngrams.bin'), acc.model.embedding.ngram_dict)
    write_hash_table(os.path.join(args.out, 'bert_vocab.bin'), vocab)
    x['bert_emb'].write(os.path.join(args.out, 'bert_emb.bin'))
    with open(os.path.join(args.out, 'exceptions.tsv'), 'w', encoding='utf-8') as f:
        for w, (s, y) in acc.exceptions.items():
            f.write(f'{w}\t{s}\t{y}\n')
    with open(os.path.join(args.out, 'homodict.tsv'), 'w', encoding='utf-8') as f:
        for w, variants in stress.homosolver.homodict.items():
            # the solver picks sorted(variants)[round(sigmoid(logit))]
            f.write(w + '\t' + '\t'.join(sorted(variants)) + '\n')

    tts = x['tts']
    ist = tts.vocoder.head.istft
    assert ist.padding == 'same'
    window = ist.window.numpy().astype('<f4')
    config = {
        'model': 'v5_5_ru',
        'symbols': pk.symbols,
        'symbol_to_id': pk.symbol_to_id,
        'speakers': pk.speaker_to_ids[0],
        'mean_std_coef': list(tts.mean_std_coef),
        'n_fft': ist.n_fft, 'hop_length': ist.hop_length, 'win_length': ist.win_length,
        'window_is_hann': bool(np.allclose(window, torch.hann_window(ist.n_fft).numpy(), atol=1e-6)),
        'pqmf': {'2': tts.vocoder.pqmf_2.H[0, 0].tolist(), '6': tts.vocoder.pqmf_6.H[0, 0].tolist()},
        'pqmf_taps': tts.vocoder.pqmf_2.taps,
        'ngram_max_len': max(len(k) for k in acc.model.embedding.ngram_dict),
        'bert': {'pad': vocab['[PAD]'], 'cls': vocab['[CLS]'], 'sep': vocab['[SEP]'], 'unk': vocab['[UNK]'],
                 'homo_start': vocab['[HOMO]'], 'homo_end': vocab['[/HOMO]'], 'max_len': max(BERT_SIZES)},
        'sizes': {'text': TEXT_SIZES, 'decoder': DECODER_SIZES, 'vocoder': VOCODER_SIZES, 'homosolver': BERT_SIZES},
        'vocoder_context': VOCODER_CONTEXT,
        'n_mels': tts.tacotron.n_mel_channels,
        'hidden': 128,
    }
    assert config['window_is_hann'], 'ISTFT window is expected to be a periodic Hann window'
    with open(os.path.join(args.out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    print('Done.')


if __name__ == '__main__':
    main()
