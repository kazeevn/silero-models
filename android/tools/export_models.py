"""Export Silero TTS v5_5_ru (torch.package) into ONNX models + frontend data
for the on-device Android engine.

The original package mixes TorchScript networks with Python glue code.  The
networks are re-implemented here as plain PyTorch modules (dropping training
only branches, padding masks that are no-ops for batch size 1 and a dead
decoder layer whose output is never used), loaded with the original weights,
verified against the TorchScript originals and exported to ONNX.  The glue
(text processing, stress, duration / pitch post-processing, ISTFT) is
re-implemented in Kotlin, see ``core/``.

Outputs (in --out):
    predictors.onnx   tokens, speaker, type_ids  -> log_dur, pitch
    acoustic.onnx     tokens, speaker, pitch, frame_idx -> mel
    vocoder.onnx      mel -> spectrum (re, im) for a 2400-point ISTFT
    homosolver.onnx   BERT homograph disambiguation
    accentor.bin      int8 fastText n-gram embeddings + MLP weights
    ngrams.bin        open-addressing hash table: n-gram -> id
    bert_vocab.bin    open-addressing hash table: wordpiece -> id
    exceptions.tsv    stress exceptions
    homodict.tsv      homograph variants
    config.json       symbols, speakers, constants

Usage:
    python export_models.py --package v5_5_ru.pt --out assets_dir
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

MODEL_URL = 'https://models.silero.ai/models/tts/ru/v5_5_ru.pt'


# ----------------------------------------------------------------------------
# Clean re-implementations
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

    def forward(self, x):
        b, l, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.reshape(b, l, self.h, self.hd).transpose(1, 2)
        k = k.reshape(b, l, self.h, self.hd).transpose(1, 2)
        v = v.reshape(b, l, self.h, self.hd).transpose(1, 2)
        att = torch.softmax(torch.matmul(q * (1.0 / math.sqrt(self.hd)), k.transpose(2, 3)), dim=-1)
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

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        y = x.transpose(1, 2)
        y = y + self.conv2(F.relu(self.conv1(y)))
        return self.norm2(y.transpose(1, 2))


class PosEnc(nn.Module):
    """Sinusoidal positional encoding computed in-graph (the original stores a
    5000 x d table; computing it saves ~8 MB and lifts the length limit)."""

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

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device).float().unsqueeze(1)
        ang = pos * self.div
        pe = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(x.shape[1], self.d)
        return x + self.scale * pe.unsqueeze(0)


class ForwardTransformer(nn.Module):
    def __init__(self, src, n_layers):
        super().__init__()
        self.pos = PosEnc(src.pos_encoder)
        self.layers = nn.ModuleList([FFTBlock(getattr(src.layers, str(i))) for i in range(n_layers)])
        self.norm = nn.LayerNorm(src.norm.weight.shape[0], eps=src.norm.eps)
        self.norm.load_state_dict(src.norm.state_dict())

    def forward(self, x):
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
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

    def forward(self, x):
        b, l, d = x.shape
        x = self.pre(self.pos(x))
        pad = (self.k - l % self.k) % self.k
        x = F.pad(x, (0, 0, 0, pad))
        down = F.avg_pool1d(x.transpose(1, 2), self.k, self.k).transpose(1, 2)
        y = self.up(self.short(down)).reshape(b, -1, self.dim)
        x = (y + x)[:, :l]
        return self.norm(self.post(x))


def copy_emb(src):
    e = nn.Embedding(*src.weight.shape)
    e.weight.copy_(src.weight)
    return e


class Predictors(nn.Module):
    """Duration + pitch predictors (they share the token / speaker inputs)."""

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

    def forward(self, tokens, speaker, type_ids):
        # tokens [1, L] int64, speaker [1] int64, type_ids [1, L] int64
        x = self.d_emb(tokens) + self.d_spk(speaker).unsqueeze(1)
        log_dur = self.d_lin(self.d_tr(x)).squeeze(-1)
        x = self.p_emb(tokens) + self.p_spk(speaker).unsqueeze(1) + self.p_type(type_ids)
        pitch = self.p_lin(self.p_tr(x)).squeeze(-1)
        return log_dur, pitch  # [1, L], [1, L]


class Acoustic(nn.Module):
    """FastPitch-style encoder / hourglass decoder producing 192-bin mels."""

    def __init__(self, tts):
        super().__init__()
        t = tts.tacotron
        assert t.pitch_emb_type == 'default' and t.pitch_strength == 1.0
        self.emb, self.spk = copy_emb(t.embedding), copy_emb(t.speaker_embedding)
        self.enc = ForwardTransformer(t.encoder, 4)
        self.pitch_proj = nn.Conv1d(1, 128, 3, padding=1)
        self.pitch_proj.load_state_dict(t.pitch_proj.state_dict())
        self.dec = HourGlass(t.decoder)
        self.lin = nn.Linear(128, t.n_mel_channels)
        self.lin.load_state_dict(t.lin.state_dict())

    def forward(self, tokens, speaker, pitch, frame_idx):
        # tokens [1, L], speaker [1], pitch [1, L] float, frame_idx [T] int64
        # (index of the token each mel frame belongs to = length regulator).
        x = self.enc(self.emb(tokens)) + self.spk(speaker).unsqueeze(1)
        x = x + self.pitch_proj(pitch.unsqueeze(1)).transpose(1, 2)
        x = torch.index_select(x, 1, frame_idx)
        return self.lin(self.dec(x)).transpose(1, 2)  # [1, 192, T]


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

    def forward(self, x):
        y = self.norm(self.dw(x).transpose(1, 2))
        y = self.pw2(F.gelu(self.pw1(y))) * self.gamma
        return x + y.transpose(1, 2)


class Vocoder(nn.Module):
    """Vocos backbone + ISTFT head up to the complex spectrum.

    The inverse FFT / overlap-add runs in Kotlin (ONNX has no complex type and
    ORT's DFT is slow for n_fft=2400), which also allows streaming.
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

    def forward(self, mel):
        x = self.embed(mel)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        x = self.out(self.final(x.transpose(1, 2))).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clamp(torch.exp(mag), max=100.0)
        return mag * torch.cos(p), mag * torch.sin(p)  # [1, 1201, T] each


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

    def forward(self, x):
        b, l, d = x.shape
        q = self.q(x).reshape(b, l, self.h, self.hd).transpose(1, 2)
        k = self.k(x).reshape(b, l, self.h, self.hd).transpose(1, 2)
        v = self.v(x).reshape(b, l, self.h, self.hd).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.hd), dim=-1)
        y = torch.matmul(att, v).transpose(1, 2).reshape(b, l, d)
        x = self.ln1(self.o(y) + x)
        return self.ln2(self.out(F.gelu(self.i(x))) + x)


FRONTEND_LETTERS = set('абвгдежзийклмнопрстуфхцчшщъыьэюяё')
FRONTEND_PUNCT = set('!+,-.:;?–…')
BERT_MAX_LEN = 512
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
    def __init__(self, hs, kept_ids):
        super().__init__()
        bert = hs.bert
        # word embeddings stay int8 inside the graph; only gathered rows are
        # dequantized
        wq = bert.embeddings.word_embeddings.weight.detach().clone().to(torch.int8)
        self.register_buffer('wq', wq[torch.tensor(kept_ids)].clone())
        self.register_buffer('scale', bert.scale.detach().clone().float().reshape(()))
        self.register_buffer('zp', bert.zero_point.detach().clone().float().reshape(()))
        # chunks sent by the engine are far below 512 wordpieces; the engine
        # windows longer inputs around the homograph
        self.register_buffer('pos', bert.embeddings.position_embeddings.weight[:BERT_MAX_LEN].detach().clone())
        self.register_buffer('tok', bert.embeddings.token_type_embeddings.weight[0].detach().clone())
        self.ln = nn.LayerNorm(312, eps=bert.embeddings.LayerNorm.eps)
        self.ln.load_state_dict(bert.embeddings.LayerNorm.state_dict())
        self.layers = nn.ModuleList([BertLayer(getattr(bert.encoder.layer, str(i))) for i in range(bert.num_hidden_layers)])
        clf = hs.homo_clf
        self.c0 = nn.Linear(624, 256)
        self.c0.load_state_dict(getattr(clf, '0').state_dict())
        self.c1 = nn.Linear(256, 1)
        self.c1.load_state_dict(getattr(clf, '3').state_dict())

    def forward(self, input_ids, starts, ends):
        # input_ids [B, S] int64, starts/ends [B] int64 (positions of [HOMO] / [/HOMO])
        s = input_ids.shape[1]
        emb = (self.wq[input_ids].float() - self.zp) * self.scale
        x = self.ln(emb + self.pos[:s].unsqueeze(0) + self.tok)
        for layer in self.layers:
            x = layer(x)
        pos = torch.arange(s, device=x.device).unsqueeze(0)
        marker = (pos == starts.unsqueeze(1)).float().unsqueeze(-1)
        inside = ((pos > starts.unsqueeze(1)) & (pos < ends.unsqueeze(1))).float().unsqueeze(-1)
        marker_emb = (x * marker).sum(1)
        word_emb = (x * inside).sum(1) / inside.sum(1)
        h = F.relu(self.c0(torch.cat([marker_emb, word_emb], dim=1)))
        return self.c1(h).squeeze(-1)  # [B]


# ----------------------------------------------------------------------------
# Verification against the TorchScript originals
# ----------------------------------------------------------------------------

def check(name, a, b, tol):
    err = (a - b).abs().max().item()
    scale = b.abs().max().item()
    print(f'  {name}: max abs err {err:.3e} (max |ref| {scale:.3e})')
    if err > tol * max(1.0, scale):
        raise SystemExit(f'verification failed for {name}')


def verify(tts, hs, preds, acoustic, vocoder, homo, symbol_to_id, tokenizer, remap):
    print('Verifying re-implementations against TorchScript ...')
    text = '|прив+ет! к+ак тво+и дел+а, др+уг? всё хорош+о.~'
    tokens = torch.tensor([[symbol_to_id[c] for c in text]])
    spk = torch.tensor([4])
    L = tokens.shape[1]
    mask = torch.zeros(1, L, dtype=torch.bool)
    type_ids = torch.randint(0, 6, (1, L))
    ref_dur = tts.dur_predictor(tokens, spk, mask, 1.0, None)
    ref_pitch = tts.pitch_predictor(tokens, spk, mask, type_ids, None)
    log_dur, pitch = preds(tokens, spk, type_ids)
    check('log_dur', log_dur, ref_dur, 1e-4)
    check('pitch', pitch.unsqueeze(1), ref_pitch, 1e-4)

    durs = torch.clamp(torch.round(torch.exp(ref_dur) - 1), min=0)
    durs[0, 0] = min(durs[0, 0].item(), 5)
    durs[0, -2] = 13
    ref_mel = tts.tacotron(tokens, spk, mask, durs.clone(), ref_pitch, None)
    frame_idx = torch.repeat_interleave(torch.arange(L), durs[0].long())
    mel = acoustic(tokens, spk, ref_pitch[:, 0], frame_idx)
    check('mel', mel, ref_mel, 1e-4)

    ref_audio = tts.vocoder(ref_mel, 48000, 0., True)
    re, im = vocoder(ref_mel)
    spec = torch.complex(re, im)
    audio = tts.vocoder.head.istft(spec)
    check('audio', audio, ref_audio, 1e-4)

    sent = 'мой замок на горе, а на двери висит замок.'
    marked = [sent[:4] + ' [HOMO] ' + sent[4:9] + ' [/HOMO] ' + sent[9:],
              sent[:-6] + ' [HOMO] ' + sent[-6:-1] + ' [/HOMO] ' + sent[-1:]]
    ids = [tokenizer(m) for m in marked]
    ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in ids], batch_first=True)
    st = torch.tensor([i.tolist().index(tokenizer.homo_start_id) for i in ids])
    en = torch.tensor([i.tolist().index(tokenizer.homo_end_id) for i in ids])
    ref = hs(ids, st, en)
    out = homo(remap[ids], st, en)
    check('homo logits', out, ref.squeeze(-1), 1e-4)


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


def export_onnx(module, args, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=17, dynamo=False, do_constant_folding=True)
    print(f'  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', default='v5_5_ru.pt', help='path to v5_5_ru.pt (downloaded if missing)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if not os.path.isfile(args.package):
        print(f'Downloading {MODEL_URL}')
        os.makedirs(os.path.dirname(os.path.abspath(args.package)), exist_ok=True)
        torch.hub.download_url_to_file(MODEL_URL, args.package)
    os.makedirs(args.out, exist_ok=True)

    imp = torch.package.PackageImporter(args.package)
    model = imp.load_pickle('tts_models', 'model')
    assert len(model.packages) == 1
    pk = model.packages[0]
    assert len(pk.models) == 1 and not pk.phons
    tts = pk.models[0]
    stress = pk.accentor
    hs_wrapper = stress.homosolver
    tokenizer = hs_wrapper.tokenizer
    # unpack the int8 BERT word embeddings the way the package does at runtime
    hs_ts = hs_wrapper.model
    bert_q = hs_ts.bert.embeddings.word_embeddings.weight.data.clone()
    vocab, kept_ids = prune_vocab(tokenizer.vocab)
    print(f'BERT vocabulary: {len(tokenizer.vocab)} -> {len(vocab)} reachable wordpieces')
    remap = torch.full((len(tokenizer.vocab),), vocab['[UNK]'], dtype=torch.long)
    remap[torch.tensor(kept_ids)] = torch.arange(len(kept_ids))
    homo = HomoSolver(hs_ts, kept_ids)
    hs_ts.bert.embeddings.word_embeddings.weight.data = hs_ts.bert.scale * (bert_q.float() - hs_ts.bert.zero_point)

    preds, acoustic, vocoder = Predictors(tts).eval(), Acoustic(tts).eval(), Vocoder(tts).eval()
    homo.eval()
    verify(tts, hs_ts, preds, acoustic, vocoder, homo, pk.symbol_to_id, tokenizer, remap)

    print('Exporting ONNX ...')
    L, T = 20, 120
    tok = torch.randint(3, 40, (1, L))
    spk = torch.tensor([0])
    export_onnx(preds, (tok, spk, torch.zeros(1, L, dtype=torch.long)), os.path.join(args.out, 'predictors.onnx'),
                ['tokens', 'speaker', 'type_ids'], ['log_dur', 'pitch'],
                {'tokens': {1: 'L'}, 'type_ids': {1: 'L'}, 'log_dur': {1: 'L'}, 'pitch': {1: 'L'}})
    export_onnx(acoustic, (tok, spk, torch.zeros(1, L), torch.randint(0, L, (T,))), os.path.join(args.out, 'acoustic.onnx'),
                ['tokens', 'speaker', 'pitch', 'frame_idx'], ['mel'],
                {'tokens': {1: 'L'}, 'pitch': {1: 'L'}, 'frame_idx': {0: 'T'}, 'mel': {2: 'T'}})
    export_onnx(vocoder, (torch.randn(1, 192, T),), os.path.join(args.out, 'vocoder.onnx'),
                ['mel'], ['re', 'im'], {'mel': {2: 'T'}, 're': {2: 'T'}, 'im': {2: 'T'}})
    export_onnx(homo, (torch.randint(5, 1000, (2, 16)), torch.tensor([3, 5]), torch.tensor([6, 8])),
                os.path.join(args.out, 'homosolver.onnx'), ['input_ids', 'starts', 'ends'], ['logits'],
                {'input_ids': {0: 'B', 1: 'S'}, 'starts': {0: 'B'}, 'ends': {0: 'B'}, 'logits': {0: 'B'}})

    print('Writing frontend data ...')
    acc = stress.accentor
    write_accentor(os.path.join(args.out, 'accentor.bin'), acc.model)
    write_hash_table(os.path.join(args.out, 'ngrams.bin'), acc.model.embedding.ngram_dict)
    write_hash_table(os.path.join(args.out, 'bert_vocab.bin'), vocab)
    with open(os.path.join(args.out, 'exceptions.tsv'), 'w', encoding='utf-8') as f:
        for w, (s, y) in acc.exceptions.items():
            f.write(f'{w}\t{s}\t{y}\n')
    with open(os.path.join(args.out, 'homodict.tsv'), 'w', encoding='utf-8') as f:
        for w, variants in hs_wrapper.homodict.items():
            # the solver picks sorted(variants)[round(sigmoid(logit))]
            f.write(w + '\t' + '\t'.join(sorted(variants)) + '\n')

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
                 'homo_start': vocab['[HOMO]'], 'homo_end': vocab['[/HOMO]'], 'max_len': int(homo.pos.shape[0])},
    }
    assert config['window_is_hann'], 'ISTFT window is expected to be a periodic Hann window'
    with open(os.path.join(args.out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    print('Done.')


if __name__ == '__main__':
    main()
