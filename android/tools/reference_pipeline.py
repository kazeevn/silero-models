"""Reference implementation of the on-device pipeline in numpy + LiteRT.

It only uses the artifacts produced by export_models.py and mirrors the Kotlin
engine (core/src/main/kotlin/...) step by step, including the static signature
sizes and the vocoder windows, and runs the .tflite models with the same LiteRT
CompiledModel runtime (XNNPACK) as the app.  It serves to validate the export
against the original torch.package, to measure the effect of optional fp16 weights
and to produce test vectors for the Kotlin unit tests.

    python reference_pipeline.py --assets DIR --compare v5_5_ru.pt
    python reference_pipeline.py --assets DIR --quality FP32_DIR
    python reference_pipeline.py --assets DIR --vectors out.json
"""
import argparse
import json
import math
import os
import re
import struct
import sys

import numpy as np

VOWELS = 'аоуыэиеяёю'


# ----------------------------------------------------------------------------
# LiteRT
# ----------------------------------------------------------------------------

class Network:
    """A .tflite model with static-size signatures, run with the LiteRT
    CompiledModel API on the CPU (XNNPACK), like the Android app."""

    def __init__(self, path, threads=4):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.cpu_options import CpuOptions
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import Options
        opts = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
        self.model = CompiledModel.from_file(path, options=opts)
        self.signatures = self.model.get_signature_list()
        self._buffers = {}

    def run(self, signature, **inputs):
        m = self.model
        if signature not in self._buffers:
            idx = m.get_signature_index(signature)
            self._buffers[signature] = (idx, m.create_input_buffers(idx), m.create_output_buffers(idx),
                                        m.get_input_tensor_details(signature), m.get_output_tensor_details(signature))
        idx, ins, outs, in_det, out_det = self._buffers[signature]
        names = self.signatures[signature]
        for name, buf in zip(names['inputs'], ins):
            d = in_det[name]
            x = np.ascontiguousarray(inputs[name], dtype=d['dtype'])
            assert list(x.shape) == list(d['shape']), (signature, name, x.shape, d['shape'])
            buf.write(x)
        m.run_by_index(idx, ins, outs)
        res = {}
        for name, buf in zip(names['outputs'], outs):
            d = out_det[name]
            res[name] = buf.read(int(np.prod(d['shape'])), d['dtype']).reshape(d['shape'])
        return res


def bucket(sizes, n):
    """Smallest static signature size that fits n."""
    for s in sizes:
        if s >= n:
            return s
    raise ValueError(f'{n} exceeds the largest signature size {sizes[-1]}')


def padded(x, size, axis=0, value=0):
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, size - x.shape[axis])
    return np.pad(x, pad, constant_values=value)


def length_mask(n, size):
    m = np.zeros((1, size), np.float32)
    m[0, :n] = 1
    return m


def vocoder_windows(total, sizes, context):
    """Vocoder windows as (a, b, s, e, size): output frames [a, b) computed
    from input frames [s, e) zero-padded to the signature size.  The first
    window is small for a low latency (mirrored in SileroEngine.kt)."""
    out, a = [], 0
    while a < total:
        cap = sizes[0] if a == 0 else sizes[-1]
        s = max(0, a - context)
        if total - s <= cap:
            b = e = total
        else:
            e = s + cap
            b = e - context
        out.append((a, b, s, e, bucket(sizes, e - s)))
        a = b
    return out


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def fnv1a(s):
    h = 0x811C9DC5
    data = s.encode('utf-16-le')
    for i in range(0, len(data), 2):
        h ^= data[i] | (data[i + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class HashTable:
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        magic, self.n_slots, n = struct.unpack_from('<4sii', data, 0)
        assert magic == b'SHT1'
        off = 12
        self.slots = np.frombuffer(data, '<i4', self.n_slots, off); off += 4 * self.n_slots
        self.values = np.frombuffer(data, '<i4', n, off); off += 4 * n
        self.offsets = np.frombuffer(data, '<i4', n + 1, off); off += 4 * (n + 1)
        self.blob = np.frombuffer(data, '<u2', self.offsets[-1], off)

    def get(self, key, default=None):
        mask = self.n_slots - 1
        s = fnv1a(key) & mask
        units = np.frombuffer(key.encode('utf-16-le'), '<u2')
        while True:
            e = self.slots[s]
            if e < 0:
                return default
            a, b = self.offsets[e], self.offsets[e + 1]
            if b - a == len(units) and np.array_equal(self.blob[a:b], units):
                return int(self.values[e])
            s = (s + 1) & mask


class Accentor:
    """Port of models/accentor.py (AccentorNgram) on the exported weights."""

    def __init__(self, assets, ngram_max_len):
        with open(os.path.join(assets, 'accentor.bin'), 'rb') as f:
            data = f.read()
        magic, v, d = struct.unpack_from('<4sii', data, 0)
        assert magic == b'ACC1'
        scale, zp = struct.unpack_from('<ff', data, 12)
        off = 20
        q = np.frombuffer(data, np.int8, v * d, off).reshape(v, d)
        off += v * d + ((-v * d) % 4)
        self.emb = (np.float32(scale) * (q.astype(np.float32) - np.float32(zp))).astype(np.float32)
        self.mlps = []
        for _ in range(2):
            n_layers, = struct.unpack_from('<i', data, off); off += 4
            layers = []
            for _ in range(n_layers):
                o, i = struct.unpack_from('<ii', data, off); off += 8
                w = np.frombuffer(data, '<f4', o * i, off).reshape(o, i); off += 4 * o * i
                b = np.frombuffer(data, '<f4', o, off); off += 4 * o
                layers.append((w, b))
            self.mlps.append(layers)
        self.ngrams = HashTable(os.path.join(assets, 'ngrams.bin'))
        self.unk = self.ngrams.get('UNK')
        self.max_n = ngram_max_len
        self.exceptions = {}
        with open(os.path.join(assets, 'exceptions.tsv'), encoding='utf-8') as f:
            for line in f:
                w, s, y = line.rstrip('\n').split('\t')
                self.exceptions[w] = (int(s), int(y))

    def embed(self, word):
        t = '<' + word + '>'
        ids = []
        for n in range(1, min(len(word) + 3, self.max_n) + 1):
            for i in range(len(t) - n + 1):
                g = self.ngrams.get(t[i:i + n])
                if g is not None:
                    ids.append(g)
        if not ids:
            ids = [self.unk]
        return self.emb[ids].mean(axis=0)

    def mlp(self, layers, x):
        for k, (w, b) in enumerate(layers):
            x = w @ x + b
            if k < len(layers) - 1:
                x = np.maximum(x, 0)
        return x

    def predict(self, word):
        e = self.embed(word)
        out = []
        for layers in self.mlps:
            logits = self.mlp(layers, e)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            k = int(np.argmax(p))
            out.append((k, float(p[k])))
        return out  # [(stress_idx, prob), (yo_idx, prob)]

    @staticmethod
    def tokenize(sentence):
        tokens, inputs, mask = [], [], []
        for word in re.split(r'([\s.,!?;:<>=()/\\]+)', sentence):
            parts = word.split('-')
            if len(parts) == 1:
                cur, cur_mask = parts, [True]
            else:
                cur = [p + '-' for p in parts[:-1]] + [parts[-1]]
                cur_mask = [True for _ in parts[:-1]] + [parts[-1] != 'то']
            cur_inputs = [re.sub(r'[^А-Яа-яёЁ]', '', t.lower()) for t in cur]
            cur_mask = [len(x) > 0 and m for x, m in zip(cur_inputs, cur_mask)]
            tokens += cur
            inputs += cur_inputs
            mask += cur_mask
        return tokens, inputs, mask

    def __call__(self, sentence, put_stress=True, put_yo=True, stress_single_vowel=True,
                 skip_stress_words=(), skip_yo_words=()):
        if not (put_stress or put_yo):
            return sentence
        raw_tokens, clean_tokens, prediction_mask = self.tokenize(sentence)
        out = []
        for raw_word, clean_word, need in zip(raw_tokens, clean_tokens, prediction_mask):
            low = raw_word.lower()
            if not need:
                out.append(raw_word)
                continue
            have_stress = '+' in low
            have_yo = 'ё' in low
            if have_stress and have_yo:
                out.append(raw_word)
                continue
            if not have_stress and have_yo and put_stress:
                if (sum(c in VOWELS for c in low) == 1 and not stress_single_vowel) or clean_word.replace('ё', 'е') in skip_stress_words:
                    out.append(raw_word)
                    continue
                pos = [i for i, x in enumerate(low) if x == 'ё']
                for i, p in enumerate(pos):
                    raw_word = raw_word[:p + i] + '+' + raw_word[p + i:]
                out.append(raw_word)
                continue
            if clean_word in self.exceptions:
                out.append(self.exception(clean_word, raw_word, have_stress))
                continue
            (s_idx, s_prob), (y_idx, y_prob) = self.predict(clean_word)
            stressed = [s_idx]
            passed_stress = s_prob > (0.5 if put_stress else 1)
            set_stress = passed_stress and not have_stress and clean_word.replace('ё', 'е') not in skip_stress_words
            yo_ids = [y_idx]
            passed_yo = y_prob > (0.5 if put_yo else 1)
            set_yo = passed_yo and clean_word.replace('ё', 'е') not in skip_yo_words
            if have_stress:
                stressed = [sum(map(part.count, VOWELS)) for part in low.split('+')]
            vowel_ids = [i for i, c in enumerate(low) if c in VOWELS]
            ye_ids = [i for i, c in enumerate(low) if c == 'е']
            stress_pos = [vowel_ids[i] for i in stressed if i < len(vowel_ids)]
            yo_pos = [ye_ids[i - 1] for i in yo_ids if i > 0 and i - 1 < len(ye_ids)]
            if not vowel_ids:
                out.append(raw_word)
                continue
            for p in yo_pos:
                if p in stress_pos and set_yo and low[p] == 'е':
                    raw_word = raw_word[:p] + ('ё' if raw_word[p].islower() else 'Ё') + raw_word[p + 1:]
            if len(vowel_ids) == 1:
                stress_pos = [vowel_ids[0]]
                set_stress = stress_single_vowel and put_stress
            if not have_stress and set_stress:
                for i, p in enumerate(stress_pos):
                    raw_word = raw_word[:p + i] + '+' + raw_word[p + i:]
            out.append(raw_word)
        return ''.join(out)

    def exception(self, clean_word, raw_word, have_stress):
        exc_stress, exc_yo = self.exceptions[clean_word]
        if have_stress:
            user = [i for i, c in enumerate(raw_word) if c == '+']
            w = raw_word.replace('+', '')
            if exc_yo != -1 and exc_yo + 1 in user:
                w = w[:exc_yo] + ('ё' if w[exc_yo].islower() else 'Ё') + w[exc_yo + 1:]
            for p in user:
                w = w[:p] + '+' + w[p:]
            return w
        if exc_yo != -1:
            raw_word = raw_word[:exc_yo] + ('ё' if raw_word[exc_yo].islower() else 'Ё') + raw_word[exc_yo + 1:]
        return raw_word[:exc_stress] + '+' + raw_word[exc_stress:]


class BertTokenizer:
    """Port of custom_tokenizers/bert_tokenizer.py restricted to the frontend charset."""

    def __init__(self, assets, cfg):
        self.vocab = HashTable(os.path.join(assets, 'bert_vocab.bin'))
        self.cfg = cfg

    @staticmethod
    def is_punct(ch):
        import unicodedata
        cp = ord(ch)
        if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
            return True
        return unicodedata.category(ch).startswith('P')

    def basic(self, text):
        out = []
        for tok in text.split():
            if tok in ('[HOMO]', '[/HOMO]'):
                out.append(tok)
                continue
            cur = ''
            for ch in tok:
                if self.is_punct(ch):
                    if cur:
                        out.append(cur)
                        cur = ''
                    out.append(ch)
                else:
                    cur += ch
            if cur:
                out.append(cur)
        return out

    def wordpiece(self, tok):
        if len(tok) > 100:
            return [self.cfg['unk']]
        ids, start = [], 0
        while start < len(tok):
            end, cur = len(tok), None
            while start < end:
                sub = tok[start:end]
                if start > 0:
                    sub = '##' + sub
                v = self.vocab.get(sub)
                if v is not None:
                    cur = v
                    break
                end -= 1
            if cur is None:
                return [self.cfg['unk']]
            ids.append(cur)
            start = end
        return ids

    def __call__(self, text):
        ids = [self.cfg['cls']]
        for tok in self.basic(text):
            ids += self.wordpiece(tok)
        return ids + [self.cfg['sep']]


class HomoSolver:
    def __init__(self, assets, cfg, net, sizes):
        self.tok = BertTokenizer(assets, cfg)
        self.cfg = cfg
        self.net = net
        self.sizes = sizes
        with open(os.path.join(assets, 'bert_emb.bin'), 'rb') as f:
            data = f.read()
        magic, v, d = struct.unpack_from('<4sii', data, 0)
        assert magic == b'BEM1'
        scale, zp = struct.unpack_from('<ff', data, 12)
        self.emb_q = np.frombuffer(data, np.int8, v * d, 20).reshape(v, d)
        self.emb_scale, self.emb_zp = np.float32(scale), np.float32(zp)
        self.homodict = {}
        with open(os.path.join(assets, 'homodict.tsv'), encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t')
                self.homodict[parts[0]] = parts[1:]
        self.pattern = re.compile(r'(?=.*[а-яё])[а-яё+]+', re.IGNORECASE)

    def logit(self, ids, start, end):
        n = len(ids)
        size = bucket(self.sizes, n)
        emb = (self.emb_q[np.array(ids)].astype(np.float32) - self.emb_zp) * self.emb_scale
        out = self.net.run(f'S{size}', embeddings=padded(emb, size)[None], mask=length_mask(n, size),
                           start=np.array([start], np.int32), end=np.array([end], np.int32))
        return out['logit'][0]

    def __call__(self, sentence, put_stress=True, put_yo=True, stress_single_vowel=True):
        if not (put_stress or put_yo):
            return sentence
        found = []
        max_len = self.cfg['max_len']
        for m in self.pattern.finditer(sentence):
            s, e = m.span()
            w = m.group()
            if w.lower() in self.homodict:
                ids = self.tok(sentence[:s] + ' [HOMO] ' + w + ' [/HOMO] ' + sentence[e:])
                st, en = ids.index(self.cfg['homo_start']), ids.index(self.cfg['homo_end'])
                if len(ids) > max_len:
                    # keep a window around the homograph (mirrors HomoSolver.kt)
                    inner = max_len - 2
                    frm = min(max(st - inner // 2, 1), len(ids) - 1 - inner)
                    ids = [self.cfg['cls']] + ids[frm:frm + inner] + [self.cfg['sep']]
                    st -= frm - 1
                    en -= frm - 1
                found.append((s, e, w, self.logit(ids, st, en)))
        out, offset = sentence, 0
        for s, e, w, lg in found:
            pred = int(np.round(1 / (1 + np.exp(-np.float32(lg)))))
            wp = self.homodict[w.lower()][pred]
            if not put_yo:
                wp = wp.replace('ё', 'е')
            n_vowels = sum(c.lower() in VOWELS for c in wp)
            si = wp.index('+')
            wp = wp.replace('+', '')
            wp = ''.join(c2.lower() if c1.islower() else c2.upper() for c1, c2 in zip(w, wp))
            s += offset
            e += offset
            if (n_vowels > 1 or stress_single_vowel) and put_stress:
                wp = wp[:si] + '+' + wp[si:]
                offset += 1
            out = out[:s] + wp + out[e:]
        return out


# ----------------------------------------------------------------------------
# Sentence type classification (port of multi_acc_v3_package.classify_text)
# ----------------------------------------------------------------------------

TYPE2ID = {'st': 0, 'wh_q': 1, 'general_q': 2, 'alternative_q': 3, 'tag_q': 4, 'exclam': 5}
WH_FORMS = set('''кто кого кому кем ком что чего чему чем чём чё чо где куда откуда когда почему зачем как
почём отчего насколько сколько скольких скольким сколькими какой какая какое какие какого каких какому
каким какими каком какую чей чья чьё чье чьи чьего чьей чьих чьему чьим чьими чьём чьем чью который
которая которое которые которого которой которых которому которым которыми котором которую каков какова
каково каковы'''.split())
LEADING_FILLERS = set('а ну и так вот слышь слушай скажите скажи пожалуйста вобще вообще типа короче'.split())
TAG_RE = re.compile('|'.join(f'(?:{p})' for p in [
    r'[,\s]+(правда|верно|да)\s*\?$',
    r'[,\s]+(не\s+так\s+ли|не\s+правда\s+ли|разве\s+не\s+так|ведь\s+так)\s*\?$',
    r'[,\s]+ведь\s*\?$',
    r'[,\s]+а\s*\?$']), re.IGNORECASE)
QUOTE_OPEN = {'"', '«', '“', '„'}
QUOTE_CLOSE = {'"', '»', '”', '’'}
WORD_RE = re.compile(r'[а-яёa-z]+', re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def strip_quotes(s):
    s = s.strip()
    while s and s[0] in QUOTE_OPEN:
        s = s[1:].strip()
    while s and s[-1] in QUOTE_CLOSE:
        s = s[:-1].strip()
    return s


def classify_sentence(s):
    s = s.strip()
    if not s:
        return 'st'
    c = strip_quotes(s)
    if not c:
        return 'st'
    tail = c
    if tail.endswith('?!') or tail.endswith('?..') or tail.endswith('?...'):
        tail = tail.rstrip('.!')
    if tail.endswith('?'):
        q = strip_quotes(tail).replace('+', '').replace('ё', 'е').replace('Ё', 'Е')
        q = re.sub(r'\s+', ' ', q).strip()
        if TAG_RE.search(q):
            return 'tag_q'
        content = []
        for w in WORD_RE.findall(q.lower())[:8]:
            if w in LEADING_FILLERS:
                continue
            content.append(w)
            if len(content) >= 4:
                break
        if any(w in WH_FORMS for w in content):
            return 'wh_q'
        if re.search(r'\bили\b', q, re.IGNORECASE):
            return 'alternative_q'
        return 'general_q'
    if c.endswith('!'):
        return 'exclam'
    return 'st'


def type_ids_for(text, seq_len):
    if not text or not text.strip():
        return np.zeros((1, seq_len), np.int64)
    sents = [s for s in SENT_SPLIT_RE.split(text.strip())]
    types = [classify_sentence(s) for s in sents if s.strip()]
    if not types:
        types = ['st']
    per_char = []
    for i, s in enumerate(sents):
        tid = TYPE2ID[types[i] if i < len(types) else types[-1]]
        per_char += [tid] * len(s)
        if i < len(sents) - 1:
            per_char.append(tid)
    out = np.full((1, seq_len), TYPE2ID[types[0]], np.int64)
    for i, t in enumerate(per_char[:seq_len - 1]):
        out[0, i + 1] = t
    return out


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------

class Pipeline:
    def __init__(self, assets, threads=4, keep_short=False):
        # keep_short: the Android engine keeps >= 1 frame per sound when speeding
        # up (the original drops sounds whose duration rounds to 0)
        self.keep_short = keep_short
        with open(os.path.join(assets, 'config.json'), encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.sizes = self.cfg['sizes']

        def net(name):
            return Network(os.path.join(assets, name + '.tflite'), threads)

        self.text_net = net('text')
        self.decoder = net('decoder')
        self.vocoder = net('vocoder')
        self.homo = HomoSolver(assets, self.cfg['bert'], net('homosolver'), self.sizes['homosolver'])
        self.acc = Accentor(assets, self.cfg['ngram_max_len'])
        self.sym = self.cfg['symbol_to_id']
        n = self.cfg['n_fft']
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)

    def clean(self, text):
        """prepare_tts_model_input + prepare_text_input (plain text, no SSML)."""
        text = text.strip()
        text = re.sub(r'\n', lambda m: ' ' if m.start() > 0 and text[m.start() - 1] in ',;:.!?' else '. ', text)
        text = text.lower().replace('—', '–').replace('‑', '-')
        allowed = set(self.cfg['symbols'][3:])
        text = ''.join(c for c in text if c in allowed)
        return re.sub(r'\s+', ' ', text).strip()

    def accentuate(self, text):
        t = self.homo(text)
        return self.acc(t)

    def durations(self, log_dur, rate):
        d = np.exp(log_dur.astype(np.float32)) - np.float32(1)
        d[d < 0] = 0
        d = np.round(d)
        d[0] = min(d[0], 5)
        if rate != 1.0:
            orig = d.copy()
            d = np.round(d / np.float32(rate))
            if self.keep_short and rate > 1:
                d[(orig >= 1) & (d < 1)] = 1
        d[0] = min(d[0], 5)
        d[-1] = min(d[-1], 7)
        d[-2] = 13
        d[-3] = min(d[-3], 13)
        return d.astype(np.int64)

    def pitch_params(self, coef, speaker_id):
        """update_pitch_coef as (scale, shift) for the text model."""
        c = np.float32(1.0 if coef == 0 else coef)
        return np.float32(coef), (c - np.float32(1)) * np.float32(self.cfg['mean_std_coef'][speaker_id])

    def mel(self, raw_text, speaker='xenia', rate=1.0, pitch=1.0):
        text = self.accentuate(self.clean(raw_text))
        seq = '|' + text + '~'
        tokens = np.array([self.sym[c] for c in seq], np.int32)
        L = len(tokens)
        spk = self.cfg['speakers'][speaker]
        type_ids = type_ids_for(raw_text, L)[0].astype(np.int32)
        scale, shift = self.pitch_params(pitch, spk)
        size = bucket(self.sizes['text'], L)
        out = self.text_net.run(f'L{size}', tokens=padded(tokens, size)[None], type_ids=padded(type_ids, size)[None],
                                speaker=np.array([spk], np.int32), mask=length_mask(L, size),
                                pitch_scale=np.array([scale], np.float32), pitch_shift=np.array([shift], np.float32))
        durs = self.durations(out['log_dur'][0, :L], rate)
        hidden = out['hidden'][0, :L]
        frames = np.repeat(hidden, durs, axis=0)
        T = frames.shape[0]
        size = bucket(self.sizes['decoder'], T)
        mel = self.decoder.run(f'T{size}', frames=padded(frames, size)[None], mask=length_mask(T, size))['mel'][0, :T]
        return text, tokens, durs, mel  # mel [T, 192]

    def spectrum(self, mel):
        """Windowed vocoder (as streamed by the engine) -> re, im [T, n_bins]."""
        T = mel.shape[0]
        n_bins = self.cfg['n_fft'] // 2 + 1
        re_ = np.zeros((T, n_bins), np.float32)
        im = np.zeros((T, n_bins), np.float32)
        for a, b, s, e, size in vocoder_windows(T, self.sizes['vocoder'], self.cfg['vocoder_context']):
            out = self.vocoder.run(f'W{size}', mel=padded(mel[s:e], size)[None], mask=length_mask(e - s, size))
            re_[a:b] = out['re'][0, a - s:b - s]
            im[a:b] = out['im'][0, a - s:b - s]
        return re_, im

    def istft(self, re_, im):
        n, hop = self.cfg['n_fft'], self.cfg['hop_length']
        spec = re_.T + 1j * im.T  # [n_bins, T]
        frames = np.fft.irfft(spec, n=n, axis=0).astype(np.float32) * self.window[:, None]
        T = frames.shape[1]
        out_len = (T - 1) * hop + n
        y = np.zeros(out_len, np.float32)
        env = np.zeros(out_len, np.float32)
        w2 = self.window ** 2
        for t in range(T):
            y[t * hop:t * hop + n] += frames[:, t]
            env[t * hop:t * hop + n] += w2
        pad = (n - hop) // 2
        return y[pad:-pad] / env[pad:-pad]

    def downsample(self, audio, sr):
        if sr == 48000:
            return audio
        h = np.array(self.cfg['pqmf']['2' if sr == 24000 else '6'], np.float32)
        stride = 2 if sr == 24000 else 6
        taps = self.cfg['pqmf_taps']
        x = np.pad(audio, (taps // 2, taps // 2))
        n_out = (len(x) - len(h)) // stride + 1
        idx = np.arange(n_out)[:, None] * stride + np.arange(len(h))[None, :]
        y = (x[idx] * h[None, :]).sum(1)
        return y

    def __call__(self, raw_text, speaker='xenia', sr=24000, rate=1.0, pitch=1.0):
        text, tokens, durs, mel = self.mel(raw_text, speaker, rate, pitch)
        audio = self.downsample(self.istft(*self.spectrum(mel)), sr)
        if np.abs(audio).max() > 1:
            audio = np.clip(audio, -1, 1)
        return {'text': text, 'tokens': tokens, 'durs': durs, 'mel': mel, 'audio': audio}


TEST_TEXTS = [
    'Меня зовут Лева Королев. Я из готов. И я уже готов открыть все ваши замки любой сложности!',
    'Привет! Как твои дела?',
    'В недрах тундры выдры в г+етрах т+ырят в вёдра ядра кедров.',
    'Мой замок стоит на горе, а на двери висит замок.',
    'Кто-то позвонил в дверь, и я пошёл открывать.',
    'Ты придёшь завтра или послезавтра?',
    'Хорошая погода сегодня, не правда ли?',
    'Съешь же ещё этих мягких французских булок, да выпей чаю.',
    'Скажите, пожалуйста, где находится ближайшая станция метро?',
    'Все мои знакомые уже давно переехали в другой город.',
    'Он вышел из комнаты и закрыл за собой дверь — тихо, без единого звука…',
    'Ёжик в тумане шёл по тропинке: медленно; осторожно.',
]


def compare(pipe, package_path, sr, speakers):
    import torch
    imp = torch.package.PackageImporter(package_path)
    model = imp.load_pickle('tts_models', 'model')
    pk = model.packages[0]
    model.apply_tts(text='Привет.', speaker=speakers[0], sample_rate=sr)  # unpacks the int8 BERT
    worst_snr, mismatches = 1e9, 0
    for i, text in enumerate(TEST_TEXTS):
        spk = speakers[i % len(speakers)]
        rate = [1.0, 1.3, 0.8][i % 3]
        pitch = [1.0, 1.15, 0.9][i % 3]
        ref_text = pk.accentor(pipe.clean(text))
        if rate == 1.0 and pitch == 1.0:
            ref = model.apply_tts(text=text, speaker=spk, sample_rate=sr).numpy()
        else:
            sign = '+' if pitch >= 1 else '-'
            ssml = (f'<speak><prosody rate="{int(round(rate * 100))}%" '
                    f'pitch="{sign}{int(round(abs(pitch - 1) * 100))}%">{text}</prosody></speak>')
            ref = model.apply_tts(ssml_text=ssml, speaker=spk, sample_rate=sr).numpy()
        out = pipe(text, spk, sr, rate, pitch)
        same_text = out['text'] == ref_text
        n = min(len(ref), len(out['audio']))
        err = np.abs(ref[:n] - out['audio'][:n]).max()
        snr = 10 * np.log10((ref[:n] ** 2).sum() / max(((ref[:n] - out['audio'][:n]) ** 2).sum(), 1e-20))
        worst_snr = min(worst_snr, snr)
        print(f'[{i}] {spk:8s} rate={rate} pitch={pitch} len ref={len(ref)} ours={len(out["audio"])} '
              f'max err={err:.2e} snr={snr:.1f}dB text_match={same_text}')
        if not same_text:
            mismatches += 1
            print('   ref :', ref_text)
            print('   ours:', out['text'])
    print(f'worst SNR {worst_snr:.1f} dB')
    return worst_snr, mismatches


def quality(pipe, ref_assets, speakers):
    """Audio of these assets vs the fp32 export: SNR and PESQ-wb (4.64 = identical)."""
    from pesq import pesq
    from scipy.signal import resample_poly
    ref = Pipeline(ref_assets)
    scores, snrs = [], []
    for text in TEST_TEXTS[:6]:
        for spk in speakers:
            a = ref(text, spk)
            b = pipe(text, spk)
            same = np.array_equal(a['durs'], b['durs'])
            n = min(len(a['audio']), len(b['audio']))
            x, y = a['audio'][:n], b['audio'][:n]
            snr = 10 * np.log10((x ** 2).sum() / max(((x - y) ** 2).sum(), 1e-20))
            q = pesq(16000, resample_poly(x, 2, 3), resample_poly(y, 2, 3), 'wb')
            scores.append(q)
            snrs.append(snr)
            print(f'{spk:8s} pesq={q:.3f} snr={snr:5.1f} dB same_durations={same}  {text[:40]}')
    print(f'PESQ-wb mean {np.mean(scores):.3f} min {np.min(scores):.3f}; SNR mean {np.mean(snrs):.1f} dB')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', required=True)
    ap.add_argument('--compare', help='original v5_5_ru.pt to compare against')
    ap.add_argument('--quality', help='fp32 export directory to measure the assets against')
    ap.add_argument('--vectors', help='write Kotlin test vectors to this json')
    ap.add_argument('--sr', type=int, default=24000)
    args = ap.parse_args()
    pipe = Pipeline(args.assets, keep_short=args.vectors is not None and not args.compare)
    speakers = list(pipe.cfg['speakers'].keys())
    if args.compare:
        worst_snr, mismatches = compare(pipe, args.compare, args.sr, speakers)
        if mismatches or worst_snr < 40:
            sys.exit('reference pipeline deviates from the original package')
    if args.quality:
        quality(pipe, args.quality, speakers)
    if args.vectors:
        cases = []
        for i, text in enumerate(TEST_TEXTS):
            spk = speakers[i % len(speakers)]
            rate = [1.0, 1.3, 0.8][i % 3]
            pitch = [1.0, 1.15, 0.9][i % 3]
            out = pipe(text, spk, 24000, rate, pitch)
            a = out['audio']
            cases.append({'text': text, 'speaker': spk, 'rate': rate, 'pitch': pitch, 'sample_rate': 24000,
                          'accented': out['text'], 'tokens': out['tokens'].tolist(), 'durs': out['durs'].tolist(),
                          'audio': [round(float(x), 6) for x in a]})
        with open(args.vectors, 'w', encoding='utf-8') as f:
            json.dump(cases, f, ensure_ascii=False)
        print(f'wrote {len(cases)} cases to {args.vectors}')


if __name__ == '__main__':
    main()
