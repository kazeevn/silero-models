"""Turn the fp32 ONNX export into the model assets shipped in the APK.

Weights are stored as fp16 and up-cast by a Cast node that ONNX Runtime
constant-folds while creating the session, so inference runs on the regular
fp32 kernels (MLAS NEON SGEMM on the Tensor G3) at full speed while the files
are half the size.  The audio is indistinguishable from fp32 (log-mel MAE
~0.01 dB, PESQ-wb 4.64 = identical).

Alternatives that were measured and rejected (PESQ-wb vs fp32, 5 speakers):
  * MatMulNBits 8-bit, int8 compute (accuracy_level 4): 4.37 mean / 4.00 min
  * dynamic int8 quantization (MatMulInteger): ~4.2 mean / 3.75 min
  * MatMulNBits 4-bit: ~3.7
  * MatMulNBits 8-bit, fp32 compute: 4.59, but without a native ARM kernel ORT
    dequantizes all weights on every run.

The BERT word embeddings are already int8 (gathered rows are de-quantized in
the graph) and stay that way.

Usage:
    python optimize_models.py --src export_dir --out android/app/src/main/assets/silero
"""
import argparse
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MODELS = ['predictors', 'acoustic', 'vocoder', 'homosolver']
# the duration predictor feeds a rounding step; keep it bit-exact (it is only 2 MB)
FP32_MODELS = {'predictors'}
DATA_FILES = ['accentor.bin', 'ngrams.bin', 'bert_vocab.bin', 'exceptions.tsv', 'homodict.tsv', 'config.json']


def fp16_storage(model, min_size=1024):
    """Store large fp32 initializers as fp16 followed by Cast(to=float)."""
    g = model.graph
    casts, converted, saved = [], [], 0
    for init in list(g.initializer):
        a = numpy_helper.to_array(init)
        if a.dtype != np.float32 or a.size < min_size:
            continue
        if np.abs(a).max() > 60000:
            raise ValueError(f'{init.name} does not fit into fp16')
        h = numpy_helper.from_array(a.astype(np.float16), init.name + '__fp16')
        casts.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + '__upcast'))
        converted.append(h)
        g.initializer.remove(init)
        saved += a.size * 2
    g.initializer.extend(converted)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='output directory of export_models.py')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name in MODELS:
        model = onnx.load(os.path.join(args.src, name + '.onnx'))
        saved = 0 if name in FP32_MODELS else fp16_storage(model)
        onnx.checker.check_model(model)
        dst = os.path.join(args.out, name + '.onnx')
        onnx.save(model, dst)
        print(f'{name}: {os.path.getsize(dst) / 1e6:.1f} MB (saved {saved / 1e6:.1f} MB)')

    for f in DATA_FILES:
        shutil.copy(os.path.join(args.src, f), os.path.join(args.out, f))
    print('Assets written to', args.out)


if __name__ == '__main__':
    main()
