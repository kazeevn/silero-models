"""Turn the fp32 LiteRT export into the model assets shipped in the APK.

Weights of the fully connected / convolution layers are stored as fp16 with
the AI Edge Quantizer's float casting (weight-only, explicit DEQUANTIZE).  The
XNNPACK delegate up-casts them once when the model is compiled, so inference
runs on the regular fp32 kernels at full speed and quality while the files are
half the size (PESQ-wb vs fp32 is measured by reference_pipeline.py --quality).

Kept fp32 (0.5M parameters):
  * the duration predictor (``d_*`` layers of text.tflite): its output is
    rounded to frame counts, and fp16 weights would occasionally flip a
    rounding (one frame more or less for a sound);
  * the pitch predictor (``p_*``): with fp16 weights the predicted pitch moves
    enough to lower PESQ-wb to 4.45 on one test sentence (>= 4.59 otherwise).

The BERT word embeddings are int8 already (bert_emb.bin, looked up by the
engine).

Integer quantization was measured and rejected for audible degradation (see
README.md).

Usage:
    python optimize_models.py --src export_dir --out android/app/src/main/assets/silero
"""
import argparse
import os
import shutil

MODELS = ['text', 'decoder', 'vocoder', 'homosolver']
# duration and pitch predictors
FP32_SCOPES = {'text': 'ForwardTransformer_(d|p)_tr|Linear_(d|p)_lin'}
DATA_FILES = ['accentor.bin', 'ngrams.bin', 'bert_vocab.bin', 'bert_emb.bin', 'exceptions.tsv', 'homodict.tsv', 'config.json']


def fp16_weights(src, dst, skip=None):
    """Weight-only fp16 float casting of FC / conv weights; ops whose output
    tensor name matches ``skip`` stay fp32."""
    from ai_edge_quantizer import qtyping, quantizer
    q = quantizer.Quantizer(src)
    fp16 = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT, explicit_dequantize=True)
    for op in [qtyping.TFLOperationName.FULLY_CONNECTED, qtyping.TFLOperationName.CONV_2D,
               qtyping.TFLOperationName.DEPTHWISE_CONV_2D]:
        q.update_quantization_recipe(regex='.*', operation_name=op, op_config=fp16,
                                     algorithm_key=quantizer.AlgorithmName.FLOAT_CASTING)
        if skip:
            q.update_quantization_recipe(regex=skip, operation_name=op, algorithm_key=quantizer.AlgorithmName.NO_QUANTIZE)
    q.quantize().export_model(dst, overwrite=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='output directory of export_models.py')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name in MODELS:
        src = os.path.join(args.src, name + '.tflite')
        dst = os.path.join(args.out, name + '.tflite')
        # op scopes are the PyTorch module paths, e.g. .../ForwardTransformer_d_tr/...
        fp16_weights(src, dst, skip=FP32_SCOPES.get(name))
        print(f'{name}: {os.path.getsize(src) / 1e6:.1f} MB -> {os.path.getsize(dst) / 1e6:.1f} MB')

    for f in DATA_FILES:
        shutil.copy(os.path.join(args.src, f), os.path.join(args.out, f))
    print('Assets written to', args.out)


if __name__ == '__main__':
    main()
