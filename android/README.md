# Silero TTS v5_5_ru as an Android system TTS engine (tuned for Pixel 8a)

An offline Android text-to-speech engine that runs the Silero `v5_5_ru`
model (5 Russian voices: `xenia`, `baya`, `kseniya`, `aidar`, `eugene`)
on-device with [LiteRT](https://ai.google.dev/edge/litert). The networks are
converted from PyTorch with [AI Edge Torch](https://developers.googleblog.com/en/ai-edge-torch-high-performance-inference-of-pytorch-models-on-mobile-devices/)
(now published as [LiteRT Torch](https://github.com/google-ai-edge/litert-torch),
`litert-torch`) and run on the CPU with XNNPACK through the LiteRT
`CompiledModel` API. After installing, select it in
*Settings → System → Languages → Text-to-speech output → Preferred engine*.
TalkBack, reading apps, navigation and anything else that uses the Android
TTS API can then speak with Silero voices. Text is never sent anywhere.

```
android/
  tools/export_models.py      torch.package -> LiteRT (.tflite) + frontend data, verified against the original
  tools/optimize_models.py    -> APK assets (fp16 weights with the AI Edge Quantizer)
  tools/reference_pipeline.py numpy + LiteRT reference of the whole pipeline, quality check, test vectors
  core/                       Kotlin engine (pure JVM, tested on the desktop with the LiteRT C API)
  app/                        TextToSpeechService on LiteRT CompiledModel, settings / benchmark screen
```

## Build

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.*"
pip install -r tools/requirements.txt

# 1. convert the networks from the original package (downloads v5_5_ru.pt, 145 MB)
python tools/export_models.py --package v5_5_ru.pt --out build/silero-export
# 2. build the APK assets
python tools/optimize_models.py --src build/silero-export --out app/src/main/assets/silero
# 3. build the APK (Android SDK 37, JDK 17+)
./gradlew :app:assembleRelease
adb install app/build/outputs/apk/release/app-release.apk
```

The release APK is signed with the debug key unless a keystore is given via
`SILERO_KEYSTORE`, `SILERO_KEYSTORE_PASSWORD`, `SILERO_KEY_ALIAS`, `SILERO_KEY_PASSWORD`
(or the `silero.keystore*` Gradle properties). APKs signed with different keys
can't update each other; uninstall first.

The `Android TTS APK` GitHub workflow runs all of the above (plus the
verification steps below) and uploads the APK as a build artifact. It also
publishes a GitHub release (tag `android-tts-<versionName>`) when the commit
message contains `[release]` or when run manually with *release* checked.
Without the `SILERO_KEYSTORE_B64` / `SILERO_KEYSTORE_PASSWORD` /
`SILERO_KEY_ALIAS` / `SILERO_KEY_PASSWORD` secrets every run signs with a new
throwaway key, so installed releases can't be updated in place.

### Verification

```bash
# LiteRT pipeline vs. the original torch package (identical stress, audio SNR > 60 dB)
python tools/reference_pipeline.py --assets build/silero-export --compare v5_5_ru.pt
# fp16 weights vs. fp32 (PESQ-wb, needs `pip install pesq`)
python tools/reference_pipeline.py --assets app/src/main/assets/silero --quality build/silero-export
# Kotlin engine vs. the reference pipeline, on the desktop LiteRT runtime
LITERT=$(python -c 'import ai_edge_litert, os; print(os.path.dirname(ai_edge_litert.__file__))')/libLiteRt.so
python tools/reference_pipeline.py --assets app/src/main/assets/silero --vectors build/vectors.json
./gradlew :core:test -Psilero.assets=$PWD/app/src/main/assets/silero -Psilero.litert=$LITERT -Psilero.vectors=$PWD/build/vectors.json
# listen: writes WAVs through the full engine
./gradlew :core:test --tests '*DemoWav*' -Psilero.litert=$LITERT -Psilero.wavOut=$PWD/build/demo
```

## How it works

The `v5_5_ru` package is a `torch.package` that mixes TorchScript networks
with Python glue. Nothing of it runs on Android as is, so:

* **Networks → LiteRT.** `export_models.py` re-implements every network as
  plain PyTorch (FastPitch-style duration/pitch predictors, the
  encoder/hourglass decoder, the Vocos vocoder, the BERT homograph model),
  loads the original weights, checks them against the TorchScript originals
  (max error ≤ 1e-5, vocoder audio bit-exact) and converts them with
  `litert_torch`. They become four models:

  | model | runs | signatures |
  |---|---|---|
  | `text` | duration + pitch predictors, pitch scaling, encoder; once per sentence | 32 … 512 tokens |
  | `decoder` | hourglass decoder → mel; once per sentence | 96 … 2304 frames (28.8 s) |
  | `vocoder` | Vocos backbone → complex spectrum; streamed in windows | 80, 160, 320 frames |
  | `homosolver` | BERT homograph classifier; once per homograph | 32 … 512 wordpieces |

* **Static shapes.** LiteRT plans memory and XNNPACK packs its kernels for
  fixed shapes, and the Android `CompiledModel` API has no input resizing, so
  each model has several fixed-size *signatures* sharing one copy of the
  weights. Inputs are zero-padded to the next size and a mask restores the
  exact unpadded result: padded keys get a −10⁴ attention bias and every
  convolution input is masked, so padded positions look exactly like the zero
  padding of the original convolutions (padded vs. unpadded PyTorch: max
  difference ≤ 4·10⁻⁶). Positions are computed from the mask at runtime, so
  the converter doesn't store a positional table per signature.
* **Glue → Kotlin.** Text cleaning, the n-gram stress model (accentor), the
  BERT tokenizer and int8 word-embedding lookup, homograph substitution,
  sentence-type classification (question / exclamation intonation), duration
  post-processing, the length regulator, the 2400-point ISTFT and the PQMF
  down-sampling to 24/8 kHz are ported to Kotlin (`core/`).
  `reference_pipeline.py` is the same pipeline in numpy on the same LiteRT
  runtime and is checked against the original package: identical stressed text
  and audio at 67–94 dB SNR on the test sentences, including speed / pitch
  changes. The Kotlin engine matches it at 105–110 dB SNR.
* **Text normalization** (not part of the original model, which silently
  drops digits and Latin letters): numbers with gender/case agreement, dates,
  years, times, decimals, percent, currencies, units, ordinals ("1-й",
  "90-х"), URLs, Latin words and acronyms, Cyrillic acronyms ("ФСБ"). A `+`
  before a vowel is a manual stress mark (Silero convention); an acute accent
  (за́мок) is converted to one.

## Optimizations

| | |
|---|---|
| **Runtime** | LiteRT 2.3 `CompiledModel` on the CPU with XNNPACK. The vocoder (most of the compute) is delegated completely, including GELU, sin/cos and the layer norms; the other networks run a few tiny ops (embedding lookups, the position cumsum, the pitch-scaling selects) on LiteRT's builtin kernels, which costs nothing measurable. The Pixel 8a's NPU can't be used: LiteRT supports the Google Tensor NPU only from Tensor G5. |
| **Model size** | 93 MB fp32 → 49 MB (text 8.8, decoder 5.1, vocoder 29.3, homosolver 5.8 MB): fully connected / convolution weights stored as fp16 by the AI Edge Quantizer (weight-only float casting); XNNPACK up-casts them once, so inference runs on the fp32 kernels at full speed. PESQ-wb vs fp32 on 30 sentences × voices: mean 4.639, min 4.591 (4.64 = identical), identical durations. Signatures share the weights, and positions are computed at runtime rather than stored per signature. The duration and pitch predictors stay fp32 (0.5M weights): durations are rounded to frames, and fp16 pitch-predictor weights lowered PESQ-wb to 4.45 on one test sentence. BERT vocabulary pruned to the 51.8k wordpieces the frontend can produce (lossless), its int8 word embeddings looked up by the engine. |
| **Quantization** | Integer quantization rejected: dynamic int8 (AI Edge Quantizer `dynamic_wi8_afp32`, XNNPACK int8 kernels) on the decoder + vocoder is 2–3× faster but drops PESQ-wb to 2.5–4.0, clearly audible. |
| **Memory / load** | Models are extracted from the APK once and loaded by path, so LiteRT memory-maps them (loading from the `AssetManager` copies them to the native heap). XNNPACK keeps the packed weights in a file-backed cache (`xnnPackWeightCachePath`): the signatures of a network share one memory-mapped copy instead of packing their own. With every signature in use that is +124 MB of private memory instead of +489 MB, and loading takes 0.14 s instead of 0.5 s (desktop); a typical sentence needs 15–25 MB of activations. Later starts skip the packing. Lookup tables are memory-mapped from the uncompressed APK and read in place. |
| **Target** | Android 17 (API 37) only, arm64-v8a only (Tensor G3): no compatibility code, other ABIs dropped; native libraries page-aligned (16 KB) and loaded from the APK. |
| **Threads** | XNNPACK uses one thread per performance core, detected from cpufreq (Pixel 8a: 4× Cortex-A715 + 1× Cortex-X3, the A510 little cores are skipped). Adjustable in the settings. |
| **Latency** | Text is split into sentences and the vocoder runs in windows (52 frames first, then 264) with 28 frames of context, which covers its receptive field exactly, so the first 0.65 s of audio is played while the rest is still being generated, with output identical to a single pass. The small signatures are warmed up right after loading. |
| **Fast speech** | Speech rate scales the predicted durations like the original SSML `prosody rate`, but keeps at least one frame per sound so screen-reader speeds don't drop phonemes. Slow speech splits sentences more finely so they fit the largest decoder signature. |

On this desktop x86 machine the Kotlin engine (LiteRT C API, default
threading) generates speech about 13× faster than real time, with 55–120 ms
to the first audio; the vocoder alone runs at ~100× real time on 4 XNNPACK
threads. **It has not been measured on a Pixel 8a**; the app's *Benchmark*
button reports model load time, first-audio latency and the real-time factor
on the device.

The first start after installing or updating extracts the models (49 MB) and
builds the XNNPACK weight cache (87 MB) in the app's private storage; later
starts memory-map both.

## Settings

Default voice, output sample rate (24 kHz default, 48 kHz native, 8 kHz),
automatic stress, homograph resolution, «ё» restoration and CPU threads. The
Android speech rate and pitch sliders are honoured; voices are exposed as
`ru-RU-x-silero-<speaker>`.

## Differences from the original package

* SSML is not supported (Android passes plain text); sentence pauses come
  from the model's punctuation handling and paragraph breaks add 250 ms.
* Each sentence is synthesized separately (the original takes the whole text
  at once), so stress/homograph context does not cross sentence boundaries.
* Peaks above full scale are hard-limited instead of `tanh` soft-clipping the
  whole utterance, which is impossible when streaming.

## License

The model is © Silero, published under
[CC BY-NC-SA 4.0](../LICENSE): **non-commercial use only**. The code in
this directory is provided under the same terms as the repository.
