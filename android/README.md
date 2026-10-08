# Silero TTS v5_5_ru as an Android system TTS engine (tuned for Pixel 8a)

An offline Android text-to-speech engine that runs the Silero `v5_5_ru`
model (5 Russian voices: `xenia`, `baya`, `kseniya`, `aidar`, `eugene`)
on-device with ONNX Runtime. After installing, select it in
*Settings → System → Languages → Text-to-speech output → Preferred engine*.
TalkBack, reading apps, navigation and anything else that uses the Android
TTS API can then speak with Silero voices. Text is never sent anywhere.

```
android/
  tools/export_models.py      torch.package -> ONNX + frontend data (verified against the original)
  tools/optimize_models.py    -> APK assets (fp16 weight storage)
  tools/reference_pipeline.py numpy/ORT reference of the whole pipeline, test vectors
  core/                       Kotlin engine (pure JVM, unit-tested on desktop ONNX Runtime)
  app/                        TextToSpeechService, settings / benchmark screen
```

## Build

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r tools/requirements.txt

# 1. export the networks from the original package (downloads v5_5_ru.pt, 145 MB)
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
# ONNX pipeline vs. the original torch package (identical stress, audio SNR > 60 dB)
python tools/reference_pipeline.py --assets build/silero-export --compare v5_5_ru.pt
# Kotlin engine vs. the reference pipeline (desktop ONNX Runtime)
python tools/reference_pipeline.py --assets app/src/main/assets/silero --vectors build/vectors.json
./gradlew :core:test -Psilero.assets=$PWD/app/src/main/assets/silero -Psilero.vectors=$PWD/build/vectors.json
# listen: writes WAVs through the full engine
./gradlew :core:test --tests '*DemoWav*' -Psilero.wavOut=$PWD/build/demo
```

## How it works

The `v5_5_ru` package is a `torch.package` that mixes TorchScript networks
with Python glue. Nothing of it runs on Android as is, so:

* **Networks → ONNX.** `export_models.py` re-implements every network as
  plain PyTorch (FastPitch-style duration/pitch predictors, the
  encoder/hourglass decoder, the Vocos vocoder, the BERT homograph model),
  loads the original weights, checks them against the TorchScript originals
  (max error ≤ 1e-5, vocoder audio bit-exact) and exports them.
* **Glue → Kotlin.** Text cleaning, the n-gram stress model (accentor), the
  BERT tokenizer and homograph substitution, sentence-type classification
  (question / exclamation intonation), duration and pitch post-processing,
  the 2400-point ISTFT and the PQMF down-sampling to 24/8 kHz are ported to
  Kotlin (`core/`). `reference_pipeline.py` is the same pipeline in numpy and
  is checked against the original package: identical stressed text and audio
  at 66–99 dB SNR on the test sentences, including speed / pitch changes.
  The Kotlin engine matches the reference at 106–110 dB SNR.
* **Text normalization** (not part of the original model, which silently
  drops digits and Latin letters): numbers with gender/case agreement, dates,
  years, times, decimals, percent, currencies, units, ordinals ("1-й",
  "90-х"), URLs, Latin words and acronyms, Cyrillic acronyms ("ФСБ"). A `+`
  before a vowel is a manual stress mark (Silero convention); an acute accent
  (за́мок) is converted to one.

## Optimizations

| | |
|---|---|
| **Model size** | 130 MB fp32 → 67 MB: weights stored as fp16 with a `Cast` that ONNX Runtime constant-folds at load, so inference runs on the fp32 NEON kernels at full speed and quality (PESQ-wb vs fp32 4.63, where 4.64 = identical). BERT vocabulary pruned to the 51.8k wordpieces the frontend can produce (lossless); positional encodings computed in-graph; a dead decoder layer whose output was discarded removed. |
| **Quantization** | Measured and rejected: int8 activations (dynamic int8, `MatMulNBits` accuracy level 4) cost 0.3–0.6 PESQ on the male voices, 4-bit ~0.9. |
| **Memory / load** | Models and lookup tables are stored uncompressed in the APK and memory-mapped (`createSession(ByteBuffer)`, n-gram/vocab hash tables read in place): no extraction, no Java heap copies, no parsing of the 126k n-grams. `libonnxruntime.so` is page-aligned (16 KB) and loaded from the APK. |
| **Target** | Android 17 (API 37) only, arm64-v8a only (Tensor G3): no compatibility code, ~100 MB of other ABIs dropped. |
| **Threads** | ONNX Runtime uses one thread per performance core, detected from cpufreq (Pixel 8a: 4× Cortex-A715 + 1× Cortex-X3, the A510 little cores are skipped); spin-waiting is disabled because synthesis is throttled by playback. Adjustable in the settings. |
| **Latency** | Text is split into sentences and the vocoder runs in windows (48 frames first, then 256) with 28 frames of context, which covers its receptive field exactly, so the first 0.6 s of audio is played while the rest is still being generated, with output identical to a single pass. |
| **Fast speech** | Speech rate scales the predicted durations like the original SSML `prosody rate`, but keeps at least one frame per sound so screen-reader speeds don't drop phonemes. |

On this desktop x86 machine (4 threads) the engine generates speech about
15–20× faster than real time. **It has not been measured on a Pixel 8a**;
the app's *Benchmark* button reports model load time, first-audio latency and
the real-time factor on the device.

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
