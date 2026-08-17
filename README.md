# LightTransformerKWS2

Lightweight keyword spotting with a configurable Transformer encoder: GQA attention, optional weight tying, dynamic INT8 quantization, and optional L1 pruning.

## Architecture

- **Input:** MFCC features (98 frames × 40 coefficients)
- **Front-end:** 2× Conv1d + BatchNorm + GELU (optional linear projection)
- **Encoder:** depth × `EncoderBlock` (Pre-LN → `MultiHeadAttentionGQA` → FFN)
- **Position:** learned absolute embeddings + optional T5-style relative position bias (weight-tied)
- **Pooling:** mean over time → linear classifier

Key options (all CLI-configurable in `train.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--dim` | 48 | Embedding dimension |
| `--heads` | 8 | Attention heads |
| `--gqa-groups` | 4 | GQA groups (`heads` must divide evenly) |
| `--layers` | 12 | Encoder depth |
| `--mlp-dim` | 96 | FFN hidden size |
| `--share-layers` | off | ALBERT-style weight tying (one block, repeated `depth` times) |
| `--enable-pruning` | off | L1 unstructured pruning before quantization |
| `--label-smooth` | off | Label smoothing loss (default: cross-entropy) |

## Setup

```bash
python -m venv kwt_env
source kwt_env/bin/activate
pip install -r requirements.txt
```

**Data:** place Google Speech Commands under `data/google-speech-commands/` (see `datagen.py` for version paths).

**`kws_streaming`:** required for data loading (`datagen.py` imports it). Clone Google's [kws_streaming](https://github.com/google-research/google-research/tree/master/kws_streaming) into this directory if missing.

## Training

```bash
python train.py my_exp --heads 8 --version 2
python train.py my_exp --share-layers --enable-pruning
sh train.sh
```

After training, `model.pth` and `model_quantized.pt` are written to the project root; logs go to `results/<experiment>/`.

## Project layout

```
LightTransformerKWS2/
├── models/kwt.py       # KWT model (MultiHeadAttentionGQA, EncoderBlock)
├── train.py            # Training + quantization + optional pruning
├── datagen.py          # GSC data pipeline (uses kws_streaming)
├── kws_streaming/      # Google KWS streaming utils (not vendored; clone separately)
├── utils/              # Warmup scheduler, label smoothing
├── train.sh            # Example launch script
└── requirements.txt
```

## License

See [LICENSE](LICENSE).
