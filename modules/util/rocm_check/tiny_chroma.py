"""A tiny Chroma model with random weights and a small masked dataset, built offline, for training runs through the real
scripts/train.py (rocm_check's end-to-end check). Same file layout and component classes as lodestones/Chroma1-HD
(diffusers folder: tokenizer, scheduler, text_encoder, vae, transformer), at a few MB."""
import json
import os
import random

import torch

_WORDS = ["a", "an", "the", "photo", "picture", "of", "red", "green", "blue", "yellow", "small", "large", "cat", "dog",
          "house", "tree", "car", "bird", "flower", "sky", "river", "mountain", "city", "street", "person", "face",
          "hand", "window", "door", "table", "chair", "book", "cup", "light", "dark", "night", "day", "sun", "moon",
          "with", "and", "on", "in", "under", "over", "near"]


def _tokenizer(path: str, tmp: str):
    from transformers import T5Tokenizer

    import sentencepiece as spm

    corpus = os.path.join(tmp, "corpus.txt")
    rng = random.Random(0)
    with open(corpus, "w") as f:
        for _ in range(2000):
            f.write(" ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 12))) + "\n")
    prefix = os.path.join(tmp, "spiece")
    # T5's special ids: pad 0, eos 1, unk 2, no bos
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=prefix, vocab_size=96, model_type="unigram",
                                   pad_id=0, eos_id=1, unk_id=2, bos_id=-1, character_coverage=1.0,
                                   hard_vocab_limit=False, minloglevel=2)
    tokenizer = T5Tokenizer(vocab_file=prefix + ".model", extra_ids=0, legacy=False, model_max_length=512)
    tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
    return len(tokenizer)


def build_tiny_chroma(path: str, seed: int = 0) -> str:
    """writes the model folder (overwriting it) and returns its path"""
    import tempfile

    from diffusers import AutoencoderKL, ChromaTransformer2DModel, FlowMatchEulerDiscreteScheduler
    from transformers import T5Config, T5EncoderModel

    os.makedirs(path, exist_ok=True)
    torch.manual_seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        vocab = _tokenizer(path, tmp)

    # no dynamic shifting: ChromaSampler sets the timesteps without the image-size dependent mu it would need
    FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False) \
        .save_pretrained(os.path.join(path, "scheduler"))

    text_dim = 32
    T5EncoderModel(T5Config(
        vocab_size=vocab, d_model=text_dim, d_kv=8, d_ff=64, num_layers=2, num_heads=4,
        relative_attention_num_buckets=8, relative_attention_max_distance=32, dropout_rate=0.0,
        feed_forward_proj="gated-gelu", decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )).save_pretrained(os.path.join(path, "text_encoder"), safe_serialization=True)

    # the Flux/Chroma VAE layout (16 latent channels, 8x downsampling, no quant convs), with narrow blocks
    AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=16,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(16, 16, 32, 32), layers_per_block=1, norm_num_groups=8,
        scaling_factor=0.3611, shift_factor=0.1159, use_quant_conv=False, use_post_quant_conv=False,
    ).save_pretrained(os.path.join(path, "vae"), safe_serialization=True)

    ChromaTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=2, num_single_layers=2, attention_head_dim=16,
        num_attention_heads=4, joint_attention_dim=text_dim, axes_dims_rope=(4, 6, 6),
        approximator_num_channels=16, approximator_hidden_dim=64, approximator_layers=2,
    ).save_pretrained(os.path.join(path, "transformer"), safe_serialization=True)

    with open(os.path.join(path, "model_index.json"), "w") as f:
        json.dump({
            "_class_name": "ChromaPipeline",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "T5Tokenizer"],
            "transformer": ["diffusers", "ChromaTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }, f, indent=2)
    return path


def build_dataset(path: str, count: int = 6, size: int = 256, seed: int = 0) -> str:
    """images with captions and masks (<stem>-masklabel.png, white where the loss counts), in two aspect ratios"""
    import numpy as np
    from PIL import Image, ImageDraw

    os.makedirs(path, exist_ok=True)
    rng = random.Random(seed)
    for i in range(count):
        w, h = (size, size) if i % 2 == 0 else (size, size * 3 // 4)
        base = np.asarray([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        image = (base + 60 * np.sin(xx / (7 + i) )[..., None] + 60 * np.cos(yy / (5 + i))[..., None]).clip(0, 255)
        img = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        draw.ellipse((w // 4, h // 4, 3 * w // 4, 3 * h // 4), fill=tuple(rng.randint(0, 255) for _ in range(3)))
        img.save(os.path.join(path, f"img{i}.png"))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rectangle((w // 8, h // 8, 7 * w // 8, 7 * h // 8), fill=255)
        mask.save(os.path.join(path, f"img{i}-masklabel.png"))
        with open(os.path.join(path, f"img{i}.txt"), "w") as f:
            f.write(" ".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 10))))
    return path
