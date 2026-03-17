import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import sys
sys.path.insert(0, "Normalized-Attention-Guidance")

import json
import argparse
import random
import numpy as np
import torch
from collections import defaultdict

from nag import NAGFluxPipeline, NAGFluxTransformer2DModel
from qwen3_vl_embedding import Qwen3VLEmbedder


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_and_split(path: str, seed: int = 42):
    with open(path) as f:
        data = [json.loads(line) for line in f]

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for entry in data:
        by_class[entry["class_name"]].append(entry)

    dev, test = [], []
    for cls, entries in by_class.items():
        shuffled = entries[:]
        rng.shuffle(shuffled)
        dev.extend(shuffled[:2])
        test.extend(shuffled[2:])

    return dev, test


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def precompute_text_embeddings(embedder, dev_set):
    """Embed all dev prompts once; returns tensor of shape (N, D)."""
    inputs = [{"text": entry["image_prompt"]} for entry in dev_set]
    with torch.no_grad():
        text_embs = embedder.process(inputs, normalize=True)  # (N, D)
    return text_embs.cpu()


def score_image(embedder, image, text_emb):
    """Cosine similarity between a generated image and a pre-embedded text."""
    with torch.no_grad():
        img_emb = embedder.process([{"image": image}], normalize=True)  # (1, D)
    img_emb = img_emb.cpu()
    return (text_emb @ img_emb.squeeze()).item()


# ---------------------------------------------------------------------------
# HP sampling
# ---------------------------------------------------------------------------

HP_RANGES = {
    "angular": {
        "nag_scale": (1.0, 10.0),
        "nag_alpha": (0.1, 0.5),
        "nag_tau":   (0.1, 0.5),   # units of pi
    },
    "norm_cfg": {
        "nag_scale": (5.0, 25.0),
        "nag_alpha": (0.1, 0.5),
        "nag_tau":   (1.0, 5.0),   # unit-vector magnitude, >= 1
    },
}

FIXED_PARAMS = {
    "angular":  {"guidance_scale": 5.0},
    "norm_cfg": {"guidance_scale": 0.0},
}


def sample_hps(method: str, rng: np.random.Generator):
    hps = {}
    for name, (lo, hi) in HP_RANGES[method].items():
        hps[name] = float(rng.uniform(lo, hi))
    return hps


# ---------------------------------------------------------------------------
# Single round: generate images + score
# ---------------------------------------------------------------------------

def run_round(pipe, embedder, dev_set, text_embs, method: str, hps: dict, seed: int):
    fixed = FIXED_PARAMS[method]
    scores = []
    for i, entry in enumerate(dev_set):
        image = pipe(
            entry["image_prompt"],
            nag_negative_prompt=entry["negative_prompt"],
            guidance_scale=fixed["guidance_scale"],
            nag_scale=hps["nag_scale"],
            nag_alpha=hps["nag_alpha"],
            nag_tau=hps["nag_tau"],
            num_inference_steps=4,
            max_sequence_length=256,
            nag_guidance_type=method,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        s = score_image(embedder, image, text_embs[i])
        scores.append(s)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def load_existing_rounds(output_path: str, method: str):
    """Read completed rounds for this method from a JSONL file."""
    if not os.path.exists(output_path):
        return []
    results = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["method"] == method:
                results.append(record)
    return results


def sweep(pipe, embedder, dev_set, text_embs, method: str, n_rounds: int,
          output_path: str, master_seed: int = 42):
    # Resume: count already-completed rounds for this method
    done_results = load_existing_rounds(output_path, method)
    done = len(done_results)
    if done:
        print(f"[{method}] Resuming from round {done} ({done} already done)")

    # Advance RNG past completed rounds so future samples stay deterministic
    rng = np.random.default_rng(master_seed)
    for _ in range(done):
        sample_hps(method, rng)

    # Append mode: each round adds exactly one line, no rewrites
    with open(output_path, "a") as f:
        for round_idx in range(done, n_rounds):
            hps = sample_hps(method, rng)
            print(f"[{method}] Round {round_idx + 1}/{n_rounds}  HPs: {hps}")

            avg_score = run_round(pipe, embedder, dev_set, text_embs, method, hps,
                                  seed=master_seed + round_idx)

            record = {"method": method, "round": round_idx, "hps": hps, "avg_score": avg_score}
            f.write(json.dumps(record) + "\n")
            f.flush()
            done_results.append(record)
            print(f"[{method}] Round {round_idx + 1} score: {avg_score:.4f}")

    # Report top-5
    top5 = sorted(done_results, key=lambda r: r["avg_score"], reverse=True)[:5]
    print(f"\n=== Top-5 HPs for {method} ===")
    for rank, r in enumerate(top5, 1):
        print(f"  #{rank}  score={r['avg_score']:.4f}  {r['hps']}")

    return done_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["angular", "norm_cfg", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="sweep_results.jsonl")
    parser.add_argument("--data", default="generated_prompts.jsonl")
    args = parser.parse_args()

    methods = ["angular", "norm_cfg"] if args.method == "both" else [args.method]

    # Load data
    print("Loading data...")
    dev_set, test_set = load_and_split(args.data, seed=args.seed)
    print(f"Dev: {len(dev_set)} prompts | Test: {len(test_set)} prompts")

    # Load NAG pipeline
    print("Loading NAG pipeline...")
    model_id = "black-forest-labs/FLUX.1-schnell"
    transformer = NAGFluxTransformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = NAGFluxPipeline.from_pretrained(
        model_id, transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")

    # Load Qwen embedder
    print("Loading Qwen3VLEmbedder...")
    embedder = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-8B", device="cuda:1")

    # Pre-compute text embeddings for dev set (shared across methods)
    print("Pre-computing text embeddings for dev set...")
    text_embs = precompute_text_embeddings(embedder, dev_set)
    print(f"Text embeddings shape: {text_embs.shape}")

    # Run sweeps
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Sweeping {method} ({args.rounds} rounds)")
        print(f"{'='*60}")
        sweep(pipe, embedder, dev_set, text_embs, method,
              n_rounds=args.rounds, output_path=args.output, master_seed=args.seed)

    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main()
