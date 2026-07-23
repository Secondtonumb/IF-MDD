#!/usr/bin/env python3
"""Build the standalone Llama-3.2 MDD-LLM L2-ARCTIC bundle."""

import argparse
from pathlib import Path

from release_bundle_builder import BundleSpec, build_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAVLM = ROOT / "pretrained_models/models--microsoft--wavlm-large/snapshots/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wavlm", type=Path, default=DEFAULT_WAVLM)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_bundle(
        BundleSpec(
            name="MDD-LLM-Llama3.2-1B-L2-ARCTIC",
            bundle_type="mdd-llm-llama3.2",
            checkpoint=args.checkpoint,
            hparams=ROOT / "hparams/MDD_LLM_Llama3_2_1B.yaml",
            output=args.output,
            root=ROOT,
            wavlm=args.wavlm,
            llama=args.llama,
            required_recoverables=("model",),
            model_signature={
                "feature_fusion": "SSL_LLM_origin_ver2",
                "LLM_DIM": 2048,
                "output_neurons": 44,
            },
            defaults={
                "prompt_system_text": "You are a phoneme transcriber.",
                "prompt_user_text": (
                    "Transcribe the preceding speech into phonemes."
                ),
            },
        )
    )
    print(output)


if __name__ == "__main__":
    main()
