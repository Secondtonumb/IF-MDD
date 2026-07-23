#!/usr/bin/env python3
"""Build the standalone CROTTC-IF L2-ARCTIC bundle."""

import argparse
from pathlib import Path

from release_bundle_builder import BundleSpec, build_bundle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAVLM = ROOT / "pretrained_models/models--microsoft--wavlm-large/snapshots/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wavlm", type=Path, default=DEFAULT_WAVLM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build_bundle(
        BundleSpec(
            name="CROTTC-IF-l2-arctic",
            bundle_type="crottc-if",
            checkpoint=args.checkpoint,
            hparams=ROOT / "hparams/CROTTC_IF.yaml",
            output=args.output,
            root=ROOT,
            wavlm=args.wavlm,
            llama=None,
            required_recoverables=("model", "perceived_ssl"),
            model_signature={
                "feature_fusion": "Trans_IFMDD_ConPCO_ver2",
                "ctc_loss_type": "ottc",
                "output_neurons": 44,
            },
            defaults={"ctc_decode_weight": 0.99},
        )
    )
    print(output)


if __name__ == "__main__":
    main()
