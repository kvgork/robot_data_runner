"""robot-data-run-check — preflight: load policy + dump expected obs/action shape.

Does NOT connect to the robot. Prints exactly what `observation.*` keys the
policy wants so the user can match `--camera <name>` flags.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .policy_loader import load_policy

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        prog="robot-data-run-check",
        description="Pre-flight: load a checkpoint and dump its expected I/O schema.",
    )
    p.add_argument("--policy-path", required=True)
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    try:
        loaded = load_policy(
            Path(args.policy_path),
            Path(args.dataset_root) if args.dataset_root else None,
            args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("policy load failed: %s", exc)
        return 2

    policy = loaded.policy
    print(f"Loaded {type(policy).__name__} on {loaded.device}")
    cfg = getattr(policy, "config", None)
    if cfg is not None:
        for attr in ("input_features", "output_features", "type"):
            val = getattr(cfg, attr, None)
            if val is not None:
                print(f"  {attr}: {val}")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  n_params: {n_params:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
