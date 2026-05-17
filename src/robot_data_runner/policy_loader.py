"""load_policy — wrap lerobot.policies.factory.make_policy with our knobs.

Soft-imports lerobot; raises ImportError with an actionable message when
the user runs without it installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedPolicy:
    """Container for a ready-to-call policy + its device + metadata source.

    ``preprocessor`` / ``postprocessor`` are the lerobot 0.5
    ``PolicyProcessorPipeline`` objects loaded from the checkpoint.
    They MUST be applied around ``select_action`` because policies like
    SmolVLA depend on the preprocessor's ``tokenizer_processor`` step to
    convert ``obs['task']`` (a string) into
    ``observation.language.tokens`` before the model forward pass.
    """

    policy: Any
    device: str
    ds_meta: Any | None
    preprocessor: Any = None
    postprocessor: Any = None


def load_policy(
    policy_path: Path,
    dataset_root: Path | None,
    seed: int = 42,
) -> LoadedPolicy:
    """Load a lerobot ``PreTrainedPolicy`` from a checkpoint dir.

    Parameters
    ----------
    policy_path:
        ``pretrained_model/`` directory.
    dataset_root:
        Path to the dataset the policy was trained on. Used to derive
        observation/action feature shapes when the checkpoint config
        does not embed them (most lerobot 0.5 checkpoints).
    seed:
        Torch seed.
    """
    try:
        import torch
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "torch is required to load a policy. Install with: pip install torch"
        ) from exc
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "lerobot >= 0.5 is required. Install with: pip install lerobot"
        ) from exc

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = Path(policy_path)

    ds_meta = None
    if dataset_root is not None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        parts = Path(dataset_root).resolve().parts
        repo_id = (
            "/".join(parts[-2:]) if len(parts) >= 2 else Path(dataset_root).name
        )
        ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=str(dataset_root))

    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.to(device)
    policy.eval()

    # Load the preprocessor / postprocessor pipelines from the checkpoint.
    # Required for SmolVLA's language path (tokenizer_processor); harmless
    # for ACT / diffusion (they still need normalization).
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(policy_path),
            dataset_stats=getattr(ds_meta, "stats", None),
        )
    except Exception:  # noqa: BLE001
        # Older lerobot or unusual checkpoint: keep going without
        # processors. select_action may still work for ACT/diffusion.
        preprocessor = None
        postprocessor = None

    return LoadedPolicy(
        policy=policy,
        device=device,
        ds_meta=ds_meta,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )
