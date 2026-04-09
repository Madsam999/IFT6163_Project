"""Exports a Brax PPO .params checkpoint to a portable policy bundle.

Bundle contents:
  - policy_bundle.npz: normalizer stats + policy MLP weights
  - policy_bundle.json: metadata needed for inference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from flax.serialization import from_bytes


def _load_params(path: Path) -> Any:
    return from_bytes(None, path.read_bytes())


def _pick_normalizer(tree: Dict[str, Any], key_hint: Optional[str]) -> Dict[str, Any]:
    if key_hint and key_hint in tree:
        candidate = tree[key_hint]
        if isinstance(candidate, dict) and "mean" in candidate and "std" in candidate:
            return candidate

    for _, value in tree.items():
        if isinstance(value, dict) and "mean" in value and "std" in value:
            return value
    raise ValueError("Could not find normalizer subtree with keys {mean,std}.")


def _pick_policy(tree: Dict[str, Any], key_hint: Optional[str]) -> Dict[str, Any]:
    if key_hint and key_hint in tree:
        candidate = tree[key_hint]
        if isinstance(candidate, dict) and "params" in candidate:
            params = candidate["params"]
            if isinstance(params, dict) and any(k.startswith("hidden_") for k in params.keys()):
                return params

    best = None
    best_out = -1
    for _, value in tree.items():
        if not isinstance(value, dict) or "params" not in value:
            continue
        params = value["params"]
        if not isinstance(params, dict):
            continue
        layer_names = sorted(
            [k for k in params.keys() if k.startswith("hidden_")],
            key=lambda x: int(x.split("_")[1]),
        )
        if not layer_names:
            continue
        out_bias = params[layer_names[-1]].get("bias")
        if out_bias is None:
            continue
        out_dim = int(np.asarray(out_bias).shape[0])
        if out_dim > best_out:
            best_out = out_dim
            best = params
    if best is None:
        raise ValueError("Could not find policy parameter subtree.")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--policy-key", type=str, default="")
    parser.add_argument("--normalizer-key", type=str, default="")
    parser.add_argument(
        "--action-head",
        type=str,
        default="normal_tanh",
        choices=["normal_tanh", "tanh", "raw"],
        help="How to interpret policy output during inference.",
    )
    args = parser.parse_args()

    params = _load_params(args.params_path)
    if not isinstance(params, dict):
        raise ValueError(f"Unsupported params root type: {type(params)} (expected dict).")

    norm = _pick_normalizer(params, args.normalizer_key or None)
    policy = _pick_policy(params, args.policy_key or None)

    layer_names = sorted([k for k in policy.keys() if k.startswith("hidden_")], key=lambda x: int(x.split("_")[1]))
    if not layer_names:
        raise ValueError("No hidden_* layers found in selected policy.")

    arrays = {
        "obs_mean": np.asarray(norm["mean"], dtype=np.float32),
        "obs_std": np.asarray(norm["std"], dtype=np.float32),
        "obs_std_eps": np.asarray(norm["std_eps"], dtype=np.float32),
    }

    hidden_sizes = []
    for i, name in enumerate(layer_names):
        w = np.asarray(policy[name]["kernel"], dtype=np.float32)
        b = np.asarray(policy[name]["bias"], dtype=np.float32)
        arrays[f"{name}.kernel"] = w
        arrays[f"{name}.bias"] = b
        if i < len(layer_names) - 1:
            hidden_sizes.append(int(w.shape[1]))

    obs_dim = int(arrays["obs_mean"].shape[0])
    output_dim = int(arrays[f"{layer_names[-1]}.bias"].shape[0])
    if args.action_head == "normal_tanh" and output_dim % 2 == 0:
        action_dim = int(output_dim // 2)
    else:
        action_dim = int(output_dim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "policy_bundle.npz"
    json_path = args.output_dir / "policy_bundle.json"
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(
        json.dumps(
            {
                "source_params_path": str(args.params_path),
                "activation": args.activation,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "policy_output_dim": output_dim,
                "hidden_sizes": hidden_sizes,
                "layer_names": layer_names,
                "action_head": args.action_head,
                "notes": "Use with puppersim/pupper_brax_policy_bundle.py for deterministic inference.",
            },
            indent=2,
        )
    )
    print(f"wrote: {npz_path}")
    print(f"wrote: {json_path}")


if __name__ == "__main__":
    main()
