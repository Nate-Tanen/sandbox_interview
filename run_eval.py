import argparse
import json
from pathlib import Path

from sandbox_eval.evaluator import evaluate_sources


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compile and correctness checks.")
    parser.add_argument("--reference", required=True, help="Path to reference source.")
    parser.add_argument("--candidate", required=True, help="Path to candidate source.")
    parser.add_argument(
        "--build-root",
        default=str(Path.cwd() / "build"),
        help="Directory for extension builds.",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of correctness trials.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic evaluation.")
    args = parser.parse_args()

    reference_path = Path(args.reference)
    candidate_path = Path(args.candidate)

    result = evaluate_sources(
        ref_src=reference_path.read_text(),
        candidate_src=candidate_path.read_text(),
        build_root=Path(args.build_root),
        num_trials=args.trials,
        seed_num=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["compile_pass"] and result["correct_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
