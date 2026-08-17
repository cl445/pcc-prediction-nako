"""CLI entry-point: ``uv run python -m smoke_test [profile|generate]``."""

from __future__ import annotations

import argparse
from pathlib import Path

_DEFAULT_PROFILE = Path("smoke_test/profiles/default.json")


def _cmd_profile(args: argparse.Namespace) -> None:
    from smoke_test.profiler import extract_profile, save_profile

    print(f"Profiling parquets in {args.input_dir} …")
    profile = extract_profile(Path(args.input_dir))
    save_profile(profile, Path(args.output))


def _cmd_generate(args: argparse.Namespace) -> None:
    from smoke_test.generators import generate_all, load_profile

    profile_path = Path(args.profile)
    if not profile_path.exists():
        raise SystemExit(f"Profile not found: {profile_path}")

    profile = load_profile(profile_path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_subjects} subjects (seed={args.seed}) …")
    frames = generate_all(profile, n_subjects=args.n_subjects, seed=args.seed)

    for name, df in frames.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name}: {len(df)} rows x {len(df.columns)} cols -> {path}")

    print(f"\nGenerated {len(frames)} parquet files in {out}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smoke_test",
        description="Profile real data or generate synthetic smoke-test parquets.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- profile ---
    p_prof = sub.add_parser("profile", help="Extract statistics from real parquets")
    p_prof.add_argument(
        "--input-dir",
        default="data/processed",
        help="Directory with real parquet files",
    )
    p_prof.add_argument(
        "--output",
        default=str(_DEFAULT_PROFILE),
        help="Output JSON path",
    )

    # --- generate ---
    p_gen = sub.add_parser("generate", help="Generate synthetic parquets from profile")
    p_gen.add_argument(
        "--profile",
        default=str(_DEFAULT_PROFILE),
        help="Path to JSON profile",
    )
    # Defaults to the smoke tree, not data/processed: generating writes
    # parquets, and the default must never be the directory that holds the
    # real ones.
    p_gen.add_argument("--output-dir", default="smoke_data")
    p_gen.add_argument("--n-subjects", type=int, default=500)
    p_gen.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "profile":
        _cmd_profile(args)
    elif args.command == "generate":
        _cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
