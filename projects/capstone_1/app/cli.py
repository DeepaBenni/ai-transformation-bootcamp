"""One entry point for every pipeline stage.

Each stage reads a file and writes a file, so a failure late in the run never
costs an expensive earlier step.
"""

from __future__ import annotations

import argparse
import sys

STAGES = ("acquire", "prepare", "seal", "features", "train", "evaluate", "report")


def main(argv: list[str] | None = None) -> int:
    """Dispatch a pipeline stage.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="SignalCraft: flight late-arrival risk (Track A).",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--start", help="acquire: first month, YYYY-MM")
    parser.add_argument("--end", help="acquire: last month, YYYY-MM")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.60,
        help="train: the recall the operating point is chosen for",
    )
    args = parser.parse_args(argv)

    if args.stage == "acquire":
        from app.data import acquire

        return acquire.main(["--start", args.start, "--end", args.end])

    if args.stage == "prepare":
        from app.data import prepare

        print(f"wrote {prepare.run()}")
        return 0

    if args.stage == "seal":
        from app.data import seal

        print(f"seal verified: {seal.verify()}")
        return 0

    if args.stage == "features":
        from app.features import build

        print(f"wrote {build.run()}")
        return 0

    if args.stage == "train":
        import pandas as pd

        from app.config import FEATURES_PARQUET
        from app.models import train

        if not FEATURES_PARQUET.exists():
            print(
                f"flights_features.parquet not found at {FEATURES_PARQUET}.\n"
                f"Run:  python -m app.cli features",
                file=sys.stderr,
            )
            return 1

        frame = pd.read_parquet(FEATURES_PARQUET)
        results = train.run_cv(frame)
        summary = train.summarise(results)
        print(summary.to_string())

        best = summary.index[0]
        model, meta = train.fit_final(frame, best)
        path = train.export(model, frame, results, family=best, target_recall=args.target_recall)
        print(f"\nshipped {best}: {path}  ({meta['rows']:,} rows)")
        return 0

    if args.stage == "evaluate":
        from app.explain import report

        return report.main()

    if args.stage == "report":
        from app.explain import docs

        return docs.main()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
