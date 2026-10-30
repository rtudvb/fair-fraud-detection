"""Command-line entry point: ``python -m frauddetect <command>``.

Commands
--------
download   Download BAF from Kaggle into ``data/raw`` and convert it to Parquet.
validate   Check the schema of downloaded variants and print a short summary.
models     List the ten registered models and whether their dependencies are installed.
smoke      Check the GPU and fit every installed model on tiny synthetic data.
run        Run one experiment (eda, reproduce, benchmark, ...) or the whole pipeline (all).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys

from frauddetect.config import load_config

log = logging.getLogger("frauddetect")

_EXTRA_MODULES = {
    "core": ["sklearn"],
    "gbdt": ["lightgbm", "xgboost", "catboost"],
    "dl": ["pytabkit"],
    "fm": ["tabpfn", "tabicl"],
}


def _cmd_download(args: argparse.Namespace) -> int:
    from frauddetect.data.download import convert_csv_to_parquet, download_from_kaggle

    config = load_config(*args.config)
    raw_dir = config["paths"]["raw"]
    download_from_kaggle(config["dataset"]["kaggle_slug"], raw_dir)
    for path in convert_csv_to_parquet(raw_dir, remove_csv=args.remove_csv):
        log.info("ready: %s", path)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from frauddetect.data.load import load_variant
    from frauddetect.data.schema import LABEL, MONTH, validate_frame

    config = load_config(*args.config)
    variants = args.variants or list(config["dataset"]["variants"])
    status = 0
    for variant in variants:
        try:
            df = load_variant(config, variant, validate=False)
        except FileNotFoundError as exc:
            log.error("%s: %s", variant, exc)
            status = 1
            continue
        report = validate_frame(df, strict=False)
        prevalence = df.groupby(MONTH)[LABEL].mean().round(4).to_dict()
        print(f"[{'OK' if report.ok else 'FAIL'}] {variant}: {report.n_rows:,} rows")
        print(f"    fraud rate by month: {prevalence}")
        for message in report.errors + report.warnings:
            print(f"    - {message}")
        status |= 0 if report.ok else 1
    return status


def _cmd_models(args: argparse.Namespace) -> int:
    from frauddetect.models.registry import MODELS

    for spec in MODELS.values():
        modules = _EXTRA_MODULES.get(spec.extra, [])
        installed = all(importlib.util.find_spec(m) is not None for m in modules)
        mark = "installed" if installed else f"missing -> pip install -e '.[{spec.extra}]'"
        print(f"{spec.name:<14} {spec.family:<14} {mark}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from frauddetect.smoke import environment_report, run_smoke

    lines, device = environment_report()
    print("\n".join(lines))
    if args.cpu:
        device = "cpu"
    print(f"\nFitting models on synthetic data (device for deep learning: {device}) ...")
    results = run_smoke(args.models, device=device)
    for r in results:
        print(f"[{'OK' if r.ok else 'FAIL'}] {r.model:<14} {r.seconds:6.1f}s  {r.detail}")
    failed = [r.model for r in results if not r.ok]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} models OK"
        + (f"; failed: {failed}" if failed else "")
    )
    return 1 if failed else 0


def _cmd_run(args: argparse.Namespace) -> int:
    from frauddetect.experiments import PIPELINE, get_runner, make_context

    names = list(PIPELINE) if args.experiment == ["all"] else args.experiment
    skip = set(args.skip or [])
    for name in names:
        if name in skip:
            log.info("skip %s", name)
            continue
        ctx = make_context(
            name,
            synthetic=args.synthetic,
            quick=args.quick,
            models=args.models,
            device=args.device,
            extra_configs=args.config,
        )
        log.info(
            "=== %s (device=%s%s) ===", name, ctx.device, ", synthetic" if ctx.synthetic else ""
        )
        get_runner(name)(ctx)
        ctx.write_meta()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frauddetect", description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="download BAF from Kaggle")
    download.add_argument("--config", nargs="*", default=[], help="extra YAML configs")
    download.add_argument("--remove-csv", action="store_true", help="keep only Parquet files")
    download.set_defaults(func=_cmd_download)

    validate = sub.add_parser("validate", help="validate downloaded variants")
    validate.add_argument("--config", nargs="*", default=[], help="extra YAML configs")
    validate.add_argument("--variants", nargs="*", help="variant keys, e.g. base variant_2")
    validate.set_defaults(func=_cmd_validate)

    models = sub.add_parser("models", help="list registered models")
    models.set_defaults(func=_cmd_models)

    smoke = sub.add_parser("smoke", help="check GPU and fit every model on tiny data")
    smoke.add_argument("--models", nargs="*", help="subset of models, e.g. lightgbm tabpfn")
    smoke.add_argument("--cpu", action="store_true", help="do not use the GPU")
    smoke.set_defaults(func=_cmd_smoke)

    run = sub.add_parser("run", help="run experiments (see docs/experiments.md)")
    run.add_argument("experiment", nargs="+", help="experiment names, or `all`")
    run.add_argument("--synthetic", action="store_true", help="use generated data (checks only)")
    run.add_argument("--quick", action="store_true", help="tiny settings for smoke runs")
    run.add_argument("--models", nargs="*", help="override the experiment's model list")
    run.add_argument("--device", help="auto | cpu | cuda:0 (default: from config)")
    run.add_argument("--skip", nargs="*", help="experiments to skip with `all`")
    run.add_argument("--config", nargs="*", default=[], help="extra YAML configs")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
