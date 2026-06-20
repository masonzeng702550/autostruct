"""``autostruct`` command-line interface: infer / validate / map."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from . import __version__, loader
from .emit import render_html, render_ksy, render_summary
from .ksy_reader import read_ksy_fields
from .model import StructModel
from .pipeline import InferConfig, infer_paths
from .validator import validate_internal

EXIT_OK = 0
EXIT_NO_SAMPLES = 2
EXIT_VALIDATION_ZERO = 3


def _add_entropy_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--window", type=int, default=256, help="entropy window size (default 256)")
    p.add_argument("--step", type=int, default=64, help="entropy window step (default 64)")
    p.add_argument("--entropy-threshold", type=float, default=0.92,
                   help="high-entropy cutoff 0..1 (default 0.92)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autostruct",
        description="Infer a Kaitai Struct (.ksy) draft from unknown binary samples.",
    )
    parser.add_argument("--version", action="version", version=f"autostruct {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # infer ------------------------------------------------------------------
    p_infer = sub.add_parser("infer", help="infer structure and emit a .ksy draft")
    p_infer.add_argument("samples", nargs="+", help="sample files (globs ok)")
    p_infer.add_argument("-o", "--output", help="write .ksy here (default stdout)")
    p_infer.add_argument("--report", help="write the HTML structure map here")
    _add_entropy_opts(p_infer)
    p_infer.add_argument("--align", choices=["equal", "prefix"], default="prefix")
    p_infer.add_argument("--min-confidence", type=float, default=0.3)
    p_infer.add_argument("--no-validate", action="store_true", help="skip closed-loop validation")
    p_infer.add_argument("--name", default="autostruct_fmt", help="meta/id of the type")
    p_infer.add_argument("-v", "--verbose", action="store_true")

    # validate ---------------------------------------------------------------
    p_val = sub.add_parser("validate", help="re-parse samples against an existing .ksy")
    p_val.add_argument("ksy", help="existing .ksy file")
    p_val.add_argument("samples", nargs="+", help="sample files (globs ok)")
    p_val.add_argument("-v", "--verbose", action="store_true")

    # map --------------------------------------------------------------------
    p_map = sub.add_parser("map", help="render the HTML structure map only")
    p_map.add_argument("samples", nargs="+", help="sample files (globs ok)")
    p_map.add_argument("--report", default="map.html", help="HTML output (default map.html)")
    _add_entropy_opts(p_map)
    p_map.add_argument("--align", choices=["equal", "prefix"], default="prefix")
    p_map.add_argument("--name", default="autostruct_fmt")
    p_map.add_argument("-v", "--verbose", action="store_true")

    return parser


def _config_from(args: argparse.Namespace) -> InferConfig:
    return InferConfig(
        window=args.window,
        step=args.step,
        entropy_threshold=args.entropy_threshold,
        align=getattr(args, "align", "prefix"),
        min_confidence=getattr(args, "min_confidence", 0.3),
        name=getattr(args, "name", "autostruct_fmt"),
        do_validate=not getattr(args, "no_validate", False),
        verbose=getattr(args, "verbose", False),
    )


def cmd_infer(args: argparse.Namespace) -> int:
    config = _config_from(args)
    try:
        model, loaded = infer_paths(args.samples, config)
    except loader.LoaderError as exc:
        print(f"autostruct: {exc}", file=sys.stderr)
        return EXIT_NO_SAMPLES

    ksy_text = render_ksy(model)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(ksy_text)
        print(render_summary(model), file=sys.stderr)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(ksy_text)

    if args.report:
        html = render_html(model, loaded.samples[0], title=config.name)
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(html)
        print(f"wrote {args.report}", file=sys.stderr)

    if model.validation is not None and model.validation.total and model.validation.passed == 0:
        print("autostruct: warning — closed-loop parsed 0 samples", file=sys.stderr)
        return EXIT_VALIDATION_ZERO
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        loaded = loader.load(args.samples, "prefix")
    except loader.LoaderError as exc:
        print(f"autostruct: {exc}", file=sys.stderr)
        return EXIT_NO_SAMPLES

    try:
        fields, meta = read_ksy_fields(args.ksy)
    except OSError as exc:
        print(f"autostruct: cannot read {args.ksy}: {exc}", file=sys.stderr)
        return EXIT_NO_SAMPLES

    model = StructModel(meta=meta, fields=fields, n_samples=loaded.n,
                        aligned_length=loaded.aligned_length)
    result = validate_internal(model, loaded.samples, loaded.paths)
    model.validation = result
    print(render_summary(model))
    if result.total and result.passed == 0:
        return EXIT_VALIDATION_ZERO
    return EXIT_OK


def cmd_map(args: argparse.Namespace) -> int:
    config = _config_from(args)
    config.do_validate = True
    try:
        model, loaded = infer_paths(args.samples, config)
    except loader.LoaderError as exc:
        print(f"autostruct: {exc}", file=sys.stderr)
        return EXIT_NO_SAMPLES

    html = render_html(model, loaded.samples[0], title=config.name)
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(render_summary(model))
    print(f"wrote {args.report}", file=sys.stderr)
    return EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "infer":
        return cmd_infer(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "map":
        return cmd_map(args)
    parser.error(f"unknown command {args.command!r}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
