# AutoStruct

**Sample-driven binary format structure inference.**

Feed AutoStruct one or more unknown binary samples of the *same* format and it
automatically produces:

- a **Kaitai Struct (`.ksy`) draft** with per-field confidence annotations,
- an **HTML hex structure map** (coloured fields + entropy + cross-sample heat),
- a **closed-loop validation** report — it re-parses the original samples against
  the inferred model and reports the pass rate.

It automates the tedious first step of reverse-engineering an unknown container,
firmware blob, save file, or private protocol message: *seeing the skeleton*. The
output is a **draft** meant for human review, not a finished spec.

## Why

Reverse-engineering an unknown binary format is manual work: staring at a hex
editor byte by byte, eyeballing which bytes are magic / length / offset, and
diffing several samples by hand to spot what is constant vs. variable. Kaitai
Struct describes a format but does not *infer* one from samples. AutoStruct fills
that gap: **N same-format samples in → a verifiable structure draft out.**

## Install

```bash
pip install -e .
# optional extras: scipy (faster autocorrelation) + kaitaistruct (external parser)
pip install -e ".[full]"
```

Only `numpy` is required. The closed-loop validator works without external
dependencies by re-parsing samples directly against the inferred model; if the
`kaitai-struct-compiler` is on `PATH` it is used for an additional syntax check.

## Usage

```bash
# Infer a structure from a batch of samples
autostruct infer samples/*.bin -o fmt.ksy --report map.html

# Re-validate an existing .ksy against samples (closed loop)
autostruct validate fmt.ksy samples/*.bin

# Just render the HTML structure map
autostruct map dump.dat --window 128
```

### Key options (`infer`)

| Option | Default | Meaning |
| --- | --- | --- |
| `-o, --output PATH` | stdout | where to write the `.ksy` draft |
| `--report PATH` | — | write the HTML structure map |
| `--window N` | 256 | entropy sliding-window size |
| `--step N` | 64 | entropy window step |
| `--entropy-threshold F` | 0.92 | normalised high-entropy cutoff (0..1) |
| `--align {equal,prefix}` | prefix | cross-sample alignment strategy |
| `--min-confidence F` | 0.3 | fields below this become `unknown bytes` |
| `--no-validate` | off | skip the closed-loop Kaitai validation |
| `--name ID` | autostruct_fmt | `meta/id` of the emitted type |
| `-v, --verbose` | off | print per-detector decisions |

Exit codes: `0` success · `2` no samples / read error · `3` closed-loop pass rate 0%.

## How it works

A byte matrix (`N` samples × aligned positions) flows through the pipeline:

1. **Load & align** samples into a `(N, L)` `uint8` matrix.
2. **Entropy segmentation** splits each sample into `structured` and
   `high_entropy` regions.
3. **Per-position stats**: distinct-value count, constant / monotonic masks.
4. **Field detectors** (pure functions over the matrix): constant/magic, integer
   typing + endianness, length/TLV correlation, repeated fixed-length records,
   and string runs — each emits a confidence and a rationale.
5. **Model** assembly into a serialization-agnostic `StructModel`.
6. **Emit** `.ksy` + HTML + terminal summary.
7. **Validate** by re-parsing the samples; the pass rate feeds back into the
   report.

## Limitations

The output is a **draft**. Single-sample inputs cannot be cross-diffed, so
confidence is capped low and clearly flagged. Encrypted / compressed content is
marked as an opaque high-entropy blob, not decoded. Naive left-alignment can
misfire on variable-length-prefixed or nested containers; the report flags low
confidence honestly rather than guessing.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
