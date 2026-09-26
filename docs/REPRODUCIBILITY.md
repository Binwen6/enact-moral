# Reproducibility protocol

## Deterministic release checks

Run the following commands from the repository root after setting
`PYTHONPATH=src`:

```bash
python -m data_validation
python -m anonymity
python scripts/reproduce.py
python -m unittest discover -s tests -v
```

The data validator checks the nine configuration counts and representative
sample coverage. The anonymity checker scans release text for machine paths,
identity fragments, email addresses, secret assignments, and private keys. The
reproduction script validates Table 3 additive identities, intervention
directionality, Figure 2 quartile ordering, and scale ordering before creating
plots.

## Model execution contract

For local weights, `--model-source local` requires `--model-root`; the selected
directory must contain the configured model subdirectory and `config.json`.
For Hub weights, `--model-source hub` requires an immutable `--revision`.
Remote-code execution is disabled. Precision and chat-template behavior are
declared in `configs/models.yaml`.

Every main intervention uses a one-indexed layer number. The relational-binding
entry point reads causal-source states at the causal-statement endpoint and
outcome-valence states at the outcome endpoint. The low-rank subspace acts on
the final-token residual of one downstream block, trains on train graphs, and
evaluates on validation graphs. Its basis is trained with Adam so that writing
the joint-run residual inside the basis reproduces the Yes-minus-No margin of
the inferred-intention readout under joint source replacement; a QR
factorization keeps the columns orthonormal. `--shuffled-control` trains a
second basis that pairs each source residual with a wrong example in the same
batch and keeps the rank, optimizer, and step count.
The frozen values used by the manuscript are listed in
`configs/paper_experiments.yaml`.

## Main evidence chain

1. `behavioral_screen.py` collects factual and judgment readouts.
2. `patient_completion.py` tests sufficiency and necessity of the inferred patient state.
3. `relational_binding.py` compares source-only, valence-only, additive, and joint replacement.
4. `mediator_knockout.py` removes downstream attention to an intermediate event and equal-length random spans.
5. `compact_interface.py` trains the low-rank basis, reports its recovery of the joint effect against the full-state write and the shuffled-supervision basis, and can export the basis as NumPy data.
6. `component_mediation.py` tests selected attention heads and MLP blocks.
7. `external_validation.py` applies the frozen basis to paired prompts from licensed external datasets.

## Numerical release

`results/paper_metrics/` stores manuscript-level measurements with the displayed
precision. `outputs/` is ignored by version control and receives generated
figures and experiment outputs. Instance-level dataset traces permit bootstrap
confidence intervals and alternative aggregations without changing the frozen
summary files.

## Hardware

The sample patient-completion check was executed with Qwen2.5-0.5B-Instruct on
one NVIDIA RTX 6000D 85GB in under one minute. Metric validation and summary
figure generation complete on CPU in under five seconds. Large-model
experiments require enough aggregate accelerator memory for the declared
bfloat16 checkpoint. The scripts expose `--device-map` and perform no automatic
precision substitution.
