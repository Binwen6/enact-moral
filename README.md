# ENACT-Moral

Code, data tooling, and released measurements for the anonymous ICLR submission
“Beyond Static Values: Causal Transition from Event Relations to Moral Judgment in LLMs.”

The repository covers the main evidence chain:

- construction and validation of MoralTrace;
- behavioral screening and moral-patient completion;
- causal-source and outcome-valence state replacement;
- low-rank interfaces and attention/MLP component mediation;
- paired-prompt transfer to licensed external evaluation sets;
- released measurements supporting Figures 2–6 and Tables 2–3.

Model weights and third-party evaluation records are excluded. Representative
MoralTrace examples are included under `data/samples/`. The complete dataset is
available from the [MoralTrace Hugging Face Dataset](https://huggingface.co/datasets/happy-holiday/MoralTrace)
and tracks the current dataset release.

## Installation

The recorded environment uses Python 3.12. Create it from the repository root:

```bash
conda env create -f environment.yml
conda activate enact-moral
export PYTHONPATH="$PWD/src"
```

The optional `circuit` dependencies are installed by `environment.yml`. Core
behavioral and state-replacement experiments only require the main dependencies
declared in `pyproject.toml`.

## Validate the release

```bash
python -m data_validation
python -m anonymity
python scripts/reproduce.py
python -m unittest discover -s tests -v
```

`scripts/reproduce.py` validates the numerical identities in the released
tables and writes five summary figures to `outputs/paper_figures/`.

## Download MoralTrace

Download MoralTrace:

```bash
python scripts/download_data.py \
  --dataset-id happy-holiday/MoralTrace \
  --output data/full/moraltrace
```

The downloader checks every configuration against
`data/manifests/moraltrace.json`. The expected total is 51,177 records across
nine configurations. Dataset structure, licensing, and split rules are listed
in `docs/DATA.md`.

## Core experiments

All experiment entry points use explicit model locations or immutable Hub
revisions. Local execution example:

```bash
python experiments/patient_completion.py \
  --data data/samples/participant_completion.jsonl \
  --model-id qwen2_5_7b_instruct \
  --model-source local \
  --model-root /path/to/open-weight-models \
  --layer 28 \
  --max-quartets 1 \
  --output outputs/patient_completion.json
```

The other principal entry points are:

- `experiments/behavioral_screen.py`: behavioral readouts and screening data;
- `experiments/relational_binding.py`: source, valence, and joint counterfactual replacement at their declared clause endpoints;
- `experiments/mediator_knockout.py`: remove downstream attention to the intermediate-event span and compare equal-length random spans;
- `experiments/compact_interface.py`: fit a low-rank interface on train graphs and evaluate it on disjoint validation graphs;
- `experiments/component_mediation.py`: restore selected attention and MLP outputs;
- `experiments/external_validation.py`: apply a frozen interface to paired external prompts.

Every experiment writes model metadata, intervention settings, and record-level
measurements. Model names, local directory names, precisions, and chat-template
settings are declared in `configs/models.yaml`. Frozen seeds, readouts, layers,
positions, ranks, and component counts are declared in
`configs/paper_experiments.yaml`.

## Tested runtime

The release checks were run in the recorded Python 3.12 environment. The
sample patient-completion command was tested with Qwen2.5-0.5B-Instruct on one
NVIDIA RTX 6000D 85GB and completed in under one minute with local weights.
The CPU-only metric validation and five summary figures complete in under five
seconds. Large-checkpoint runtime and memory follow the selected model size;
the scripts preserve the declared bfloat16 precision and expose `--device-map`.

The project fits intervention directions and low-rank bases from open-weight
model activations. It trains no new language-model checkpoint and distributes
no model weights.

## Results and paper mapping

The compact numerical release lives in `results/paper_metrics/`:

- Figure 2: `figure2_patient_completion.csv`;
- Figure 3 and Table 3: `table3_relational_binding.csv`;
- Figure 4: `mediator_knockout.csv`;
- Figure 5: `compact_interface.csv` and `component_mediation.csv`;
- Figure 6: `scale_transition.csv`;
- Table 2: `table2_patient_intervention.csv`;
- external validation and normative criterion controls: `external_validation.csv` and `criterion_conditioning.csv`.

The CSV files retain the precision reported by the manuscript. Full
record-level traces are distributed with the dataset release so confidence
intervals and alternative aggregations can be recomputed.

## Licenses

Source code is released under Apache-2.0. Original MoralTrace records and
metadata are released under CC BY 4.0. Third-party datasets and model weights
remain governed by their original terms. See `LICENSE`, `DATA_LICENSE`, and
`NOTICE`.

Citation metadata remains anonymous during review and is stored in
`CITATION.cff`.
