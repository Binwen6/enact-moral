# MoralTrace data release

MoralTrace contains 51,177 instances in nine configurations. The authoritative
counts are stored in `data/manifests/moraltrace.json` and checked by
`python -m data_validation`.

## Configurations

- `single_chain_core`: 9,600 factorial instances and 300 construct controls;
- `converging_core`: 9,600 factorial instances and 300 construct controls;
- `relation_reassignment`: 28,800 two-agent routing instances;
- `variable_isolation`: 423 variable-specific controls;
- `participant_completion`: 400 factorial instances and 189 extensions;
- `operator_alignment`: 395 operator and checkpoint comparisons;
- `outcome_valence`: 389 outcome-conditioned instances;
- `patient_mind_gradient`: 430 patient-mind controls;
- `node_collapse`: 351 intermediate-event controls.

Core factorial instances retain graph identifiers, semantic settings, declared
manipulations, causal bindings, ordered text segments, behavioral queries,
mechanism-trace prompts, split metadata, and construction provenance.

## Split discipline

Graph templates determine the split. All surface variants of one graph stay in
the same split. Subspace fitting uses train graphs, layer and rank selection
uses validation graphs, and the sealed test split is read once for the frozen
analysis. Downloaded configurations preserve their named splits. Each
configuration contains the Arrow representation and a `jsonl/` directory.
Experiment entry points consume one named JSONL split directly.

## Representative samples

`data/samples/samples.jsonl` contains one real representative row from every
configuration. `data/samples/participant_completion.jsonl` contains a complete
validated quartet for running the patient-completion entry point.
`data/samples/formal_graphs.jsonl` contains a reviewed frozen graph with all
three semantic settings for exercising the surface-generation pipeline.
`data/samples/external_pairs.jsonl` contains one paired-prompt instance for
checking the frozen-subspace transfer entry point.
`data/samples/node_collapse.jsonl` contains one complete direct, controllable,
and uncontrollable mediator unit for the attention-knockout entry point.

## Generation pipeline

`dataset/generate/generate_contexts.py` creates surface contexts after formal
graphs have been frozen. It reads `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and
`DATAGEN_MODEL` from the process environment. Generated contexts are rejected
when graph identifiers change, settings are missing, length constraints fail,
or causal and moral cue terms appear.
`.env.example` documents the required variable names and contains no values.

## Dataset access

The complete release is available from the
[MoralTrace Hugging Face Dataset](https://huggingface.co/datasets/happy-holiday/MoralTrace).
The download command in `README.md` checks every configuration against the
local manifest before writing the Arrow and JSONL representations.

## External evaluation data

MoCa, AITA, Scruples, ETHICS-UniMoral, and other third-party corpora remain
under their source licenses and access rules. Their instances are excluded from
this distribution. `experiments/external_validation.py` accepts locally
prepared JSONL rows with three required fields:

```json
{"record_id":"dataset/split/item","target_prompt":"...","donor_prompt":"..."}
```

Target and donor prompts must form the declared counterfactual pair while
retaining the same readout format. The script records target, donor,
low-rank-interface, and full-state scores for every pair.

## License boundary

Original MoralTrace content is distributed under CC BY 4.0. The data license
does not cover third-party instances, model outputs whose upstream terms impose
additional restrictions, or model weights.
