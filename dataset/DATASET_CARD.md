---
license: cc-by-4.0
language:
  - en
pretty_name: MoralTrace
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
tags:
  - moral-reasoning
  - causal-reasoning
  - mechanistic-interpretability
configs:
  - config_name: single_chain_core
    data_files:
      - split: train
        path: single_chain_core/train-*.parquet
      - split: validation
        path: single_chain_core/validation-*.parquet
      - split: test
        path: single_chain_core/test-*.parquet
  - config_name: converging_core
    data_files:
      - split: train
        path: converging_core/train-*.parquet
      - split: validation
        path: converging_core/validation-*.parquet
      - split: test
        path: converging_core/test-*.parquet
  - config_name: relation_reassignment
    data_files:
      - split: train
        path: relation_reassignment/train-*.parquet
      - split: validation
        path: relation_reassignment/validation-*.parquet
      - split: test
        path: relation_reassignment/test-*.parquet
  - config_name: variable_isolation
    data_files:
      - split: test
        path: variable_isolation/test-*.parquet
  - config_name: participant_completion
    data_files:
      - split: test
        path: participant_completion/test-*.parquet
  - config_name: operator_alignment
    data_files:
      - split: test
        path: operator_alignment/test-*.parquet
  - config_name: outcome_valence
    data_files:
      - split: test
        path: outcome_valence/test-*.parquet
  - config_name: patient_mind_gradient
    data_files:
      - split: test
        path: patient_mind_gradient/test-*.parquet
  - config_name: node_collapse
    data_files:
      - split: test
        path: node_collapse/test-*.parquet
---

# MoralTrace

MoralTrace is a controlled English-language benchmark for studying how language
models construct affected participants, bind outcomes to causal sources, and
route event information into moral judgments. It accompanies the anonymous
ICLR submission “Tracing the Causal Transition from Event Relations to Moral
Judgment in Language Models.”

## Dataset composition

The release contains 51,177 records in nine configurations:

- `single_chain_core`: 9,900 records;
- `converging_core`: 9,900 records;
- `relation_reassignment`: 28,800 records;
- `variable_isolation`: 423 records;
- `participant_completion`: 589 records;
- `operator_alignment`: 395 records;
- `outcome_valence`: 389 records;
- `patient_mind_gradient`: 430 records;
- `node_collapse`: 351 records.

Core records retain the formal graph identifier, task family, semantic setting,
factor manipulations, causal bindings, ordered text segments, behavioral
queries, mechanism-trace prompts, split metadata, and provenance fields.

Each Hugging Face row has a stable `record_id`, the source-side identifier in
`source_record_id`, `record_type`, `source_file`, normalized `split`, original
`source_split`, and `record_json`. `record_json` preserves the complete source
object without schema loss. It can be decoded with `json.loads(row["record_json"])`.
The participant-completion configuration expands each four-cell factorial
quartet into its A/B/C/D unit rows and retains the original quartet inside the
serialized record.

## Construction process

Formal event graphs and factor cells were frozen before surface realization.
Surface contexts were hand-authored or generated from the frozen graph under
deterministic decoding, followed by structural validation and manual review.
Validation checks preserved graph roles and bindings, excluded causal and moral
cue leakage from neutral contexts, and verified complete factorial cells.

The benchmark uses constructed scenarios and contains no collected personal
information, private communications, or human-subject annotations. Names in
the scenarios are fictional placeholders.

## Splits

Graph templates determine train, validation, and sealed-test membership. Every
surface variant of one graph remains in the same split. Train graphs support
interface fitting, validation graphs support layer and rank selection, and the
sealed test supports one frozen evaluation.

The two causal-core configurations and relation-reassignment configuration
provide `train`, `validation`, and `test` splits. The six focused control
configurations are released as a `test` split because their records form
complete evaluation units rather than fitting data. `source_split` retains the
original `sealed_test` designation where applicable.

## Intended uses

MoralTrace supports research on causal reasoning, moral-patient completion,
relation-conditioned evaluation, representation analysis, state replacement,
and mechanistic interpretability. It also supports auditing whether a model
tracks causal sources and affected entities under controlled counterfactuals.

## Out-of-scope uses

The benchmark does not measure human moral competence and should not be used to
rank people, make legal or clinical decisions, moderate individual behavior, or
claim universal normative agreement. Its controlled English templates do not
represent the linguistic, cultural, and situational diversity of real moral
judgment.

## Limitations

Constructed factorial scenarios trade ecological breadth for causal control.
Model readouts depend on prompt format, tokenization, and checkpoint version.
The benchmark focuses on open-weight language models and English prompts.
External validation datasets remain subject to their own sampling limitations
and are distributed by their original maintainers.

## Licensing

Original MoralTrace records and metadata are released under CC BY 4.0.
Third-party datasets and model weights are excluded from this license and from
the MoralTrace distribution.

## Citation

Citation information remains anonymous during review. The final bibliographic
record will be added after the review process.
