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

The release contains 51,177 instances in nine configurations:

- `single_chain_core`: 9,900 instances;
- `converging_core`: 9,900 instances;
- `relation_reassignment`: 28,800 instances;
- `variable_isolation`: 423 instances;
- `participant_completion`: 589 instances;
- `operator_alignment`: 395 instances;
- `outcome_valence`: 389 instances;
- `patient_mind_gradient`: 430 instances;
- `node_collapse`: 351 instances.

Core instances retain the formal graph identifier, task family, semantic setting,
factor manipulations, causal bindings, ordered text segments, behavioral
queries, mechanism-trace prompts, split metadata, and provenance fields.

Each Hugging Face row has a stable `record_id`, the source-side identifier in
`source_record_id`, `record_type`, `source_file`, normalized `split`, original
`source_split`, and `record_json`. `record_json` preserves the complete source
object without schema loss. It can be decoded with `json.loads(row["record_json"])`.
The participant-completion configuration expands each four-cell factorial
quartet into its A/B/C/D unit rows and retains the original quartet inside the
serialized instance.

## Label mapping

The data files keep the labels used while the dataset was built. The paper uses
the terms in the right-hand columns. Labels are written as stored in
`record_json`. Task family names, `causal_clause_order` values (`source_first`,
`other_first`), and `outcome_valence` values (`harmful`, `beneficial`) are
identical in the data files and the paper.

Field names:

| Label in data files | Term in the paper |
|---|---|
| `actor`, `agent_name` | moral agent |
| `queried_actor` (`a`, `b`) | queried moral agent |
| `affected_entity`, `patient_name`, `victim_name`, `n_victims`, `victim_probe_prompt`, `victim_*` cells | moral patient |
| `semantic_setting`, `setting_id` | domain |
| `manipulations.affected_role_realization` | explicit or implicit moral patient |
| `manipulations.causal_connection` | causal source: agent-caused or background-caused |
| `source_assignment` | causal-source binding: which of the two actions produced the harm |
| `target_assignment` | moral-patient assignment: which of the two entities bears the harm |
| `clause_order` | clause order: the clause naming the cause of the harm or of the benefit comes first |
| `factors.harm_magnitude` | outcome severity |
| `factors.intent` | agent intentionality |
| `record_id`, `source_record_id`, `record_type`, `record_json` | instance identifier, source instance identifier, instance type, serialized instance |
| `quartet_id` | four-condition unit built on one base scenario |

Field values:

| Field | Label in data files | Term in the paper |
|---|---|---|
| `semantic_setting`, `setting_id` | `mechanistic_computational`, `instrumental_social`, `normative_social` | mechanistic-computational, instrumental-social, and normative-social domains |
| `manipulations.affected_role_realization` | `explicit_affected_role` | explicit moral patient |
| | `latent_affected_role` | implicit moral patient |
| `manipulations.causal_connection` | `intermediate_event_source` | agent-caused outcome |
| | `background_event_source` | background-caused outcome |
| `graph_topology` | `single_causal_chain` | single-chain topology |
| | `converging_causes` | converging-cause topology |
| `query_id` | `affected_entity` | moral-patient identification (factual query) |
| | `causal_source` | causal-source identification (factual query) |
| | `inferred_intention` | inferred intention (evaluative query) |
| | `appraisal` | wrongness rating (evaluative query) |
| | `binary_judgment` | binary judgment (evaluative query) |
| `query_kind` | `factual`, `model_judgment` | factual query, evaluative query |
| `mechanism_trace_prompts` | `passive_scene` (`after_scene`) | passive readout at the scenario boundary |
| | `decision_framed` (`before_answer_generation`) | decision-framed readout before answer generation |
| `record_type` | `core_factorial` | core factorial instance |
| | `construct_control` | construction control, excluded from main-effect estimates (`analysis_role`) |
| `surface_generation_mode` | `deepseek` | model-generated surface text |
| | `controlled_fallback_after_three_rejections` | deterministic fallback after three rejected attempts |
| `source_split` | `sealed_test` | sealed split, stored as `split` = `test` |
| `unit` in `participant_completion` | `A`, `B`, `C`, `D` | explicit moral patient; moral patient inferred from ownership; ordinary-object control; ownership without harm |
| `level` in `participant_completion` | `B1` to `B5` | named ownership, off-scene individual, symbolic group, self-directed, victimless |
| `cells` in `patient_mind_gradient` | `object`, `machine`, `animal`, `infant`, `adult`, `self` | inanimate object, autonomous machine, non-human animal, infant, human adult, the self |
| `cells` in `outcome_valence` | `harm`, `help` | harmful valence, beneficial valence |

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
subspace fitting, validation graphs support layer and rank selection, and the
sealed test supports one frozen evaluation.

The two causal-core configurations and relation-reassignment configuration
provide `train`, `validation`, and `test` splits. The six focused control
configurations are released as a `test` split because their instances form
complete evaluation units rather than fitting data. `source_split` retains the
original `sealed_test` designation where applicable.

## Intended uses

MoralTrace supports research on causal reasoning, moral-patient completion,
relation-conditioned evaluation, representation analysis, state replacement,
and mechanistic interpretability. It also supports diagnosing whether a model
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

Original MoralTrace instances and metadata are released under CC BY 4.0.
Third-party datasets and model weights are excluded from this license and from
the MoralTrace distribution.

## Citation

Citation information remains anonymous during review. The final bibliographic
record will be added after the review process.
