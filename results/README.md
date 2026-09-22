# Released paper metrics

`paper_metrics/` contains the manuscript-level values needed for rapid result
inspection and summary-figure regeneration. Values retain the displayed paper
precision. CSV files use one row per model, domain, stage, or external-dataset
readout according to the corresponding table.

`python scripts/reproduce.py` checks the algebraic and ordering constraints and
writes Figures 2–6 summaries under `outputs/paper_figures/`.

The metric files contain aggregate model measurements and no model-generated
scenario text. Record-level outputs are produced by the experiment entry points
and remain associated with their MoralTrace `record_id` values. This separation
keeps the source archive compact while preserving the join key needed for
independent aggregation.

External validation rows report `n_min` and `n_max` because the number of valid
Scruples items varies slightly across models. Equal endpoints indicate a fixed
sample count.
