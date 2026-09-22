# Anonymous release checklist

Before every public update:

1. Run `python -m anonymity` from the repository root.
2. Run the checker again with `--deny` for each author surname, username, institution, and private host fragment.
3. Inspect `CITATION.cff`, repository metadata, commit author settings, issue templates, and generated artifacts.
4. Confirm that `.env`, local paths, model caches, outputs, weights, tensors, and NumPy basis files remain ignored.
5. Confirm that dataset and repository identifiers refer only to anonymous review resources.
6. Confirm that external datasets and model weights are absent.
7. Create the public archive from a clean checkout and rerun all release checks inside that archive.

The scanner excludes `outputs/` because runtime metadata can contain local model
paths. Generated output files must remain outside the anonymous source archive.
