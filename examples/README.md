# Examples

Two ways to walk through the pipeline:

- **`notebook_example.ipynb`** -- a narrated, section-by-section notebook
  running the full pipeline once on the bundled Corel5k + Swin Transformer
  sample (`../data/corel5k/`): UMAP projection, Ball Tree ranking, CPRR
  re-ranking, Borda Count aggregation, post re-ranking, and evaluation.
  Good starting point if you want to see the whole thing end to end with
  explanations alongside the code.
- **The numbered scripts below** -- the same pipeline stages as standalone,
  reusable CLI scripts. Good for scripting a full experiment (e.g. looping
  over datasets/descriptors) or as a reference for the `manifold_rank_fusion`
  API.

All commands below are run from the repository root and operate on the
bundled `data/corel5k/` sample, so they work right after cloning -- see the
main [README](../README.md#bundled-sample-data) for what's included and
the installation steps (`numpy`/`scikit-learn`/`umap-learn` for scripts 01
and 03; pyUDLF, installed from source, additionally for 02, 04, and 05).

## 1. Generate rankings (no pyUDLF needed)

Original-feature ranking and UMAP-projection ranking. `data/corel5k/corel5k_swintf.txt`
and `corel5k_swintf_umap.txt` are the precomputed outputs of these two
commands -- diff against them to check reproducibility.

```bash
python examples/01_generate_ranking.py \
    --features data/corel5k/features_swintf_corel5k.npy \
    --output output/corel5k_swintf.txt

python examples/01_generate_ranking.py \
    --features data/corel5k/features_swintf_corel5k.npy \
    --output output/corel5k_swintf_umap.txt \
    --umap
```

## 2. Rank-based manifold learning re-ranking (requires pyUDLF)

Re-ranks the original-feature ranking with CPRR. `data/corel5k/corel5k_swintf_CPRR.txt`
is the precomputed output of this command -- skip this step if you don't
have pyUDLF installed and just use the bundled file.

```bash
python examples/02_rerank_with_pyudlf.py \
    --method CPRR --dataset corel5k \
    --input data/corel5k/corel5k_swintf.txt \
    --output output/corel5k_swintf_CPRR \
    --size-dataset 5000 \
    --lists-file data/corel5k/corel5k_lists.txt \
    --classes-file data/corel5k/corel5k_classes.txt
```

## 3. Aggregate rankings with Borda Count (no pyUDLF needed)

Combines the UMAP ranking with the CPRR re-ranking. Runs directly on the
bundled precomputed files.

```bash
python examples/03_aggregate_rankings.py \
    --umap-ranking data/corel5k/corel5k_swintf_umap.txt \
    --rerank-ranking data/corel5k/corel5k_swintf_CPRR.txt \
    --output output/borda_corel5k_swintf_CPRR.txt
```

## 4. Post re-rank + evaluate (requires pyUDLF)

Applies an additional CPRR pass to the aggregated ranking from step 3, then
reports MAP / Precision@k / Recall@k.

```bash
python examples/04_post_rerank_and_evaluate.py \
    --input output/borda_corel5k_swintf_CPRR.txt \
    --post-method CPRR --dataset corel5k \
    --size-dataset 5000 \
    --lists-file data/corel5k/corel5k_lists.txt \
    --classes-file data/corel5k/corel5k_classes.txt
```

## 5. Compare Borda Count vs. RRF vs. CombSUM (requires pyUDLF for evaluation)

Aggregates the same UMAP + CPRR rankings with all three fusion strategies
and reports MAP / Precision@k for each.

```bash
python examples/05_compare_aggregation_methods.py \
    --umap-ranking data/corel5k/corel5k_swintf_umap.txt \
    --rerank-ranking data/corel5k/corel5k_swintf_CPRR.txt \
    --size-dataset 5000 \
    --lists-file data/corel5k/corel5k_lists.txt \
    --classes-file data/corel5k/corel5k_classes.txt \
    --output-dir output/comparison_corel5k_swintf_CPRR
```
