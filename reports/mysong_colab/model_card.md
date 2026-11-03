# MySong large-corpus model card

- trained: 2026-10-27 23:32:15 on `Tesla T4` (cuda)
- project commit: 0ac4975
- trainer: `mysong/neural.py::train_tagger_sequences` (2,014,524 parameters,
  256 hidden units, 4581 training sequences,
  398 validation sequences)

## Data

| corpus | songs | role |
|---|---:|---|
| OpenEWLD (public-domain EWLD subset) | 502 | structural tables + in-domain training; composer-disjoint 80/10/10 split |
| AutoHarmonizer / Wikifonia archive | 4571 | training only, deduplicated by title and transposition-invariant melody signature |

Held-out split sizes: train 409 / dev 49 / test 44.
Melody-signature overlap between the auxiliary corpus and the whole OpenEWLD corpus: 0
(verified in this notebook; title matching alone missed 40 of the 44 test tunes).

## Training

- epochs requested: 60 (chunk size 10), seed 0
- auxiliary scores used: all 4571
- OpenEWLD sequences repeated 1x to keep the in-domain corpus from being drowned
- ensemble: second tagger, seed 1
- early stopping on a slice of the training songs; dev/test never used for gradients or stopping
- best validation root accuracy: 0.516

## Held-out result (test split, 44 songs)

| metric | value |
|---|---:|
| root | 0.604 |
| core | 0.576 |
| full | 0.312 |
| key (exact) | 0.977 |
| chord changes F1 | 0.591 |

Dev split: root 0.683 / core 0.660 / key 0.898.
The paper's generative observation with everything else unchanged scores root 0.249
on the same split, the discriminative tagger is what buys the accuracy.

## Limits

- the auxiliary archive is Wikifonia-derived; its files do not carry a blanket open
  licence for the underlying songs, so check before publishing derived weights;
- one global key per song, symbolic supervision, and Western popular/jazz harmony;
- accuracy against one reference is a lower bound on musical quality, listening
  is the arbiter, and a held-out take has been rendered and listened to above.

## Installed in this checkout (2026-10-27)

The four files above were copied into `data/mysong/model/`. They are also
promoted to the package defaults, which is what the app, `python -m mysong.cli`
and `tests/test_mysong.py` load:

| shipped name | default name |
|---|---|
| `mysong_large.npz` | `mysong.npz` |
| `mysong_large_tagger.npz` | `mysong_tagger.npz` |
| `mysong_large_tagger.pt` | `mysong_tagger.pt` |
| `mysong_large_tagger2.npz` | `mysong_tagger2.npz` |

The previous defaults are kept in `data/mysong/model/mysong_pre_large_backup/`.

Installed file hashes:

- `data/mysong/model/mysong_large.npz` `0e9201a3d7856323fc392d5461aadcc8b69a03b934170d69a6688d099aa97d6a`
- `data/mysong/model/mysong_large_tagger.npz` `454788247d767b4c4f8aebe848d36c309239c87e9ba170832a287c521f7db6db`
- `data/mysong/model/mysong_large_tagger2.npz` `8f23ba9f627adf0c594f6f51f75c77561ad58ef4296a5e70a8404dda4b4b1d77`
- `data/mysong/model/mysong_tagger.npz` `454788247d767b4c4f8aebe848d36c309239c87e9ba170832a287c521f7db6db`
- `data/mysong/model/mysong_tagger2.npz` `8f23ba9f627adf0c594f6f51f75c77561ad58ef4296a5e70a8404dda4b4b1d77`

Two things were corrected after the run, before installation:

* the structural model's `meta.config["tagger"]` still described the previous
  local 192-unit tagger that the base checkpoint was fitted with; it now records
  the attached 256-unit, 2-member ensemble (the notebook's `assemble_model`
  rewrites this block, so a re-run does it by itself);
* the single-tagger and ensemble runs were scored on the test split to check the
  ensemble is real: root 0.592 single vs 0.604 ensemble, core 0.560 vs 0.576,
  key 0.955 vs 0.977.  Both numbers reproduce exactly from the shipped files with
  `python -m mysong.cli evaluate --model data/mysong/model/mysong_large.npz --split test`.

Reports: `reports/mysong_colab/evaluate_dev.json`, `reports/mysong_colab/evaluate_test.json`.
