**What happened.** The first E2c run (2026-11-17) used MedSAM's loss as is: per-box soft Dice + BCE on
*every* box, with half of each training batch lesion-free tiles. Training loss plateaued at ≈ 0.49 in all
five folds and the validation predictions were unusable (folds 1–2 almost empty, fold 0 145–224 false
positives per patient). It was replaced by the corrected run above; its numbers are kept here for the
record. Per-box Dice of an empty target is ≈ 0 only for an exactly empty output, so the empty tiles push
every box towards empty output.

**Diagnosis, corrected after the rerun was set up** (training patients only, box diagnostics table below):
the collapse was in the **tiles**, not in every box. With fold 1's final checkpoint only ≈ 5 % of
lesion-containing tiles gave any output, while the whole-brain box still marked most lesion voxels and
over-segmented heavily. At inference each pixel takes the mean probability over the brain box and the 1–4
tiles covering it, so empty tiles outvoted the brain box (fold 1: almost nothing predicted), or the brain
box flooded through where the tiles were weak (fold 0). The earlier reading that the ≈ 0.5 loss plateau is
"an empty output for every box" was wrong: it is the average of a well-fitted empty-tile term and a poor
lesion-box term. A 3,000-step replay of the first-run loss on fold 1 (`--dice-on all_boxes`) still had
58 % of lesion boxes non-empty at step 3,000, so the collapse built up later. That is why the collapse
check runs from step 3,000 to the end, for the brain box and the tiles separately.

**Correction (post hoc, training folds only, no test data):** Dice only on boxes whose target contains lesion
voxels, BCE on every box (`finetune.dice_on: lesion_boxes`), the same for E2a, E2b, E2c and E8. E2a's loss is
identical under both settings (every E2a box contains lesion voxels), so it was not rerun.
