> ⚠ **E2c (and therefore the E8 decision below) is not valid yet: its training collapsed.** In all five
> folds the E2c training loss plateaus at ≈ 0.49 from step 1,000 to 10,000 (E2a falls steadily to ≈ 0.16).
> 0.5 is exactly the loss of a model that outputs an empty mask for every box: half of each batch are
> empty tiles, where per-box soft Dice is 0 only for an exactly empty output, and the lesion boxes then
> cost ≈ 1. Predictions match this: folds 1–2 output almost nothing for all six patients, fold 0
> sits just above threshold (145–224 FP per patient). The cause is the loss we chose (MedSAM's per-box
> Dice + BCE, designed for boxes that always contain the target) combined with 50 % empty boxes; it is a
> defect of our setup, not a property of MedSAM. E2a has no empty boxes and is unaffected. The E2c and
> E8 numbers are kept here unchanged for the record; see `experiments/E2/README.md`.
