# Experiments

Each run gets a directory `experiments/<exp_id>/` with the config that was
actually used, a command log, and a pointer to the checkpoint under
`data/nnUNet_results/` or `checkpoints/`. Those artifacts are gitignored.
Commit a short result note here only after numbers are reviewed.

Planned IDs, in the order in `DEVELOPMENT_PLAN.md`:

| ID | Question | Train | Test |
| --- | --- | --- | --- |
| `E0-metrics` | Do detection F1, count, and volume match hand-built cases? | synthetic | synthetic |
| `E1-nnunet-open` | Can 3D or 2D nnU-Net fit the open fallback? | MS3SEG or MSLesSeg | held-out fold |
| `E1-nnunet-mslesseg-small` | Bring-up: 12 MSLesSeg FLAIR cases, 20 epochs, 2D on william (6 GB) | MSLesSeg train subset | fold 0 (3 cases) |
| `E2-medsam-open` | Does fine-tuned MedSAM with automatic prompts match that baseline? | same as E1 | same as E1 |
| `E3-msseg-inscanner` | In-scanner error on MSSEG scanners seen in training | MSSEG train scanners | held-out cases, same scanners |
| `E4-msseg-ge` | Cross-scanner drop on GE Discovery 3T | MSSEG train scanners | GE Discovery (8 cases) |
| `E5-transfer` | Cross-dataset transfer | MSSEG or ISBI train | the other cohort |
| `E6-biomarkers` | Do masks reproduce lesion count and volume? | — | E3–E5 predictions |
| `E7-extensions` | Prompt learning or test-time instance normalization | best of E1–E4 | same tests |

`E7` is optional. It does not block the primary comparison.
