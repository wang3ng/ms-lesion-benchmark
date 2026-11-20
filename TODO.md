# TODO

## Cleanup once the experiments are frozen

- [ ] Delete the E8 zoomed-tile embedding cache (MSSEG training cases only, ~101 GB on the NVMe system
      disk of the lab RTX 4090 machine, created 2026-11-19 by `scripts/cache_e8_tiles.py`):

      rm -rf /home/kirby/msbench_e8_cache

      After deleting, E8 can no longer be retrained without re-running `scripts/cache_e8_tiles.py`
      (~30 GPU-min); its trained decoders and predictions in `../medsam_runs/E8` are unaffected.
