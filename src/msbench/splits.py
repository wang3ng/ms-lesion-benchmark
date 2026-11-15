"""Patient-level, scanner-stratified K-fold splits.

All cases of one subject (e.g. ISBI / MSLesSeg timepoints) land in the same
fold. Subjects are dealt round-robin into folds within each stratum
(scanner) after a seeded shuffle, so with MSSEG training (5 patients per
scanner, 3 scanners) every fold holds exactly one patient per scanner.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def patient_kfold(
    case_ids: Sequence[str],
    subject_ids: Sequence[str],
    strata: Sequence[str],
    n_folds: int,
    seed: int,
) -> list[dict[str, list[str]]]:
    """Return nnU-Net style splits: [{"train": [...], "val": [...]}, ...]."""
    if not (len(case_ids) == len(subject_ids) == len(strata)):
        raise ValueError("case_ids, subject_ids and strata must have equal length")
    subj_cases: dict[str, list[str]] = defaultdict(list)
    subj_stratum: dict[str, str] = {}
    for c, s, st in zip(case_ids, subject_ids, strata):
        subj_cases[s].append(c)
        if subj_stratum.setdefault(s, st) != st:
            raise ValueError(f"subject {s} has cases in more than one stratum")
    if len(subj_cases) < n_folds:
        raise ValueError(f"{len(subj_cases)} subjects < {n_folds} folds")

    rng = np.random.default_rng(seed)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for s in sorted(subj_cases):
        by_stratum[subj_stratum[s]].append(s)

    fold_subjects: list[list[str]] = [[] for _ in range(n_folds)]
    offset = 0  # continue dealing where the previous stratum stopped, to balance fold sizes
    for st in sorted(by_stratum):
        subs = list(by_stratum[st])
        rng.shuffle(subs)
        for i, s in enumerate(subs):
            fold_subjects[(offset + i) % n_folds].append(s)
        offset += len(subs)

    all_cases = sorted(case_ids)
    splits = []
    for k in range(n_folds):
        val = sorted(c for s in fold_subjects[k] for c in subj_cases[s])
        splits.append({"train": [c for c in all_cases if c not in set(val)], "val": val})
    return splits
