from collections import Counter

import pytest

from msbench.splits import patient_kfold


def _msseg_like():
    cases, subjects, scanners = [], [], []
    for sc in ("siemens_verio_3t", "siemens_aera_1p5t", "philips_ingenia_3t"):
        for p in range(5):
            c = f"{sc}_p{p}"
            cases.append(c), subjects.append(c), scanners.append(sc)
    return cases, subjects, scanners


def test_one_patient_per_scanner_per_fold_and_full_coverage():
    cases, subjects, scanners = _msseg_like()
    splits = patient_kfold(cases, subjects, scanners, 5, seed=20261011)
    sc = dict(zip(cases, scanners))
    vals = [c for s in splits for c in s["val"]]
    assert sorted(vals) == sorted(cases)          # every case validated exactly once
    for s in splits:
        assert Counter(sc[c] for c in s["val"]) == Counter({k: 1 for k in set(scanners)})
        assert not set(s["train"]) & set(s["val"])
        assert len(s["train"]) + len(s["val"]) == len(cases)


def test_deterministic_for_seed_and_changes_with_seed():
    args = _msseg_like()
    assert patient_kfold(*args, 5, seed=20261011) == patient_kfold(*args, 5, seed=20261011)
    assert patient_kfold(*args, 5, seed=20261011) != patient_kfold(*args, 5, seed=1)


def test_timepoints_of_one_subject_stay_together():
    cases = [f"s{i}_t{t}" for i in range(6) for t in range(3)]
    subjects = [c.split("_")[0] for c in cases]
    splits = patient_kfold(cases, subjects, ["x"] * len(cases), 3, seed=0)
    for s in splits:
        assert not {c.split("_")[0] for c in s["val"]} & {c.split("_")[0] for c in s["train"]}


def test_subject_in_two_strata_is_rejected():
    with pytest.raises(ValueError, match="more than one stratum"):
        patient_kfold(["a", "b", "c"], ["s", "s", "t"], ["x", "y", "x"], 2, seed=0)
