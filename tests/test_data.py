"""Converters on tiny synthetic release trees (no real data)."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from msbench.data import ConversionError, build_index, convert_case
from msbench.data import isbi2015, msseg2016, mslesseg

SHAPE = (12, 14, 10)
AFF = np.diag([0.5, 0.5, 1.2, 1.0])


def _save(path, arr, affine=AFF):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


def _img(seed):
    return np.random.default_rng(seed).random(SHAPE).astype(np.float32)


def _lesions(n_blobs=2, value=1):
    m = np.zeros(SHAPE, np.uint8)
    for i in range(n_blobs):
        m[2 + 4 * i : 5 + 4 * i, 3:6, 3:6] = value  # 27 voxels, separated blobs
    return m


def _msseg_case(root, release, center, patient, mask_affine=AFF, mask_value=1):
    d = root / "msseg2016" / f"MSSEG-{release}" / release / f"Center_{center}" / f"Patient_{patient}"
    for mod in ("FLAIR", "T1", "T2", "DP", "GADO"):
        _save(d / "Preprocessed_Data" / f"{mod}_preprocessed.nii.gz", _img(len(mod)))
    _save(d / "Masks" / "Consensus.nii.gz", _lesions(value=mask_value), mask_affine)
    _save(d / "Masks" / "Brain_Mask.nii.gz", np.ones(SHAPE, np.uint8))
    for k in range(1, 8):
        _save(d / "Masks" / f"ManualSegmentation_{k}.nii.gz", _lesions(1 + k % 2), mask_affine)


def test_msseg_layout_scanners_and_index(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    _msseg_case(raw, "Training", "01", "01")
    _msseg_case(raw, "Testing", "03", "01")
    specs = msseg2016.discover(raw)
    assert [s.case_id for s in specs] == ["msseg_train_c01_p01", "msseg_test_c03_p01"]
    assert specs[1].scanner == "ge_discovery_3t"
    for s in specs:
        convert_case(s, out, raw)

    case = out / "msseg2016" / "msseg_train_c01_p01"
    names = sorted(p.name for p in case.iterdir())
    assert names == sorted(["flair.nii.gz", "t1.nii.gz", "t2.nii.gz", "lesion.nii.gz",
                            "brain_mask.nii.gz", "meta.json",
                            *[f"lesion_rater{k}.nii.gz" for k in range(1, 8)]])
    assert not any("dp" in n or "gado" in n for n in names)  # FLAIR/T1/T2 only
    les = nib.load(str(case / "lesion.nii.gz"))
    assert les.get_data_dtype() == np.uint8 and np.allclose(les.affine, AFF)

    index, stats = build_index(out)
    row = index.set_index("case_id").loc["msseg_train_c01_p01"]
    assert row.n_lesions == 2 and row.lesion_voxels == 54
    assert row["spacing_mm"] == "0.5x0.5x1.2" and row["shape"] == "12x14x10"
    assert len(stats) == 2 * 8  # consensus + 7 experts per case
    assert (out / "index.csv").exists() and (out / "lesion_stats.csv").exists()


def test_small_affine_drift_is_fixed_and_recorded(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    drift = AFF.copy()
    drift[0, 3] += 0.001
    _msseg_case(raw, "Testing", "07", "07", mask_affine=drift)
    meta = convert_case(msseg2016.discover(raw)[0], out, raw)
    assert any("affine" in f for f in meta["header_fixes"])
    les = nib.load(str(out / "msseg2016" / meta["case_id"] / "lesion.nii.gz"))
    assert np.allclose(les.affine, AFF, atol=1e-6)


def test_mask_copies_flair_qform_and_sform_when_they_differ(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    _msseg_case(raw, "Training", "01", "04")
    flair_path = (raw / "msseg2016/MSSEG-Training/Training/Center_01/Patient_04"
                  / "Preprocessed_Data/FLAIR_preprocessed.nii.gz")
    img = nib.load(str(flair_path))
    q = AFF.copy()
    q[1, 3] += 0.3                      # qform != sform, as in one MSSEG release file
    img.set_qform(q, code=2)
    img.set_sform(AFF, code=1)
    nib.save(img, str(flair_path))
    meta = convert_case(msseg2016.discover(raw)[0], out, raw)
    les = nib.load(str(out / "msseg2016" / meta["case_id"] / "lesion.nii.gz")).header
    fl = nib.load(str(out / "msseg2016" / meta["case_id"] / "flair.nii.gz")).header
    assert np.allclose(les.get_qform(), fl.get_qform(), atol=1e-5) and les["qform_code"] == 2
    assert np.allclose(les.get_sform(), fl.get_sform(), atol=1e-5) and les["sform_code"] == 1


def test_grid_mismatch_fails_case_and_leaves_no_output(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    shifted = AFF.copy()
    shifted[0, 3] += 2.0
    _msseg_case(raw, "Training", "07", "02", mask_affine=shifted)
    with pytest.raises(ConversionError, match="grid differs"):
        convert_case(msseg2016.discover(raw)[0], out, raw)
    assert not (out / "msseg2016").exists() or not any((out / "msseg2016").iterdir())


def test_mask_stored_as_255_is_binarised(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    _msseg_case(raw, "Training", "08", "01", mask_value=255)
    meta = convert_case(msseg2016.discover(raw)[0], out, raw)
    assert meta["stats"]["lesion"]["lesion_voxels"] == 54
    assert any("255" in f for f in meta["header_fixes"])


def test_conversion_is_idempotent(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    _msseg_case(raw, "Training", "01", "03")
    spec = msseg2016.discover(raw)[0]
    convert_case(spec, out, raw)
    meta_path = out / "msseg2016" / spec.case_id / "meta.json"
    mtime = meta_path.stat().st_mtime_ns
    convert_case(spec, out, raw)
    assert meta_path.stat().st_mtime_ns == mtime


def test_isbi_two_raters_and_subject_grouping(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    base = raw / "isbi2015" / "training_final_v4" / "training" / "training01"
    for tp in ("01", "02"):
        for mod in ("flair", "mprage", "t2", "pd"):
            _save(base / "preprocessed" / f"training01_{tp}_{mod}_pp.nii", _img(len(mod)))
        _save(base / "masks" / f"training01_{tp}_mask1.nii", _lesions(1).astype(np.float32))
        _save(base / "masks" / f"training01_{tp}_mask2.nii", _lesions(2).astype(np.float32))
    specs = isbi2015.discover(raw)
    assert [s.case_id for s in specs] == ["isbi_training01_01", "isbi_training01_02"]
    assert {s.subject_id for s in specs} == {"isbi_training01"}
    meta = convert_case(specs[0], out, raw)
    assert meta["references"] == ["lesion_rater1", "lesion_rater2"]
    assert meta["stats"]["lesion_rater2"]["n_lesions"] == 2
    case = out / "isbi2015" / "isbi_training01_01"
    assert not (case / "lesion.nii.gz").exists()  # no consensus in ISBI
    assert (case / "t1.nii.gz").exists()          # MPRAGE -> t1


def test_mslesseg_split_timepoints_and_scanner(tmp_path):
    raw = tmp_path / "raw"
    root = raw / "mslesseg" / mslesseg.RELEASE_DIR
    for stem in ("train/P1/T1/P1_T1", "train/P1/T2/P1_T2", "test/P54/P54"):
        for mod in ("FLAIR", "T1", "T2"):
            _save(root / f"{stem}_{mod}.nii.gz", _img(len(mod)))
        _save(root / f"{stem}_MASK.nii.gz", _lesions(1))
    csv = root / "info_dataset" / "patient_scanners_info" / "patient_scanners_info_FLAIR.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("Patient;Timepoint;MANUFACTURER;SCANNER\n"
                   "P1;T1;Philips Medical Systems;Ingenia\nP1;T2;NONE;NONE\nP54;T1;GE MEDICAL SYSTEMS;Signa HDxt\n")
    specs = {s.case_id: s for s in mslesseg.discover(raw)}
    assert set(specs) == {"mslesseg_P1_T1", "mslesseg_P1_T2", "mslesseg_P54_T1"}
    assert specs["mslesseg_P1_T1"].scanner == "philips_ingenia"
    assert specs["mslesseg_P1_T2"].scanner == "unknown"
    assert specs["mslesseg_P54_T1"].release_split == "test"
    assert specs["mslesseg_P1_T1"].subject_id == specs["mslesseg_P1_T2"].subject_id
    meta = convert_case(specs["mslesseg_P1_T1"], tmp_path / "processed", raw)
    assert json.loads(json.dumps(meta))["primary_reference"] == "lesion"
