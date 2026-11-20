import pytest

from msbench.config import load_config, paths

cfg = load_config()
pytestmark = pytest.mark.skipif(not (paths(cfg)["data_processed"] / "index.csv").exists(), reason="no data index")


def test_combined_training_set():
    from msbench.e9 import excluded_cases, medsam_steps, training_cases
    tr = training_cases(cfg)
    assert len(tr) == 99 and tr.subject_id.nunique() == 64
    assert set(tr.dataset) == {"msseg2016", "mslesseg"} and (tr.dataset == "msseg2016").sum() == 15
    assert not tr.scanner.str.startswith("ge_").any() and "unknown" not in set(tr.scanner)
    ex = excluded_cases(cfg)
    assert len(ex) == 9 and set(ex.scanner) <= {"unknown", "ge_signa_hdxt", "ge_signa_explorer", "ge_optima_mr450w"}
    assert medsam_steps(cfg, 12) == 10000 and medsam_steps(cfg, 99) == 82500


def test_hardnet_split(tmp_path):
    import json
    from msbench.e9 import hardnet_split
    d = json.loads(hardnet_split(cfg, tmp_path / "s.json").read_text())
    assert len(d["fold0"]["train"]) == 99 and d["fold0"]["val"] == []
    assert set(d["fold0"]["train"][0]) == {"flair", "t1", "t2", "mask", "case_id"}
