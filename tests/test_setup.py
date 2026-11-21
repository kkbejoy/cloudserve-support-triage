"""One-command setup after a clean checkout (B-19, A1, NFR-09). Offline with the fake embedder."""
import json

import pytest

import scripts.setup as setup
from tests.fake_embedder import FakeEmbedder
from tests.test_classify import _tickets
from tests.test_pipeline_log import DOCS


def test_T19a_builds_index_and_classifier_once_then_skips(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(setup, "REPORTS_DIR", str(tmp_path / "reports"))
    emb = FakeEmbedder()
    chroma, model = str(tmp_path / "store" / "chroma"), str(tmp_path / "store" / "clf.joblib")
    tickets = tmp_path / "t.json"
    tickets.write_text(json.dumps(_tickets(n_per_class=12)), encoding="utf-8")

    assert setup.ensure_index(chroma, str(DOCS), "section", emb) == "built"
    assert setup.ensure_index(chroma, str(DOCS), "section", emb) == "skipped"         # unchanged docs
    assert setup.ensure_index(chroma, str(DOCS), "section", emb, force=True) == "built"

    assert setup.ensure_classifier(model, str(tickets), emb, folds=3) == "built"
    assert setup.ensure_classifier(model, str(tickets), emb, folds=3) == "skipped"
    assert list((tmp_path / "reports").glob("classifier_comparison_*.md"))          # report kept out of git

    setup.self_test(emb, chroma_path=chroma, docs_path=str(DOCS), model_path=model, strategy="section",
                    threshold=0.0, text="locked out password login")
    assert "self-test" in capsys.readouterr().out


def test_T19b_changed_documentation_triggers_a_rebuild(tmp_path):
    emb = FakeEmbedder()
    docs = tmp_path / "docs.json"
    docs.write_text(DOCS.read_text(encoding="utf-8"), encoding="utf-8")
    chroma = str(tmp_path / "chroma")
    setup.ensure_index(chroma, str(docs), "section", emb)
    data = json.loads(docs.read_text(encoding="utf-8"))
    (data if isinstance(data, list) else data["documents"])[0]["content"] += "\n\nA new note."
    docs.write_text(json.dumps(data), encoding="utf-8")
    assert setup.ensure_index(chroma, str(docs), "section", emb) == "built"


def test_T19c_missing_env_is_created_from_example_and_reported(tmp_path, capsys):
    example = tmp_path / ".env.example"
    example.write_text("GROQ_API_KEY=your_key_here\n", encoding="utf-8")
    assert setup.check_env(tmp_path / ".env", example) is False
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "GROQ_API_KEY=your_key_here\n"
    assert "created" in capsys.readouterr().out


def test_T19d_problems_stop_setup_with_a_clear_message(tmp_path):
    with pytest.raises(setup.SetupError, match="3.10 or newer"):
        setup.check_python((3, 9, 1))
    with pytest.raises(setup.SetupError, match="pip install -r requirements.txt"):
        setup.check_packages(["a-package-that-does-not-exist-xyz"])
    with pytest.raises(setup.SetupError, match="clone completed"):
        setup.check_data([str(tmp_path / "missing.json")])
    with pytest.raises(setup.SetupError, match="both missing"):
        setup.check_env(tmp_path / ".env", tmp_path / ".env.example")
    assert setup.classifier_is_usable(str(tmp_path / "nothing.joblib")) is False


def test_T19e_main_returns_1_on_a_setup_problem(monkeypatch, capsys):
    def boom(**kw):
        raise setup.SetupError("Packages not installed: torch")
    monkeypatch.setattr(setup, "run", boom)
    assert setup.main([]) == 1 and "SETUP STOPPED" in capsys.readouterr().err
