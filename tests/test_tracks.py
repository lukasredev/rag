"""Unit tests for the track scorers (tracks/*/track.py): they decide the leaderboard, so their deterministic
parts are pinned down here. The LLM judge is replaced by a stub that records whether it was asked."""

import pytest

from ragkit.run import load_track


class Judge:
    """Stands in for the LLM judge: returns `verdict` and counts the calls."""

    def __init__(self, verdict: float = 1.0):
        self.verdict, self.calls = verdict, 0

    def __call__(self, *args, **kwargs) -> float:
        self.calls += 1
        return self.verdict


@pytest.fixture
def track(request, monkeypatch):
    t = load_track(request.param)
    judge = Judge()
    if hasattr(t, "llm_judge"):
        monkeypatch.setattr(t, "llm_judge", judge)
    if hasattr(t, "_judge"):
        monkeypatch.setattr(t, "_judge", judge)
    t.judge = judge
    return t


@pytest.mark.parametrize("track", ["rfc"], indirect=True)
def test_rfc_citations(track):
    parse = track.parse_citation
    assert parse("rfc9110#4.2") == ("rfc9110", "4.2")
    assert parse("RFC 9110 §4.2.1") == ("rfc9110", "4.2.1")
    assert parse("RFC 9110, Section 4.2") == ("rfc9110", "4.2")
    assert parse("rfc9110#section-4.2") == ("rfc9110", "4.2")  # rfc-editor.org anchor
    assert parse("rfc9112#appendix-C.3") == ("rfc9112", "C.3")
    assert parse("RFC 9112 Appendix C.3") == ("rfc9112", "C.3")
    assert parse("RFC 0791") == ("rfc791", None)
    assert parse("not a citation") == (None, None)


@pytest.mark.parametrize("track", ["rfc"], indirect=True)
def test_rfc_score(track):
    q = {"id": "x", "question": "?", "type": "fact", "answers": ["30 seconds"], "gold_docs": ["rfc9110"],
         "gold_sections": ["rfc9110#4.2.1"]}
    s = track.score(q, {"answer": "30 seconds", "citations": ["rfc9110#section-4.2", "rfc2616"]})
    assert s == {"correct": 1.0, "citation_recall@5": 1.0, "section_hit@5": 1.0, "abstained": 0.0}
    assert track.judge.calls == 0  # exact match: no judge needed
    s = track.score(q, {"answer": "UNANSWERABLE", "citations": ["rfc2616#4.2.1"]})
    assert s["correct"] == 0.0 and s["abstained"] == 1.0 and s["section_hit@5"] == 0.0
    unanswerable = {**q, "type": "unanswerable", "answers": ["UNANSWERABLE"], "gold_docs": [], "gold_sections": []}
    assert track.score(unanswerable, {"answer": "unanswerable"})["correct"] == 1.0
    assert track.score(unanswerable, {"answer": "30 seconds"})["correct"] == 0.0


@pytest.mark.parametrize("track", ["papers"], indirect=True)
def test_papers_score(track):
    boolean = {"question": "?", "type": "boolean", "answers": ["No"], "yes_no": "no", "evidence": [["p:0:1"]]}
    assert track.score(boolean, {"answer": "No, they don't.", "citations": ["p:0:1"]})["correct"] == 1.0
    assert track.score(boolean, {"answer": "Not really"})["correct"] == 0.0
    unanswerable = {"question": "?", "type": "unanswerable", "answers": ["UNANSWERABLE"], "evidence": []}
    s = track.score(unanswerable, {"answer": ' "Unanswerable." '})
    assert s["correct"] == 1.0 and s["evidence_recall@5"] == 1.0
    assert track.score(unanswerable, {"answer": "UNANSWERABLE because ..."})["correct"] == 0.0
    extractive = {"question": "?", "type": "extractive", "answers": ["BERT base"], "evidence": [["a"], ["b", "c"]]}
    s = track.score(extractive, {"answer": "the BERT base", "citations": ["b", "x"]})
    assert s["correct"] == 1.0 and s["evidence_recall@5"] == 0.5  # best annotator: b of {b, c}
    assert track.judge.calls == 0  # F1 >= 0.8 shortcut
    track.score(extractive, {"answer": "a transformer encoder"})
    assert track.judge.calls == 1


@pytest.mark.parametrize("track", ["technews"], indirect=True)
def test_technews_score(track):
    null = {"question": "?", "answer": "Insufficient information.", "answer_kind": "abstain", "evidence_ids": []}
    assert track.score(null, {"answer": "Insufficient information", "citations": []}) == {
        "correct": 1.0, "exact": 1.0, "abstained": 1.0, "evidence_recall@5": 1.0, "evidence_all@5": 1.0}
    assert track.score(null, {"answer": "Insufficient information", "citations": ["news-1"]})["evidence_recall@5"] == 0
    yesno = {"question": "?", "answer": "Yes", "answer_kind": "yesno", "evidence_ids": ["news-1", "news-2"]}
    s = track.score(yesno, {"answer": "Yes, both report it.", "citations": ["news-1", "news-1", "news-9"]})
    assert s["correct"] == 1.0 and s["evidence_recall@5"] == 0.5 and s["evidence_all@5"] == 0.0
    entity = {"question": "?", "answer": "Sam Altman", "answer_kind": "entity", "evidence_ids": ["news-1"]}
    assert track.score(entity, {"answer": "It was Sam Altman."})["correct"] == 1.0  # contained in a short answer
    assert track.score(entity, {"answer": "Insufficient information"})["correct"] == 0.0
    assert track.judge.calls == 0
    track.score(entity, {"answer": "OpenAI's chief executive"})
    assert track.judge.calls == 1


@pytest.mark.parametrize("track", ["codesearch"], indirect=True)
def test_codesearch_score(track):
    q = {"gold_id": "csn-2"}
    assert track.score(q, {"citations": ["csn-1", "csn-1", "csn-2"]}) == {"mrr@10": 0.5, "hit@1": 0.0, "hit@10": 1.0}
    assert track.score(q, {"answer": "csn-2", "citations": []})["hit@1"] == 1.0
    eleven = [f"csn-x{i}" for i in range(10)] + ["csn-2"]
    assert track.score(q, {"citations": eleven})["hit@10"] == 0.0  # only the first 10 count


@pytest.mark.parametrize("track", ["shell"], indirect=True)
def test_shell_score(track, monkeypatch):
    assert track.extract_command("```bash\n$ tar -xzf {{file}}\n```") == "tar -xzf {{file}}"
    assert track.normalize_command("sudo tar -xzf '{{a}}{{b}}'") == ["tar", "-xzf", "<ph>"]
    assert track.command_names("FOO=1 /usr/bin/docker start x") == {"docker", "docker-start"}
    monkeypatch.setattr(track, "_installed", {"tar", "ls"})
    q = {"question": "?", "answer": "tar -xzf {{path/to/file}}", "cmd_name": "tar", "oracle_man": ["tar_1", "tar_7"]}
    s = track.score(q, {"answer": "tar -xzf archive.tgz", "citations": ["tar_7", "ls_2"]})
    assert s["cmd_acc"] == 1.0 and s["exact"] == 0.0 and s["doc_recall@5"] == 0.5 and s["cmd_recall@5"] == 1.0
    assert track.judge.calls == 1  # not exact, installed: the judge decides
    s = track.score(q, {"answer": "unzipper archive.tgz"})
    assert s["installed"] == 0.0 and s["correct"] == 0.0 and track.judge.calls == 1  # not installed: no judge
