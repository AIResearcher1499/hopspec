from hopspec.data.pilot_corpus import build_pilot_corpus
from hopspec.data.reservoir_sampling import reservoir_sample
from hopspec.data.retriever import Document


def make_docs(n):
    return [Document(str(i), f"Title {i}", f"text {i}") for i in range(n)]


# ---- reservoir_sample ----

def test_reservoir_size():
    assert len(reservoir_sample(range(100), 10, seed=0)) == 10


def test_reservoir_smaller_stream_keeps_everything():
    assert reservoir_sample(range(3), 10) == [0, 1, 2]


def test_reservoir_deterministic():
    assert reservoir_sample(range(100), 5, seed=1) == reservoir_sample(range(100), 5, seed=1)


def test_reservoir_seed_changes_sample():
    assert reservoir_sample(range(1000), 5, seed=1) != reservoir_sample(range(1000), 5, seed=2)


def test_reservoir_rejects_negative_k():
    import pytest

    with pytest.raises(ValueError):
        reservoir_sample(range(5), -1)


def test_reservoir_elements_come_from_stream():
    sample = reservoir_sample(range(50), 8, seed=3)
    assert all(0 <= x < 50 for x in sample)
    assert len(set(sample)) == 8


# ---- build_pilot_corpus ----

def test_corpus_keeps_all_gold():
    docs = make_docs(100)
    gold = {"Title 3", "Title 42", "Title 99"}
    corpus = build_pilot_corpus(docs, gold, num_background=10, seed=0)
    titles = {d.title for d in corpus}
    assert gold <= titles


def test_corpus_background_count():
    docs = make_docs(100)
    gold = {"Title 0"}
    corpus = build_pilot_corpus(docs, gold, num_background=10, seed=0)
    assert len(corpus) == 1 + 10


def test_corpus_no_duplicates():
    docs = make_docs(50)
    corpus = build_pilot_corpus(docs, {"Title 1"}, num_background=20, seed=0)
    assert len({d.doc_id for d in corpus}) == len(corpus)


def test_corpus_small_background_pool():
    docs = make_docs(5)
    corpus = build_pilot_corpus(docs, {"Title 0"}, num_background=100, seed=0)
    assert len(corpus) == 5


def test_corpus_deterministic():
    docs = make_docs(200)
    first = build_pilot_corpus(docs, {"Title 5"}, num_background=10, seed=4)
    second = build_pilot_corpus(docs, {"Title 5"}, num_background=10, seed=4)
    assert [d.doc_id for d in first] == [d.doc_id for d in second]
