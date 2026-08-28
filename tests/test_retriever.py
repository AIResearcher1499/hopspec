from hopspec.data.retriever import Document, InMemoryBM25Retriever

DOCS = [
    Document("1", "Paris", "Paris is the capital of France."),
    Document("2", "Berlin", "Berlin is the capital of Germany."),
    Document("3", "Cheese", "France is famous for cheese and wine."),
]


def test_relevant_document_ranked_first():
    retriever = InMemoryBM25Retriever(DOCS)
    results = retriever.search("capital of Germany")
    assert results[0].doc_id == "2"


def test_k_limits_results():
    retriever = InMemoryBM25Retriever(DOCS)
    assert len(retriever.search("capital France cheese", k=2)) == 2


def test_no_match_returns_empty():
    retriever = InMemoryBM25Retriever(DOCS)
    assert retriever.search("zebra quantum") == []


def test_title_terms_are_searchable():
    retriever = InMemoryBM25Retriever(DOCS)
    results = retriever.search("cheese")
    assert results[0].doc_id == "3"


def test_deterministic_tie_break():
    docs = [Document("b", "T", "same words here"), Document("a", "T", "same words here")]
    retriever = InMemoryBM25Retriever(docs)
    results = retriever.search("same words")
    assert [d.doc_id for d in results] == ["a", "b"]


def test_document_is_frozen():
    import dataclasses

    import pytest

    doc = Document("1", "t", "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.title = "y"
