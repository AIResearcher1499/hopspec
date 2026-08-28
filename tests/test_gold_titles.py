from hopspec.data.gold_titles import (
    GOLD_TITLE_EXTRACTORS,
    hotpotqa_gold_titles,
    musique_gold_titles,
    twowiki_gold_titles,
)


def test_hotpotqa_dict_of_lists():
    example = {"supporting_facts": {"title": ["A", "B", "A"], "sent_id": [0, 1, 2]}}
    assert hotpotqa_gold_titles(example) == {"A", "B"}


def test_twowiki_list_of_pairs():
    example = {"supporting_facts": [["A", 0], ["C", 1]]}
    assert twowiki_gold_titles(example) == {"A", "C"}


def test_musique_supporting_paragraphs():
    example = {"paragraphs": [
        {"title": "A", "is_supporting": True},
        {"title": "B", "is_supporting": False},
        {"title": "C", "is_supporting": True},
    ]}
    assert musique_gold_titles(example) == {"A", "C"}


def test_extractor_registry():
    assert set(GOLD_TITLE_EXTRACTORS) == {"hotpotqa", "2wikimultihopqa", "musique"}
    assert "bamboogle" not in GOLD_TITLE_EXTRACTORS  # eval-only, no gold titles
