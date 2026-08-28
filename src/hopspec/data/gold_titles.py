"""Per-benchmark supporting-fact gold-title extractors."""

from __future__ import annotations


def _titles_from_supporting_facts(supporting_facts) -> set[str]:
    # HF mirrors expose supporting facts either as a dict of parallel lists
    # ({"title": [...], "sent_id": [...]}) or as a list of [title, sent_id].
    if isinstance(supporting_facts, dict):
        return set(supporting_facts["title"])
    return {pair[0] for pair in supporting_facts}


def hotpotqa_gold_titles(example: dict) -> set[str]:
    return _titles_from_supporting_facts(example["supporting_facts"])


def twowiki_gold_titles(example: dict) -> set[str]:
    return _titles_from_supporting_facts(example["supporting_facts"])


def musique_gold_titles(example: dict) -> set[str]:
    return {p["title"] for p in example["paragraphs"] if p.get("is_supporting")}


GOLD_TITLE_EXTRACTORS = {
    "hotpotqa": hotpotqa_gold_titles,
    "2wikimultihopqa": twowiki_gold_titles,
    "musique": musique_gold_titles,
    # Bamboogle has no supporting-facts annotation; it is eval-only.
}
