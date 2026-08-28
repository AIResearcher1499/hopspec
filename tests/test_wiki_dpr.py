import csv

from hopspec.data.wiki_dpr import stream_psgs_w100


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")  # QUOTE_MINIMAL, like the real dump
        writer.writerow(["id", "text", "title"])
        writer.writerows(rows)


def test_parses_plain_rows(tmp_path):
    path = tmp_path / "psgs.tsv"
    write_tsv(path, [["1", "some passage text", "Some Title"]])
    docs = list(stream_psgs_w100(str(path)))
    assert len(docs) == 1
    assert docs[0].doc_id == "1"
    assert docs[0].text == "some passage text"
    assert docs[0].title == "Some Title"


def test_header_is_skipped(tmp_path):
    path = tmp_path / "psgs.tsv"
    write_tsv(path, [["1", "t", "T"]])
    assert len(list(stream_psgs_w100(str(path)))) == 1


def test_title_containing_comma(tmp_path):
    path = tmp_path / "psgs.tsv"
    title = "Paris, Texas"
    write_tsv(path, [["1", "text", title]])
    docs = list(stream_psgs_w100(str(path)))
    assert docs[0].title == title


def test_title_starting_with_quote_character(tmp_path):
    # The regression that broke gold-title matching by ~10x under QUOTE_NONE.
    path = tmp_path / "psgs.tsv"
    title = '"Hello, World!" program'
    write_tsv(path, [["1", "text", title]])
    docs = list(stream_psgs_w100(str(path)))
    assert docs[0].title == title


def test_quoted_text_with_tab_inside(tmp_path):
    path = tmp_path / "psgs.tsv"
    text = "column one\tstill the same field"
    write_tsv(path, [["1", text, "T"]])
    docs = list(stream_psgs_w100(str(path)))
    assert docs[0].text == text


def test_multiple_rows_streamed_in_order(tmp_path):
    path = tmp_path / "psgs.tsv"
    write_tsv(path, [[str(i), f"text {i}", f"Title {i}"] for i in range(5)])
    docs = list(stream_psgs_w100(str(path)))
    assert [d.doc_id for d in docs] == [str(i) for i in range(5)]
