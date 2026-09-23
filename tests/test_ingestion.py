from ingest_cli import process_markdown_or_text_file
from my_agent.utils.tools import encode_image_data_uri


def test_markdown_parsing_returns_named_page_records(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("First page\n---\nSecond page", encoding="utf-8")

    records = process_markdown_or_text_file(str(source), "guide")

    assert [record.page_num for record in records] == [1, 2]
    assert [record.native_text for record in records] == ["First page", "Second page"]
    assert all(record.image_rel_url == "" for record in records)
    assert all(record.image_disk_path == "" for record in records)
    assert all(record.has_visuals is False for record in records)


def test_encode_image_data_uri_uses_detected_mime_type(tmp_path):
    image = tmp_path / "diagram.png"
    image.write_bytes(b"png-data")

    data_uri = encode_image_data_uri(str(image))

    assert data_uri == "data:image/png;base64,cG5nLWRhdGE="
