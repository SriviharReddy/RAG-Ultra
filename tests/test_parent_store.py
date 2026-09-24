from pathlib import Path

from core.database import ParentStore


def test_parent_store_put_and_get(tmp_path: Path):
    store = ParentStore(str(tmp_path))
    key = store.put("doc_1", 1, "Page 1 content")
    assert key == "doc_1::1"
    assert store.get(key) == "Page 1 content"
    assert store.get("nonexistent") == ""


def test_parent_store_persistence(tmp_path: Path):
    store1 = ParentStore(str(tmp_path))
    key1 = store1.put("doc_a", 1, "Content A")
    key2 = store1.put("doc_a", 2, "Content B")
    store1.flush()

    # Re-instantiate from same directory
    store2 = ParentStore(str(tmp_path))
    assert store2.get(key1) == "Content A"
    assert store2.get(key2) == "Content B"


def test_parent_store_clear_doc(tmp_path: Path):
    store = ParentStore(str(tmp_path))
    k1 = store.put("doc_1", 1, "Doc 1 P1")
    k2 = store.put("doc_1", 2, "Doc 1 P2")
    k3 = store.put("doc_2", 1, "Doc 2 P1")

    store.clear_doc("doc_1")
    assert store.get(k1) == ""
    assert store.get(k2) == ""
    assert store.get(k3) == "Doc 2 P1"


def test_parent_store_clear_all(tmp_path: Path):
    store = ParentStore(str(tmp_path))
    k = store.put("doc_x", 1, "Text")
    store.flush()
    assert (tmp_path / "parent_store.json").exists()

    store.clear()
    assert store.get(k) == ""
    assert not (tmp_path / "parent_store.json").exists()


def test_parent_store_corrupted_file_handling(tmp_path: Path):
    bad_file = tmp_path / "parent_store.json"
    bad_file.write_text("NOT_JSON{", encoding="utf-8")

    # Should not crash on invalid JSON
    store = ParentStore(str(tmp_path))
    assert store.get("any_key") == ""
    # Should still allow writing new entries
    k = store.put("doc_new", 1, "Fresh")
    assert store.get(k) == "Fresh"
