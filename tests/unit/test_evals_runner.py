"""Unit tests for evals.runner (discover_pairs)."""

from pathlib import Path

from evals.runner import discover_pairs


def test_discover_pairs_finds_matching_pairs(evals_fixture_dir: Path) -> None:
    """discover_pairs returns one pair per image stem that has a matching JSON; sorted by sample_id."""
    pairs = discover_pairs(
        images_dir=evals_fixture_dir / "images",
        json_dir=evals_fixture_dir / "json-files",
    )
    assert len(pairs) == 2
    sample_ids = [s.sample_id for s in pairs]
    assert sample_ids == ["meal_a", "meal_b"]
    for s in pairs:
        assert s.image_path.exists()
        assert s.json_path.exists()
        assert s.image_path.stem == s.json_path.stem


def test_discover_pairs_dedupes_by_stem(evals_fixture_dir: Path) -> None:
    """When the same stem appears as both .jpeg and .jpg, only one pair is returned."""
    pairs = discover_pairs(
        images_dir=evals_fixture_dir / "images",
        json_dir=evals_fixture_dir / "json-files",
    )
    meal_a_pairs = [p for p in pairs if p.sample_id == "meal_a"]
    assert len(meal_a_pairs) == 1


def test_discover_pairs_ignores_image_without_json(tmp_path: Path) -> None:
    """Images with no matching JSON file are not included."""
    (tmp_path / "images").mkdir()
    (tmp_path / "json-files").mkdir()
    (tmp_path / "images" / "no_pair.jpeg").write_bytes(b"\xff\xd8\xff")
    pairs = discover_pairs(
        images_dir=tmp_path / "images",
        json_dir=tmp_path / "json-files",
    )
    assert len(pairs) == 0


def test_discover_pairs_sorted_by_sample_id(evals_fixture_dir: Path) -> None:
    """Results are sorted by sample_id for stable order."""
    pairs = discover_pairs(
        images_dir=evals_fixture_dir / "images",
        json_dir=evals_fixture_dir / "json-files",
    )
    assert pairs == sorted(pairs, key=lambda s: s.sample_id)
