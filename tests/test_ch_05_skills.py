"""Unit tests for ch_05 skill functions (no LLM calls)."""

from pathlib import Path

from ch_05_skills import (
    _discover_skills,
    _parse_frontmatter,
    get_skill_descriptions,
    load_skill_body,
)

# ---------------------------------------------------------------------------
# _parse_frontmatter()
# ---------------------------------------------------------------------------


def test_parse_frontmatter_valid():
    """Should parse YAML frontmatter and return (meta, body)."""
    text = "---\nname: test\ndescription: A test skill.\n---\n# Body\nContent here."
    meta, body = _parse_frontmatter(text)
    assert meta["name"] == "test"
    assert meta["description"] == "A test skill."
    assert body == "# Body\nContent here."


def test_parse_frontmatter_no_frontmatter():
    """Should return ({}, full_text) when no frontmatter delimiters exist."""
    text = "# Just a body\nNo frontmatter here."
    meta, body = _parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_frontmatter_skips_lines_without_colon():
    """Should silently skip frontmatter lines that lack a colon."""
    text = "---\nname: test\ninvalid line\ndescription: ok\n---\nBody."
    meta, body = _parse_frontmatter(text)
    assert meta == {"name": "test", "description": "ok"}
    assert "invalid line" not in body


# ---------------------------------------------------------------------------
# _discover_skills()
# ---------------------------------------------------------------------------


def test_discover_skills_returns_all():
    """_discover_skills should find all three example skills."""
    skills = _discover_skills()
    names = sorted(skills.keys())
    assert "code_review" in names
    assert "doc_writer" in names
    assert "test_generator" in names


def test_discover_skills_nonexistent_dir():
    """_discover_skills should return {} for a non-existent directory."""
    skills = _discover_skills(Path("/nonexistent/dir"))
    assert skills == {}


# ---------------------------------------------------------------------------
# get_skill_descriptions()
# ---------------------------------------------------------------------------


def test_skill_descriptions_for_system_prompt():
    """get_skill_descriptions should return a compact one-liner per skill."""
    desc = get_skill_descriptions()
    assert "code_review:" in desc
    assert "doc_writer:" in desc
    assert "test_generator:" in desc


def test_skill_descriptions_empty():
    """get_skill_descriptions should return placeholder for empty skills."""
    assert get_skill_descriptions({}) == "(no skills available)"


# ---------------------------------------------------------------------------
# load_skill_body()
# ---------------------------------------------------------------------------


def test_skill_loading_returns_content():
    """load_skill_body should return the SKILL.md body wrapped in <skill> tags."""
    content = load_skill_body("code_review")
    assert "<skill" in content
    assert "</skill>" in content
    assert 'name="code_review"' in content
    assert "Code Review" in content
    assert "Checklist" in content


def test_skill_rejects_path_traversal():
    """load_skill_body should reject names with path traversal sequences."""
    for bad_name in ["../etc/passwd", "foo/bar", "..\\windows", ".."]:
        result = load_skill_body(bad_name)
        assert "Error" in result
        assert "path traversal" in result.lower()


def test_skill_nonexistent_returns_available():
    """Unknown skill name should return error listing available skills."""
    result = load_skill_body("nonexistent_skill_xyz")
    assert "Error" in result
    assert "Available" in result
