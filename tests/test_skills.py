"""Tests for agent/skills.py — frontmatter parsing, skill loading, formatting."""

from tests.conftest import BaseTest
from agent.skills import SkillLoader


class TestParseFrontmatter(BaseTest):
    """Test YAML frontmatter parsing from skill markdown files."""

    def setUp(self):
        super().setUp()
        self.loader = SkillLoader(skills_dir=self._tmpdir)

    def test_parses_valid_frontmatter(self):
        content = '---\nname: test-skill\ndescription: A test skill\n---\n\n## Steps\nDo something.'
        meta, body = self.loader._parse_frontmatter(content)
        assert meta["name"] == "test-skill"
        assert meta["description"] == "A test skill"
        assert "Do something" in body

    def test_parses_empty_body(self):
        content = "---\nname: empty\n---\n"
        meta, body = self.loader._parse_frontmatter(content)
        assert meta["name"] == "empty"
        assert body == ""

    def test_no_frontmatter_returns_empty_meta(self):
        content = "# Just a heading\nNo frontmatter here."
        meta, body = self.loader._parse_frontmatter(content)
        assert meta == {}
        assert "Just a heading" in body

    def test_strips_quotes_from_values(self):
        content = '---\nname: "quoted-skill"\ndescription: "A quoted description"\n---\nBody.'
        meta, body = self.loader._parse_frontmatter(content)
        assert meta["name"] == "quoted-skill"
        assert meta["description"] == "A quoted description"

    def test_single_quotes_stripped(self):
        content = "---\nname: 'single-quoted'\n---\nBody."
        meta, body = self.loader._parse_frontmatter(content)
        assert meta["name"] == "single-quoted"


class TestListSkills(BaseTest):
    """Test listing available skill files."""

    def setUp(self):
        super().setUp()
        self.loader = SkillLoader(skills_dir=self._tmpdir)

    def test_empty_directory(self):
        assert self.loader.list_skills() == []

    def test_lists_skill_files(self):
        (self._tmpdir / "skill-a.md").write_text("---\nname: a\n---\n")
        (self._tmpdir / "skill-b.md").write_text("---\nname: b\n---\n")
        skills = self.loader.list_skills()
        assert sorted(skills) == ["skill-a", "skill-b"]

    def test_ignores_non_md_files(self):
        (self._tmpdir / "readme.txt").write_text("not a skill")
        (self._tmpdir / "skill.md").write_text("---\nname: s\n---\n")
        assert self.loader.list_skills() == ["skill"]


class TestLoadSkill(BaseTest):
    """Test loading individual skills."""

    def setUp(self):
        super().setUp()
        self.loader = SkillLoader(skills_dir=self._tmpdir)

    def test_load_existing_skill(self):
        content = "---\nname: my-skill\n---\nSkill body here."
        (self._tmpdir / "my-skill.md").write_text(content, encoding="utf-8")
        meta, body = self.loader.load_skill("my-skill")
        assert meta is not None
        assert meta["name"] == "my-skill"
        assert body == "Skill body here."

    def test_load_missing_skill(self):
        meta, body = self.loader.load_skill("nonexistent")
        assert meta is None
        assert body is None

    def test_caching(self):
        content = "---\nname: cached\n---\nOriginal body."
        (self._tmpdir / "cached.md").write_text(content, encoding="utf-8")
        meta1, body1 = self.loader.load_skill("cached")

        # Modify file on disk
        (self._tmpdir / "cached.md").write_text(
            "---\nname: cached\n---\nModified body.", encoding="utf-8"
        )

        # Should return cached version
        meta2, body2 = self.loader.load_skill("cached")
        assert body2 == "Original body."


class TestLoadAll(BaseTest):
    """Test loading all skills into a formatted block."""

    def setUp(self):
        super().setUp()
        self.loader = SkillLoader(skills_dir=self._tmpdir)

    def test_no_skills_returns_none(self):
        assert self.loader.load_all() is None

    def test_single_skill(self):
        (self._tmpdir / "example.md").write_text(
            "---\nname: example\ndescription: An example\n---\n## Steps\nDo stuff.",
            encoding="utf-8",
        )
        result = self.loader.load_all()
        assert "## Available Skills" in result
        assert "### example (An example)" in result
        assert "Do stuff" in result

    def test_multiple_skills(self):
        (self._tmpdir / "a.md").write_text(
            "---\nname: a\ndescription: First\n---\nBody A.", encoding="utf-8"
        )
        (self._tmpdir / "b.md").write_text(
            "---\nname: b\ndescription: Second\n---\nBody B.", encoding="utf-8"
        )
        result = self.loader.load_all()
        assert "Body A" in result
        assert "Body B" in result
