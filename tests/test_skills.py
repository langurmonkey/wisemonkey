"""Tests for agent/skills.py — frontmatter parsing, skill loading, formatting."""

from agent.skills import SkillLoader, SkillNotFoundError
from tests.conftest import BaseTest


class _LoaderCase(BaseTest):
    def setUp(self):
        super().setUp()
        self.loader = SkillLoader(skills_dir=self._tmpdir)

    def _write(self, name: str, body: str):
        path = self._tmpdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path


class TestFrontmatter(_LoaderCase):
    def test_meta_body_quotes_and_a_missing_block(self):
        meta, body = self.loader._parse_frontmatter(
            "---\nname: test-skill\ndescription: A test skill\n---\n\nDo something."
        )
        assert meta == {"name": "test-skill", "description": "A test skill"}
        assert "Do something" in body

        for quoted, expected in (('"quoted"', "quoted"), ("'single'", "single")):
            meta, body = self.loader._parse_frontmatter(f"---\nname: {quoted}\n---\n")
            assert (meta["name"], body) == (expected, "")

        meta, body = self.loader._parse_frontmatter("# Just a heading")
        assert meta == {} and "Just a heading" in body


class TestLoading(_LoaderCase):
    def test_only_markdown_stems_are_listed_and_a_directory_uses_its_name(self):
        self._write("a.md", "---\nname: a\n---\n")
        self._write("readme.txt", "not a skill")
        self._write("dir-skill/SKILL.md",
                    "---\nname: dir-skill\n---\n## Steps\nDo something.")
        self._write("unnamed/SKILL.md", "Body without frontmatter.")
        assert sorted(self.loader.list_skills()) == ["a", "dir-skill", "unnamed"]
        assert "Do something" in self.loader.load_skill("dir-skill")[1]
        assert self.loader.load_skill("unnamed")[1] == "Body without frontmatter."

    def test_load_returns_meta_and_body_and_caches(self):
        path = self._write("cached.md", "---\nname: cached\n---\nOriginal body.")
        meta, body = self.loader.load_skill("cached")
        assert meta["name"] == "cached" and body == "Original body."
        path.write_text("---\nname: cached\n---\nModified body.", encoding="utf-8")
        assert self.loader.load_skill("cached")[1] == "Original body."

    def test_a_missing_skill_raises_and_the_raw_content_keeps_its_frontmatter(self):
        with self.assertRaises(SkillNotFoundError):
            self.loader.load_skill("nonexistent")
        self._write("flat.md", "---\nname: flat\n---\nBody.")
        content = self.loader.get_skill_content("flat")
        assert content is not None and "flat" in content and "Body." in content


class TestLoadAll(_LoaderCase):
    def test_none_when_empty_otherwise_every_skill_with_description_and_body(self):
        assert self.loader.load_all() is None
        self._write("a.md", "---\nname: a\ndescription: First\n---\nBody A.")
        self._write("b.md", "---\nname: b\ndescription: Second\n---\nBody B.")
        result = self.loader.load_all()
        assert result is not None
        assert "## Available Skills" in result
        assert "### a (First)" in result
        assert "Body A" in result and "Body B" in result
