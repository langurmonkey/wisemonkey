"""Load skills from markdown files.

Each skill is a .md file in the skills/ directory with YAML frontmatter:

    ---
    name: my-skill
    description: What this skill does
    ---

    ## Steps
    ...

The frontmatter is parsed for metadata; the body is injected into the
system prompt.
"""

import re
from pathlib import Path

class SkillLoader:
    """Load and manage skills from markdown files."""

    def __init__(self, skills_dir=None):
        self.skills_dir = Path(skills_dir) if skills_dir else Path(__file__).parent.parent / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = {}


    def list_skills(self):
        """List all available skill files."""
        if not self.skills_dir.exists():
            return []
        return [f.stem for f in self.skills_dir.glob("*.md")]

    def get_skills_str(self):
        """Prints out the current skills"""
        skills = self.list_skills()
        result = ""
        for name in skills:
            skill_path = self.skills_dir / f"{name}.md"
            if skill_path.exists():
                content = skill_path.read_text()
                meta, _ = self._parse_frontmatter(content)
                result += f"⚔ [cyan]{meta['name']}[/]\n"
                result += f"[grey39]{meta['description']}[/]\n"
        return result

    def _parse_frontmatter(self, content):
        """Extract YAML frontmatter and body from skill content."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", content, re.DOTALL)
        if not match:
            return {}, content.strip()

        meta = {}
        for line in match.group(1).strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta, match.group(2).strip()


    def load_skill(self, name):
        """Load a skill by name (without .md extension).

        Returns (metadata_dict, body_str).
        """
        if name in self._loaded:
            return self._loaded[name]

        skill_path = self.skills_dir / f"{name}.md"
        if skill_path.exists():
            content = skill_path.read_text()
            meta, body = self._parse_frontmatter(content)
            self._loaded[name] = (meta, body)
            return meta, body

        return None, None

    def load_all(self):
        """Load all skills and return formatted block for system prompt."""
        skills = self.list_skills()
        if not skills:
            return None

        lines = ["## Available Skills"]
        for skill_name in skills:
            meta, body = self.load_skill(skill_name)
            if body:
                desc = meta.get("description", "")
                lines.append(f"\n### {skill_name} ({desc})\n\n{body}")

        return "\n".join(lines)
