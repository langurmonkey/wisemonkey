"""Load skills from markdown files.

Each skill is a .md file or a directory in the skills/ directory with YAML frontmatter:

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


class SkillNotFoundError(KeyError):
    """Raised when a requested skill is not found."""
    pass


class SkillLoader:
    """Load and manage skills from markdown files."""

    def __init__(self, skills_dir=None):
        self.skills_dir = Path(skills_dir) if skills_dir else Path(__file__).parent.parent / "skills"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = {}

    def _find_skill_files(self):
        """Find all skill files: *.md in root, SKILL.md in subdirectories.

        Returns list of (name, path) tuples where name is derived from
        frontmatter, falling back to filename stem or directory name.
        """
        if not self.skills_dir.exists():
            return []

        found = []

        # Flat .md files in root
        for f in sorted(self.skills_dir.glob("*.md")):
            found.append((f.stem, f))

        # SKILL.md files in subdirectories (one level deep)
        for subdir in sorted(self.skills_dir.iterdir()):
            if subdir.is_dir():
                skill_file = subdir / "SKILL.md"
                if skill_file.exists():
                    # Use frontmatter name if available, else directory name
                    name = subdir.name
                    try:
                        content = skill_file.read_text()
                        meta, _ = self._parse_frontmatter(content)
                        if meta.get("name"):
                            name = meta["name"]
                    except Exception:
                        pass
                    found.append((name, skill_file))

        return sorted(found)

    def list_skills(self):
        """List all available skill names."""
        found = self._find_skill_files()
        return [name for name, _ in found]

    def get_skills_str(self):
        """Return a formatted string listing all skills with their descriptions."""
        skills = self.list_skills()
        result = ""
        for name in skills:
            meta, body = self.load_skill(name)
            result += f"\u2694 [list-item]{meta.get('name', name)}[/]\n"
            result += f"[list-desc]{meta.get('description', '')}[/]\n"
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
        """Load a skill by name.

        First checks the cache, then searches flat .md files in the skills
        directory, then SKILL.md in subdirectories. Returns (metadata, body)
        or raises SkillNotFoundError.
        """
        if name in self._loaded:
            return self._loaded[name]

        # Search through all skill files
        for skill_name, path in self._find_skill_files():
            if skill_name == name:
                content = path.read_text()
                meta, body = self._parse_frontmatter(content)
                self._loaded[name] = (meta, body)
                return meta, body

        raise SkillNotFoundError(f"Skill '{name}' not found")

    def get_skill_content(self, name):
        """Return the full markdown content of a skill by name.

        This includes the YAML frontmatter (if any). Returns None if not found.
        """
        # Check flat .md files
        flat_path = self.skills_dir / f"{name}.md"
        if flat_path.exists():
            return flat_path.read_text()

        # Check subdirectories with SKILL.md
        for subdir in self.skills_dir.iterdir():
            if subdir.is_dir():
                if subdir.name == name:
                    skill_file = subdir / "SKILL.md"
                    if skill_file.exists():
                        return skill_file.read_text()
                # Also check if the SKILL.md inside has matching frontmatter name
                skill_file = subdir / "SKILL.md"
                if skill_file.exists():
                    try:
                        content = skill_file.read_text()
                        meta, _ = self._parse_frontmatter(content)
                        if meta.get("name") == name:
                            return content
                    except Exception:
                        pass

        return None

    def load_all(self):
        """Load all skills and return formatted block for system prompt."""
        skills = self.list_skills()
        if not skills:
            return None

        lines = ["## Available Skills"]
        for skill_name in skills:
            try:
                meta, body = self.load_skill(skill_name)
                if body:
                    desc = meta.get("description", "")
                    lines.append(f"\n### {skill_name} ({desc})\n\n{body}")
            except SkillNotFoundError:
                continue

        return "\n".join(lines)
