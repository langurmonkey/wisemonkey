---
name: readme-gen
description: Generates a README.md from the project structure. Use 
  when the user asks to write a README, document the project, or 
  says "generate docs."
---

# README Generator

When asked to generate a README:

1. Scan the project root for:
   - package.json, Cargo.toml, pyproject.toml, go.mod (determine 
     language/framework)
   - Docker files (docker-compose.yml, Dockerfile)
   - CI config (.github/workflows/, .gitlab-ci.yml)
   - Environment files (.env.example)
   - License file

2. Read the main entry point to understand what the project does

3. Generate a README with these sections:

## Project Name
One-paragraph description of what the project does and who it's for.

## Getting Started

### Prerequisites
List runtime requirements found in the project.

### Installation
Step-by-step based on the actual package manager and setup files.

### Running
Based on scripts in package.json, Makefile, or common conventions.

### Environment Variables
If .env.example exists, list each variable with a description.

## Tech Stack
List frameworks and major dependencies found in package files.

## Project Structure
Show the top-level directory structure with one-line descriptions.

## Contributing
Standard contributing section if no CONTRIBUTING.md exists.

## License
Based on the LICENSE file if present.

## Rules
- Only include sections that apply to this project
- All commands must be based on actual project files, not guesses
- If something is unclear, note it rather than making it up
