# Wisemonkey — justfile
# Usage: just <recipe> [args...]

# Run the agent (pass arguments through, e.g. just run --onboard)
run *args:
    uv run wisemonkey {{args}}

# Run all tests
test:
    uv run python -m unittest discover -s tests -v

# Run a specific test file (e.g. just test-file test_core)
test-file name:
    uv run python -m unittest tests.{{name}} -v

# Run a single test (e.g. just test-one test_core.TestFindWorkspaceRoot.test_finds_agents_md_in_parent)
test-one path:
    uv run python -m unittest {{path}} -v

# Install dependencies
install:
    uv sync

# Build the package
build:
    uv build

# Run the interactive onboarding configuration
onboard:
    uv run wisemonkey --onboard

# Run from source (alias for `run`)
dev *args:
    uv run wisemonkey {{args}}
