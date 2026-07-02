---
name: code-review
description: Reviews code for bugs, security issues, and best 
  practices. Use when the user asks for a code review, mentions 
  reviewing changes, or says "check this code."
---

# Code Review

When asked to review code:

1. Identify which files changed (check git diff or ask the user)
2. Read each changed file completely

## Check for these issues

### Security (Critical)
- SQL injection via string concatenation
- XSS from unescaped user input
- Authentication or authorization bypasses
- Hardcoded secrets or API keys
- Insecure deserialization

### Logic (Critical)
- Off-by-one errors
- Null/undefined access without checks
- Race conditions in async code
- Unhandled promise rejections
- Incorrect boolean logic

### Performance (Warning)
- N+1 database queries
- Missing database indexes for query patterns
- Unnecessary re-renders in React components
- Large objects in memory that could be streamed

### Style (Suggestion)
- Inconsistent naming conventions
- Functions longer than 50 lines
- Deeply nested conditionals (3+ levels)
- Dead code or unused imports

## Output format

Group findings by severity: Critical, Warning, Suggestion.
For each finding:
- File and line number
- What the issue is
- Why it matters
- A concrete fix (show code)

If no issues found, say so explicitly.
