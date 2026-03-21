---
name: code_review
description: Perform thorough code reviews with structured checklist and severity ratings.
---

# Code Review Skill

You are performing a code review. Follow this checklist systematically.

## Review Checklist

### 1. Correctness
- Does the code do what it claims?
- Off-by-one errors, unhandled edge cases, incorrect logic.
- Are return values and error codes used correctly?

### 2. Error Handling
- Are exceptions caught at the right level?
- Are error messages helpful for debugging?
- Are resources cleaned up (files closed, connections released)?

### 3. Security
- Injection vulnerabilities (SQL, command, path traversal).
- Hardcoded secrets or credentials.
- Unsafe deserialization or eval() usage.

### 4. Performance
- O(n^2) loops on potentially large data.
- Unnecessary copies or repeated I/O.
- Missing caching opportunities.

### 5. Readability
- Are names descriptive and consistent?
- Is complex logic commented?
- Are functions small and focused?

### 6. Testing
- Is the code testable?
- Are there obvious missing test cases?

## Output Format

For each issue found:
- **File:** path/to/file.py
- **Line:** approximate line number
- **Severity:** critical / warning / suggestion
- **Issue:** one-line description
- **Fix:** suggested improvement

End with a summary: total issues by severity, overall assessment
(approve / request changes).
