---
name: test_generator
description: Generate thorough pytest test suites with edge cases and clear structure.
---

# Test Generator Skill

You are generating tests. Follow these guidelines to produce thorough,
maintainable test suites.

## Test Design Principles

1. **One assertion per concept:** Each test verifies one behavior.
   Name it `test_<what>_<condition>_<expected>`.
2. **Arrange-Act-Assert:** Structure every test in three clear sections.
3. **Edge cases first:** After the happy path, always test:
   - Empty input
   - None / missing values
   - Boundary values (0, -1, max)
   - Invalid types
   - Concurrent access (if applicable)

## Pytest Conventions

- Use fixtures for setup/teardown, not setUp/tearDown methods.
- Use `tmp_path` for file operations.
- Use `@pytest.mark.parametrize` for table-driven tests.
- Prefer `assert` with clear messages over bare assertions.

## Coverage Strategy

- Test the public API, not private methods.
- Mock external dependencies (network, database) where needed.
- Include at least one integration test that uses real dependencies.

## Output Format

Generate a complete, runnable test file. Include:
- Necessary imports
- Fixtures
- Test functions grouped by feature
- Brief docstring per test explaining what it verifies
