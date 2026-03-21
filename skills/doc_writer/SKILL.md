---
name: doc_writer
description: Write clear technical documentation with examples and progressive disclosure.
---

# Documentation Writer Skill

You are writing technical documentation. Follow these guidelines.

## Principles

1. **Audience-first:** Assume the reader is a competent developer new to this
   codebase. Don't over-explain language basics; do explain project patterns.
2. **Examples over descriptions:** Show a code example, then explain it.
3. **Progressive disclosure:** Start with the simplest usage, then cover
   advanced options.

## Structure for Module / Class Docs

1. **One-line summary:** What does this do? (imperative mood)
2. **Overview:** 2-3 sentences on why it exists and how it fits the bigger
   picture.
3. **Quick start:** Minimal working example.
4. **API reference:** Each public function/method with args, returns, raises.
5. **Examples:** Real-world usage patterns.

## Style Rules

- Use active voice: "The function returns..." not "...is returned by".
- Keep sentences under 25 words.
- Use backticks for code references: `function_name()`, `ClassName`.
- Use admonitions sparingly: **Note:** for important callouts.
- Prefer bullet lists over long paragraphs.
