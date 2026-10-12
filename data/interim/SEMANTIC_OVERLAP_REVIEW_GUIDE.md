# Semantic Overlap Manual Review Guide

## Purpose

Embedding similarity is used only to identify candidate adaptation-benchmark
pairs for manual inspection. A high similarity score is not itself evidence
of training leakage.

## Labels

### near_duplicate

Use when the adaptation item and benchmark item test essentially the same
specific content or problem, with only superficial changes such as:

- paraphrasing,
- minor wording changes,
- translation,
- trivial entity substitutions,
- the same factual question,
- the same missing-information setup.

### template_overlap

Use when the two items share a task template or reasoning pattern but require
meaningfully different computation, facts, entities, or inference.

Examples include:

- two arithmetic word problems with different operations,
- two syllogisms using the same logical form but different propositions,
- two contextual-word-meaning questions about different words.

Template overlap is not automatically treated as leakage.

### not_overlap

Use when similarity is mainly topical or lexical and the tasks are
substantively different.

## Review policy

Judge the actual information required to solve the items, not the embedding
score alone.

When uncertain between `near_duplicate` and `template_overlap`, prefer
`template_overlap` unless solving one item would substantially reveal the
answer or solution structure of the other.

The review is author-reviewed and may be AI-assisted. It must not be described
as independent human validation.
