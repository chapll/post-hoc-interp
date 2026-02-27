---
phase: quick-004
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - paper/milestone.tex
autonomous: true
---

<objective>
Condense milestone.tex from ~6 pages (319 lines) to ~2 pages (~105 lines) while preserving all key content.
</objective>

<tasks>
<task type="auto">
  <name>Task 1: Rewrite milestone.tex to fit ~2 pages</name>
  <files>paper/milestone.tex</files>
  <action>
Aggressively condense all sections:
- Abstract: ~100 words (was ~200)
- Approach: Merge 7 subsections into 5 tight paragraphs, cut implementation details (dual matching, prompt templates, LABEL_MAPPINGS), keep equations and key ideas
- Experiments: Merge data/eval/details into tight paragraphs, cut dataset table, keep 1 combined placeholder figure (was 3), keep results table
- Future Work: 3 short paragraphs (was 4 long ones)
- Remove algorithm/algpseudocode/xcolor packages (unused)
  </action>
</task>
</tasks>
