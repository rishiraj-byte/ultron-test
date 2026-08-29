# Paper build

Three commands, in order. Nothing in the results section is typed by hand.

```bash
python -m benchmark.run
```

Trains every variant across every task and seed, writes `results/results.json`.
Use `--quick` first (2 seeds, 6 epochs) to check the plumbing before committing
to the full protocol.

```bash
python -m benchmark.tables
```

Turns `results/results.json` into `paper/generated/*.tex` and `damage.pdf`.
`main.tex` `\input{}`s these, so the paper cannot drift from the results after
a rerun.

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs a LaTeX distribution (MiKTeX on Windows) and `IEEEtran.cls`, which ships
with most distributions and is otherwise available from IEEE.

---

## Before submitting

**Every `\todo{}` in `main.tex` is load-bearing.** They mark the sentences that
interpret numbers, and they are deliberately empty because writing them before
seeing results is how papers end up claiming things their own tables contradict.
Search for `\todo` and resolve all of them.

The ones that matter most:

- **Related Work is a stub.** It has the required topics listed but no prose.
  This is the section that positions the contribution, and a reviewer will read
  it first. It cannot be skipped.
- **`refs.bib` is a starting point.** Every entry is a real, well-known work,
  but verify each against the publisher record — and it is missing recent work
  entirely, which a reviewer will notice.
- **Affiliation fields** are placeholders.

## If the modular model loses

Report it. The ablations are informative either way, and the paper is
structured so that a negative multimodal result is still a contribution: it
would show that on this task, at this scale, cortical-style specialisation does
not pay — which is a finding, not a failure. What would *not* be publishable is
quietly dropping the tasks where it lost.

The single-modality controls (CIFAR-100, 20 Newsgroups) are expected to show no
modular advantage. That is the point of including them. Saying so plainly is
what makes the GridNav result credible if it holds.
