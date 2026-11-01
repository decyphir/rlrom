# RLROM — Thesis Implementation Chapter Plan

Working plan for the **implementation chapter** of a PhD thesis that presents
**RLROM** (Reinforcement Learning with Robust Online Monitoring), the library we
developed to test and train RL agents with robust online monitoring of Signal
Temporal Logic formulas. The chapter is illustrated with the (updated) CartPole
configuration files.

> This version supersedes the earlier "tool paper" skeleton. It is restructured as
> a single-column **thesis implementation chapter**, not a 4–5 page conference
> paper.

## Chapter goal

Give a self-contained description of the *design and implementation* of RLROM —
as the concrete artefact that produces the experimental results of the surrounding
chapters — rather than "advertising a tool". The framing emphasizes:

- how design requirements (one formalism for testing + training, config-driven
  experiments, wrapper-based integration, composability, a usable workflow) shape
  the architecture;
- how each feature is implemented and configured, illustrated on CartPole;
- where the library stands relative to the rest of the thesis and how it plugs
  into the overall experimental loop.

## Deliverable structure (files)

```
latex/                              # compile harness (standalone)
├── rlrom_chapter.tex               # main file: preamble + section inputs
├── rlrom.bib                       # bibliography stubs (placeholders)
├── plan.md                         # this plan
└── sections/
    ├── thesis_objectives.tex       # chapter intro + objectives/requirements
    ├── thesis_background.tex       # brief STL / RL / shaping / RM background
    ├── thesis_design.tex           # principles, wrapper pipeline, config system
    ├── thesis_specs.tex            # signals + cartpole.stl
    ├── thesis_testing.tex          # testing against STL specs (cfg0_hug_specs)
    ├── thesis_training.tex         # reward shaping + obs augmentation (cfg0tr_ppo_specs)
    ├── thesis_rewardmachines.tex   # reward machines (cfg0tr_rm_stay_left_right)
    ├── thesis_workflow.tex         # train/select/test/plot, CLI, repeat_training
    ├── thesis_extension.tex        # beyond CartPole, limitations, lessons
    ├── thesis_relatedwork.tex      # positioning within the thesis
    ├── thesis_summary.tex          # chapter summary & outlook
    └── thesis_thesismap.tex        # notes on merging into the full thesis
```

## Current status / what is illustrated

Feature             | Section (file)        | Config file used
--------------------|-----------------------|-------------------------------------
Minimal config      | design (minimal)      | `cfg0_min.yml`
Signal declaration  | specs                 | `obs_names` / `action_names` snippet
STL formula library | specs                 | `cartpole.stl` (updated: `cart_in_A`
                      |                       | `-0.7..-0.5`, `stay_A`, `not_A`)
Testing vs STL      | testing               | `cfg0_hug_specs.yml`
Reward shaping      | training              | `cfg0tr_ppo_specs.yml`
Obs augmentation    | training (obs)        | (TODO snippet, see below)
Reward machines     | rewardmachines        | `cfg0tr_rm_stay_left_right.yml`
Labels              | rewardmachines        | `label_formulas` block snippet
Workflow            | workflow              | `repeat_training`, `select-model`, `plot`

## Figures / tables produced or planned

- **F1 Fig:pipeline** (TikZ, done): env → SpecWrapper → (RM | Flatten) → agent.
- **F2 Fig:rm** (TikZ, done): state graph of the left–stay–right reward machine.
- **T1 Tab:config**: top-level configuration keys. (done)
- **T2 Tab:cli**: command-line surface. (done)
- **F3 (TODO)**: signal traces with robustness bands (from `rlr test` / notebook).
- **F4 (TODO)**: learning-curve envelopes with/without STL shaping (from `rlr plot`).

## Open points / placeholders

- **Chapter number & title** — currently "Chapter N"; renumber and title after the
  thesis outline is fixed.
- **Cross-chapter references** — `chap:prelim` (background) and `chap:xxx`
  (experiment chapters) are placeholders.
- **Bibliography** — entries in `rlrom.bib` are stubs (`stlrom`, `stl`,
  `gymnasium`, `stable`, `rewardmachines`); fill with real data, and in the real
  thesis use the main thesis `.bib`.
- **`obs_formulas` snippet** — a canonical CartPole snippet may be added once a
  config with `obs_formulas` is confirmed.
- **Preamble / class** — the harness uses `article` + geometry/titlesec for
  standalone preview; merge into the thesis preamble when integrating.
- **Front matter** — the centered title block in `rlrom_chapter.tex` is preview-only
  and must be removed in the thesis.

## Compile

```
cd latex
latexmk -pdf rlrom_chapter.tex
```

Produces `rlrom_chapter.pdf` (currently ~15 pages). The 2 remaining warnings are
the intentional placeholder cross-references (`chap:prelim`, `chap:xxx`).

## Implementation order still to do

1. Add experimental figures (F3/F4) generated from `rlr plot` / `rlr test`.
2. Fill bibliography with real references.
3. Resolve chapter number, title, cross-references and preamble integration with the
   main thesis.
4. Optionally tighten length (say, target ~8–12 pages of core content) if needed.
