# docs/

Reference material for working in this repository without re-reading the whole code base.

- [`CODEBASE_OVERVIEW.md`](CODEBASE_OVERVIEW.md) — code base organization, core concepts
  (SpecWrapper, Reward Machines, config system), canonical example configs, known
  issues, and conventions. **Read this first in any new session.**
- [`STLROM_OVERVIEW.md`](STLROM_OVERVIEW.md) — overview of the external `stlrom` submodule (C++ core, Python bindings) used for STL parsing and robustness evaluation.

Update policy: keep `CODEBASE_OVERVIEW.md` current incrementally — when a task changes
structure, APIs, or known issues, update the relevant section in the same change.
