# STLRom — Library Overview

*How the `stlrom` library looks from rlrom's side. Companion to
[`CODEBASE_OVERVIEW.md`](CODEBASE_OVERVIEW.md) — read that first for the rlrom
side. Describes the 0.4.0 wheel installed in `.venv` (what rlrom actually
runs).*

## What it is

STLRom is a **C++ library with pybind11 Python bindings for robust online
monitoring of Signal Temporal Logic (STL)**. It parses STL specifications,
stores sampled signals, and computes (interval) robustness — loosely following
Deshmukh et al., "Robust online monitoring of signal temporal logic".

- **Version in use: 0.4.0** — rlrom depends on `stlrom >= 0.4.0`, resolved
  from **PyPI** in `uv.lock`, installed in `.venv`. This interface is treated
  as stable and is what rlrom codes against.
- `stlrom/` in this repo is a **separate git checkout** of the same library
  (not a submodule, untracked by rlrom's git), currently on branch `main` at
  the 0.4.0 lineage. Other branches in that checkout (`rc`, `tubes`, ...) hold
  work in progress — a `Tube` robustness-tube API, `get_rob_tube`,
  `plot_tube`, ~15 commits ahead of `main`. **None of it is used by rlrom**;
  do not document or rely on branch-only APIs here.
- `stlrom/` has its own `.venv`, `uv.lock`, CMake build (`build/`).

**The rlrom-critical surface** (keep these stable in mind when upgrading):
`STLDriver.parse_string`, `add_sample`, `get_signals_names`, `get_monitor`,
`get_online_rob`, `get_rob`, `get_rob_signal`, `get_param`/`set_param`, and the
`STLData`/`Signal`/`Sample` accessors listed below. `get_online_rob` is the
central one: every per-step reward/label/observation value in rlrom comes from
it.

## Repository layout (`stlrom/`)

```
stlrom/
├── include/            # C++ headers
│   ├── stl_data.h      #   STLData (the data container — see below)
│   ├── signal.h        #   Point, Sample, Signal (piecewise-linear signal)
│   ├── stl_driver.h    #   STLDriver (parser + data + formula registry)
│   ├── stl_monitor.h   #   STLMonitor (per-formula monitor)
│   ├── transducer.h    #   transducer (compiled formula tree, holds robustness signal z)
│   ├── interval.h, robustness.h, scanner.h, parser.hpp, tools.h, signal_expr.h
├── src/                # C++ implementation + parser (bison/flex: parser.y, scanner.l)
│   ├── pybind11/main.cpp   # ALL Python bindings (single file, ~400 lines)
│   └── stlrom/           # Python glue package: __init__.py, signal.py,
│                         #   stl_driver.py, signal_gen.py
├── test/               # C++ tests (test_driver.cpp, test_stl_data.cpp,
│                       #   test_monitor.cpp, test_until.cpp, test_stress.cpp,
│                       #   test_prog.cpp), Python tests (test_driver.py,
│                       #   test_monitor.py, test_eval_rob.py, test_online_rob_t0.py,
│                       #   test_U.py, test_until.py, test_gt/lt/abs/alw/ev/or/implies.py,
│                       #   test_memory.py, test_SetBigM.py, ...) and many
│                       #   exploratory notebooks (tutorial_notebook1/2.ipynb,
│                       #   test_data_vector.ipynb, test_stl_monitor*.ipynb,
│                       #   test_time_rob.ipynb, ...)
├── docs/stl_grammar.md # full grammar reference for the STL dialect
└── CMakeLists.txt      # scikit-build-core + pybind11 build
```

Build/install (inside `stlrom/`): needs `bison`, `flex`, a C++ toolchain;
`uv venv && uv pip install -e .` (or `pip install -e .`). Tests: `make
test-cpp` / `make test-python` in `build/`.

## The data model: `STLData`

A driver/monitor does not hold a flat sample array; it owns an **`STLData`**,
which holds:

- `data_vector` — `std::vector<Signal>`, one `Signal` per declared signal
  (time is *not* a signal; each `Sample` carries its own `time`);
- `signal_map` — `dict[str, int]` signal name → index.

Python-visible `STLData` API:

| member | meaning |
|---|---|
| `data_vector` | list of `Signal` (read/write) |
| `signal_map` | name → index (read/write) |
| `add_sample([t, v1, ...])` | append one sample to **all** signals; **time is element 0**; size must be `n_signals + 1`; default linear interpolation |
| `add_signal_sample(name, t, v[, d[, interp]])` | append to one signal |
| `get_signal(idx_or_name)` | returns a `Signal` (by copy) |
| `get_signal_idx(name)` / `get_signame_from_idx(idx)` | name ↔ index |

**`STLData` is NOT iterable, has no `len()`, no `__eq__`, and no reset method
bound.** `for s in data`, `len(data)` and `data == []` all fail or silently
misbehave. (The C++ header also declares `reset_signal_data()`, `is_empty()`,
`get_size()`, `set_data_vector()` — none of them are exposed to Python.)

### Critical pybind11 behavior: `driver.data` is a *live handle*

`STLDriver.data` (and `STLMonitor.data`) is a C++ value member, but pybind11's
object registry returns **the same Python object** for the same underlying C++
object. Observed:

```python
d = stlrom.STLDriver(); d.parse_string("signal x, y\nphi := x[t] > 0.5")
d.add_sample([0., 1., 2.])
snap = d.data
d.add_sample([1., 1.5, 2.])
len(snap.get_signal(0).get_samples_list())   # 2 — snap sees new samples!
id(snap) == id(d.data)                       # True
m = d.get_monitor("phi"); id(m.data) == id(d.data)   # True — monitor shares it
```

So `episode['stl_data'] = stl_driver.data` (what `SpecWrapper.reset_monitor`
does) is a **live alias** of the driver's internal data, *not* a snapshot —
this is what makes per-episode replay work in rlrom. It is an undocumented
pybind11 side effect; treat it as fragile. Conversely,
`stl_driver.data = other_stl_data` **deep-copies** `other_stl_data` into the
driver (copy-assignment), and `stl_driver.data = []` raises `TypeError`.

## `Signal`

Piecewise-linear, right-continuous signal = `std::deque<Sample>` where
`Sample = (time, value, derivative)`. (Robustness *tubes* are a separate
`Tube` class available only on the development branches — not in 0.4.0, not
used by rlrom.)

Python API:

- construction: `Signal()` only — the 2-arg `Signal(t, v)` binding is a
  placeholder that **raises**; build with `append_*` instead.
- `append_sample(t, v[, d[, interp]])`, `append_constant_sample(t, v)`
  (derivative 0 → step function), `append_linear_sample(t, v[, d])`.
- `get_samples_list()` → list of `Sample` (each with `.time`, `.value`,
  `.derivative`).
- `value_at(t)`, `resize(t0, tf)`, `empty()`, `begin_time` / `end_time`
  (read/write).
- arithmetic: `+ - * /` between signals; `compute_not/and/or/boolean`,
  `compute_left_time_rob`, `compute_right_time_rob`.
- statics: `set_semantics/get_semantics` (`SPACE` (default), `LEFT_TIME`,
  `RIGHT_TIME`, `BOOLEAN`), `set_BigM/get_BigM`, `set_Eps/get_Eps`.
  **Semantics is global (static) — setting it affects every Signal.**
- `plot(...)` — Python-side (see below).

## `STLDriver`

Parse + store a context of signals/formulas + data; the main entry point.

- `parse_string(s)` / `parse_file(path[, verbose])` — the STL dialect
  (see `docs/stl_grammar.md`): `signal x, y` declarations, `param a=1, b=2`,
  formula definitions `phi := alw_[0,5] (x[t] > 0.5) and ev_[1,2] (y[t] < a)`,
  `until`, `not/and/or/=>`, `abs`, comparisons, `t` for time. Statements end
  with `.` or EOF. (rlrom test note: bare negative literals like `x[t] < -0.2`
  have been a parser limitation.)
- `get_signals_names()` → `"x y ..."` (space-separated, declaration order).
- `add_sample([t, v1, ...])` — forwards to `data` (time first).
- `get_monitor(phi)` → `STLMonitor` for the formula (shares the driver's data).
- **Robustness queries:**
  - `get_rob(phi, t0=0.)` → `float` — offline robustness at `t0` (uses all data).
  - `get_online_rob(phi, t0=0.)` → `list[float]` of **3**:
    `[estimate, lower, upper]` — interval robustness as if monitoring online
    at `t0` with the data currently in the driver.
  - `get_rob_signal(phi[, t0[, tf]])` → `Signal` — offline robustness trace.
  - `get_online_rob_signal(phi[, t0[, tf]])` → `list[Signal]` (estimate,
    lower, upper traces).
  - `get_robustness_map(phi)` / `get_online_robustness_map(phi)` →
    `dict[depth][formula_str] = {'z': Signal}` — robustness of every
    subformula, grouped by depth (for `plot_rob_map`).
- `set_param/get_param`, `set_interval/get_interval_bounds` (intervals can be
  parameters), `set_signals`, `load_csv/write_csv`, `disp()`, `__str__`.
- `data` — the `STLData` (live handle, see above).

## `STLMonitor`

Per-formula monitor: `formula` (transducer), `data` (shared `STLData`),
`start_time`/`end_time`, `up_to_date`, `rob`/`lower_rob`/`upper_rob`,
`param_map`; `add_sample`, `set_eval_time(t0, tf)`, `get_rob_signal`,
`get_online_rob_signal`, `get_robustness_map`, `set_param/get_param`,
`display_formula`. `transducer` exposes `z` (its robustness `Signal`),
`get_formula_string()`, `get_child[L/R]()`.

## Python glue (`stlrom/src/stlrom/`)

`__init__.py` re-exports the C++ module and **monkey-patches** helpers onto the
C++ classes:

- `Signal.plot(label, ax, **kwargs)` — matplotlib (Python side); kwargs
  `draw_samples`, `plot_sat`, `plot_rob`, `legend`, `title`, `draw_canvas`;
  samples interpolated between sample times via `value_at`; `plot_sat=True`
  overlays the Boolean satisfaction on a twin axis.
- `STLDriver` / `STLMonitor` gain: `get_signal_idx`, `get_signal`,
  `get_signame_from_idx`, `plot_signal(sig, ...)`, `plot_signals(names,
  same_ax, ...)`, `plot_rob(phi, t0, tf, ...)`, `plot_sat(phi, t0, tf, ...)`.
- `signal.py`: `plot`, `plot_rob_map(rob_map, max_depth,
  to_plot='robustness'|'lower'|'upper', ax, same_figure, title)`,
  `plot_rob_map_widget` (ipywidgets).
- `signal_gen.py`: `SignalGen` (samples a python function into a `Signal`),
  `OscillSignalGen`, `PWCSignalGen` (piecewise-constant), `PWLSignalGen`
  (piecewise-linear), `RandSignalGen` (random walk, seeded) — all with
  `set_param(**kw)` / `get_signal(t0, tf, ...)`.

## How rlrom uses stlrom (integration points)

All in `src/rlrom/wrappers/spec_wrapper.py` (see `CODEBASE_OVERVIEW.md`):

- one `STLDriver` per `SpecWrapper`, built from `cfg_specs.stl_specs`
  (or `specs`); **recreated on every `reset()`** (no reset API in stlrom).
- per env step: `get_sample()` evaluates the cfg signal expressions and calls
  `stl_driver.add_sample([t, v1, ...], False)` (no interpolation).
- per-step formula evaluation: `stl_driver.get_online_rob(f, t0)` →
  `[estimate, lower, upper]` (used by `eval_formula_cfg`; semantics options
  `rob`/`lower_rob`/`upper_rob`/`bool`, bounds, `past_horizon`).
- `episode['stl_data']` = the driver's `STLData` live handle; `get_ep_len`,
  `get_time`, `get_sig` read it via `get_signal(idx_or_name)`.
- `SpecWrapper.get_rob(formula, horizon, online)`: offline = per-step
  `stl_driver.get_rob(formula, t0)` over the full trace; online = replay of
  the trace into a **fresh `STLDriver`** (one `add_sample` + `get_online_rob`
  per step) so no future data is visible; unknown formulas are rejected up
  front via `get_monitor(f).formula is None` (see gotcha 9).
- plotting: `get_signal_from_str` returns the driver's `stlrom.Signal`
  directly for raw mapped signals; other tokens are re-encoded (constant
  samples for discrete signals, linear for continuous) and uses `Signal.plot`.
- `stlrom.Signal.get_BigM()` seeds the default `BigM` for obs-formula spaces.

## Gotchas checklist (verified at runtime against the installed wheel)

1. `STLData`: no iteration, no `len()`, no `==`, no bound reset — so
   `for s in driver.data`, `len(driver.data)`, `driver.data == []` and
   `monitor.data = []` all fail.
2. `driver.data` / `monitor.data` are live shared handles (same Python
   object); assigning a new `STLData` into `driver.data` deep-copies.
3. `add_sample` expects `[time, v1, ..., vn]` (time first, `n+1` elements).
4. `get_online_rob` returns a 3-element list `[estimate, lower, upper]`.
5. `Signal` semantics/BigM/Eps are **static** (global) settings; default
   semantics is `SPACE`, not Boolean.
6. `Signal(t, v)` constructor raises; use `Signal()` + `append_*`.
7. "Online" robustness at `t0` uses **all data currently in the driver** —
   if you loaded a whole trace, evaluating at an early `t0` can see the
   future. True online replay requires re-adding samples one by one to a
   fresh driver/monitor (what `SpecWrapper.get_rob(online=True)` does).
8. The `stlrom/` checkout may sit on a development branch with extra APIs
   (tubes, `plot_tube`, ...). rlrom only uses the 0.4.0 wheel surface; check
   `git branch --show-current` in `stlrom/` before drawing conclusions from
   that tree.
9. **Unknown formulas do not raise**: `get_monitor(name)` prints a warning
   and returns a monitor with `formula is None`; `get_rob(name, t0)` returns
   `0.0`; `get_online_rob(name, t0)` returns `[]`. Validate with
   `driver.get_monitor(name).formula is None` if you need a hard error.
