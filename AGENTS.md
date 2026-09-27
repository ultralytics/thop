# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

THOP (`ultralytics-thop` on PyPI, imported as `thop`, AGPL-3.0) is a PyTorch model profiler that computes MACs and parameter counts to measure deep learning model complexity. It supports Python>=3.8 and torch>=1.8 (the CI floor; `torch` itself is unpinned).

## Core Principles (CRITICAL)

**Less is more. The simplest solution is the best solution.** The action hierarchy for every change: **Delete > Replace > Add**.

1. **Solve at the owner**: Put behavior in the code path that owns or observes it. For fixes, never guard a symptom with a staleness check, initialization flag, skip-first-call branch, or `try/except` around broken logic; relocate the trigger and delete the wrong path. For features, extend the existing owner rather than creating a parallel abstraction.
2. **Search and reuse first**: Search the whole repository before creating a feature, component, helper, workflow, or utility. Reuse or adapt what exists, consolidate in-scope duplication in the shared owner, and delete duplicate paths. Three similar lines beat a helper nobody else calls.
3. **Delete and modify existing code before creating new code**: Bugfixes are net-negative by default unless deletion and relocation are demonstrably impossible. A new file must first prove it cannot fit cleanly in an existing owner.
4. **Keep scope minimal**: Implement only the simplest complete solution. Avoid impossible-state handling, speculative flags, compatibility shims, policy scaffolding, and unrelated cleanup. Tests are out of scope by default — rely on existing coverage and focused validation; only an uncovered, high-risk regression path justifies minimal new test code.
5. **Ship zero-regression, production-ready changes**: Understand what you remove instead of retaining broken code as insurance. Remove unused imports, functions, types, files, and comments; run relevant cleanup checks; and thoroughly debug and validate the changed owner. Do not break existing features or workflows unless the PR intentionally removes them with evidence.

**Review gate:** for every addition, the reviewer decides whether deleting or changing existing code would have fixed the problem instead — if it would, that is a blocking finding. A missing or thin PR description is never itself a finding.

NEVER push to `main`. NEVER force push. Always start work in a new git worktree (`git worktree add`) on a feature branch and open a PR — never edit the primary checkout directly, it may hold in-flight work.

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Review the full diff in-session against the Core Principles, performance, and the review gate above, then batch the fixes into one commit and push. After each round of bot or human commits, pull and resume the same reviewer on `<last-reviewed-sha>..HEAD` plus anything that delta could have invalidated. Repeat until the local head matches the live head.
3. Hand off or merge only on a clean final pass: one cold full-diff review returning LGTM with no findings, on a head that is still live at merge time.
4. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never reset or revert commits you did not author.
5. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

## Commands

```bash
# Install in editable mode with the test runner (the README examples also need torchvision)
uv pip install -e . pytest

# Run all tests
python -m pytest tests/

# Run one test
python -m pytest tests/test_conv2d.py::TestUtils::test_conv2d_no_bias

# Format/lint — the main Ruff/Prettier steps from ultralytics/actions@main invoked by format.yml (its action.yml is
# the source of truth; CI additionally runs docstring and Markdown code-block formatters, and its auto-format commit
# on the PR covers anything missed locally)
ruff check --fix --unsafe-fixes --extend-select F,I,D,UP,RUF,FA --target-version py38 --ignore BLE001,D100,D104,D203,D205,D212,D213,D401,D406,D407,D413,RUF001,RUF002,RUF012,S110 .
ruff format --line-length 120 .
npx prettier@3.8.5 --write --print-width 120 "**/*.{yml,yaml,json,md}"
```

- `ci.yml` runs `tests/` on every pull request, on pushes to `main`, and nightly, at the `requires-python` floor (Python 3.8 with torch 1.8.0) and at the ceiling (3.14, latest torch), both CPU-only; coverage is not measured. Keep `--target-version py38` so pyupgrade never emits syntax that breaks the floor. The other workflows are `format.yml` (autoformat, AI labels, summaries and review on PRs), `cla.yml` (CLA signing) and `publish.yml` (see Conventions).

## Architecture

`thop/profile.py` holds the `register_hooks` dict mapping `nn.Module` types to counting functions (resolved through the MRO, `custom_ops` first) and the two entry points: `profile()` (hooks every module via `model.apply(add_hooks)` and sums each hooked module's `total_ops` once, plus the functional matrix products a `TorchFunctionMode` counts outside ruled modules on torch>=1.13 — `dfs_count` runs only to build the per-layer tree when `ret_layer_info=True`) and the legacy `profile_origin()` (which does skip non-leaf modules that carry no rule of their own). Counting functions live in `thop/vision/basic_hooks.py` (formulas in `thop/vision/calc_func.py`) and `thop/rnn_hooks.py` for RNN/GRU/LSTM; `thop/utils.py` provides `clever_format`. `benchmark/evaluate_famous_models.py` regenerates the README results table and hard-requires `ultralytics==8.4.106`.

`profile(..., stride=...)` is the path Ultralytics `get_flops` uses: it profiles one-stride-tall proxies of increasing width (starting at `min_cells` strides) and extrapolates to the target's cell count by Newton forward differences — one proxy for spatial-only models, two when a `fixed_ops` layer adds a size-independent cost, three plus an exact fourth-point check for `nn.MultiheadAttention`, `custom_ops` or functional products — and profiles the target directly when proxies are unsuitable, disagree, or `ret_layer_info=True`. A new layer whose cost does not scale with image area must be covered by `fixed_ops`, or the single-proxy path scales it by the cell count.

`profile()` accumulates into a plain `total_ops` int written straight into each module's `__dict__` (`profile_origin()` still uses a float64 `register_buffer`), so a rule adds into `m.total_ops` — a plain number is cheapest, and a one-element tensor works too because the traversal reduces it with `float()`. Parameter counts are not hooked: both entry points read them from `nn.Module.parameters()`, which deduplicates shared weights and covers module types that have no counting rule. With `ret_layer_info=True` each node reports the parameters its own subtree holds, deduplicated within that node but not across nodes.

## Conventions

- Ultralytics-owned PyPI packages use `MAJOR.MINOR.PATCH` versions only; no suffixes.
- Every source file starts with the header `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds it automatically; don't add or revert it manually.
- Ruff at line length 120 with Google-style docstrings; Prettier at print width 120 for YAML/JSON/Markdown — all applied automatically by `format.yml` on PRs.
- Tests are plain pytest, each file wrapping its tests in a `TestUtils` class with exact-value asserts on op counts; there is no conftest or pytest config, and no test hits the network.
- To release, bump `__version__` in `thop/__init__.py` (read dynamically by setuptools) in a PR. On pushes to main by `glenn-jocher`, `publish.yml` compares it against PyPI and, when it is ahead, tags `v<version>`, creates the GitHub release, and publishes to PyPI — so merging a version bump IS the release, and it does not wait for `ci.yml`.
