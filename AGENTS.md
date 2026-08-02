# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

THOP (`ultralytics-thop` on PyPI, imported as `thop`, AGPL-3.0) is a PyTorch model profiler that computes MACs and parameter counts to measure deep learning model complexity. Supported Python versions are 3.8 through 3.14.

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
# Install in editable mode (deps: numpy, torch)
uv pip install -e .

# Test deps (tests need pytest; benchmark scripts also need torchvision)
uv pip install pytest torchvision

# Run all tests
python -m pytest tests/

# Run one test
python -m pytest tests/test_conv2d.py::TestUtils::test_conv2d_no_bias

# Format/lint — the main Ruff/Prettier steps from ultralytics/actions@main invoked by format.yml (its action.yml is
# the source of truth; CI additionally runs docstring and Markdown code-block formatters, and its auto-format commit
# on the PR covers anything missed locally)
ruff check --fix --unsafe-fixes --extend-select F,I,D,UP,RUF,FA --target-version py38 --ignore BLE001,D100,D104,D203,D205,D212,D213,D401,D406,D407,D413,RUF001,RUF002,RUF012,S110 .
ruff format --line-length 120 .
npx prettier --write --print-width 120 "**/*.{yml,yaml,json,md}"
```

- `ci.yml` runs `tests/` on every pull request, on pushes to `main`, and nightly, at the `requires-python` floor (Python 3.8 with torch 1.8.0) and at the ceiling (3.14, latest torch) — run them locally before pushing all the same. Coverage is not measured, and `publish.yml` (PyPI release) is gated on the version bump alone, so a release does not imply a green suite; the other workflows are `format.yml` (autoformat + AI labels/summaries on PRs) and `cla.yml` (CLA signing).
- `requires-python = ">=3.8"` in pyproject.toml; classifiers cover Python 3.8–3.14. The `--target-version py38` above matches CI (ultralytics/actions targets py38 for pyupgrade), which keeps pyupgrade from rewriting code into syntax that breaks Python 3.8.

## Architecture

THOP (PyPI package `ultralytics-thop`, import name `thop`) computes MACs of PyTorch models via forward hooks, and parameter counts from the module tree. `thop/profile.py` holds the `register_hooks` dict mapping `nn.Module` types to counting functions and exposes the two entry points: `profile()` (DFS traversal that counts each leaf module once) and the legacy `profile_origin()`. Counting functions live in `thop/vision/basic_hooks.py` (formulas in `thop/vision/calc_func.py`) and `thop/rnn_hooks.py` for RNN/GRU/LSTM; `thop/utils.py` provides `clever_format`. `benchmark/` scripts regenerate the README results table.

`profile()` accumulates into a plain `total_ops` int written straight into each module's `__dict__` (`profile_origin()` still uses a float64 `register_buffer`), so a rule adds into `m.total_ops` — a plain number is cheapest, and a one-element tensor works too because the traversal reduces it with `float()`. Parameter counts are not hooked: both entry points read them from `nn.Module.parameters()`, which deduplicates shared weights and covers module types that have no counting rule. With `ret_layer_info=True` each node reports the parameters its own subtree holds, deduplicated within that node but not across nodes.

Releases are gated in `publish.yml`: it runs on every push to main but only for actor `glenn-jocher`, and compares `thop.__version__` (in `thop/__init__.py`, read dynamically by setuptools) against PyPI via ultralytics-actions `check_pypi_version`. If the local version is ahead it tags `v<version>`, creates an AI-summarized GitHub release, builds, publishes to PyPI via trusted publishing, uploads an SBOM, and notifies Slack — so merging a version bump to main IS the release trigger.

## Conventions

- Every source file starts with the header `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds it automatically; don't add or revert it manually.
- Ruff formatting at line length 120 with Google-style docstrings (single-line imperative summaries); Prettier at print width 120 for YAML/JSON/Markdown — all applied automatically by `format.yml` on PRs.
- Tests are plain pytest with class-based grouping (each file wraps tests in a `TestUtils` class) and exact-value asserts on op counts; there is no conftest or pytest config, and no test hits the network.
- To release: bump `__version__` in `thop/__init__.py` in a PR; publishing happens automatically on merge to main (see Architecture).
