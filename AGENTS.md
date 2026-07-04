# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, etc.) when working with code in this repository. CLAUDE.md is a symlink to this file.

## Core Principles (CRITICAL)

Respecting these principles is critical for every PR.

**Less is more. The simplest solution is the best solution.**

The action hierarchy for every change: **Delete > Replace > Add**. The best code change is a deletion. The second best is modifying what exists. Adding new code is the last resort.

1. **Minimal**: The simplest solution that works. Do not over-engineer, over-abstract, or add code just in case. Three similar lines beat a premature abstraction. Avoid error handling for impossible states, feature flags, compatibility shims, or policy scaffolding unless they are truly required.
2. **Solve at the source**: Do not hack fixes. Solve problems at their root. If something is broken, fix or remove the broken thing. Never patch over a broken abstraction, add workarounds, or add synchronization code for state that should not be duplicated.
3. **Delete ruthlessly**: When replacing code, delete what it replaced. Remove unused imports, functions, types, files, and commented-out code. Git preserves history. Run the repo's relevant dead-code or cleanup check when available.
4. **Replace > Add**: Modify existing code over adding new code. Edit existing files, extend existing components or functions with minimal parameters, and reuse existing utilities. If creating a new file, first prove it cannot fit cleanly in an existing file.
5. **Check existing**: Search the entire repo before creating anything new. If a feature, component, helper, responder, workflow, or utility already solves a similar problem, reuse or adapt it and delete the duplicate path.
6. **Deduplicate**: Do not duplicate existing code when updating the repo. Consolidate or refactor duplicates you find when it is in scope and low risk.
7. **Zero Regression**: Do not break existing features or workflows unless the PR intentionally removes them with evidence.
8. **Production ready**: All changes must be thoroughly debugged, validated, and production ready.

**When fixing bugs, ask: "What can I delete?" before "What can I replace?" before "What should I add?"**

## PR Workflow

After opening a PR:

1. Wait for the automated PR review and auto-format commit from Ultralytics Actions (`format.yml`), then pull and address every finding.
2. Launch an independent adversarial review agent with cold context (just the PR diff and this file) to hunt for bugs, regressions, and Core Principles violations — use the Codex CLI, one fresh `codex exec` run per round. Fix, push, and repeat until a fresh run reports LGTM.
3. Never fight other commits: Ultralytics Actions pushes auto-format and header commits, and multiple users may work on the same PR. `git pull --rebase` before pushing; never force-push, reset, or revert commits you did not author.
4. After the PR merges, clean up: remove local worktrees and branches for it, then `git checkout main && git pull`.

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

# Format/lint — mirrors what ultralytics/actions@main runs in format.yml (its action.yml is the source of truth)
ruff check --fix --unsafe-fixes --extend-select F,I,D,UP,RUF,FA --target-version py39 --ignore D100,D104,D203,D205,D212,D213,D401,D406,D407,D413,RUF001,RUF002,RUF012 .
ruff format --line-length 120 .
npx prettier --write --print-width 120 "**/*.{yml,yaml,json,md}"
```

- There is no test or coverage CI: the only workflows are `format.yml` (autoformat + AI labels/summaries on PRs), `cla.yml` (CLA signing), and `publish.yml` (PyPI release) — run tests locally before pushing.
- `requires-python = ">=3.8"` in pyproject.toml; classifiers cover Python 3.8–3.14.

## Architecture

THOP (PyPI package `ultralytics-thop`, import name `thop`) computes MACs and parameter counts of PyTorch models via forward hooks. `thop/profile.py` holds the `register_hooks` dict mapping `nn.Module` types to counting functions and exposes the two entry points: `profile()` (DFS traversal that counts each leaf module once) and the legacy `profile_origin()`. Counting functions live in `thop/vision/basic_hooks.py` (formulas in `thop/vision/calc_func.py`) and `thop/rnn_hooks.py` for RNN/GRU/LSTM; `thop/utils.py` provides `clever_format`. `thop/fx_profile.py` is an alternative `torch.fx`-based profiler not exported from `thop/__init__.py`. `benchmark/` scripts regenerate the README results table.

Releases are gated in `publish.yml`: it runs on every push to main but only for actor `glenn-jocher`, and compares `thop.__version__` (in `thop/__init__.py`, read dynamically by setuptools) against PyPI via ultralytics-actions `check_pypi_version`. If the local version is ahead it tags `v<version>`, creates an AI-summarized GitHub release, builds, publishes to PyPI via trusted publishing, uploads an SBOM, and notifies Slack — so merging a version bump to main IS the release trigger.

## Conventions

- Every source file starts with the header `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license` — Ultralytics Actions adds it automatically; don't add or revert it manually.
- Ruff formatting at line length 120 with Google-style docstrings (single-line imperative summaries); Prettier at print width 120 for YAML/JSON/Markdown — all applied automatically by `format.yml` on PRs.
- Tests are plain pytest with class-based grouping (each file wraps tests in a `TestUtils` class) and exact-value asserts on op counts; there is no conftest or pytest config, and no test hits the network.
- To release: bump `__version__` in `thop/__init__.py` in a PR; publishing happens automatically on merge to main (see Architecture).
