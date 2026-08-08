---
name: dependency-update
description: Run this to update project dependencies
disable-model-invocation: true
---

I need you to update the dependencies in pyproject.toml.
The process to go through is:

1. Notice how we do versioning. We make sure that we do not auto upgrade to the major version. For example, "openai[aiohttp]>=2.9,<3.0", means we will never upgrade to v3.x. You will just be updating minor versions. Note that for dev dependencies we don't need to pin to be less than the major version. Do not touch anything outside `dependencies`, `[dependency-groups]`, or `[build-system].requires`.
2. Do not look up versions on PyPI to decide the pins. Ask uv instead. Our pins are floors, so the current floor never limits what uv picks, and uv already knows both our ceilings and the `exclude-newer` window configured in `[tool.uv]`. Leave `pyproject.toml` untouched and run `uv sync -U --all-extras --all-groups` from the root. This resolves every dependency to the newest version we are actually allowed to install.
3. Read the resolved versions with `uv tree --depth 1`, then bump each floor to the version shown. For example, if the current version in the pyproject.toml is `>=1.05,<2.0`, and `uv tree` shows v1.11, change the dependency to `>=1.11,<2.0`. Keep the existing ceiling as is. Because the resolution already accounted for `exclude-newer`, every pin you write is guaranteed to install, so you should never need to retry a bump.
4. Run `uv lock` after editing the pins so the lock file's `requires-dist` matches the new floors. The resolved versions will not change.
5. uv only considers versions inside our ceilings, so it will never surface a major upgrade on its own. Run `uv pip list --outdated` to find them. Anything it reports is a version our constraints currently exclude, which is either a major upgrade or a package held back by a parent's pin. The `Latest` column is the version itself, so you do not need to look it up on PyPI. If you notice a major version upgrade (ex v2 to v3), let the user know of each of those cases, but do not make the change yourself.
6. Be aware that uv also hides any release published inside the `exclude-newer` window, so a dependency can sit a version or two behind its true latest. That is expected and the next run will pick it up once the release ages past the window. Never use a version from PyPI that uv held back, because it will not resolve. If you want to note the held back versions in your summary, check the PyPI JSON site for the direct dependencies only.
7. Bump the `uv_build` pin in `[build-system].requires` to the latest version on PyPI (https://pypi.org/pypi/uv-build/json), using the same `>=X.Y.Z,<X.(Y+1).0` style as the other pins. Major version bumps are allowed here. Note that `uv_build` is resolved at build time into an isolated environment, so it is never installed in the venv and never appears in `uv.lock`. Neither `uv tree` nor `uv pip list --outdated` can see it, which is why this is one of the few places you still need PyPI. Run `uv build` from the root afterwards. If the build fails to resolve the pin, the error names the highest version available under the `exclude-newer` window, so use that one instead. Otherwise confirm it does not emit the `build_system.requires ... does not contain the current uv version` warning.
8. Update the `rev` fields in `.pre-commit-config.yaml`. These revs come from GitHub and are not subject to `exclude-newer`, so take the `ruff` hook rev from the version step 3 already resolved rather than from a newer release the project cannot install. The `uv-pre-commit` rev has no installed counterpart for uv to compare against, so check https://pypi.org/pypi/uv/json for that one. Run `prek run --all-files` to verify the hooks still pass.
9. Update the `uses:` action versions in every workflow file under `.github/workflows/`. Do not guess pins from release tags or major versions. For each action we use, read its upstream README and copy the exact `uses:` value from the documented examples:
    - `actions/checkout`: `https://raw.githubusercontent.com/actions/checkout/main/README.md`
    - `astral-sh/setup-uv`: `https://raw.githubusercontent.com/astral-sh/setup-uv/main/README.md`
    - `pypa/gh-action-pypi-publish`: `https://raw.githubusercontent.com/pypa/gh-action-pypi-publish/unstable/v1/README.md`
    If a workflow later adds a different action, apply the same rule: find that action's README Usage/examples section and pin to whatever it shows there.
10. Make sure all the checks still pass by running `uv run ruff format && uv run ruff check --fix && uv run ty check` from the root. A `ty` upgrade can surface new diagnostics in code you did not touch. Fix them, and call them out separately in your summary so they are not mistaken for a dependency change.
11. Set up local reference repositories for the supported provider SDKs:
    - Ensure `ai_working/` exists.
    - For each repository below, determine the version currently installed in the project's uv environment with `uv run python -c "from importlib.metadata import version; print(version('<package>'))"`:
      - `openai-python`: package `openai`, repository `https://github.com/openai/openai-python.git`
      - `python-genai`: package `google-genai`, repository `https://github.com/googleapis/python-genai.git`
      - `anthropic-sdk-python`: package `anthropic`, repository `https://github.com/anthropics/anthropic-sdk-python.git`
    - Remove any existing checkout at the corresponding `ai_working/<repository>` path.
    - Prefer a shallow, single-branch clone of the tag matching the installed version. Confirm the exact tag name first with `git ls-remote --tags <repo-url>` because repositories may use either `v<version>` or `<version>` tags. Clone the matching tag with `git clone --depth 1 --single-branch --branch <tag> <repo-url> <local-path>`.
    - If no tag matches the installed version, shallow-clone the default branch with `git clone --depth 1 --single-branch <repo-url> <local-path>` and explicitly report the version mismatch.
    - Check `AGENTS.md` to ensure each cloned repository is listed. Add any missing repository names without additional notes.
12. Briefly summarize the updated dependencies, hook and workflow action versions, and reference repository revisions. Also report any major upgrade you found in step 5.
