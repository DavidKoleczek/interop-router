I am about to release a new version of this package. Please take the following steps to make sure it is successful:

Understand changes
1. Look at my currently staged changes to identify what changes were made.

Checks
1. Make sure the pyproject.toml is updated with a new version number.
2. Ensure there are no spelling and grammar mistakes.
3. First run `uv sync --all-extras --all-groups` to make sure the lock file is up to date.
4. Run all formatting, linting, and type checking: `uv run ruff format && uv run ruff check --fix && uv run ty check`
5. Run `uv build` to make sure the package builds correctly.
6. If any of these checks fail, please stop and inform me about the issues so we can fix them before proceeding.

Draft release notes
1. Look at the previous release logs at https://github.com/DavidKoleczek/interop-router/releases your draft release MUST follow the same style and structure.
2. Create a draft release description based on the recent code changes and place it in `ai_working/draft_release_{version}.md`. Here are some additional guidelines:
  - Don't list dependency updates individual. Just say "Updated dependencies".
3. 
4. At the end of the release notes be sure to include:
```
**Full Changelog**: https://github.com/DavidKoleczek/interop-router/compare/vx1.y1.z1...vx2.y2.z2
Available on [PyPI](https://pypi.org/project/interop-router/x.y.z/).
```
