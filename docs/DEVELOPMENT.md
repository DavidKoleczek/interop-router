# Development

## Git Hooks

This project uses [prek](https://github.com/j178/prek) for git hooks. See [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) for the full configuration.

Run all hooks manually:

```bash
prek run --all-files
```


## Code Quality

Format code:

```bash
uv run ruff format
```

Lint code:

```bash
uv run ruff check --fix
```

Type check:

```bash
uv run ty check
```

## Testing

Run tests:

```bash
uv run pytest
```

Run notebooks and save outputs in-place:

```bash
uv run scripts/run_notebooks.py --timeout 600 --max-concurrency 3
```

## AI Development Tools

Several commands (currently only work for Claude Code) are available in [.claude/commands/](../.claude/commands/)
- `/dependency-update`: Assists in updating dependencies in the project to their latest minor version
- `/supported-model-update`: Assists in updating the SupportedModel type to include new models released by providers
