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

Run all tests:

```bash
uv run pytest
```

Run unit or integration suites separately:

```bash
uv run pytest tests/unit
uv run pytest tests/integration
```

Filter by file or name:

```bash
uv run pytest tests/integration/test_chat_completions.py
uv run pytest tests/integration/test_chat_completions.py -k openai
uv run pytest tests/integration/test_chat_completions.py -k local
```

### Environment variables

Integration tests call live APIs. The main suite expects:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

Optional Azure OpenAI test (`test_azure_openai_client`):
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_AD_TOKEN`

Optional local chat completions tests (`test_chat_completions.py`, `-k local`). These skip when unset:
- `CHAT_COMPLETIONS_BASE_URL` (for example `http://10.1.10.5:8000/v1`)
- `CHAT_COMPLETIONS_MODEL` (for example `nvidia/Qwen3.6-27B-NVFP4`)

For local development, copy [`.env.example`](../.env.example) to `.env` (gitignored) and set the values there. VS Code loads `.env` for the Testing panel and Debug Tests via `python.envFile`.

PowerShell example for the local suite without `.env`:

```powershell
$env:CHAT_COMPLETIONS_BASE_URL="http://10.1.10.5:8000/v1"
$env:CHAT_COMPLETIONS_MODEL="nvidia/Qwen3.6-27B-NVFP4"
uv run pytest tests/integration/test_chat_completions.py -k local
```

Run notebooks and save outputs in-place:

```bash
uv run scripts/run_notebooks.py --timeout 600 --max-concurrency 3
```

## AI Development Tools

Several commands (currently only work for Claude Code) are available in [.claude/commands/](../.claude/commands/)
- `/dependency-update`: Assists in updating dependencies in the project to their latest minor version
- `/supported-model-update`: Assists in updating the SupportedModel type to include new models released by providers
