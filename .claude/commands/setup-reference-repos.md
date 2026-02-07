Follow these steps to clone repos of key dependencies for ai context.
1. Read the `pyproject.toml` to understand which versions are being used.
1.  Create directory: Ensure `ai_working/` exists.
1. Check for existing repos: If any of these directories exist in `ai_working/`, delete them completely before proceeding:
   - `ai_working/openai-python`
   - `ai_working/anthropic-sdk-python`
   - `ai_working/python-genai`
2. Clone each of the repos with the following template in parallel: `git clone --depth 1 --single-branch <repo-url> <local-path>`
   - `openai-python`: https://github.com/openai/openai-python.git
   - `python-genai`: https://github.com/googleapis/python-genai.git
   - `anthropic-sdk-python`: https://github.com/anthropics/anthropic-sdk-python.git
3. Check the AGENTS.md to see if each of the repos being cloned is mentioned there. If any are missing, add them without any notes.
4. Briefly summarize what was done.
