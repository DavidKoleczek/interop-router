# General Instructions
- InteropRouter is a production-grade Python package. You must *always* follow best open-source Python practices. 
- Shortcuts are not appropriate. When in doubt, you must work with the user for guidance.
- Any documentation you write, including in the README.md, should be clear, concise, and accurate like the official documentation of other production-grade Python packages.
- Make sure any comments in code are necessary. A necessary comment captures intent that cannot be encoded in names, types, or structure. Comments should be reserved for the "why", only used to record rationale, trade-offs, links to specs/papers, or non-obvious domain insights. They should add signal that code cannot.
- The current code in the package should be treated as an example of high quality code. Make sure to follow its style and tackle issues in similar ways where appropriate.
- Do not run tests automatically unless asked since they take a while.
- When writing documentation
  - Keep it very concise
  - No emojis or em dashes.

# Python Development Instructions
- `ty` by Astral is used for type checking. Always add appropriate type hints such that the code would pass ty's type check.
- Follow the Google Python Style Guide.
- After each code change, checks are automatically run. Fix any issues that arise.
- **IMPORTANT**: The checks will remove any unused imports after you make an edit to a file. So if you need to use a new import, be sure to use it FIRST (or do your edits at the same time) or else it will be automatically removed. DO NOT use local imports to get around this.
- Always prefer pathlib for dealing with files. Use `Path.open` instead of `open`. 
- When using pathlib, **always** Use `.parents[i]` syntax to go up directories instead of using `.parent` multiple times.
- When writing tests, use pytest and pytest-asyncio.
- NEVER use `# type: ignore`. It is better to leave the issue and have the user work with you to fix it.

# Dependency Context
- For libraries that are new or change frequently you must use the source code of libraries or their documentation as your source of truth for how to use them.
- There is a select set of repos cloned that constitute the dependencies for this project at `ai_working`. You must explore it directly when needed. It can be updated through the command `/setup-reference-repos`  in this project.
  - `openai-python`
  - `anthropic-sdk-python`
  - `python-genai` - Google Gen AI SDK
- When looking for something specific that might take a while, use a sub-agent to find it. Tell the sub-agent return the location (paths) of what is found so it can be referenced easily later.


# Key Files

@README.md

@pyproject.toml

@src/interop_router/types.py

@src/interop_router/router.py
