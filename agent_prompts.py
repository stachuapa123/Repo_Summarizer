from providers import ToolSpec

SYSTEM_PROMPT = """You are a helpful coding assistant.
You can explore the user's project with tools: list_files shows which files exist,
read_file shows a file's contents. Use them instead of guessing; never invent file names.
All paths are relative to the project root. Keep answers short and concrete."""


# ---------------------------------------------------------------------------
# 1. Tool descriptions: what the model sees
# ---------------------------------------------------------------------------
LIST_FILES_TOOL = ToolSpec(
    name="list_files",
    description=(
        "List files in the project, skipping .git, node_modules and virtualenvs. "
        "Returns one path per line, relative to the project root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Folder to list, relative to the project root. Use '.' for the whole project.",
            },
            "pattern": {
                "type": "string",
                "description": "Only list files whose name matches this pattern, e.g. '*.py'. Default: all files.",
            },
        },
        "required": [],
    },
)

READ_FILE_TOOL = ToolSpec(
    name="read_file",
    description="Read the full text of a file in the project. Use list_files first to find the right path.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the project root, e.g. 'calculator.py' or 'src/main.py'.",
            },
        },
        "required": ["path"],
    },
)

TOOL_SPECS = [LIST_FILES_TOOL, READ_FILE_TOOL]