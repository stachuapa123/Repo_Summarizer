import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from file_tools import Workspace

if len(sys.argv) != 2:
    sys.exit("Usage: python mcp_server.py <project folder>")
ws = Workspace(sys.argv[1])
server = MCPServer("file-tools")

@server.tool()
def list_files(path: str = ".", pattern: str = "*") -> str:
    """List files in the project (skips .git, node_modules, virtualenvs).
    Paths are relative to the project root; use '.' for the whole project
    and a pattern like '*.py' to filter by file name."""
    try:
        return ws.list_files(path, pattern)
    except ValueError as e:
        raise ToolError(str(e)) from e

@server.tool()
def read_file(path: str = ".", pattern: str = "*") -> str:
    """Read the full text of a file, e.g. 'calculator.py' or 'src/main.py'.
    Paths are relative to the project root. Use list_files first to find files."""
    try:
        return ws.read_file(path)
    except ValueError as e:
        raise ToolError(str(e)) from e

if __name__ == "__main__":
    server.run("stdio")