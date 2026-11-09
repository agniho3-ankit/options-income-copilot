"""Phase 5 multi-agent workflow: a LangGraph supervisor, CrewAI role agents, and a local MCP
server over the deterministic tools. Privacy switches are applied on import, before any
framework module loads."""
from . import privacy

privacy.configure()
