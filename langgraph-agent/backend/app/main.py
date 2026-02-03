from dotenv import load_dotenv
load_dotenv(override=True)
import logging
import sys
import types

# --- LANGCHAIN COMPATIBILITY SHIM FOR LANGFUSE V2 ---
# Must run before any langfuse imports
try:
    import langchain.callbacks.base
except ImportError:
    if "langchain" not in sys.modules:
        sys.modules["langchain"] = types.ModuleType("langchain")
    if "langchain.callbacks" not in sys.modules:
        sys.modules["langchain.callbacks"] = types.ModuleType("langchain.callbacks")
    
    shim_module = types.ModuleType("langchain.callbacks.base")
    try:
        from langchain_core import callbacks as core_callbacks
        shim_module.BaseCallbackHandler = core_callbacks.BaseCallbackHandler
    except ImportError:
        pass
    sys.modules["langchain.callbacks.base"] = shim_module

    if "langchain.schema" not in sys.modules:
        sys.modules["langchain.schema"] = types.ModuleType("langchain.schema")
    if "langchain.schema.agent" not in sys.modules:
        agent_module = types.ModuleType("langchain.schema.agent")
        try:
            from langchain_core import agents as core_agents
            agent_module.AgentAction = core_agents.AgentAction
            agent_module.AgentFinish = core_agents.AgentFinish
        except ImportError:
            pass
        sys.modules["langchain.schema.agent"] = agent_module

    if "langchain.schema.document" not in sys.modules:
        doc_module = types.ModuleType("langchain.schema.document")
        try:
            from langchain_core import documents as core_docs
            doc_module.Document = core_docs.Document
        except ImportError:
            pass
        sys.modules["langchain.schema.document"] = doc_module
# --- END SHIM ---

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import chat, graph

app = FastAPI(title="LangGraph Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(graph.router)
app.include_router(chat.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
