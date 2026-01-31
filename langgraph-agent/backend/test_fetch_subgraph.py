from app.services.agent import get_graph_structure
import traceback

try:
    print("Fetching graph for 'chatbot'...")
    structure = get_graph_structure(subgraph_id="chatbot")
    print("Success!")
    print(structure)
except Exception:
    traceback.print_exc()
