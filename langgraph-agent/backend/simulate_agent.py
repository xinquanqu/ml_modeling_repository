
import asyncio
import websockets
import json
import sys

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

async def run_simulation():
    uri = "ws://127.0.0.1:8000/ws"
    print(f"{BLUE}--- Starting Simulation ---{RESET}")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"{GREEN}Connected to WebSocket{RESET}")
            
            # Test 1: Simple Chat
            print(f"\n{YELLOW}Test 1: Simple Chat ('Hello'){RESET}")
            await websocket.send(json.dumps({"message": "Hello"}))
            
            # Expect: start -> chatbot -> response
            expected_flow = ["start", "chatbot", "end"] # end is implicit in response
            received_flow = []
            
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                if data.get("type") == "state_update":
                    node = data.get("node")
                    print(f"  -> State Update: {node}")
                    received_flow.append(node)
                
                elif data.get("type") == "response":
                    print(f"  -> {GREEN}Final Response: {data.get('content')}{RESET}")
                    # Validate flow
                    break
                
                elif data.get("type") == "error":
                    print(f"{RED}  -> ERROR: {data.get('error')}{RESET}")
                    break
            
            # Test 2: Tool Use
            print(f"\n{YELLOW}Test 2: Tool Use ('Check the weather'){RESET}")
            await websocket.send(json.dumps({"message": "Check the weather"}))
            
            tool_flow_detected = False
            
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                if data.get("type") == "state_update":
                    node = data.get("node")
                    print(f"  -> State Update: {node}")
                    if node == "tool_executor":
                        tool_flow_detected = True
                        print(f"     {BLUE}Tool execution confirmed!{RESET}")
                
                elif data.get("type") == "response":
                    content = data.get("content")
                    print(f"  -> {GREEN}Final Response: {content}{RESET}")
                    
                    if "72°F" in content or "weather" in content.lower():
                         print(f"     {GREEN}Content verified!{RESET}")
                    else:
                         print(f"     {RED}Content missing expected weather info.{RESET}")
                    break

            if tool_flow_detected:
                print(f"\n{GREEN}SUCCESS: Tool usage flow verified.{RESET}")
            else:
                print(f"\n{RED}FAILURE: Tool usage flow NOT detected.{RESET}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"{RED}Connection closed unexpectedly: {e.code} {e.reason}{RESET}")
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")

if __name__ == "__main__":
    asyncio.run(run_simulation())
