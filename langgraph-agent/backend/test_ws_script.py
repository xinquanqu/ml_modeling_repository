import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Send a message that triggers a tool (assuming a calculator or similar exists)
        message = json.dumps({"message": "What is 25 * 4?"})
        await websocket.send(message)
        print(f"Sent: {message}")
        
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Received: {json.dumps(data, indent=2)}")
                
                if data.get("type") == "response":
                    print("Received final response, closing connection.")
                    break
            except Exception as e:
                print(f"Error: {e}")
                break

if __name__ == "__main__":
    asyncio.run(test_ws())
