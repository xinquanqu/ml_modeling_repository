
import asyncio
import websockets
import datetime

async def test_idle_connection():
    uri = "ws://127.0.0.1:8000/ws"
    print(f"Connecting to {uri} at {datetime.datetime.now()}")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected. Waiting for disconnect...")
            while True:
                try:
                    # Wait for message or just sleep
                    msg = await asyncio.wait_for(websocket.recv(), timeout=60)
                    print(f"Received: {msg}")
                except asyncio.TimeoutError:
                    print("No message for 60s, sending ping")
                    await websocket.ping()
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed by server at {datetime.datetime.now()}: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_idle_connection())
