import asyncio, websockets, json, random

async def test_client():
    async with websockets.connect("ws://localhost:8092") as ws:
        while True:
            data = {"test": random.random()}
            await ws.send(json.dumps(data))
            print(f"Отправлено: {data}")
            await asyncio.sleep(2)

asyncio.run(test_client())