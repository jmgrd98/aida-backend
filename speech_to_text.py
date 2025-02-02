import assemblyai as aai
import pyaudio
from dotenv import load_dotenv
import websockets
import asyncio
import json
import base64
import os

load_dotenv()

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

FRAMES_PER_BUFFER = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']} - Input Channels: {device_info['maxInputChannels']}")
p.terminate()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER
)

url = 'wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000'

async def send_receive():
    print(f'Connecting websocket to {url}')
    async with websockets.connect(
        url,
        extra_headers=(("Authorization", aai.settings.api_key),),
        ping_interval=5,
       ping_timeout=20
    ) as _ws:
        
        await asyncio.sleep(0.1)
        print("Receiving SessionBegins...")

        session_begins = await _ws.recv()
        print(session_begins)
        print("Sending messages...")

        async def send():
            while True:
                try:
                    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow = False)
                    data = base64.b64encode(data).decode("utf8")
                    json_data = json.dumps({"audio_data": str(data)})
                    await _ws.send(json_data)
                
                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    assert False, "Not a websocket 4008 error"

                await asyncio.sleep(0.01)

            return True


        async def receive():
            while True:
                try:
                    result_str = await _ws.recv()
                    print(json.loads(result_str)['text'])

                except websockets.exceptions.ConnectionClosedError as e:
                    print(e)
                    assert e.code == 4008
                    break
                except Exception as e:
                    assert False, "Not a websocket 4008 error"

        send_result, receive_result = await asyncio.gather(send(), receive())

while True:
    asyncio.run(send_receive())
