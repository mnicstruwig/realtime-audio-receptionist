import base64
import os
import json
import time
import pyaudio
import websocket

import base64
import json
import struct
import soundfile as sf
import numpy as np


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    "Authorization: Bearer " + OPENAI_API_KEY,
    "OpenAI-Beta: realtime=v1"
]

incoming_audio_data = []

def on_open(ws):
    print("Connected to server.")

def on_message(ws, message):
    data = json.loads(message)
    print("Received event:", data.get("type"))

    if data.get("type") == "session.created":
        print("Session created:", data.get("session"))
        time.sleep(1)
        record_and_stream(ws)

    if data.get("type") == "response.audio.delta":
        audio_data = base64.b64decode(data.get("delta"))
        #audio_array = np.frombuffer(audio_data, dtype=np.int16)
        incoming_audio_data.append(audio_data)
        print("AUDIO DATA LENGTH:", len(incoming_audio_data))

    if data.get("type") == "response.done":
        #print("Received response.done", json.dumps(data, indent=2))
        print("AUDIO DATA LENGTH:", len(incoming_audio_data))
        if incoming_audio_data:
            print("Trying to save received audio...")
            combined_audio = b''.join(incoming_audio_data)
            print("Converted to bytes")
            combined_audio = np.frombuffer(combined_audio, dtype=np.int16)
            print("Converted to numpy array")
            sf.write("received_audio.wav", combined_audio, 24000)
            print("Saved received audio to received_audio.wav")


def float_to_16bit_pcm(float32_array):
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16

def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded


def record_and_stream(ws, duration=8, sample_rate=24000, chunk_size=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    
    time.sleep(0.5)
    print("Recording...")
    all_audio_data = []
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size, exception_on_overflow=False)
        all_audio_data.append(data)
        #audio_chunk = np.frombuffer(data, dtype=np.float32)
        #base64_chunk = base64_encode_audio(float_to_16bit_pcm(audio_chunk))
        base64_chunk = base64.b64encode(data).decode('ascii') #base64_encode_audio(float_to_16bit_pcm(audio_chunk))
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64_chunk
        }
        ws.send(json.dumps(event))
        time.sleep(0.01)

    print("Done.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save file to desk for debugging
    print("Writing to file...")
    audio_data = b''.join(all_audio_data)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    sf.write("input.wav", audio_array, sample_rate)
    print("Done.")



ws = websocket.WebSocketApp(
    url,
    header=headers,
    on_open=on_open,
    on_message=on_message,
)


files = [
    './path/to/sample1.wav',
    './path/to/sample2.wav',
    './path/to/sample3.wav'
]

# for filename in files:
#     data, samplerate = sf.read(filename, dtype='float32')  
#     channel_data = data[:, 0] if data.ndim > 1 else data
#     base64_chunk = base64_encode_audio(channel_data)
    
#     # Send the client event
#     event = {
#         "type": "input_audio_buffer.append",
#         "audio": base64_chunk
#     }
#     ws.send(json.dumps(event))

ws.run_forever()