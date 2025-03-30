import base64
import os
import json
import queue
import time
import pyaudio

import soundfile as sf
import numpy as np
import threading

import prompts
from websockets.sync.client import connect

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


class AudioService:
    def __init__(self):
        self.audio_output_queue = queue.Queue()
        self.playback_running = False
        self.audio_output_stream = None
        self.transcript = []

        self._url = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        )
        self._headers = {
            "Authorization": "Bearer " + OPENAI_API_KEY,
            "OpenAI-Beta": "realtime=v1",
        }

    def handle_messages(self, ws):
        human_transcript = ""
        for message in ws:
            data = json.loads(message)
            print("Received event:", data.get("type"))

            if data.get("type") == "error":
                print("Error:", data)
                break

            if (
                data.get("type")
                == "conversation.item.input_audio_transcription.completed"
            ):
                role = "human"
                human_transcript += data["transcript"]
                self.transcript.append({"role": role, "text": human_transcript})
                human_transcript = ""

            if data.get("type") == "response.done":
                # Store the transcript
                output = data.get("response", {}).get("output", [])
                for item in output:
                    if item["type"] == "message":
                        role = item.get("role")
                        for content_item in item["content"]:
                            if "transcript" in content_item:
                                self.transcript.append(
                                    {"role": role, "text": content_item["transcript"]}
                                )

                try:
                    response = data.get("response", {})
                    output = response.get("output")
                    if len(output) > 0:
                        for item in output:
                            if item.get("type") == "function_call":
                                function_name = item.get("name")
                                print("Function call:", function_name)
                                print("Arguments:", item.get("arguments"))
                                if function_name == "end_call":
                                    with open("transcript.txt", "w") as f:
                                        for item in self.transcript:
                                            f.write(f"{item['role']}: {item['text']}\n")
                except Exception as e:
                    print("Error parsing response:", data)
                    print("Error:", e)

            if data.get("type") == "session.updated":
                print("Session updated:", data.get("session"))
                threading.Thread(
                    target=self.record_and_stream, args=(ws,), daemon=True
                ).start()

                event = {
                    "type": "response.create",
                    "response": {
                        "instructions": prompts.SYSTEM_PROMPT,
                    },
                }
                ws.send(json.dumps(event))

            if data.get("type") == "session.created":
                event = {
                    "type": "session.update",
                    "session": {
                        "instructions": prompts.SYSTEM_PROMPT,
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "interrupt_response": True,
                            "create_response": True,
                            "silence_duration_ms": 1000,
                        },
                        "input_audio_transcription": {
                            "language": "en",
                            "model": "gpt-4o-transcribe",
                        },
                        "tools": [
                            {
                                "type": "function",
                                "name": "store_call_information",
                                "description": "Store the call information collected from the patient.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "The name of the patient.",
                                        },
                                        "phone_number": {
                                            "type": "string",
                                            "description": "The phone number of the patient. (Save without any spaces)",
                                        },
                                        "email": {
                                            "type": "string",
                                            "description": "The email address of the patient.",
                                        },
                                        "enquiry": {
                                            "type": "string",
                                            "description": "The enquiry of the patient.",
                                        },
                                        "referring_doctor": {
                                            "type": "string",
                                            "description": "The name of the referring doctor, if any.",
                                        },
                                        "extra_notes": {
                                            "type": "string",
                                            "description": "Any extra notes about the call or conversation that would be useful for the human answering the message.",
                                        },
                                    },
                                    "required": ["name", "phone_number"],
                                },
                            },
                            {
                                "type": "function",
                                "name": "take_a_message",
                                "description": "Store the message from the patient if they do not want to talk to an AI.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "message": {
                                            "type": "string",
                                            "description": "The message from the patient.",
                                        },
                                    },
                                    "required": ["message"],
                                },
                            },
                            {
                                "type": "function",
                                "name": "end_call",
                                "description": "If you have recorded and stored all the information you need, and they have no further questions, notify them that they're all done, and then you can end the call by calling this function.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": [],
                                },
                            },
                        ],
                        "tool_choice": "auto",
                    },
                }
                ws.send(json.dumps(event))
                time.sleep(0.01)

            if data.get("type") == "input_audio_buffer.speech_started":
                pass
                if self.playback_running:
                    self.audio_output_queue = queue.Queue()
                    self.audio_output_queue.put("STOP")
                    self.playback_running = False

            if data.get("type") == "conversation.item.created":
                time.sleep(0.5)
                if not self.playback_running:
                    print("New speech started... starting playback")
                    self.playback_running = True
                    self.initialize_audio_output()
                    threading.Thread(
                        target=self.audio_playback_thread, daemon=True
                    ).start()

            if data.get("type") == "response.audio.delta":
                audio_data = base64.b64decode(data.get("delta"))
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                self.audio_output_queue.put(audio_array)

    def initialize_audio_output(self):
        p = pyaudio.PyAudio()
        self.audio_output_stream = p.open(
            format=pyaudio.paInt16, channels=1, rate=24000, output=True
        )
        return self.audio_output_stream

    def audio_playback_thread(self):
        if self.audio_output_stream is None:
            self.initialize_audio_output()

        buffer_size = 4800
        audio_buffer = np.array([], dtype=np.int16)
        while True:
            try:
                chunk = self.audio_output_queue.get(timeout=0.01)
                if isinstance(chunk, str) and chunk == "STOP":
                    break

                audio_buffer = np.append(audio_buffer, chunk)

                if len(audio_buffer) > buffer_size:
                    self.audio_output_stream.write(audio_buffer.tobytes())
                    audio_buffer = np.array([], dtype=np.int16)
            except queue.Empty:
                if len(audio_buffer) > 0:
                    self.audio_output_stream.write(audio_buffer.tobytes())
                    audio_buffer = np.array([], dtype=np.int16)
                time.sleep(0.01)

    def record_and_stream(self, ws, duration=1000, sample_rate=24000, chunk_size=1024):
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount")
        input_devices = []

        print("Available input devices:")
        for i in range(num_devices):
            device_info = p.get_device_info_by_index(i)
            if device_info.get("maxInputChannels") > 0:
                print(f"  {i}: {device_info.get('name')}")
                input_devices.append(i)

        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            input_device_index=1,
            frames_per_buffer=chunk_size,
        )

        print("Recording...")
        all_audio_data = []
        for _ in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size, exception_on_overflow=False)
            all_audio_data.append(data)
            base64_chunk = base64.b64encode(data).decode("ascii")
            event = {"type": "input_audio_buffer.append", "audio": base64_chunk}
            ws.send(json.dumps(event))
            time.sleep(0.01)

        print("Done.")
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save file to desk for debugging
        print("Writing to file...")
        audio_data = b"".join(all_audio_data)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sf.write("input.wav", audio_array, sample_rate)
        print("Done.")

    def run(self):
        with connect(self._url, additional_headers=self._headers) as ws:
            self.handle_messages(ws)


def store_patient_information(
    name: str, phone_number: str, email: str, enquiry: str, extra_notes: str
):
    print("Storing patient information...")


def main():
    audio_service = AudioService()
    audio_service.run()


if __name__ == "__main__":
    main()
