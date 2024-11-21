import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

speech_file_path = "speech_english.mp3"

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="If your app runs for a long time and constantly caches functions, you might run into two problems:",
) as response:
    response.stream_to_file(speech_file_path)
