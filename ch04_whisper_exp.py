import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

audio_file = open("speech_english.mp3", "rb")


transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text",
)

print(transcript)
