import base64
import os

import numpy as np
import streamlit as st
from audiorecorder import audiorecorder
from openai import OpenAI


def STT(audio, client):
    filename = "input.mp3"
    wav_file = open(filename, "wb")
    wav_file.write(audio.export().read())
    wav_file.close()

    audio_file = open(filename, "rb")

    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

        audio_file.close()
        os.remove(filename)
    except Exception as e:
        transcript = str(e)
    return transcript


def TTS(response, client):
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=response,
    ) as response:
        filename = "output.mp3"
        response.stream_to_file(filename)

    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()

        md = f"""
<audio autoplay="True">
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
</audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    os.remove(filename)


def ask_gpt(prompt, client: OpenAI):
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=prompt)
    return response.choices[0].message.content


def main():
    client: OpenAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="음성 비서 프로그램", page_icon="🤖", layout="wide")

    if "check_audio" not in st.session_state:
        st.session_state.check_audio = []

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": """You are a thoughtful assistant.
                            Response to all input in 25 words and answer in koreans""",
            }
        ]

    col1, col2 = st.columns(2)

    with col1:
        st.header("AI Assistant")
        st.image("ai.png", width=200)
        st.markdown("---")

        flag_start = False

        audio = audiorecorder("질문", "녹음중")
        if len(audio) > 0 and not np.array_equal(audio, st.session_state.check_audio):
            st.audio(audio.export().read())

            question = STT(audio, client)

            st.session_state.messages = st.session_state.messages + [
                {"role": "user", "content": question}
            ]

            st.session_state.check_audio = audio
            flag_start = True

    with col2:
        st.subheader("대화기록")
        if flag_start:
            response = ask_gpt(st.session_state.messages, client)
            st.session_state.messages = st.session_state.messages + [
                {"role": "assistant", "content": response}
            ]

            for message in st.session_state.messages:
                if message["role"] != "system":
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            TTS(response, client)


if __name__ == "__main__":
    main()
