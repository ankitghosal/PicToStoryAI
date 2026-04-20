import os
import time
from typing import Any
from urllib import response
import io
import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from huggingface_hub import InferenceClient
from utils.custom import css_code
from gtts import gTTS


# Load env
load_dotenv(find_dotenv())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ------------------ Progress Bar ------------------
def progress_bar(amount_of_time: int) -> Any:
    progress_text = "Please wait, Generative models hard at work..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(amount_of_time):
        time.sleep(0.03)
        my_bar.progress(percent_complete + 1, text=progress_text)

    time.sleep(0.5)
    my_bar.empty()


# ------------------ Image → Text (BLIP) ------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


def generate_text_from_image(image_path: str) -> str:
    processor, model = load_blip()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


# ------------------ Text → Story (Gemini) ------------------
def generate_story_from_text(scenario: str) -> str:

    prompt = PromptTemplate.from_template(
        """
You are a creative storyteller.

Create a short story (max 50 words) based on the given context.

CONTEXT: {scenario}

STORY:
"""
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.9,
        google_api_key=GEMINI_API_KEY
    )

    chain = prompt | llm
    response = chain.invoke({"scenario": scenario})

    return response.content


# ------------------ Text → Speech (FIXED) ------------------
# def generate_speech_from_text(message):

    API_URL = "hexgrad/Kokoro-82M"   # keeping API_URL name intact

    try:
        client = InferenceClient(
            provider="fal-ai",
            api_key=HUGGINGFACE_API_TOKEN
        )

        # returns audio bytes
        audio_bytes = client.text_to_speech(
            message,
            model=API_URL
        )

        # ❌ API failure
        if response.status_code != 200:
            st.error(f"TTS API Error: {response.text}")
            return None
        
        # ❌ Empty response
        if not audio_bytes:
            st.error("No audio generated.")
            return None

        return audio_bytes

    except Exception as e:
        st.error(f"TTS Exception: {e}")
        return None

# -------------------
def generate_speech_from_text(message):

    API_URL = "gtts"

    try:
        audio_buffer = io.BytesIO()

        tts = gTTS(
            text=message,
            lang="en"
        )

        tts.write_to_fp(audio_buffer)

        audio_buffer.seek(0)

        return audio_buffer.read()

    except Exception as e:
        st.error(f"TTS Exception: {e}")
        return None

# ------------------ Main App ------------------
def main() -> None:

    st.set_page_config(
        page_title="Image to Story Converter",
        page_icon="🖼️"
    )

    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.image("img/profile.png")
        st.write("---")
        st.write("Image to Story App created by ANKIT GHOSAL")

    st.header("PicToStory Converter")

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Save image temporarily
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        progress_bar(100)

        # Step 1: Image → Caption
        scenario = generate_text_from_image(file_path)

        # Step 2: Caption → Story
        story = generate_story_from_text(scenario)

        # Step 3: Story → Speech
        audio_bytes = generate_speech_from_text(story)

        # Display
        with st.expander("Generated Image Scenario"):
            st.write(scenario)

        with st.expander("Generated Story"):
            st.write(story)

        # ✅ Safe audio playback
        if audio_bytes:
            st.audio(audio_bytes)
        else:
            st.warning("Audio could not be generated.")


if __name__ == "__main__":
    main()