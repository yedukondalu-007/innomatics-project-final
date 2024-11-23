import streamlit as st
import pyttsx3
import base64
from PIL import Image
import requests
from io import BytesIO
import time
from transformers import AutoProcessor, BlipForConditionalGeneration
import pytesseract
from gtts import gTTS
import os
from tenacity import retry, stop_after_attempt, wait_exponential

# Set up Google Gemini API key (replace 'YOUR_API_KEY' with actual key)
GENAI_API_KEY = "AIzaSyDqui1f0QXykGcKYpHzZIlA16JLEQfLmzc"
# Assuming you're configuring for Google Gemini
import google.generativeai as genai
genai.configure(api_key=GENAI_API_KEY)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Load BLIP model for image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate image caption using BLIP
def generate_image_caption(image):
    image = image.convert('RGB')
    inputs = processor(images=image, text="Describe the image", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to extract text from image using OCR (Tesseract)
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to convert text to speech using gTTS
def text_to_speech(text):
    try:
        tts = gTTS(text, lang="en")
        audio_file = "output.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        return f"Error generating speech: {str(e)}"

# Streamlit configuration
st.set_page_config(page_title="AI-Powered Image Description and Text-to-Speech", page_icon="🖼🎙", layout="wide")

# Sidebar for User Instructions and Features
st.sidebar.title("Welcome to the AI Image Description and Text-to-Speech App")
st.sidebar.markdown("""
    This app helps visually impaired users by:
    1. Extracting text from images and converting it to speech.
    2. Generating detailed descriptions of images based on emotions, actions, and visual elements.
    
    📝 Instructions:
    1. Upload an image containing text or an image to describe.
    2. The app will generate a description (if an image) and convert any text (from image) to speech.
    3. Receive a detailed description suitable for accessibility.
""")

# Load BLIP captioning section at the top
st.title("AI-Powered Image Description")
st.write("""
    Upload an image, and the app will generate a description of the image, including actions, emotions, and visual elements.
""")

# File uploader for the image (Captioning)
uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate image caption
    with st.spinner("Generating description..."):
        caption = generate_image_caption(image)
        st.subheader("Generated Caption:")
        st.write(caption)

    # Function to generate scene description using Google Gemini API
    def generate_scene_description_with_gemini(caption):
        try:
            prompt = f"Generate an emotionally rich and action-based description of the following scene: {caption}"
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            ai_assistant = model.start_chat(history=[])
            response = ai_assistant.send_message(prompt)
            return response.text.strip() if response and response.text else "No description generated."
        except Exception as e:
            return f"Error generating description: {str(e)}"

    # Get the scene description from Google Gemini
    description = generate_scene_description_with_gemini(caption)
    if "Error" in description:
        st.error(description)
    else:
        st.subheader("Generated Description:")
        st.write(description)

# ------------------------------- OCR and Text-to-Speech Section -------------------------------
st.title("OCR and Text-to-Speech Conversion")
st.write("""
    Upload an image containing text, and the app will extract the text and convert it to speech.
""")

# File uploader for the image (OCR)
ocr_uploaded_file = st.file_uploader("Choose an image with text to extract...", type=["jpg", "jpeg", "png"], label_visibility="visible")

if ocr_uploaded_file:
    # Load the uploaded image
    image = Image.open(ocr_uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract text from image
    with st.spinner("Extracting text from image..."):
        text = extract_text_from_image(image)

    # Display extracted text and convert to speech
    if text:
        st.subheader("Extracted Text:")
        st.write(text)

        # Convert the extracted text to speech
        with st.spinner("Converting text to speech..."):
            audio_file = text_to_speech(text)

        # Play the audio
        if os.path.exists(audio_file):
            st.subheader("Audio Playback:")
            audio = open(audio_file, "rb")
            st.audio(audio, format="audio/mp3")
            audio.close()  # Ensure the file is closed after playback
            os.remove(audio_file)  # Clean up the generated audio file
    else:
        st.warning("No text found in the image. Please try another image with visible text.")

# Footer
st.markdown("---")
st.markdown("🔍 Powered by Tesseract OCR, gTTS, and Google's Generative AI | Built with ❤ using Streamlit")
