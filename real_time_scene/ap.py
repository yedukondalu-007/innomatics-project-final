import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import logging
import os

# Static Google API Key
GOOGLE_API_KEY = "your_api_key"

# Initialize models through LangChain with correct model names
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Error handling function
def handle_error(error):
    logging.error(error)
    st.error(f"Error: {str(error)}")

# Scene understanding function
def scene_understanding(image):
    try:
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes = image_bytes.getvalue()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Describe this image for visually impaired individuals, including:
                    - Scene layout
                    - Main objects
                    - People and actions
                    - Colors and lighting
                    - Points of interest"""
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                }
            ]
        )
        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        handle_error(e)

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
st.set_page_config(page_title="AI Image Description & Speech Conversion", page_icon="🖼🎙️", layout="wide")

# ---------------------------- Custom CSS for Animation and Styles ----------------------------
# Inject custom CSS for navbar, footer, and animation
st.markdown("""
    <style>
        /* Navbar styles */
        .navbar {
            position: sticky;
            top: 0;
            background-color: #333;
            padding: 15px;
            text-align: center;
            color: white;
            font-size: 20px;
            font-weight: bold;
        }
        .navbar a {
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            margin: 0 10px;
        }
        .navbar a:hover {
            background-color: #ddd;
            color: black;
        }

        /* Border Animation */
        .border-animated {
            border: 5px solid;
            border-image: linear-gradient(45deg, red, yellow, green, cyan, blue, magenta);
            border-image-slice: 1;
            animation: borderAnimation 4s linear infinite;
        }

        /* Keyframe for border animation */
        @keyframes borderAnimation {
            0% {
                border-image-source: linear-gradient(45deg, red, yellow, green, cyan, blue, magenta);
            }
            100% {
                border-image-source: linear-gradient(45deg, blue, cyan, green, yellow, red, magenta);
            }
        }

        /* Footer Styles */
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #333;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }

        /* Adjust padding for the main content to avoid overlap with navbar and footer */
        .content {
            padding-top: 60px;
            padding-bottom: 60px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- Navbar ----------------------------
st.markdown('<div class="navbar">AI Image and Speech Converter</div>', unsafe_allow_html=True)

# Sidebar for User Instructions and Features
st.sidebar.title("AI Image Description & Speech Conversion App")
st.sidebar.markdown(""" 
    This app helps visually impaired users by:
    1. Generating descriptions of images based on emotions, actions, and visual elements.
    2. Extracting text from images and converting it to speech.

    📝 *Instructions*:
    1. Upload an image for description or text extraction.
    2. The app will either generate a description of the image or convert the extracted text to speech.
""")

# ---------------------------- Main Content ----------------------------
# Add content container with animation border
with st.container():
    st.markdown('<div class="border-animated">', unsafe_allow_html=True)

    # ---------------------------- Image Description Section ----------------------------
    st.markdown("<h2 style='text-align: center;'>Image Description</h2>", unsafe_allow_html=True)
    st.write("""
        Upload an image to get a description of the scene, including actions, emotions, and visual elements.
    """)

    # File uploader for the image (Captioning)
    uploaded_file = st.file_uploader("Choose an image for description...", type=["jpg", "jpeg", "png"], label_visibility="visible")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate image caption using Google Gemini
        with st.spinner("Generating description..."):
            description = scene_understanding(image)
            st.subheader("Generated Description:")
            st.write(description)

    # ----------------------------- OCR and Text-to-Speech Section ------------------------------
    st.markdown("<h2 style='text-align: center;'>OCR and Text-to-Speech</h2>", unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- Footer ----------------------------
st.markdown('<div class="footer">🔍 Powered by Tesseract OCR, gTTS, and Google\'s Generative AI | Built with ❤ using Streamlit</div>', unsafe_allow_html=True)
