from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

os.getenv("GOOGLE_API_KEY")

def get_gemini_response(input, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input!="":
        response = model.generate_content([input,image])
    else:
        response = model.generate_content(image)

    response = response.text
    return response

# Streamlit app
def upload_picture():
    uploaded_file = st.file_uploader("Upload Picture", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Picture', use_column_width=True)
        return image

# Function to display the sample picture options
def use_sample_picture():
    sample_option = st.selectbox('Select a sample picture:', ['Sample 1', 'Sample 2', 'Sample 3'])
    sample_images = {
        'Sample 1': 'E:/Gemini ChatBot/dog.jpeg',
        'Sample 2': 'E:/Gemini ChatBot/auditorium.png',
        'Sample 3': 'E:/Gemini ChatBot/jungle.png'
    }
    if sample_option in sample_images:
        image = Image.open(sample_images[sample_option])
        st.image(image, caption=sample_option, use_column_width=True)
        return image

st.set_page_config(page_title="TALK WITH IMAGES", page_icon="log.png")
st.markdown("""
    <style>
        .header {
            font-size: 36px;
            color: #1E90FF; /* Change the color as per your preference */
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2); /* Add shadow effect */
        }
        .response {
            font-size: 24px;
            color: green; /* Change the color as per your preference */
            text-align: left;
            margin-bottom: 20px;
        }
        .ouput {
            font-size: 29px;
            color: #333333; /* Change the color as per your preference */
            line-height: 1.5;
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f0f0; /* Change the background color as per your preference */
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="header">Talk With Images</h1>', unsafe_allow_html=True)
input_text = st.text_input("Input Prompt", key='input')

choice = st.radio("Choose an option:", ("Upload Picture", "Use Sample Picture"))
image = None
if choice == "Upload Picture":
    image = upload_picture()
elif choice == "Use Sample Picture":
    image = use_sample_picture()

submit = st.button("Ask from Image")
if submit and input_text and image:
    response = get_gemini_response(input_text, image)
    st.markdown('<h1 class="response">Response</h1>', unsafe_allow_html=True)
    st.write(response)
