import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import sqlite3
from datetime import datetime

# Configure Google Generative AI
genai.configure(api_key="AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0")
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0"

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model2 = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
    system_instruction="You are a helpful personal assistant chatbot",
)

chat = model2.start_chat()

def chat_with_me(question):
    try:
        response = chat.send_message(question)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Database setup
def init_db():
    conn = sqlite3.connect('disease_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            prediction TEXT,
            timestamp TEXT
        )''')
    conn.commit()
    conn.close()

def insert_record(image_path, prediction):
    conn = sqlite3.connect('disease_history.db')
    c = conn.cursor()
    c.execute('''INSERT INTO history (image_path, prediction, timestamp)
                 VALUES (?, ?, ?)''', (image_path, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_records():
    conn = sqlite3.connect('disease_history.db')
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY timestamp DESC')
    records = c.fetchall()
    conn.close()
    return records

# Initialize the database
init_db()

background_image_url = "https://th.bing.com/th/id/OIP.LAOaWuloBHvVV7ZQRBwcowHaE7?rs=1&pid=ImgDetMain"

# Streamlit UI Setup
st.markdown(f"""
    <style>
    .main {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Sidebar content above "Connect with Us"
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Chat Support", "History"])

# Spacer to push "Connect with Us" to the bottom
st.sidebar.markdown("<br>" * 14, unsafe_allow_html=True)

# "Connect with Us" at the bottom inside the sidebar
st.sidebar.markdown("""
<div style="position: relative; bottom: 0; width: 100%; text-align: center;">
    <h4>Connect with Us</h4>
    <a href="https://github.com/Yashaswini0707/Plant-Disease-Detection-system" target="_blank">
        <img src="https://img.icons8.com/material-outlined/24/ffffff/github.png" style="vertical-align: middle;"/>
    </a>
    <a href="https://www.linkedin.com/in/your-linkedin-profile/" target="_blank">
        <img src="https://img.icons8.com/material-outlined/24/ffffff/linkedin.png" style="vertical-align: middle;"/>
    </a>
    <a href="https://www.instagram.com/your-instagram-profile/" target="_blank">
        <img src="https://img.icons8.com/material-outlined/24/ffffff/instagram-new.png" style="vertical-align: middle;"/>
    </a>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import sqlite3
from datetime import datetime

# Configure Google Generative AI
genai.configure(api_key="AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0")
os.environ["GOOGLE_API_KEY"] = "AIzaSyC9ofeMhsLxxB6pw6bENBZUPlveLY_osz0"

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model2 = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
    system_instruction=(
        "You are a helpful personal assistant chatbot"
    ),
)

chat = model2.start_chat()

def chat_with_me(question):
    try:
        response = chat.send_message(question)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Database setup
def init_db():
    conn = sqlite3.connect('disease_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            image_path TEXT,
            prediction TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_record(image_path, prediction):
    conn = sqlite3.connect('disease_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO history (image_path, prediction, timestamp)
        VALUES (?, ?, ?)
    ''', (image_path, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def fetch_records():
    conn = sqlite3.connect('disease_history.db')
    c = conn.cursor()
    c.execute('SELECT * FROM history ORDER BY timestamp DESC')
    records = c.fetchall()
    conn.close()
    return records

# Initialize the database
init_db()

background_image_url = "https://th.bing.com/th/id/OIP.LAOaWuloBHvVV7ZQRBwcowHaE7?rs=1&pid=ImgDetMain"

# Streamlit UI Setup
st.markdown(f"""
    <style>
    .main {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Spacer to push "Connect with Us" to the bottom
st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# Home Page
if app_mode == "Home":
    st.markdown("""
    <div class="typewriter">
        <h1>KRISHI AVARANAM</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    Welcome to KRISHI AVARANAM! 🌿🔍

    AI DRIVEN CROP DISEASE PREDICTION AND MANAGEMENT SYSTEM.
    
    Our mission is to help in identifying plant diseases efficiently.
    Discover the future of plant disease detection! Upload a plant image, and our state-of-the-art system will rapidly evaluate it for any disease signs.
    Partner with us to enhance crop health and secure a thriving harvest through innovative, precise analysis. Let’s work together for healthier, more resilient plants.
    """)

# About Project Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                This project focuses on leveraging machine learning to detect plant diseases from images. It is built using a combination of TensorFlow for model prediction and Google Generative AI for chatbot support. The system is designed to assist farmers and researchers in diagnosing plant health efficiently.
                #### Dataset
                The dataset used in this project is an augmented version of an original dataset, which consists of about 87K RGB images of healthy and diseased crop leaves. These images are categorized into 38 different classes, including various crops and diseases.

                Dataset Structure:
                1. Train: 70295 images
                2. Test: 33 images
                3. Validation: 17572 images

                #### Key Features
                - Advanced ML Models: The project utilizes cutting-edge machine learning models to ensure high accuracy in disease detection.
                - Real-time Chat Support: Integrated Google Generative AI for real-time support, helping users with their queries related to plant diseases.

                #### Achievements
                - Model Optimization: Improved the model's performance and prediction accuracy by fine-tuning the architecture.
                - User Experience: Developed an intuitive interface, making it easy for users to interact with the system.

                #### Future Goals
              - Data Acquisition and validation from the farmers.
              - Region wise crop data analysis.
              - Analysis of the data by the expertise and coming up with the solutions.
              - Implementing the LLM model.
              - Implementing Wireless multimedia sensor networks (WMSNs) and drone technology for scalability.
    """)

# Define the function to display video stream
def display_video_stream(ip_address):
    st.write("Streaming video from mobile camera...")
    # Display the video stream using the provided IP address
    video_stream_url = f"http://{ip_address}/video"  # Adjust the URL path as necessary
    st.video(video_stream_url)

# Disease Recognition Page
if app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    
    # Use camera input for image capture
    test_image = st.camera_input("Capture an Image:")
    
    # Use file uploader for dataset image input
    uploaded_image = st.file_uploader("Or choose an Image from your dataset:", type=["jpg", "jpeg", "png"])
    
    # Determine which image to use for prediction
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        selected_image = test_image
    elif uploaded_image is not None:
        st.image(uploaded_image, use_column_width=True)
        selected_image = uploaded_image
    else:
        selected_image = None
    
    # Prediction button
    if st.button("Predict"):
        if selected_image is not None:
            result_index = model_prediction(selected_image)
            class_name = [
                'Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 
                'Blueberry_healthy', 'Cherry(including_sour)_Powdery_mildew', 
                'Cherry(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn(maize)_Common_rust', 'Corn(maize)_Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 
                'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach_Bacterial_spot', 
                'Peach_healthy', 'Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy', 
                'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 
                'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 
                'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 
                'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
                'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 
                'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato_Tomato_mosaic_virus', 'Tomato_healthy'
            ]
            
            prediction = class_name[result_index]
            st.success("Model is predicting it's a {}".format(prediction))

            # Save record to database
            insert_record(selected_image.name, prediction)

            # Ask chatbot about disease management
            management_info = chat_with_me(f"What are the management practices for {prediction}?")
            st.info(f"Management Information: {management_info}")
        else:
            st.warning("Please capture an image or upload an image from your dataset before attempting to predict.")


# Chat Support Page
elif app_mode == "Chat Support":
    st.header("Agri LifeLine")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def display_chat():
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.write(f"You: {msg['content']}")
            else:
                st.write(f"Bot: {msg['content']}")

    display_chat()

    def send_message():
        user_message = st.session_state.chat_input
        if user_message:
            st.session_state.messages.append({"role": "user", "content": user_message})
            response = chat_with_me(user_message)
            st.session_state.messages.append({"role": "bot", "content": response})
            st.session_state.chat_input = ""
            st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

    user_input = st.text_input("Type your message here:", key="chat_input", on_change=send_message)
    st.button("Send", on_click=send_message)


# History Page
elif app_mode == "History":
    st.header("Prediction History")
    records = fetch_records()
    if records:
        for record in records:
            st.write(f"ID: {record[0]}")
            st.write(f"Image Path: {record[1]}")
            st.write(f"Prediction: {record[2]}")
            st.write(f"Timestamp: {record[3]}")
            st.write("---")
    else:
        st.write("No records found.")