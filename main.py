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

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Chat Support", "History"])

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
    #### About the Project
    This project focuses on leveraging machine learning to detect plant diseases from images.
    Welcome to KRISHI AVARANAM! 🌿🔍
    A AI DRIVEN CROP DISEASE PREDICTION AND MANAGEMENT SYSTEM.
    Our mission is to help in identifying plant diseases efficiently.
    Discover the future of plant disease detection! Upload a plant image, and our state-of-the-art system will rapidly evaluate it for any disease signs.
    Partner with us to enhance crop health and secure a thriving harvest through innovative, precise analysis. Let’s work together for healthier, more resilient plants.
    ### How It Works
    1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.
    3. Results: View the results and recommendations for further action.
    ### Why Choose Us?
    - Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - User-Friendly: Simple and intuitive interface for seamless user experience.
    - Fast and Efficient: Receive results in seconds, allowing for quick decision-making.
    ### Get Started
    Navigate to the Disease Recognition page in the sidebar to upload your plant image and witness the capabilities of our cutting-edge Plant Disease Recognition System. This powerful tool will analyze your image in-depth, providing you with accurate insights and disease detection. Explore the technology that’s transforming plant health management and optimize your crop care with just a few clicks.
    ### About Us
    Learn more about the project, our team, and our goals on the About page.
    ### Recent Work
    - Successfully integrated Google Generative AI for providing chatbot support within the application.
    - Enhanced the machine learning model for better accuracy and faster predictions. 

    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.warning("Please upload an image before attempting to display it.")

    if st.button("Predict"):
        if test_image is not None:
            st.snow()
            result_index = model_prediction(test_image)
            class_name = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 
                          'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew', 
                          'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy', 
                          'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 
                          'Grape_healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 
                          'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 
                          'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 
                          'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 
                          'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 
                          'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
                          'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 
                          'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
                          'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']
            prediction = class_name[result_index]
            st.success("Model is Predicting it's a {}".format(prediction))
            
            # Save record to database
            insert_record(test_image.name, prediction)

        else:
            st.warning("Please upload an image before attempting to predict.")

# History Page
elif app_mode == "History":
    st.header("Disease Prediction History")
    records = fetch_records()
    if records:
        for record in records:
            st.write(f"Image: {record[1]}, Predicted Disease: {record[2]}, Timestamp: {record[3]}")
    else:
        st.write("No records found.")

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