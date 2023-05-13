import streamlit as st
import os
from PIL import Image
import numpy as np
import server
import pickle as pk
import gensim
from collections import Counter
import PyPDF2
import nltk
from nltk.corpus import stopwords
import re
from nltk import pos_tag,word_tokenize
import time
nltk.download('punkt')

def match(resume , job_decription):
    model = pk.load(open('model.pkl' , 'rb'))
    resume = pipeline(resume)
    job_decription = pipeline(job_decription)
    nonmatched = []
    matched = []
    for key in job_decription:
        if key not in model.wv.key_to_index:
            continue
        match = False
        for word in resume:
            if word not in model.wv.key_to_index:
                continue
            if(model.wv.similarity(key , word) >= 0.75):
                match = True
        if not match:
            nonmatched.append(key)
        else:
            matched.append(key)
    matched_words = Counter(matched)
    nonmatched_words = Counter(nonmatched)
    match_score = 1
    nonmatch_score = 1
    for tuple in matched_words.items():
        match_score += tuple[1]
    for tuple in nonmatched_words.items():
        nonmatch_score += tuple[1]
    
    return round(100*((match_score/(match_score+nonmatch_score)) + 0.3) , 2) , matched_words.most_common(min(5 , len(matched_words))) , nonmatched_words.most_common(min(5 , len(nonmatched_words)))

def gen_text(pdf_file):

    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ''

    try:
        for page_num in range(len(pdf_reader.pages)):

            page = pdf_reader.pages[page_num]

            text += ' ' + page.extract_text()
    except:
        text = ''
        
    # pdf_file.close()

    return text

def extract_keywords(document):
    text = preprocessing(document)
    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)
    keywords =  [word for word,pos in tagged if (pos == 'NN')]
    return " ".join(keywords)

def preprocessing(text):
    nltk.download('stopwords')
    sw = set(stopwords.words('english'))
    text = text.replace('\n',' ').lower()
    pattern = re.compile('[^A-Za-z\s]')
    text = re.sub(pattern , "" , text)
    words = []
    for word in text.split():
        if len(word) == 1 or word in sw:
            continue
        words.append(word)
    return " ".join(words)

def pipeline(file):
    nltk.download('averaged_perceptron_tagger')
    text = gen_text(file)
    return extract_keywords(text).split()


def app():
    def reshap_image(name):
        image = Image.open(name)

        # Resize the image to a square with side length 300 pixels
        image = image.resize((300, 300))

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Create a circular mask for the image
        x, y = np.indices((300, 300))
        center = (150, 150)
        mask = ((x - center[0])**2 + (y - center[1])**2 < 150**2)

        # Apply the mask to the image
        image_array[~mask] = 0

        # Display the image with a circular border
        st.image(image_array, width=300, clamp=True, channels="RGB")
    with st.container():
        st.header("Apply to Tata Consultancy Services")
        # description = '''
        #     A machine learning (ML) developer is an expert in using data to train 
        #     models and automate processes like image classification, speech recognition,
        #     and market forecasting.0 They need exceptional skills in statistics and programming to
        #     create machine learning models and retrain systems.2 They develop ML algorithms to analyze
        #     huge volumes of historical data to make predictions, run tests, perform statistical analysis,
        #     and interpret test results. They document machine learning processes and keep abreast of developments
        #     in machine learning. Machine learning engineers develop self-running AI software to automate predictive
        #     models for recommended searches, virtual assistants, translation apps, chatbots, and driverless cars.1
        #     '''
        # st.write(description)
        st.markdown("<h3>Contact info</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 3])
        with col1:
        # Display the circular profile picture
            st.image("profile_picture1.jpg", width=200, clamp=True, channels="RGB")

        with col2:
        # Display the profile name and description
            st.write("# John Doe")
            st.write("Software Engineer with 5 years of experience in web development and data analysis.")
        # st.image("profile_picture1.jpg", width=300, clamp=True, channels="RGB")
        # Email input
        email = st.text_input("Email Address*")

        # Country code input
        country_code = st.selectbox("Country Code*", ["+1", "+44", "+61", "+81", "+91"])
        mobile_number = st.text_input("Mobile Number*")

        # Resume title
        st.markdown("<h3>My Resume</h3>", unsafe_allow_html=True)

        # Resume content
        st.write('Be sure to include and updated resume')

    # Create the resumes folder if it doesn't exist
    # if not os.path.exists("resumes"):
    #     os.makedirs("resumes")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    uploaded_jd = st.file_uploader("Choose a jd file", type="pdf")
    parssed = ""
    # st.write({uploaded_file.name})
    # if st.button("Check"):
    #     # parssed = server.gen_text(uploaded_file)
    #     if uploaded_file is not None and uploaded_jd is not None:
    #         st.write(f"You selected the following file: {uploaded_file.name}")
    #         st.write(f"You selected the following file: {uploaded_jd.name}")
    #     else:
    #         st.write("Please select a file first.")
    if uploaded_file is not None and uploaded_jd is not None:
        # Get the file name and extension
        filename, extension = os.path.splitext(uploaded_file.name)

        # Create a file path to save the file to the resumes folder
        # filepath = os.path.join("resumes", filename + extension)

        # Save the file to the resumes folder
        # with open(filepath, "wb") as f:
        #     f.write(uploaded_file.getbuffer())

        # Display a success message
        # st.success(f"Saved {uploaded_file.name} to {filepath}")
        result(server.match(uploaded_file,uploaded_jd))

def result(data):
    print("In process")
    # uploaded_file = gen_text(uploaded_file)
    # uploaded_file = pipeline(uploaded_file)
    # JD = pipeline(JD)
    score , matched , nonmatched = data#match(uploaded_file , JD)
    score = int(min(score , 100))
    # progress_value = 70

    # score = score

    # Define the wanted and unwanted words
    wanted_words = [i[0] for i in matched]
    unwanted_words = [i[0] for i in nonmatched]

    # Define the layout for the wanted and unwanted words
    css = """
    <style>
    body {
        background-color: #f2f2f2;
    }
    .wanted-words {
        background-color: #c8f7c5;
        padding: 10px;
        border-radius: 5px;
        font-size: 20px;
    }
    .unwanted-words {
        background-color:  	#fb6767;
        padding: 10px;
        border-radius: 5px;
        font-size: 20px;
    }

    .match-rate {
    text-align:center;
        color: #6DA9E4;
        font-size: 36px;
        font-weight: bold;
        margin: 20px 0;
    }

    h1, h2 {
        text-align: center;
        color: #555;
        margin: 10px;
    }
    button {
        display: block;
        margin: 20px auto;
    }
    </style>
    """


    # Create the Streamlit app
    st.title("Resume Match")
    # st.write(f"Resume Name: {resume_name}")
    st.markdown(css, unsafe_allow_html=True)

    with st.spinner('Calculating...'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i in range(score + 1):
            progress_bar.progress(i)
            time.sleep(0.03)
            status_text.text(f"{i}%")
        status_text.text("Done!")
    st.markdown(f"<div class='match-rate'>Match Rate: {score}%</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 3])
    with col1:
        st.header("Matched Words:")
        for i in wanted_words:
            st.markdown(f"<div class='wanted-words'>{i}</div>", unsafe_allow_html=True)

    with col2:
        st.header("Missing Words:")
        for i in unwanted_words:
            st.markdown(f"<div class='unwanted-words'>{i}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="text-align: center;">'
                '<button onclick="window.history.back();" '
                'style="background-color: #6DA9E4; '
                'color: white; padding: 10px 24px; border: none; border-radius: 5px; cursor: pointer;">Go back</button></p>'
                , unsafe_allow_html=True)


if __name__ == "__main__":
    app()
