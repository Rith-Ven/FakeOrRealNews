import streamlit as st
import pickle

st.set_page_config(page_title = "Fake News Detector" page_icon = "📰", layout = "centered")

st.write("Enter an article or headline below to analyze its credibility.")

# Load in the saved model and vectorizer from train_and_save.py

# Keeps the model IN MEMORY so it doesn't reload on each click
@st.cache_resource 

def load_assets():
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    with open('model.pkl','rb') as model_file:
        model = pickle.load(model_file)
    return vectorizer, model

try: 
    vectorizer, model = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Run 'train_and_save.py' first.")
    st.stop()

#Text input
user_input = st.text_area("Enter News Article here:", height = 200, placeholder = "Type or paste news content...")

#Predict button
if st.button("Analyze Article"):
    if user_input.strip() == "":
        st.warning("Enter text before analyzing.")
    else:
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        st.write("---")
        if prediction == "REAL":
            st.success("This news looks REAL!")
        else:
            st.error("This news looks FAKE.")
        