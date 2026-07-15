import streamlit as st
import pickle

st.set_page_config(page_title = "Fake News Detector", page_icon = "📰", layout = "centered")

st.write("Analyze news articles by pasting a URL link or the whole article below.")

# Load in the saved model and vectorizer from train_and_save.py

# Keeps the model IN MEMORY so it doesn't reload on each click
@st.cache_resource 
@st.cache_resource
def download_nltk_data():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()

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
#User_input = st.text_area("Enter News Article here:", height = 200, placeholder = "Type or paste news content...")

tab1, tab2 = st.tabs(["Analyze via URL", "Analyze via Text"])
# Prior to trying to handle firewalls
# with tab1:
    
#     with st.form(key = "url_form"):
#         url_input = st.text_input("Paste News Article URL", placeholder = "https://www.example.com/new-article")
#         submit_button = st.form_submit_button(label = "Fetch and Analyze URL")
    
#     if submit_button:
#         if not url_input.strip():
#             st.warning("Please enter a valid URL.")
#         else:
#             with st.spinner("Fetching article content..."):
#                 try:
#                     import newspaper

#                     config = newspaper.Config() #Create custom configuration to bypass bot blockers
#                     #Tells the website that we are a normal Chrome browser, not a Python script
#                     config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
#                     config.request_timeout = 20

#                     #Initialize article and fetch its content
#                     article = newspaper.Article(url_input.strip(), config = config)
#                     article.download()
#                     article.parse()
#                     #Run NLP for summary and keywords
#                     article.nlp()
#                     article_text = article.text

#                     if not article_text or not article_text.strip():
#                         st.error("Could not extract any text from this URL.")
#                     else:
#                         st.info(f"**Extracted Title:** {article.title}")
#                         st.write("---")
#                         transformed_input = vectorizer.transform([article_text])
#                         prediction = model.predict(transformed_input)[0]

#                         if prediction == "REAL":
#                             st.success("This news looks **REAL**")
#                         else:
#                             st.error("Warning: This news looks **FAKE**")
#                         st.write("---")
#                         st.subheader(f"{article.title}")
                        
#                         if article.keywords:
#                             st.write("**Key Topics**: " + " ".join([f"'{kw}'" for kw in article.keywords[:8]]))
                        
#                         if article.summary:
#                             with st.expander("Show Article Summary", expanded = True):
#                                 st.write(article.summary)

                        
#                         with st.expander("Show scraped text snippet"):
#                             st.write(article_text[:1000] + "...")
                        
#                 except Exception as e:
#                     st.error(f"Error fetching article. Check the URL or try pasting the raw text. (Error: {e})")
# 
with tab1:
    with st.form(key="url_form"):
        url_input = st.text_input("Paste News Article URL", placeholder="https://www.example.com/new-article")
        submit_button = st.form_submit_button(label="Fetch and Analyze URL")
        
    if submit_button:
        if not url_input.strip():
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Bypassing firewall and retrieving article..."):
                target_url = url_input.strip()
                html_content = None
                used_archive = False
                
                # --- TRY 1: Direct Safe Fetch with Cloudscraper ---
                try:
                    import cloudscraper
                    scraper = cloudscraper.create_scraper(
                        browser={
                            'browser': 'chrome',
                            'platform': 'windows',
                            'desktop': True
                        }
                    )
                    response = scraper.get(target_url, timeout=10)
                    if response.status_code == 200:
                        # Ensure we didn't just get a blank block page
                        if "captcha" not in response.text.lower() and len(response.text) > 2000:
                            html_content = response.text
                except Exception:
                    pass

                # --- TRY 2: Internet Archive (Wayback Machine) Fallback ---
                if not html_content:
                    try:
                        used_archive = True
                        # We query the Archive.org API to get the closest saved snapshot
                        import requests
                        archive_api = f"https://archive.org/wayback/available?url={target_url}"
                        archive_resp = requests.get(archive_api, timeout=5).json()
                        
                        if archive_resp.get("archived_snapshots", {}).get("closest", {}).get("available"):
                            snapshot_url = archive_resp["archived_snapshots"]["closest"]["url"]
                            
                            # Fetch the HTML from the archive snapshot
                            fallback_scraper = cloudscraper.create_scraper()
                            response = fallback_scraper.get(snapshot_url, timeout=10)
                            if response.status_code == 200:
                                html_content = response.text
                    except Exception as fallback_error:
                        pass

                # --- PROCESS THE EXTRACTED HTML ---
                if html_content:
                    try:
                        import newspaper
                        article = newspaper.Article(target_url)
                        article.download(input_html=html_content)
                        article.parse()
                        
                        # Run NLP for summary and keywords
                        article.nlp()
                        article_text = article.text

                        if not article_text or not article_text.strip():
                            st.warning("Could not extract clean text. Please try Tab 2 and paste the raw text.")
                        else:
                            if used_archive:
                                st.caption("*Bypassed firewall using Internet Archive (Wayback Machine).*")
                            
                            st.write("---")
                            
                            # Step 1: Run AI Prediction
                            transformed_input = vectorizer.transform([article_text])
                            prediction = model.predict(transformed_input)[0]

                            # Display prediction banner
                            if prediction == "REAL":
                                st.success("### Prediction: This news looks **REAL**")
                            else:
                                st.error("### Warning: This news looks **FAKE**")
                            
                            st.write("---")
                            
                            # Step 2: Display Extracted Info and Summary
                            st.subheader(f"📋 Title: {article.title}")
                            
                            # Display keywords as styled tags/pills
                            if article.keywords:
                                st.write("**Key Topics:** " + " ".join([f"`{kw}`" for kw in article.keywords[:8]]))
                            
                            # Display Summary
                            if article.summary:
                                with st.expander("Show Article Summary", expanded=True):
                                    st.write(article.summary)
                            
                            # Original Text Snip
                            with st.expander("Show raw scraped text"):
                                st.write(article_text[:1000] + "...")
                                
                    except Exception as parse_error:
                        st.error(f"Parsing failed: {parse_error}")
                else:
                    st.error("This article is strictly protected by a firewall or paywall and is not archived yet. Please go to **Tab 2** and paste the raw text manually.")
with tab2:
    user_input = st.text_area("Paste news article text here: ", height=200, placeholder = "Type or paste text here...")

    if st.button("Analyze raw text"):
        if not user_input.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            transformed_input = vectorizer.transform([user_input])
            prediction = model.predict(transformed_input)[0]
            st.write("---")
            if prediction == "REAL":
                st.success("This news looks **REAL**")
            else:
                st.error("Warning: This news looks **FAKE**")
#Predict button
# if st.button("Analyze Article"):
#     if user_input.strip() == "":
#         st.warning("Enter text before analyzing.")
#     else:
#         transformed_input = vectorizer.transform([user_input])
#         prediction = model.predict(transformed_input)[0]
#         st.write("---")
#         if prediction == "REAL":
#             st.success("This news looks REAL!")
#         else:
#             st.error("Careful! This news looks FAKE.")
        