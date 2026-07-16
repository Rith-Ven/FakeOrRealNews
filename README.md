# AI Fake News Detector

An intelligent, machine learning-powered web application that analyzes news articles to determine their authenticity. This application uses natural language processing (NLP) to scrape articles, generate summaries, extract key keywords, and run predictions using a custom-trained Scikit-Learn classifier.

## Live Demo
Access the live web application here:  
**[AI Fake News Detector Live App](https://realorfakenewsdetector.streamlit.app/)**

---

## Features

- **Dual-Input Analysis:**
  - **Analyze via URL:** Paste any news article link. The system automatically fetches, parses, and cleans the text.
  - **Analyze via Text (Fallback):** Paste raw text directly for instant evaluation (perfect for bypassed paywalls or highly secured sites).
- **Anti-Scraping Bypass & Resiliency:**
  - Integrates `cloudscraper` to bypass strict browser/bot firewalls (like Cloudflare) used by major publishers (e.g., *The Washington Post*, *The New York Times*).
  - Features an **Internet Archive (Wayback Machine)** API fallback engine. If a live article is blocked, the app automatically attempts to fetch its latest archived snapshot.
- **NLP Text Insights:**
  - **Key Topic Badges:** Automatically identifies and highlights the most important keywords in the text.
  - **Article Summary:** Generates an AI-driven, TL;DR executive summary directly on your dashboard.
  - **Raw Scraped Text Viewer:** Toggleable expander to review exactly what the AI read.
- **Robust Caching:** Leverages Streamlit’s `@st.cache_resource` to ensure instant model loads and seamless NLTK resource checking on startup.

---

## Tech Stack

- **Frontend Dashboard:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (TF-IDF Vectorizer + Classification Model)
- **Web Scraping & Parsing:** [newspaper4k](https://github.com/miso-belica/newspaper/) & [cloudscraper](https://github.com/VeNoMouS/cloudscraper)
- **Natural Language Processing:** [NLTK](https://www.nltk.org/) (Natural Language Toolkit)

---

## Project Structure

```bash
FakeOrRealNews/
├── app.py                 # Streamlit frontend & web scraping logic
├── train_and_save.py      # Model training script (saves .pkl files)
├── model.pkl              # Saved trained machine learning model
├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── requirements.txt       # Python package dependencies
└── .gitignore             # Prevents venv/ and heavy model uploads to GitHub
Local Installation & Setup
If you want to run this project locally on your machine, follow these steps:

1. Clone the Repository
Bash
git clone https://github.com/Rith-Ven/FakeOrRealNews.git
cd fakeorrealnews
2. Create and Activate a Virtual Environment
macOS / Linux:

Bash
python3 -m venv venv
source venv/bin/activate
Windows (PowerShell):

PowerShell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Streamlit App
Bash
python -m streamlit run app.py
Deployment Configuration (Streamlit Community Cloud)
To deploy this application seamlessly online:

Ensure your .gitignore is active and ignores the local virtual environment folder (venv/).

Push your code repository to GitHub (excluding local venv/ files).

Connect your repository to Streamlit Community Cloud.

Crucial: In your App Settings on the Streamlit dashboard, set your Python Version to 3.11 (or 3.10). This avoids compilation errors with packages like lxml on newer, uncompiled environments.

License
This project is open-source and available under the MIT License.