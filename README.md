<div align="center">
  <h1>🔥 Algerian Forest Fire Prediction API</h1>
  <h3>An End-to-End ML Pipeline Built with Flask & AWS</h3>

  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
  [![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey.svg)](https://flask.palletsprojects.com/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-orange.svg)](https://scikit-learn.org/)
  [![AWS Elastic Beanstalk](https://img.shields.io/badge/AWS-Elastic_Beanstalk-FF9900.svg)](https://aws.amazon.com/elasticbeanstalk/)
</div>

---

## 📺 AWS Deployment & Demo

*(Note: To avoid recurring cloud billing, the live AWS Elastic Beanstalk instance has been spun down. However, you can see the fully working deployed app and CI/CD pipeline in action below.)*

**🔗 [Watch the Deployment Demo Video Here]** *(https://drive.google.com/file/d/1btZchLDUWf5uDWYBlGo-IjTPLp8N1obW/view?usp=drive_link)*

**A quick note on the UI:** I wasn't a fan of the default, boring form templates you usually see in ML projects. I used GenAI (Gemini) to help me custom-style the frontend, making it look much cleaner and more responsive.

<img width="557" height="350" alt="Screenshot 2026-04-23 at 00 27 46" src="https://github.com/user-attachments/assets/01346d5a-90ef-4401-8f78-5fa154a48c7e" />
https://docs.google.com/document/d/1FwrPpFGJTHdVHTz-skB7Asvx7BeZr3SlvRk66EgNOg0/edit?usp=sharing*

---

## 📖 What is this project?
I built this project to predict the **Fire Weather Index (FWI)** for the Algerian Forest Fires dataset. But my main goal wasn't just to train a model in a Jupyter Notebook and call it a day. I wanted to build a complete pipeline—from handling messy raw data to deploying a working API on the cloud.

### Why I chose this tech stack:
* **Ridge Regression:** I tested a few models (Lasso, ElasticNet), but Ridge handled the multicollinearity in this specific dataset the best, giving me a solid 98.4% accuracy score.
* **Flask:** It’s lightweight and does exactly what I need to serve the model without unnecessary bloat.
* **AWS CodePipeline & Elastic Beanstalk:** I wanted to set up a real CI/CD flow. Any code pushed to the `main` branch of this repo automatically triggered a build and deployed straight to Beanstalk.

---

## 🏗️ How it works (The Pipeline)

1. **Data Cleaning:** The raw CSV was pretty messy (e.g., hidden spaces in column headers, string values in numerical columns, and that one corrupt row 122). I cleaned it up and mapped categorical variables.
2. **Feature Engineering:** I checked the Pearson correlation matrix. To avoid overfitting, I set a strict `0.85` correlation threshold and dropped highly overlapping features (like `BUI` or `DC`). Then, I scaled everything using `StandardScaler`.
3. **Training & Pickling:** Trained the Ridge model, pickled both the model (`ridge.pkl`) and the scaler (`scaler.pkl`), and saved them in the `/models` directory.
4. **The Web API:** Wrote a Flask backend to take user inputs from the UI, transform them through the scaler, and return the FWI prediction.

---

## 🧪 Testing & Validation (My QA Background)

Coming from a Quality Engineering and SDET background, I can't just write code and hope it works. I made sure to add some reliability checks:
* **Edge Cases:** The Flask routes and frontend are structured so that missing or weird boundary-value inputs won't just crash the app.
* **Pipeline Checks:** Monitored the AWS CodePipeline logs to ensure health markers stayed green during deployment updates.

---

## 💻 Run it locally

If you want to pull this down and run it on your own machine:

```bash
# 1. Clone it
git clone [https://github.com/Ravinder-Labs/My_ML_Project.git](https://github.com/Ravinder-Labs/My_ML_Project.git)
cd My_ML_Project

# 2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install the requirements
pip install -r requirements.txt

# 4. Start the Flask server
python application.py

---
### 📝 Detailed Project Breakdown
I have written a comprehensive guide on how I handled data cleaning and multicollinearity for this project. 
**👉[Read the full article on Hashnode](https://ravinderlabs.hashnode.dev/breaking-down-multicollinearity-how-i-cleaned-messy-data-for-an-ml-pipeline)**
