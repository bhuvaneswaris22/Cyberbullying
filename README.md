

# 🛡️ Cyberbullying Detection in Social Media Platforms using Deep Learning

This project detects cyberbullying content on social media platforms like **Twitter** and **Wikipedia** using advanced **Natural Language Processing (NLP)** and **Deep Learning (DL)** techniques. It helps to identify harmful text such as hate speech or personal attacks in real-time and supports actions to reduce online abuse.


## 
## 🔍 Key Features

- Real-time detection of cyberbullying
- Supports Twitter and Wikipedia-based text datasets
- Feature extraction using:
  - Bag of Words (BoW)
  - TF-IDF
  - N-grams
- Multiple classification models:
  - Logistic Regression
  - SVM
  - Random Forest
  - Deep Learning Models
- Profanity filtering using `better_profanity`
- Admin & user login system
- Visual performance graphs for each model


##  🎯 Objectives

- Automatically detect cyberbullying in social media posts
- Classify tweets/comments using ML classifiers
- Analyze and visualize model performance
- Support real-time detection and moderation
- Provide a user-friendly web interface
##  🔤 Text Preprocessing
- Stopword removal
- Lowercasing
- Tokenization
- Stemming
- Punctuation & number removal
## 🛠️ Tech Stack

| Layer       | Tools & Frameworks                     |
|-------------|-----------------------------------------|
| Language    | Python                                  |
| Web Framework | Flask                                 |
| Frontend    | HTML, CSS                               |
| DL/NLP      | Scikit-learn, NLTK, Pandas, NumPy       |
| Data Visualization | Matplotlib                       |
| Database    | MySQL                                   |


## 🖼️ Screenshots

![1](https://github.com/user-attachments/assets/628d2954-4d6e-4619-a16b-42d39294df3d)


![2](https://github.com/user-attachments/assets/5fce2374-9f71-42b0-871b-7a604ea5d8f8)


![3](https://github.com/user-attachments/assets/d5a4fc22-cbde-4559-b7cf-9b97c030b302)
## ⚙️ Installation

1. Clone the repository

git clone https://github.com/your-username/cyberbullying-detection.git
cd cyberbullying-detection


2. (Optional) Create and activate a virtual environment

python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows


3. Install the required Python libraries

pip install -r requirements.txt


4. Set up the MySQL database

Create a database named cyber

Update your MySQL credentials in app.py:

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="cyber"
)



5. Run the Flask web application

python app.py


6. Open the app in your browser

http://127.0.0.1:2000


7. ✅ Done! You can now:

Register/Login as a user or admin

Post tweets

Upload datasets

View model predictions and graphs
    
