import time
import json
from better_profanity import profanity
from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
import nltk
import re, string
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, roc_curve, classification_report
from imblearn.over_sampling import RandomOverSampler

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

user_data_file = 'users.json'
tweets_file = 'tweets.json'
vulgar_words_count = {}
account_blocked_until = 0
permanent_blocked = False

def load_users():
    if os.path.exists(user_data_file):
        with open(user_data_file, 'r') as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(user_data_file, 'w') as file:
        json.dump(users, file)

def load_tweets():
    if os.path.exists(tweets_file):
        with open(tweets_file, 'r') as file:
            return json.load(file)
    return []

def save_tweets(tweets):
    with open(tweets_file, 'w') as file:
        json.dump(tweets, file)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def user_login():
    global data1
    if request.method == 'POST':
        data1 = request.form.get('name')
        data2 = request.form.get('password')

        print("Username:", data1)  # Debug statement
        print("Password:", data2)  # Debug statement

        if data2 is None:
            return render_template('login.html', msg='Password not provided')

        users = load_users()
        if data1 in users and users[data1]['password'] == data2:
            session["uname"] = data1
            return redirect(url_for('about'))
        else:
            return render_template('login.html', msg='Invalid username or password')

@app.route('/NewUser')
def newuser():
    return render_template('NewUser2.html')

@app.route('/reg', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        password = request.form.get('psw')

        users = load_users()
        if name in users:
            return render_template('NewUser2.html', msg='Username already exists')

        users[name] = {'phone': phone, 'password': password}
        save_users(users)
        return render_template('login.html', msg='Registration successful')
    else:
        return render_template('NewUser2.html')

@app.route('/twitter')
def twitter():
    uname = session.get('uname')  # Retrieve username from session
    users = load_users()
    followers = [f for f in users if users[f].get('following') == uname]
    non_followers = [u for u in users if u != uname and users[u].get('following') != uname]

    return render_template('twitter.html', btn_value='Remove', btn_value1='Follow', data=followers, data1=non_followers)

@app.route('/follow', methods=['POST', 'GET'])
def follow():
    if request.method == 'POST':
        name = request.form.get('name')
        status = request.form.get('status')
        uname = session.get('uname')

        users = load_users()
        if status == 'Follow':
            users[name]['following'] = uname
        else:
            users[name].pop('following', None)

        save_users(users)
        return redirect('twitter')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/send', methods=['POST', 'GET'])
def send():
    global account_blocked_until, permanent_blocked, vulgar_words_count, user_name

    if request.method == 'POST':
        msg = request.form.get('msg')
        censored = profanity.censor(msg)
        now = time.time()

        data1 = session.get('uname')
        user_name = data1
        print("data", data1)
        print("your user name ", user_name)

        if permanent_blocked:
            return render_template('twitter.html', view='style=display:block', value='Your account is blocked due to inappropriate behavior.')

        if account_blocked_until > now:
            return render_template('twitter.html', view='style=display:block', value=f'Your account is temporarily blocked until {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(account_blocked_until))} due to inappropriate behavior.')

        if '*' in censored:
            for word in censored.split():
                if '*' in word:
                    vulgar_words_count[word] = vulgar_words_count.get(word, 0) + 1
                    if vulgar_words_count[word] == 2:
                        return render_template('twitter.html', view='style=display:block', value='Hello user! You have used inappropriate language twice. Please refrain from using such language.')
                    elif vulgar_words_count[word] >= 3:
                        account_blocked_until = now + (5 * 24 * 60 * 60)
                        users = load_users()
                        users.pop(user_name, None)
                        save_users(users)
                        vulgar_words_count.clear()
                        return render_template('login.html', view='style=display:block', value='Hello user! You have used inappropriate language thrice. Please refrain from using such language.')

            return render_template('twitter.html', view='style=display:block', value='Hello user! You have used inappropriate language. Please refrain from using such language.')

        if not permanent_blocked:
            tweets = load_tweets()
            tweets.append({'name': data1, 'date': now, 'tweet': msg})
            save_tweets(tweets)
            return render_template('twitter.html', view='style=display:block', value='Post Tweeted')

    return redirect(url_for('twitter'))

@app.route('/delete_account/<username>')
def delete_account(username):
    users = load_users()
    users.pop(username, None)
    save_users(users)
    return redirect(url_for('login'))

@app.route('/tweet')
def tweet():
    tweets = load_tweets()
    if tweets:
        return render_template('tweet.html', data=tweets)
    return render_template('tweet.html', msg='No tweets')

@app.route('/upload.html')
def up():
    return render_template('upload.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    global df
    if request.method == 'POST':
        if os.path.exists('static/file/perform.png'):
            os.remove('static/file/perform.png')
        if os.path.exists('static/file/abc.png'):
            os.remove('static/file/abc.png')
        if os.path.exists('static/file/dtc.png'):
            os.remove('static/file/dtc.png')
        if os.path.exists('static/file/gnb.png'):
            os.remove('static/file/gnb.png')
        if os.path.exists('static/file/lgr.png'):
            os.remove('static/file/lgr.png')
        if os.path.exists('static/file/rfc.png'):
            os.remove('static/file/rfc.png')
        file1 = request.files['jsonfile']
        if file1:
            jsonfile = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(jsonfile)
        else:
            jsonfile = 'static/file/Dataset.json'
        df = pd.read_json(jsonfile)
        for i in range(0, len(df)):
            if df.annotation[i]['label'][0] == '1':
                df.annotation[i] = 1
            else:
                df.annotation[i] = 0
        df.drop(['extras'], axis=1, inplace=True)
        df['annotation'].value_counts().sort_index().plot.bar()
        plt.savefig('static/file/perform.png')

        # pre processing
        nltk.download('stopwords')
        stop = stopwords.words('english')
        regex = re.compile('[%s]' % re.escape(string.punctuation))

        def test_re(s):
            return regex.sub('', s)

        df['content_without_stopwords'] = df['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        df['content_without_puncs'] = df['content_without_stopwords'].apply(lambda x: regex.sub('', x))
        del df['content_without_stopwords']
        del df['content']

        # Stemming
        porter_stemmer = PorterStemmer()
        nltk.download('punkt')
        tok_list = []
        size = df.shape[0]
        for i in range(size):
            word_data = df['content_without_puncs'][i]
            nltk_tokens = nltk.word_tokenize(word_data)
            stem_words = [porter_stemmer.stem(w) for w in nltk_tokens]
            tok_list.append(stem_words)
        df['final_text'] = tok_list

        df.to_json('static/file/output.json')
        return render_template('select.html')

@app.route('/download')
def download():
    p = 'static/file/output.json'
    return send_file(p, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
