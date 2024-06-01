# -*- coding: utf-8 -*-
# coding: utf-8
from flask import Flask, render_template, request
import os
import io
import numpy
from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

#Function to read files (emails) from the local directory
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

#An empty dataframe with 'message' and 'class' headers
data = pd.DataFrame(columns=['message', 'class'])

#Including the email details with the spam/ham classification in the dataframe
dfs = [dataFrameFromDirectory('C:\\Users\\vaish\\Downloads\\Email-Spam-Classifier-Using-Naive-Bayes-2\\emails\\spam', 'spam'),
       dataFrameFromDirectory('C:\\Users\\vaish\\Downloads\\Email-Spam-Classifier-Using-Naive-Bayes-2\\emails\\ham', 'ham')]

data = pd.concat(dfs, ignore_index=True)

vectoriser = CountVectorizer()
count = vectoriser.fit_transform(data['message'].values)
target = data['class'].values

classifier = MultinomialNB()
classifier.fit(count, target)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        excount = vectoriser.transform([message])
        prediction = classifier.predict(excount)
    return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
