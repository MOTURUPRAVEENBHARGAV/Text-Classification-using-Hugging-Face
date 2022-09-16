
from flask import Flask,redirect, url_for, render_template,request,jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from model_build import model_predict
from labels import change_labels
import warnings
warnings.filterwarnings("ignore")
#WSGI Application
app= Flask(__name__) #Flask App Object

#Decorator
@app.route('/') #Home Page
def pop_results():
    tokenizer = AutoTokenizer.from_pretrained("./model")
    corpus = "Question1","Self","Good"
            "Question2","Manager","Bad"
    multi_model = TFAutoModelForSequenceClassification.from_pretrained(r"./model",num_labels=5)
    pred = model_predict(multi_model,tokenizer,lis=corpus)
    pred = change_labels(pred)
    return jsonify({"Results":pred})

@app.route('/postData', methods=['POST', 'GET'])
def postData():
    content = request.data
    if request.method == "POST":
        science = 1
    return content


#Read the data
# manager_data= pd.read_excel(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model_questions.xlsx",sheet_name='Manager')
# peer_data = pd.read_excel(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model_questions.xlsx",sheet_name='Peers')
# self_data = pd.read_excel(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model_questions.xlsx",sheet_name='Self')


if __name__ == '__main__':
    app.run(debug = True)





# @app.route('/login', methods = ["POST"]) #login Page
# def login():
#     username = request.form("fname")
#     passoword = request.form("password")
#     return "Welcome %s" %username

