from time import time
from flask import Flask,redirect, url_for, render_template,request,jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from model_build import model_predict
from labels import change_labels
import warnings
warnings.filterwarnings("ignore")
import json

#load the MODEL 
tokenizer = AutoTokenizer.from_pretrained(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model")
multi_model = TFAutoModelForSequenceClassification.from_pretrained(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model",num_labels=5)
#WSGI Application
app= Flask(__name__) #Flask App Object

#Decorator
@app.route('/') #Home Page
def welcome():
    return "Welcome"

@app.route('/model',methods=['POST', 'GET']) #Model Page
def pop_results():
    # data = request.json()
    # data= req.content
    if request.method == "POST":
        data = request.data
        data= json.loads(data)
        print(data)
    # data = {"inputs":[{"Question":"",
    #             "Employer":"Manager",
    #             "Response":"Good"},
    #              {"Question":" ",
    #               "Employer": "Peers",
    #               "Response":"Behaviour is very furious"},
    #              {"Question":"",
    #              "Employer": "Self",
    #               "Response":"Very Rude to colleagues"}
    #             ]}
        l=[]                
        for input in data["inputs"]:
            l.append(input["Response"])
        pred = model_predict(multi_model,tokenizer,lis=l)
        pred = change_labels(pred)
        for i in range(0,len(l)):
            data["inputs"][i]["Polarity"] = pred[i]
    print(data)

    return data



    # multi_model = TFAutoModelForSequenceClassification.from_pretrained(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model",num_labels=5)
    # pred = model_predict(multi_model,tokenizer,lis=corpus)
    # pred = change_labels(pred)
    # return jsonify({"Results":pred})

@app.route('/postData', methods=['POST', 'GET'])
def postData():
    req = request.get("URL")
    content = req.content
    if request.method == "POST":
        l=[]                
        for input in req["inputs"]:
            l.append(input["Response"])
        pred = model_predict(multi_model,tokenizer,lis=l)
        pred = change_labels(pred)
        for i in range(0,len(l)):
            req["inputs"][i]["Polarity"] = pred[i]

    return content

if __name__ == '__main__':
    app.run(debug = True)





# @app.route('/login', methods = ["POST"]) #login Page
# def login():
#     username = request.form("fname")
#     passoword = request.form("password")
#     return "Welcome %s" %username

