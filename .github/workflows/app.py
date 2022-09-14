import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from model_build import model_predict
from labels import change_labels
import warnings
warnings.filterwarnings("ignore")

#Read the data
# manager_data= pd.read_excel(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model_questions.xlsx",sheet_name='Manager')
# peer_data = pd.read_excel(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model_questions.xlsx",sheet_name='Peers')
# self_data = pd.read_excel(r"D:\DESKT\INTERNSHIPS\Talent Spotify\NLP Tasks\Sentiment Analysis\bert pretrained Model\model_questions.xlsx",sheet_name='Self')

tokenizer = AutoTokenizer.from_pretrained(r"D:/DESKT/INTERNSHIPS/Talent Spotify/NLP Tasks/Sentiment Analysis/bert pretrained Model")

corpus = ["This meeting is adorable and wonderful","You are looking so furious"]

multi_model = TFAutoModelForSequenceClassification.from_pretrained(r"D:/DESKT/INTERNSHIPS/Talent Spotify/NLP Tasks/Sentiment Analysis/bert pretrained Model",num_labels=5)
pred = model_predict(multi_model,tokenizer,lis=corpus)
pred = change_labels(pred)
print(pred)
