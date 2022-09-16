from tqdm import tqdm
import numpy as np
def change_labels(arr):
  l=[]
  dic = {0: "More Negative",1 :"Negative",2:"Neutral",3:"Positive",4: "More Positive"}
  for i in tqdm(arr):
    l.append(dic[i])
  return l