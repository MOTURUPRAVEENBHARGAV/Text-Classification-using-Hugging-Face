import tensorflow as tf
import numpy as np
def model_predict(model,tokenizer,data=None,col=None,lis=None):
    if lis:
        corpus = list(lis)
        #corpus = data[col].tolist()
        encoding = tokenizer(corpus,padding=True,truncation=True,max_length=256,return_tensors= 'tf')
        tf_outputs = model(encoding)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels=np.argmax(tf_predictions,axis=-1)
        return labels
        
    else:
        corpus = data[col].tolist()
        encoding = tokenizer(corpus,padding=True,truncation=True,max_length=256,return_tensors= 'tf')
        tf_outputs = model(encoding)
        tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
        labels=np.argmax(tf_predictions,axis=-1)
        return labels