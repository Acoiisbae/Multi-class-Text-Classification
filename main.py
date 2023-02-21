#%%
import os
import re
import json
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Dropout, LSTM, Dense, Embedding, Bidirectional
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

model_path = os.path.join(os.getcwd(),'model','model_h5')
ohe_path = os.path.join(os.getcwd(),'model','ohe.pkl')
tokenizer_path = os.path.join(os.getcwd(),'model','tokenizer.json')

#%% Data Loading

df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv')
text = df['text'] #features
category = df['category'] #target

#%% EDA

df.head()
df.tail()
print(df['text'][8])

#%% Data Cleaning
for index, temp in enumerate(text):
    temp = re.sub('\[.*?\]', '', temp)
    # clean numbers and punctuation
    temp = re.sub('[^a-zA-Z]',' ',temp)
    
    text[index] = temp.lower()

#%% Data Preprocessing

vocab_num = 5000
oov_token = '<OOV>' # OUT OF VOCAB
tokenizer = Tokenizer(num_words=vocab_num,oov_token=oov_token)

#training the tokenizer to learn the words
tokenizer.fit_on_texts(text)

# view the converted data
text_index = tokenizer.word_index
print(list(text_index.items())[0:10])

# to convert the text into numbers
text = tokenizer.texts_to_sequences(text)
# %%
maxlen = []
for i in text:
    maxlen.append(len(i))

maxlen = int(np.ceil(np.percentile(maxlen, 75)))

# padding and truncating
text = pad_sequences(text,maxlen=maxlen,padding='post',truncating='post')

# OHE --> tARGET
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

# %% Train Test Split
X_train,X_test,y_train,y_test = train_test_split(text,category,train_size=0.7,shuffle = True, random_state=123)

#%% Model Development
nb_classes = len(np.unique(y_train, axis=0))
embedding_dims = 64
model = Sequential()
model.add(Embedding(vocab_num,embedding_dims))
model.add(Bidirectional(LSTM(embedding_dims)))
model.add(Dropout(0.3))
model.add(Dense(nb_classes,activation = 'softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics = 'accuracy')


#%% tensorboard
log_dir = os.path.join(os.getcwd(),datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_callback = TensorBoard(log_dir=log_dir)

#early stopping
es_callback = EarlyStopping(monitor='loss',patience=5)
hist = model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test),callbacks=[tb_callback,es_callback])

# %%Model Analysis
print(hist.history.keys())

def plot_hist(hist):
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

    plt.figure()
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.show()


#%% Model deployment
y_pred = model.predict(X_test)
y_true = y_test

y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_true,axis=1)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

# %%
model.save(model_path)

with open(ohe_path,'wb')as f:
    pickle.dump(ohe,f)

token_json = tokenizer.to_json()
with open(tokenizer_path,'w')as json_file:
    json.dump(token_json,json_file)

