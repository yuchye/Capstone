# Importing of libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
import regex as re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from flask import Flask, jsonify, request
from joblib import load

app = Flask(__name__)

# Load the various joblib objects previously saved
ss = load("ss.joblib") # StandardScaler
pca = load("pca.joblib") # PCA
model = load("baseline_clf.joblib") # Baseline classifier

# Load the complete set of features used for training
model_cols = pd.read_csv("model_cols.csv", header=None, names=['feature'])

# We define a function to preprocess the text
def preprocess(input_string):
    text_lower = BeautifulSoup(input_string).get_text().lower()
    return re.sub("[^a-zA-Z]", " ", text_lower) # we lower the case, and remove punctuation

# We define a function to perform POS (Parts-of-Speech) tagging which should improve the lemmatisation output
def get_wordnet_pos(word):
    # Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN) # Return wordnet.NOUN if the 'tag' key is not found

def lemmatize(input_string):

    # Lemmatise each word of text
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input_string)
    lemmatizer = WordNetLemmatizer()
    tokens_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]

    # Remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    new_stop_words = ['mutation', 'figure', 'fig', 'tumor', 'cell', 'patient', \
                     'use', 'et', 'al', 'also', 'cancer', 'show']
    stopwords.extend(new_stop_words)
    stops = set(stopwords)
    meaningful_words = [w for w in tokens_lem if not w in stops]

    # Join the words back into one string separated by space, and return the result
    return(" ".join(meaningful_words))

@app.route('/predict-class', methods=['POST'])
def predict_class():

    input_gene = request.form["gene"]
    input_variation = request.form["variation"]
    input_text = request.form["text"]

    # input_gene = request.args.get("gene")
    # input_variation = request.args.get("variation")
    # input_text = request.args.get("text")

    if (input_gene and input_variation and input_text):

        # Create a dataframe comprising the features and initialise it
        # We will use the dataframe to make our prediction
        predict_df = pd.DataFrame(columns=model_cols['feature'])
        predict_df.loc[0] = 0

        text_preproc = preprocess(input_text)
        lemm_text = lemmatize(text_preproc)

        # populate the prediction dataframe
        try:
            predict_df.loc[0, ['gene_'+input_gene]] = 1
        except:
            pass
        try:
            predict_df.loc[0, ['variation_'+input_variation]] = 1
        except:
            pass

        # Create a word frequency dictionary
        word_freq = {}
        for word in lemm_text.split():
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1

        # Populate our prediction dataframe with the word counts
        # from the word frequency dictionary
        for word, freq in word_freq.items():
            try:
                predict_df.loc[0, [word]] = freq
            except:
                pass

        # Transform the prediction dataframe using our previously fitted StandardScaler
        predict_df = ss.transform(predict_df)

        # Transform the scaled dataframe using our fitted PCA
        # and use the first 1,800 principle components
        predict_df = pca.transform(predict_df)
        predict_df = pd.DataFrame(predict_df)
        predict_df = predict_df.iloc[:,:1800]

        # Make the prediction!
        pred_class = str((model.predict(predict_df))[0])

        result = {
            "response": "ok",
            "predicted_class": pred_class
        }
    else:
        result = {
            "response": "not found",
            "message": "Please provide gene, variation and text parameters to predict!"
        }

    return jsonify(result)

@app.route('/')
def placeholder():
    return 'This is just a placeholder!'
