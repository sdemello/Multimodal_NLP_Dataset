import re
import string
from bs4 import BeautifulSoup
import os
import pandas as pd
import numpy as np
import sys
import contractions
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# 1) Text Cleaning
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove html tags
    return text

trim = True
if trim is True:
    text_file = "C:/Users/User/OneDrive/Documents/ResearchProjectTrimmedText.csv"
else:
    text_file = "C:/Users/User/OneDrive/Documents/ResearchProjectText.csv"
text_pd = pd.read_csv(text_file, index_col=0)
text_transcript = text_pd["Transcript Text"]
text_transcript_list = text_transcript.tolist()
print("text_transcript_list")
print(text_transcript_list)
text_transcript_list = [str(doc) for doc in text_transcript_list]

print("\n\n\ncleaned_text_transcript_list")
cleaned_text_transcript_list = [clean_text(doc) for doc in text_transcript_list]
[print(clean_text(doc)) for doc in text_transcript_list]


# 2) Contractions
expanded_corpus = [contractions.fix(doc) for doc in cleaned_text_transcript_list]
print("\n\n\nexpanded_corpus")
print(expanded_corpus)


# 3) Tokenization
tokenized_corpus = [word_tokenize(doc) for doc in expanded_corpus]
print("\n\n\ntokenized_corpus")
print(tokenized_corpus)


# 4) Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_corpus = [[word for word in doc if word not in stop_words] for doc in tokenized_corpus]
print("\n\n\nfiltered_corpus")
print(filtered_corpus)


# 5) Stemming and Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_corpus = [[lemmatizer.lemmatize(word) for word in doc] for doc in filtered_corpus]
print("\n\n\nlemmatized_corpus")
print(lemmatized_corpus)

text_lemmatized_corpus = [" ".join(words) for words in lemmatized_corpus]
print("\n\n\ntext_lemmatized_corpus")
print(text_lemmatized_corpus)

# with open("C:/Users/User/Downloads/Text_Preprocessing.txt", "w") as f:
    # f.write(str(text_lemmatized_corpus))

# 6) Convert lemmatizations to numerical embeddings
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

inputs = tokenizer(text_lemmatized_corpus, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1)
if trim is True:
    torch.save(embeddings, "C:/Users/User/Downloads/text_embeddings_trimmed.pt")
    np.save("C:/Users/User/Downloads/text_embeddings_trimmed.npy", embeddings.numpy())
else:
    torch.save(embeddings, "C:/Users/User/Downloads/text_embeddings.pt")
    np.save("C:/Users/User/Downloads/text_embeddings.npy", embeddings.numpy())


def text_preprocessing(text_body):
    # 1) Text Cleaning
    cleaned_text_transcript_list = [clean_text(doc) for doc in text_body]
    # 2) Contractions
    expanded_corpus = [contractions.fix(doc) for doc in cleaned_text_transcript_list]
    # 3) Tokenization
    tokenized_corpus = [word_tokenize(doc) for doc in expanded_corpus]
    # 4) Stop Words Removal
    stop_words = set(stopwords.words('english'))
    filtered_corpus = [[word for word in doc if word not in stop_words] for doc in tokenized_corpus]
    # 5) Stemming and Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_corpus = [[lemmatizer.lemmatize(word) for word in doc] for doc in filtered_corpus]
    return lemmatized_corpus












