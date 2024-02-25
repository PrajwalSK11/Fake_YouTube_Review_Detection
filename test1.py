#VENUGOPAL ADEP NOTEBOOK
#prajwal
# Data processing packages
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 200)

# Visualization packages
import matplotlib.pyplot as plt
import seaborn as sns

# NLP packages
from textblob import TextBlob
import spacy
import nltk
import warnings
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
warnings.filterwarnings("ignore")
#from nltk.tokenize import word_tokenize

# Download NLTK resources (stopwords, punkt)
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Porter Stemmer
porter = PorterStemmer()

# Function to preprocess text (remove stopwords and apply stemming) 
'''
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    if isinstance(text, str):
        words = nltk.word_tokenize(text)
        filtered_words = [porter.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
        return ' '.join(filtered_words)
    else:
        return ''  # Return an empty string for NaN values
'''

nlp = spacy.load("en_core_web_sm")
'''
# Function to preprocess text (remove stopwords and apply lemmatization)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    if isinstance(text, str):
        doc = nlp(text)
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
        return ' '.join(lemmatized_words)
    else:
        return ''  # Return an empty string for NaN values
'''    
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    if isinstance(text, str):
        doc = nlp(text)
        lemmatized_words = [token.lemma_ for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
        filtered_text = ' '.join(lemmatized_words)
        
        # Check if the sentence has non-English characters
        if not filtered_text.encode('utf-8').isascii():
            return None  # Return None for non-English sentences
        else:
            return filtered_text
    else:
        return None  # Return None for NaN values

# Initialize the Progress Bar (tqdm) for visualizing the progress
tqdm.pandas()

# Testing NLP - Sentiment Analysis using TextBlob
TextBlob("The movie is ok").sentiment


# Importing YouTube comments data
comm = pd.read_csv('D:\Prajwal\PCCOE\Major project\Youtube\Codes\data.csv', encoding='utf8', error_bad_lines=False)
df = pd.DataFrame(comm)
df.head()
df = df.drop(['UserName', 'Time', 'Likes', 'Reply Count'], axis=1)
df.head()

tqdm.pandas()

# Apply the preprocess_text function to the 'Comment' column
df['Processed_Comment'] = df['Comment'].apply(preprocess_text)
'''
from googletrans import Translator
from tqdm.notebook import tqdm

def hinglish_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='hi', dest='en')
    return translation.text
'''
from googletrans import Translator
import json

def hinglish_to_english(text):
    if text is None or pd.isnull(text) or not text.strip():
        return ''  # Return an empty string for None, NaN, or empty values

    translator = Translator()
    try:
        translation = translator.translate(text, src='hi', dest='en')
        return translation.text
    except Exception as e:
        print(f"Error translating text: {text}")
        print(f"Error details: {e}")
        return ''



# Apply the translation function to the 'Text' column and create a new column
df['English_Translation'] = df['Processed_Comment'].apply(hinglish_to_english)


# Calculating the Sentiment Polarity.
pol = []  # list which will contain the polarity of the comments
for i in df.Processed_Comment.values:
    try:
        analysis = TextBlob(i)
        pol.append(analysis.sentiment.polarity)
    except:
        pol.append(0)

# Adding the Sentiment Polarity column to the data
df['pol'] = pol

# Converting the polarity values from continuous to categorical
df['pol'][df.pol == 0] = 0
df['pol'][df.pol > 0] = 1
df['pol'][df.pol < 0] = -1

# Displaying the POSITIVE comments
df_positive = df[df.pol == 1]
df_positive.head(10)

# Displaying the NEUTRAL comments
df_neutral = df[df.pol == 0]
df_neutral.head(10)

# Displaying the NEGATIVE comments
df_negative = df[df.pol == -1]
df_negative.head(10)


# Using Matplotlib to plot the bar plot
plt.figure(figsize=(8, 6))
ax = df['pol'].value_counts().plot(kind='bar', color=['blue', 'orange', 'blue'])

# Replace numeric labels with corresponding labels
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Neutral', 'Positive', 'Negative'], rotation=0)

plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()
df.pol.value_counts()

#df.pol.value_counts().plot.bar()

# Use raw string literals to avoid escape characters
df.to_csv(r'D:\Prajwal\PCCOE\Major project\Youtube\Codes\filee.csv')
