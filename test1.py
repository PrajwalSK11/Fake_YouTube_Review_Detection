# VENUGOPAL ADEP NOTEBOOK
# %%
# Data processing packages
import random
from wordcloud import WordCloud
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
from textblob import TextBlob
import unicodedata
import re
import warnings
import nltk
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', 200)

# Visualization packages

# NLP packages
#import string
warnings.filterwarnings("ignore")


# Download NLTK resources (stopwords, punkt)
nltk.download('stopwords')
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")
stop_words = stopwords.words('english')
lzr = WordNetLemmatizer()
# %%

# %%
# Importing YouTube comments data
comm = pd.read_csv('D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\youtube_review_dataset.csv',
                   encoding='utf8', error_bad_lines=False)
df = pd.DataFrame(comm)
df.head()
df = df.drop(['UserName', 'Time', 'Likes', 'Reply Count'], axis=1)
df.head()
# %%

# %%


def text_processing(text):
    # convert text into lowercase
    text = text.lower()

    # Normalize the string
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove repeated "Y" characters
    text = re.sub(r'(Y)\1+', r'\1', text)

    # Remove repeated "y" characters (case-insensitive)
    text = re.sub(r'(y)\1+', r'\1', text, flags=re.IGNORECASE)

    # remove new line characters in text
    text = re.sub(r'\n', ' ', text)

    # remove weird characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)

    # Remove numeric values from text
    text = re.sub(r'\d+', '', text)

    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    text = re.sub(r"http\S+", "", text)

    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # remove special characters from text
    text = re.sub(r'\W', ' ', text)

    # adding lemma in it
    text = ' '.join([word for word in word_tokenize(text)
                    if word not in stop_words])

    # lemmatizer using WordNetLemmatizer from nltk package
    text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text


#%%

sample = "ℍ𝕚 𝔼𝕧𝕖𝕣𝕪𝕠𝕟𝕖,,,,,,,.,./;/';à¤…à¤—à¤° à¤¯à¥‡ à¤¸à¤²à¤¾à¤° à¤®à¥‚à¤µà¥€ à¤¹à¥ˆ à¤¤à¥‹ à¤®à¥ˆ à¤¸à¤²à¤®à¤¾à¤¨ à¤–à¤¾à¤‚à¤¨ à¤¹à¥‚à¤,,,à¤¸à¤¾à¤¹à¥‹ à¤®à¥‚à¤µà¥€ à¤•à¤¾ðŸ˜‚ðŸ˜‚ðŸ˜‚'𝕀 𝕒𝕞 𝔸𝕟𝕜𝕚𝕥 𝔾𝕦𝕡𝕥𝕒 ðŸŒðŸ 𝕙𝕒𝕧𝕚𝕟𝕘 𝕥𝕙𝕖 𝕗𝕠𝕝𝕝𝕠𝕨𝕚𝕟𝕘 𝕂𝕒𝕘𝕘𝕝𝕖 𝕡𝕣𝕠𝕗𝕚𝕝𝕖 \n https://www.kaggle.com/nkitgupta 𝕒𝕟𝕕, 𝕀 𝕒𝕞 😊 𝕥𝕠 𝕔𝕣𝕖𝕒𝕥𝕖 𝕥𝕙𝕚𝕤 𝕟𝕠𝕥𝕖𝕓𝕠𝕠𝕜."
print(text_processing(sample))

# Testing NLP - Sentiment Analysis using TextBlob
#TextBlob("The movie is ok").sentiment
# %%
# Initialize the Progress Bar (tqdm) for visualizing the progress
tqdm.pandas()

df['Processed_Comment'] = df['Comment'].fillna(
    '').progress_apply(text_processing)
df = df[df['Processed_Comment'].str.len() > 0]


# Extract 5 sequential samples from the `Processed_Comment` column
sequential_samples = df['Processed_Comment'].iloc[0:10]

sequential_samples.shape
print(sequential_samples)
# %%
# %%


# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(
    df['Processed_Comment'].str.cat(sep=' '))

# Display the generated image
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# %%
# %%
'''
from googletrans import Translator
from tqdm.notebook import tqdm

def hinglish_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='hi', dest='en')
    return translation.text
'''
#import json

from googletrans import Translator
def hinglish_to_english(text):
    if text is None or pd.isnull(text) or not text.strip():
        return ''  # Return an empty string for None, NaN, or empty values

    translator = Translator()
    try:
        translator = Translator()
        translation = translator.translate(text, src='hi', dest='en')
        return translation.text
    except Exception as e:
        print(f"Error translating text: {text}")
        print(f"Error details: {e}")
        return ''

# try

from joblib import Parallel, delayed

# Set the number of cores to use for parallel processing
n_cores = 4

# Apply the translation function using parallel processing
df['English_Translation'] = Parallel(n_jobs=n_cores)(delayed(hinglish_to_english)(text) for text in df['Processed_Comment'].values)


df['English_Translation'] = df['Processed_Comment'].applymap(
    hinglish_to_english)
#
# Hinglish sentence
hinglish_sentence = " khaana achha nahi hai pr service accha nahi tha"

# Translating Hinglish to English
english_translation = hinglish_to_english(hinglish_sentence)

# Printing the result
print("Hinglish Sentence:", hinglish_sentence)
print("English Translation:", english_translation)
# Apply the translation function to the 'Text' column and create a new column
df['English_Translation'] = sequential_samples.progress_apply(
    hinglish_to_english)


# Apply the translation function to the 'Processed_Comment' column and create a new column
df['English_Translation'] = sequential_samples.progress_apply(
    hinglish_to_english)

# Save the DataFrame to a CSV file
df.to_csv('translated_data1.csv', index=False)


df['English_Translation'] = df['Processed_Comment'].progress_apply(
    hinglish_to_english)
# %%
# %%
# Calculating the Sentiment Polarity
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
ax = df['pol'].value_counts().plot(
    kind='bar', color=['blue', 'orange', 'blue'])

# Replace numeric labels with corresponding labels
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Neutral', 'Positive', 'Negative'], rotation=0)

plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()
df.pol.value_counts()

# df.pol.value_counts().plot.bar()
# %%
# Use raw string literals to avoid escape characters
df.to_csv(
    r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\b.csv')
# %%
