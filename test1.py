# VENUGOPAL ADEP NOTEBOOK
# %%

# Data processing packages
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
import pickle
pd.set_option('display.max_colwidth', 200)
warnings.filterwarnings("ignore")

# Download NLTK resources (stopwords, punkt)
nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')
lzr = WordNetLemmatizer()

# %%

# %%

# Importing YouTube comments data
'''
comm = pd.read_csv('D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\youtube_review_dataset.csv',
                   encoding='utf8', error_bad_lines=False)
'''
comm = pd.read_csv('D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\youtube_review_dataset1.csv',
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

# Testing NLP - Sentiment Analysis using TextBlob
#TextBlob("The movie is ok").sentiment

# %%

# Initialize the Progress Bar (tqdm) for visualizing the progress
tqdm.pandas()

df['Processed_Comment'] = df['Comment'].fillna(
    '').progress_apply(text_processing)
df = df[df['Processed_Comment'].str.len() > 0]
df.to_csv('cleaned_data.csv', index=False)

# Read the processed data from the CSV file
#df = pd.read_csv(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\cleaned_data.csv')

# Save the processed DataFrame to a Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\cleaned_data.pkl', 'wb') as file:
    pickle.dump(df, file)

# Load the data from the Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\cleaned_data.pkl', 'rb') as file:
    df = pickle.load(file)

'''
# Extract 5 sequential samples from the `Processed_Comment` column
sequential_samples = df['Processed_Comment'].iloc[0:10]

sequential_samples.shape
print(sequential_samples)
'''

# %%

# %%

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

'''
# Apply the translation function to the 'Processed_Comment' column and create a new column
df['English_Translation'] = sequential_samples.progress_apply(
    hinglish_to_english)
'''

# Initialize the Progress Bar (tqdm) for visualizing the progress
tqdm.pandas()
df['English_Translation'] = df['Processed_Comment'].progress_apply(
    hinglish_to_english)

# Save the DataFrame to a CSV file
df.to_csv('english_translated_data.csv', index=False)

# Read the processed data from the CSV file
#df = pd.read_csv(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\english_translated_data.csv')

# Save the processed DataFrame to a Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\processed_data.pkl', 'wb') as file:
    pickle.dump(df, file)

# Load the data from the Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\processed_data.pkl', 'rb') as file:
    df = pickle.load(file)

# %%

# %%

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(
    df['English_Translation'].str.cat(sep=' '))

# Display the generated image
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# %%

#%%

from nltk import FreqDist
def tokenize_and_pos(text):
    # Check for NaN values
    if pd.isna(text):
        return []

    # Tokenize and add POS tagging
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# Initialize the Progress Bar (tqdm) for visualizing the progress
tqdm.pandas()
# Tokenize and add POS tagging
df['Tokenized_POS'] = df['English_Translation'].progress_apply(tokenize_and_pos)

# Display the top words based on frequency
all_tokens = [token for tokens_pos in df['Tokenized_POS'] for token, pos in tokens_pos]
freq_dist = FreqDist(all_tokens)
top_words = freq_dist.most_common(10)

# Plot a bar chart for the top words
plt.figure(figsize=(12, 8))
plt.bar(range(len(top_words)), [count for word, count in top_words], align='center')
plt.xticks(range(len(top_words)), [word for word, count in top_words], rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top Words Frequency Distribution')
plt.show()

# Save the DataFrame to a CSV file
#df.to_csv('processed_data_with_POS.csv', index=False)

# Save the processed DataFrame to a Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\processed_data_with_POS.pkl', 'wb') as file:
    pickle.dump(df, file)

# Load the data from the Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\processed_data_with_POS.pkl', 'rb') as file:
    df = pickle.load(file)

# Display the DataFrame with POS tagging
#print(df[['English_Translation', 'Tokenized_POS']])

#%%

# %%

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tqdm.pandas()

# Initialize the SentimentIntensityAnalyzer
sent_analyser = SentimentIntensityAnalyzer()

def polarity(text):
    # Check if the input is a float
    if isinstance(text, float):
        return 0  # Return a neutral sentiment for float values

    # Calculate the compound polarity score using VADER
    return sent_analyser.polarity_scores(text)["compound"]
# Apply the sentiment analysis function to the "English_Translation" column
df["Polarity"] = df["English_Translation"].progress_apply(polarity)

def sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"
df["Sentiment"] = df["English_Translation"].progress_apply(sentiment)

# Plot the countplot
plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")
ax = sns.countplot(x="Sentiment", data=df, palette=dict(Neutral="blue", Positive="green", Negative="red"))

sentiment_counts = df['Sentiment'].value_counts()
print("Sentiment Counts:\n" + str(sentiment_counts))

nlp = spacy.load("en_core_web_sm")
def pos_with_sentiment(comment):
    doc = nlp(comment)
    aspects = [token.text for token in doc if token.pos_ == "NOUN"]
    #sentiment = df['Sentiment']
    Sentiment = sentiment(comment)
    return {'Aspects': aspects, 'Sentiment': Sentiment}

# Apply the function to create a new column 'Aspects_Sentiment'
df['Aspects_Sentiment'] = df['English_Translation'].progress_apply(pos_with_sentiment)

#df.to_csv('b.csv', index=False)
'''
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
'''
# %%

#%%

# Use raw string literals to avoid escape characters
df.to_csv(
    r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\a.csv')

# %%
