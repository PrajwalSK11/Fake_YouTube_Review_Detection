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

# Importing YouTube API
from googleapiclient.discovery import build

# YouTube API key
api_key = "AIzaSyBu3nBiDk0_FjHbdoVYZlKuTlBBm8VzsKk"

# Function to extract comments from a YouTube video
def extract_youtube_comments(video_url, api_key):
    # Extracting video ID from URL
    video_id = video_url.split("=")[-1]
    
    # Building YouTube service
    youtube_service = build("youtube", "v3", developerKey=api_key)
    
    # Retrieving video title
    video_request = youtube_service.videos().list(
        part="snippet",
        id=video_id
    )
    video_response = video_request.execute()
    video_title = video_response["items"][0]["snippet"]["title"]
    
    # Retrieving comments for the video
    comments_list = []
    next_page_token = None
    while True:
        comments_request = youtube_service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        comments_response = comments_request.execute()
        for item in comments_response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            comments_list.append((video_title, comment))  # Append title along with comment
        next_page_token = comments_response.get("nextPageToken")
        if not next_page_token:
            break
    
    return comments_list

# Sample YouTube video URL
youtube_video_url = "https://www.youtube.com/watch?v=EHo0iFU1Fkc"

# Extracting comments from the specified YouTube video
comments = extract_youtube_comments(youtube_video_url, api_key)

# Create a DataFrame from the extracted comments
df = pd.DataFrame(comments, columns=["Video Title", "Comment"])

# Display the DataFrame
print(df.head())

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

'''
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

# Initialize the Progress Bar (tqdm) for visualizing the progress
tqdm.pandas()
df['English_Translation'] = df['Processed_Comment'].progress_apply(
    hinglish_to_english)

'''
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
'''
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
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
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

'''
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
'''
#%%

# %%

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tqdm.pandas()
#df.fillna(method='bfill', inplace=True)

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
    #analysis = TextBlob(text)
    Polarity=polarity(text)
    if Polarity > 0:
        return "Positive"
    elif Polarity < 0:
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

'''
df.to_csv('till_senti.csv', index=False)

# Save the processed DataFrame to a Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\till_senti.pkl', 'wb') as file:
    pickle.dump(df, file)
    
# Load the data from the Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Code\Fake_YouTube_Review_Detection\till_senti.pkl', 'rb') as file:
    df = pickle.load(file)
'''
import spacy
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

'''
def pos_with_sentiment(comment):
    # Analyze the comment using spaCy
    doc = nlp(comment)
    
    # Initialize an empty dictionary to store aspect sentiments
    aspect_sentiments = {}
    
    # Iterate through each token in the comment
    for token in doc:
        # If the token is a noun, analyze its sentiment along with its surrounding adjectives
        if token.pos_ in ["NOUN"]:
            aspect = token.text
            sentiment = None
            
            # Get the index of the token in the document
            token_index = token.i
            
            # Find adjectives in a window around the noun
            adjectives = []
            for i in range(max(token_index - 3, 0), min(token_index + 4, len(doc))):
                if doc[i].pos_ == "ADJ":
                    adjectives.append(doc[i].text)
            
            # Analyze sentiment using TextBlob
            analysis = TextBlob(" ".join(adjectives + [aspect]))
            polarity = analysis.sentiment.polarity
            
            # Convert numerical polarity to sentiment label
            if polarity > 0:
                sentiment = "Positive"
            elif polarity < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            # Store the aspect sentiment in the dictionary
            aspect_sentiments[aspect] = sentiment
    
    return aspect_sentiments
'''
def extract_aspect_terms_with_sentiment(comment):
    aspect_terms_with_sentiment = {}
    compound_terms = []
    for token in nlp(comment):
        if token.pos_ == 'NOUN':
            aspect = token.text
            polarity = TextBlob(aspect).sentiment.polarity
            sentiment = "Positive" if polarity > 0 else ("Negative" if polarity < 0 else "Neutral")
            aspect_terms_with_sentiment[aspect] = sentiment
        elif token.dep_ == 'compound':
            compound_terms.append(token.text)
    return aspect_terms_with_sentiment, compound_terms

# Call the function to extract aspect terms and sentiments
aspect_terms_with_sentiment_list = []
compound_terms_list = []
for comment in tqdm(df['English_Translation']):
    aspect_terms_with_sentiment, compound_terms = extract_aspect_terms_with_sentiment(comment)
    aspect_terms_with_sentiment_list.append(aspect_terms_with_sentiment)
    compound_terms_list.append(compound_terms)

# Add extracted aspect terms and aspect sentiments to the DataFrame
df['Aspect_with_Sentiment'] = aspect_terms_with_sentiment_list
df['Compound_Terms'] = compound_terms_list

# Display the DataFrame with added aspect terms and sentiments
print(df.head())

df.to_csv('final_youtube_comments_dataset.csv', index=False)


# %%