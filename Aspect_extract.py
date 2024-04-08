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
from googleapiclient.discovery import build
from googletrans import Translator  # Import Translator here for Hinglish to English translation
from nltk import FreqDist  # Import FreqDist for word frequency calculation
from pyabsa import available_checkpoints  # Import ABSA related modules
from pyabsa import ATEPCCheckpointManager

# Download NLTK resources (stopwords, punkt)
nltk.download('stopwords')
nltk.download('punkt')

# Set up data processing configurations
pd.set_option('display.max_colwidth', 200)
warnings.filterwarnings("ignore")

# Initialize NLTK and spaCy resources
stop_words = stopwords.words('english')
lzr = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# YouTube API key (from the first snippet)
api_key = "YOUR_API"

def extract_youtube_comments(video_url, api_key, max_comments=40):
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
    total_comments = 0
    while total_comments < max_comments:
        comments_request = youtube_service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments - total_comments, 100),  # Limit comments per page
            pageToken=next_page_token
        )
        comments_response = comments_request.execute()
        for item in comments_response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            comments_list.append((video_title, comment))  # Append title along with comment
            total_comments += 1
            if total_comments >= max_comments:
                break
        next_page_token = comments_response.get("nextPageToken")
        if not next_page_token:
            break
    
    return comments_list

# Function for Hinglish to English translation
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

# Function for text preprocessing (from the first snippet)
def text_processing(text):
    # convert text into lowercase
    text = text.lower()

    # Normalize the string
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove URLs
    text = re.sub(r'(https|http)?:\/(\w|\.|\/|\?|\=|\&|\%)*\b',' ', text)
    text = re.sub(r'www\.\S+\.com',' ',text)

    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,."," ", text)
    text = re.sub(r"http\S+"," ", text)
    text = re.sub(r'#\S+',' ', text)

    #user mention removes
    text = re.sub(r'@\S+', ' ', text)

    #emoji
    text = re.sub(r'[^\x00-\x7F]+',' ', text)

    #html tags
    text = re.sub(r'<.*?>',' ', text)

    #removes extra spaces
    text = re.sub(r' +',' ', text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+',' ', text)

    # Remove repeated "Y" characters
    text = re.sub(r'(Y)\1+', r'\1', text)

    # Remove repeated "y" characters (case-insensitive)
    text = re.sub(r'(y)\1+', r'\1', text, flags=re.IGNORECASE)

    # remove new line characters in text
    text = re.sub(r'\n',' ', text)

    # remove weird characters
    text = re.sub(r'[^\x00-\x7F]+','', text)

    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation)," ", text)

    # Remove numeric values from text
    text = re.sub(r'\d+',' ', text)

    # remove multiple spaces from text
    text = re.sub(r'\s+',' ', text, flags=re.I)

    # remove special characters from text
    text = re.sub(r'\W',' ', text)

    # adding lemma in it
    text = ' '.join([word for word in word_tokenize(text)
                    if word not in stop_words])

    # lemmatizer using WordNetLemmatizer from nltk package
    text = ' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text


# Extracting comments from the specified YouTube video (from the first snippet)
youtube_video_url = "youtube_URL"
#comments = extract_youtube_comments(youtube_video_url, api_key)
max_comments = 30
comments_list = extract_youtube_comments(youtube_video_url, api_key, max_comments)


# Create a DataFrame from the extracted comments (from the first snippet)
#df = pd.DataFrame(comments, columns=["Video Title", "Comment"])
df = pd.DataFrame(comments_list, columns=["Video Title", "Comment"])
df.head()
# Preprocess comments and translate Hinglish comments to English (from the first snippet)
tqdm.pandas()
df['Processed_Comment'] = df['Comment'].fillna('').progress_apply(text_processing)
df = df[df['Processed_Comment'].str.len() > 0]
df['English_Translation'] = df['Processed_Comment'].progress_apply(hinglish_to_english)

import pandas as pd
import missingno as msno
plt.figure(figsize=(25, 20))
msno.matrix(df, color=[0.2, 0.4, 1])
plt.show()

# Visualize word cloud (from the first snippet)
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(
    df['English_Translation'].str.cat(sep=' '))
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Tokenize and add POS tagging (from the first snippet)
nlp = spacy.load("en_core_web_sm")
def tokenize_and_pos(text):
    # Check for NaN values
    if pd.isna(text):
        return []
    # Tokenize and add POS tagging
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# Initialize the Progress Bar (tqdm) for visualizing the progress
tqdm.pandas()
df['Tokenized_POS'] = df['English_Translation'].progress_apply(tokenize_and_pos)

# Calculate word frequency and plot bar chart (from the first snippet)
all_tokens = [token for tokens_pos in df['Tokenized_POS'] for token, pos in tokens_pos]
freq_dist = FreqDist(all_tokens)
top_words = freq_dist.most_common(10)
plt.figure(figsize=(12, 8))
plt.bar(range(len(top_words)), [count for word, count in top_words], align='center')
plt.xticks(range(len(top_words)), [word for word, count in top_words], rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top Words Frequency Distribution')
plt.show()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
tqdm.pandas()
df["Polarity"] = df["Processed_Comment"].progress_apply(polarity)

plt.figure(figsize=(12, 8))
sns.histplot(df['Polarity'], bins=20, kde=True)
plt.title('Distribution of Polarity Scores')
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.show()

# Function to extract aspects and sentiments for each comment
def extract_aspects_and_sentiments(comments):
    # Initialize an empty list to store the results
    aspect_sentiment_results = []
    
    # Initialize the ABSA model
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english', auto_device=False)
    
    # Iterate through each comment
    for comment in tqdm(comments):
        # Extract aspects and sentiments for the current comment
        result = aspect_extractor.extract_aspect(inference_source=[comment], pred_sentiment=True)
        
        # Append the result to the list
        aspect_sentiment_results.append(result)
    
    return aspect_sentiment_results

# Apply aspect extraction to the 'English_Translation' column of the DataFrame
aspect_results = extract_aspects_and_sentiments(df['English_Translation'])

# Initialize an empty list to store aspect-sentiment pairs as dictionaries
aspect_sentiment_pairs = []

# Iterate through each result in the aspect_results list
for result in aspect_results:
    # Check if the result list is empty
    if result:
        # Extract aspect and sentiment from the first dictionary in the list
        aspect = result[0].get('aspect', None)
        sentiment = result[0].get('sentiment', None)
        
        # Create a dictionary with aspect and sentiment as key-value pairs
        aspect_sentiment_dict = {'Aspect': aspect, 'Sentiment': sentiment}
    else:
        # If result list is empty, create a dictionary with None for aspect and sentiment
        aspect_sentiment_dict = {'Aspect': None, 'Sentiment': None}
    
    # Append the aspect-sentiment dictionary to the list
    aspect_sentiment_pairs.append(aspect_sentiment_dict)

# Add the aspect-sentiment pairs as a new column to the DataFrame
df['Aspect_Sentiment'] = aspect_sentiment_pairs

# Display the DataFrame with the added aspect-sentiment column
print(df.head())

# Filter the DataFrame to remove rows where either 'Aspect' or 'Sentiment' is absent
df = df[df['Aspect_Sentiment'].apply(lambda x: x['Aspect'] != [] and x['Sentiment'] != [])]

# Reset the index of the filtered DataFrame
df.reset_index(drop=True, inplace=True)


# Save the final DataFrame to a CSV file
df.to_csv('final_youtube_comments_dataset.csv', index=False)

# Read the processed data from the CSV file
df = pd.read_csv(r'D:\Prajwal\PCCOE\Major project\Youtube\Fake_YouTube_Review_Detection\pyabsa\final_youtube_comments_dataset.csv')

# Save the processed DataFrame to a Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Fake_YouTube_Review_Detection\test_aspects.pkl', 'wb') as file:
    pickle.dump(df, file)

# Load the data from the Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Fake_YouTube_Review_Detection\test_aspects.pkl', 'rb') as file:
    df = pickle.load(file)