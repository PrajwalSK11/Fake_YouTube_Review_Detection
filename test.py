import re
import unicodedata
from string import punctuation
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
from googleapiclient.discovery import build
from tqdm import tqdm
from wordcloud import WordCloud
    

# Download NLTK resources (stopwords, punkt)
nltk.download('stopwords')
nltk.download('punkt')

# Set up NLTK and spaCy
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

# Initialize Google Translator and Sentiment Analyzer
translator = Translator()
sent_analyzer = SentimentIntensityAnalyzer()
tqdm.pandas()

# Function to process text data
def process_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'(Y)\1+', r'\1', text)
    text = re.sub(r'(y)\1+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\n', ' ', text)
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    text = re.sub(r'\d+', '', text)
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub(r'\W', ' ', text)
    text = ' '.join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words])
    return text

# Function to translate Hinglish text to English
def hinglish_to_english(text):
    if text is None or pd.isnull(text) or not text.strip():
        return ''
    try:
        translation = translator.translate(text, src='hi', dest='en')
        return translation.text
    except Exception as e:
        print(f"Error translating text: {text}")
        print(f"Error details: {e}")
        return ''

# Function to tokenize and perform POS tagging
def tokenize_and_pos_tag(comment):
    # Tokenize the comment
    tokens = nltk.word_tokenize(comment)
    
    # Perform POS tagging
    pos_tags = nltk.pos_tag(tokens)
    
    return pos_tags
    
# Function to generate a word cloud
def generate_word_cloud(aspect_terms):
    # Concatenate aspect terms into a single string
    text = ' '.join(aspect_terms)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
# Function to calculate sentiment polarity
def calculate_polarity(text):
    if isinstance(text, float):
        return 0
    return sent_analyzer.polarity_scores(text)["compound"]

# Function to classify sentiment
def classify_sentiment(text):
    polarity_score = calculate_polarity(text)
    if polarity_score > 0:
        return "Positive"
    elif polarity_score < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to extract comments from a YouTube video
def extract_youtube_comments(video_url, api_key):
    video_id = video_url.split("=")[-1]
    youtube_service = build("youtube", "v3", developerKey=api_key)
    video_request = youtube_service.videos().list(part="snippet", id=video_id)
    video_response = video_request.execute()
    video_title = video_response["items"][0]["snippet"]["title"]
    
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
            comments_list.append((video_title, comment))
        next_page_token = comments_response.get("nextPageToken")
        if not next_page_token:
            break
    return comments_list

# Function to extract aspect terms from a comment along with sentiment analysis
def extract_aspect_terms_with_sentiment(comment):
    aspect_sentiments = {}
    aspect_terms = []
    compound_terms = []
    lines = comment.split('.')
    for line in lines:
        doc = nlp(line)
        for token in doc:
            if token.pos_ == 'NOUN':
                aspect = token.text
                analysis = TextBlob(aspect)
                polarity = analysis.sentiment.polarity
                if polarity > 0:
                    sentiment = "Positive"
                elif polarity < 0:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                aspect_terms.append(aspect)
                aspect_sentiments[aspect] = sentiment
            elif token.dep_ == 'compound':
                compound_terms.append(token.text)
    return aspect_terms, aspect_sentiments, compound_terms


# Function to process comments
def process_comments(comments):
    df = pd.DataFrame(comments, columns=["Video Title", "Comment"])
    df['Processed_Comment'] = df['Comment'].fillna('').progress_apply(process_text)
    df = df[df['Processed_Comment'].str.len() > 0]
    df['English_Translation'] = df['Processed_Comment'].progress_apply(hinglish_to_english)
    df['POS_Tags'] = df['English_Translation'].progress_apply(tokenize_and_pos_tag)
    #df['Polarity'] = df['English_Translation'].progress_apply(calculate_polarity)
    #df['Sentiment'] = df['English_Translation'].progress_apply(classify_sentiment)
    return df

# Main function
def main():
    # YouTube API key and video URL
    api_key = "AIzaSyBu3nBiDk0_FjHbdoVYZlKuTlBBm8VzsKk"
    youtube_video_url = "https://www.youtube.com/watch?v=EHo0iFU1Fkc"
    
    # Extract comments
    comments = extract_youtube_comments(youtube_video_url, api_key)
    
    # Process and analyze comments
    df = process_comments(comments)
    
    # Extract aspect terms and sentiments
    aspect_terms_list = []
    aspect_sentiments_list = []
    compound_terms_list = []
    for comment in tqdm(df['English_Translation']):
        aspect_terms, aspect_sentiments, compound_terms = extract_aspect_terms_with_sentiment(comment)
        aspect_terms_list.append(aspect_terms)
        aspect_sentiments_list.append(aspect_sentiments)
        compound_terms_list.append(compound_terms)
    
    # Add extracted aspect terms, aspect sentiments, and compound terms to the DataFrame
    df['Compound_Terms'] = compound_terms_list
    df['Aspect_Terms'] = aspect_terms_list
    df['Aspect_Sentiments'] = aspect_sentiments_list
    generate_word_cloud(df['Aspect_Terms'][0])
    
    # Display the DataFrame with added aspect terms and sentiments
    print(df.head())
    
    # Save DataFrame to a CSV file
    df.to_csv('youtube_comments_dataset.csv', index=False)

# Call the main function
if __name__ == "__main__":
    main()
    
