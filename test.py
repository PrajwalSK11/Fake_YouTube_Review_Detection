# %%

# Data processing packages
from wordcloud import WordCloud
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
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
from pyabsa import available_checkpoints  # Import ABSA related modules
from pyabsa import ATEPCCheckpointManager
warnings.filterwarnings("ignore")

# Download NLTK resources (stopwords, punkt)
nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')
lzr = WordNetLemmatizer()

# %%

# %%

# Importing YouTube comments data
comm = pd.read_csv('D:\Prajwal\PCCOE\Major project\Youtube\Fake_YouTube_Review_Detection\youtube_review_dataset.csv',
                   encoding='utf8')
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
    text = re.sub(r"http\S+", " ", text)

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove repeated "Y" characters
    text = re.sub(r'(Y)\1+', r'\1', text)

    # Remove repeated "y" characters (case-insensitive)
    text = re.sub(r'(y)\1+', r'\1', text, flags=re.IGNORECASE)

    # remove new line characters in text
    text = re.sub(r'\n', ' ', text)

    # remove weird characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), " ", text)

    # Remove numeric values from text
    text = re.sub(r'\d+', ' ', text)

    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", " ", text)
    text = re.sub(r"http\S+", " ", text)

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
#df.to_csv('english_translated_data.csv', index=False)

# Read the processed data from the CSV file
#df = pd.read_csv(r'D:\Prajwal\PCCOE\Major project\Youtube\Fake_YouTube_Review_Detection\english_translated_data.csv')

# Save the processed DataFrame to a Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Fake_YouTube_Review_Detection\english.pkl', 'wb') as file:
    pickle.dump(df, file)

# Load the data from the Pickle file
with open(r'D:\Prajwal\PCCOE\Major project\Youtube\Fake_YouTube_Review_Detection\english.pkl', 'rb') as file:
    df = pickle.load(file)
'''
import pandas as pd
import missingno as msno
plt.figure(figsize=(25, 20))
msno.matrix(df, color=[0.2, 0.4, 1])
plt.title('Visualization of Missing Values', fontsize=24)
plt.xlabel('Columns', fontsize=24)
plt.ylabel('Rows', fontsize=24)
plt.show()
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

#%%

# %%

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
tqdm.pandas()
df.fillna(method='bfill', inplace=True)

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

plt.figure(figsize=(12, 8))
sns.histplot(df['Polarity'], bins=20, kde=True)
plt.title('Distribution of Polarity Scores')
plt.xlabel('Polarity Score')
plt.ylabel('Frequency')
plt.show()

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract aspects and sentiments for each comment
def extract_aspects_and_sentiments(comments):
    # Initialize an empty list to store the results
    aspect_sentiment_results = []
    
    # Initialize the ABSA model
    aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english', auto_device=True)
    
    # Iterate through each comment
    for comment in tqdm(comments):
        # Extract aspects and sentiments for the current comment
        result = aspect_extractor.extract_aspect(inference_source=[comment], pred_sentiment=True)
        
        # Append the result to the list
        aspect_sentiment_results.append(result)
    
    return aspect_sentiment_results

# Extract the some entries from a specific column (e.g., 'Processed_Comment')
comments_subset = df['English_Translation'].head(10)

# Now, you can pass this subset of comments to your processing function
aspect_results = extract_aspects_and_sentiments(comments_subset)

# Apply aspect extraction to the 'English_Translation' column of the DataFrame
#aspect_results = extract_aspects_and_sentiments(df['English_Translation'])

# Initialize an empty list to store aspect-sentiment pairs as dictionaries
aspect_sentiment_pairs = []

# Iterate through each comment in the DataFrame and extract the corresponding aspect-sentiment pair
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
plt.figure(figsize=(12, 8))
df_aspect_sentiment = df.explode('Aspect_Sentiment')
sns.countplot(x='Aspect_Sentiment', data=df_aspect_sentiment)
plt.title('Aspect-Sentiment Distribution')
plt.xlabel('Aspect-Sentiment Pair')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()

df.to_csv('test_aspects_result.csv', index=False)

# %%