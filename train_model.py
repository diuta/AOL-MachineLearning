import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
import pickle
import nltk

# Download necessary NLTK data if not already present
# This is good practice for a standalone script, though in a controlled environment
# you might manage this separately.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
# Add other nltk downloads if your full preprocessing (not shown in detail in NB source) requires them.

# --- Configuration ---
FILE_PATH_EMOTION = './dataset/combined_emotion.csv'
# FILE_PATH_SONG = './dataset/data_moods.csv' # Not used for training this specific model pipeline
OUTPUT_PIPELINE_FILENAME = 'musicmood_pipeline.pkl'
TEST_SIZE = 0.3
RANDOM_STATE = 42

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the emotion dataset."""
    df_emotion = pd.read_csv(file_path)
    
    # Basic preprocessing as seen in the notebook
    # Consolidate emotion categories
    df_emotion['emotion'] = df_emotion['emotion'].replace({
        'fear': 'Sad', 
        'sad': 'Sad',
        'love': 'Happy',
        'joy': 'Happy',
        'suprise': 'Energetic', # 'surprise' is likely intended, but matching notebook
        'anger': 'Energetic'
    })
    # Ensure no NaN values in 'sentence' or 'emotion' if they are critical
    df_emotion.dropna(subset=['sentence', 'emotion'], inplace=True)
    return df_emotion['sentence'], df_emotion['emotion']

def train_and_save_pipeline(X, y, filename):
    """Trains the TF-IDF + ComplementNB pipeline and saves it."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Define the vectorizer and model (as in the notebook)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    model = ComplementNB()

    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('nb', model)
    ])

    # Train the pipeline
    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Save the pipeline
    with open(filename, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Pipeline saved to {filename}")

    # Optional: Evaluate the model (as in the notebook)
    accuracy = pipeline.score(X_test, y_test)
    print(f'Accuracy on the test set: {accuracy*100:.2f}%')

if __name__ == '__main__':
    print("Starting model training process...")
    sentences, emotions = load_and_preprocess_data(FILE_PATH_EMOTION)
    train_and_save_pipeline(sentences, emotions, OUTPUT_PIPELINE_FILENAME)
    print("Model training script finished.") 