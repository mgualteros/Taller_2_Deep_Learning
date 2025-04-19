import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def load_and_prepare_data(url, max_features=10000, maxlen=300, test_size=0.2):
    # Cargar dataset
    csv_path = tf.keras.utils.get_file("twitter_sentiment.csv", url)
    df = pd.read_csv(csv_path)
    tweets = df['tweet']
    labels = df['label']

    # Tokenización y padding
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    x_data = pad_sequences(sequences, maxlen=maxlen)
    y_data = labels.values

    # Split Train/Test
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=42, stratify=y_data
    )

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    x_resampled, y_resampled = ros.fit_resample(x_train, y_train)

    return x_resampled, y_resampled, x_test, y_test, tokenizer
