#Import Librari yang digunakan
import pandas as pd #digunakan untuk memuat dataset
import re
import nltk
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Load dataset
df = pd.read_csv('dataset_shopee2.csv', encoding='latin1')  # untuk encoding jika ada karakter khusus
print(df.head())

# Cek distribusi label
print(df['SENTIMEN'].value_counts())

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

def preprocess_text(text):
    # Case folding
    text = text.lower()
    # Menghapus karakter khusus dan angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisasi
    tokens = word_tokenize(text)
    # Menghapus stopwords
    stop_words = set(stopwords.words('english')) 
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatisasi
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_review'] = df['Review'].apply(preprocess_text)
print(df['cleaned_review'].head())

# Tokenisasi
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_review'])
sequences = tokenizer.texts_to_sequences(df['cleaned_review'])

max_len = 100  # Panjang maksimum sequence
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Konversi label ke numerik
df['label'] = df['SENTIMEN'].apply(lambda x: 1 if x == 'POSITIF' else 0)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['label'], test_size=0.2, random_state=42)

# Model LSTM
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluasi
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot akurasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return "POSITIF" if prediction > 0.5 else "NEGATIF"

# Contoh prediksi
print(predict_sentiment("barangnya bagus dan pengiriman cepat"))  # Output: POSITIF
print(predict_sentiment("aplikasi sering error dan lemot"))  # Output: NEGATIF

model.save('sentiment_model.h5')