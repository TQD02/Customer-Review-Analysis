
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import zipfile
import os



with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_files')
    
print("Files extracted to 'extracted_files' folder")

#List the extracted files
for root, dirs, files_list in os.walk('extracted_files'):
    for file in files_list:
        print(os.path.join(root, file))



review_data = pd.read_csv("extracted_files/Reviews.csv")
print(review_data.head)


#Inspect the data
print(review_data.shape)
print(review_data.columns)
print(review_data.info())
print("\nScore Distribution")
print(review_data['Score'].value_counts().sort_index())
print(f"\nAverage review length: {review_data['Text'].str.len().mean():.0f} characters")

print("\nScore Distribution:")
print(review_data['Score'].value_counts().sort_index())

review_data['review_length'] = review_data['Text'].str.len()
review_data['summary_length'] = review_data['Summary'].str.len()

print(f"\nAverage review length: {review_data['review_length'].mean():.0f} characters")

plt.figure(figsize=(8, 5))
plt.hist(review_data['Score'], bins=5, edgecolor='black')
plt.title("Distribution of Review Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(review_data['review_length'], bins=50, edgecolor='black')
plt.title("Distribution of Review Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(review_data['summary_length'], bins=50, edgecolor='black')
plt.title("Distribution of Summary Lengths")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


#Data cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower() #Convert lower case
    text = re.sub(r'http\S+', '', text) #Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text) #Remove special chars w/o basic punctuation
    text = re.sub(r'\s+', ' ', text).strip() #Remove white noise
    return text

review_data['Text_Clean'] = review_data['Text'].apply(clean_text)
review_data['Summary_Clean'] = review_data['Summary'].apply(clean_text)
review_data['Combined_Text'] = review_data['Summary_Clean'] + ' ' + review_data['Text_Clean']
review_data = review_data.dropna(subset=['Combined_Text', 'Score'])
print(f"Dataset after cleaning: {review_data.shape[0]} reviews")
df_selected = review_data[['Score', 'Combined_Text']].copy()
print(df_selected.info())


#Score Classificaiton 
review_data['Sentiment_Binary'] = (review_data['Score'] >= 4).astype(int) #Binary Classification (0-1)
review_data['Score_Class'] = review_data['Score'] - 1  #Multi-class Classification (0-4) 
TARGET_TYPE = 'multi_class' 
if TARGET_TYPE == 'binary': #Switch between 2 types of classification 
    labels = review_data['Sentiment_Binary'].values
    num_classes = 2
else:
    labels = review_data['Score_Class'].values
    num_classes = 5

print(review_data['Score_Class'].value_counts())


#Tokenization
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 128
texts = review_data['Combined_Text'].values

print("\nTokenizing text...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

#Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)

#Pad sequences to same length
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
print(f"Shape of X: {X.shape}")


#Spit training, validation and testing sets
X_train, X_tv, y_train, y_tv = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_tv, y_tv, test_size = 0.5, random_state=42, stratify=y_tv)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

#Model 1: Baseline CNN for text classification
def create_cnn_model(max_words=10000, max_len=100, embedding_dim=128, num_classes=2):

    model = keras.Sequential([
        layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len), #Embedding
        
        layers.SpatialDropout1D(0.2), #Dropout for regularization
        
        #Multiple Conv1D layers with different kernel sizes to capture different n-grams
        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=128, kernel_size=4, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        
        layers.Dense(128, activation='relu'),  #Dense layers
        layers.Dropout(0.5),
        
        layers.Dense(num_classes if num_classes > 2 else 1,  #Output layer
                    activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    return model

#Model 2: Baseline LTSM for text classification 
def create_lstm_model(max_words=10000, max_len=100, embedding_dim=128, num_classes=2):
    
    model = keras.Sequential([
        layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len), # Embedding layer
        layers.Dropout(0.2), # Dropout
        layers.LSTM(128, return_sequences=True),  # LSTM layers
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'), # Dense layers
        layers.Dropout(0.5),
        
        layers.Dense(num_classes if num_classes > 2 else 1,  # Output layer
                    activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    return model

#Model 3: Bidirectional LSTM
def create_bilstm_model(max_words=10000, max_len=100, embedding_dim=128, num_classes=2):
    
    model = keras.Sequential([
        layers.Embedding(input_dim=max_words, # Embedding layer
                        output_dim=embedding_dim, 
                        input_length=max_len),
        
        layers.Dropout(0.2),
        
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)), # Bidirectional LSTM layers
        layers.Dropout(0.2),
        
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'), # Dense layers
        layers.Dropout(0.5),
        
        layers.Dense(num_classes if num_classes > 2 else 1, # Output layer
                    activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    return model


#Multi-class focal loss
def focal_loss(gamma=2., alpha=0.25, num_classes=5):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=num_classes)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)  
        cross_entropy = -y_true_onehot * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return loss_fn

#Binary focal loss
def binary_focal_loss(gamma=2., alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        loss = -alpha * y_true * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)                - (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return loss_fn


def compile_and_train(model, X_train, y_train, X_val, y_val, 
                      num_classes=2, epochs=10, batch_size=256,class_weight=None):
    if num_classes == 2:
        loss_fn = binary_focal_loss()
        metrics = ['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
    else:
        loss_fn = focal_loss(num_classes=num_classes)
        metrics = ['accuracy']

    #Compile model
    model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)
    #Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001),
        ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)]
    #Train model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,   
        verbose=1)
    return history

model1 = create_cnn_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM, num_classes)
model2 = create_lstm_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM, num_classes)
model3 = create_bilstm_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM, num_classes)
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
print("Class weights:", class_weights)


#Train CNN model
history_cnn = compile_and_train(
    model1, 
    X_train, y_train, 
    X_val, y_val,
    num_classes=num_classes, #adjust TARGET_TYPE Variable for different score classification
    epochs=10,
    batch_size=256,
    class_weight = class_weights
)


#Evaluate the CNN model on test set
test_loss_cnn, test_acc_cnn  = model1.evaluate(X_test, y_test, verbose=0)

y_pred_probs_cnn = model1.predict(X_test)
if num_classes == 2:
    y_pred_cnn = (y_pred_probs_cnn > 0.5).astype(int).flatten()
else:
    y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)

test_precision_cnn = precision_score(y_test, y_pred_cnn, average='weighted') 
test_recall_cnn = recall_score(y_test, y_pred_cnn, average='weighted')
print(f"\nTest Loss: {test_loss_cnn:.4f}")
print(f"Test Accuracy: {test_acc_cnn:.4f}")
 
print(f"\nTest Precsion: {test_precision_cnn:.4f}")
print(f"Test Recall: {test_recall_cnn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cnn))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_cnn))


#Train LSTM model
history_lstm = compile_and_train(
    model2, 
    X_train, y_train, 
    X_val, y_val,
    num_classes=num_classes, #adjust TARGET_TYPE Variable for different score classification
    epochs=10,
    batch_size=256,
    class_weight = class_weights
)

#Evaluate LSTM model on test set
test_loss_lstm, test_acc_lstm  = model2.evaluate(X_test, y_test, verbose=0)
y_pred_probs = model2.predict(X_test)
if num_classes == 2:
    y_pred_lstm = (y_pred_probs > 0.5).astype(int).flatten()
else:
    y_pred_lstm = np.argmax(y_pred_probs, axis=1)

test_precision_lstm = precision_score(y_test, y_pred_lstm, average='weighted') 
test_recall_lstm = recall_score(y_test, y_pred_lstm, average='weighted')

print(f"\nTest Loss: {test_loss_lstm:.4f}")
print(f"Test Accuracy: {test_acc_lstm:.4f}")
print(f"Test Precision: {test_precision_lstm:.4f}")
print(f"Test Recall: {test_recall_lstm :.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lstm))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_lstm))


#Train BiLSTM model
history_bilstm = compile_and_train(
    model3, 
    X_train, y_train, 
    X_val, y_val,
    num_classes=num_classes, #adjust TARGET_TYPE Variable for different score classification
    epochs=10,
    batch_size=256
)


#Evalutate BiLSTM model on test set
test_loss_bilstm, test_acc_bilstm  = model3.evaluate(X_test, y_test, verbose=0)
y_pred_probs_bilstm = model3.predict(X_test)
if num_classes == 2:
    y_pred_bilstm = (y_pred_probs_bilstm > 0.5).astype(int).flatten()
else:
    y_pred_bilstm = np.argmax(y_pred_probs_bilstm, axis=1)

test_precision_bilstm = precision_score(y_test, y_pred_bilstm, average='weighted') 
test_recall_bilstm = recall_score(y_test, y_pred_bilstm, average='weighted')

print(f"\nTest Loss: {test_loss_bilstm:.4f}")
print(f"Test Accuracy: {test_acc_bilstm:.4f}")
print(f"Test Precision: {test_precision_bilstm:.4f}")
print(f"Test Recall: {test_recall_bilstm :.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_bilstm))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_bilstm))


f1_mc_cnn =     [0.76, 0.38, 0.54, 0.42, 0.91]
f1_mc_lstm =    [0.76, 0.30, 0.51, 0.46, 0.91]
f1_mc_bilstm =  [0.75, 0.39, 0.50, 0.38, 0.91]

classes_mc = ["0", "1", "2", "3", "4"]
x = np.arange(len(classes_mc))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, f1_mc_cnn, width, label="CNN")
plt.bar(x,         f1_mc_lstm, width, label="LSTM")
plt.bar(x + width, f1_mc_bilstm, width, label="Bi-LSTM")

plt.xticks(x, classes_mc)
plt.xlabel("Class")
plt.ylabel("F1-score")
plt.title("F1-score Comparison Across Models (Multiclass Classification)")
plt.legend()
plt.tight_layout()
plt.show()


f1_bc_cnn =     [0.88, 0.97]
f1_bc_lstm =    [0.86, 0.97]
f1_bc_bilstm =  [0.91, 0.96]
classes_bc = ["0", "1"]
x = np.arange(len(classes_bc))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width , f1_bc_cnn, width, label="CNN")
plt.bar(x,  f1_bc_lstm, width, label="LSTM")
plt.bar(x + width, f1_bc_bilstm, width, label="Bi-LSTM")

plt.xticks(x, classes_bc)
plt.xlabel("Class")
plt.ylabel("F1-score")
plt.title("F1-score Comparison Across Models (Binary Classification)")
plt.legend()
plt.tight_layout()
plt.show()




