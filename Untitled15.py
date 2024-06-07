#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[2]:


import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# # preprocessing

# In[21]:


import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Load data from CSV file
file_path = "/Users/Zahra's computer/Desktop/textmining/twitter_training.csv"
df = pd.read_csv(file_path, encoding='latin-1')

# Drop rows with NaN values in 'Tweet content'
df = df.dropna(subset=['Tweet content'])

# Function to preprocess text
def preprocess_text(text):
    # Remove special characters
    pattern = r'[^A-Za-z0-9\s]'
    text = re.sub(pattern, '', str(text))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)
    
    return text

# Apply preprocessing to 'Tweet content'
df['Tweet content'] = df['Tweet content'].apply(preprocess_text)

# Tokenization
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
df['Tweet content'] = df['Tweet content'].apply(tokenizer.tokenize)
df['Tweet content'] = df['Tweet content'].apply(lambda tokens: ' '.join(tokens))
df = df.dropna(subset=['Tweet content'])
output_file_path = "/Users/Zahra's computer/Desktop/textmining/twitter_training_preprocessed.xlsx"
df.to_excel(output_file_path, index=False)


# In[22]:


df


# ## تبدیل داده های پردازش شده به اکسل:

# In[23]:


df = df.to_csv("twitterpreprocessed.csv")


# ## bag of words:
# 

# In[8]:


df = pd.read_csv("twitterpreprocessed.csv")

df["Tweet content"] = df["Tweet content"].fillna("")

documents = df["Tweet content"].tolist()

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(documents)



# # tf_idf:
# 

# In[9]:


documents = df['Tweet content'].astype(str)

tfidf_vectorizer = TfidfVectorizer()

X_tfidf = tfidf_vectorizer.fit_transform(documents)

df_tfidf = pd.DataFrame.sparse.from_spmatrix(X_tfidf, columns=tfidf_vectorizer.get_feature_names_out())


# # Linear SVC for training model
# 

# In[10]:


X = df_tfidf
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svc_model = LinearSVC()
svc_model.fit(X_train, y_train)
svc_predictions = svc_model.predict(X_test)
print("Linear SVC Accuracy:", accuracy_score(y_test, svc_predictions))
print("Linear SVC Classification Report:\n", classification_report(y_test, svc_predictions))



# # training with naiive bayes and svc
# 

# In[11]:


X = df_tfidf
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print(f"\nEvaluation metrics for {model_name}:\n")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_rep}")

    return predictions


svc_model = LinearSVC()
svc_predictions = train_evaluate_model(svc_model, X_train, y_train, X_test, y_test, 'Linear SVC')

nb_model = MultinomialNB()
nb_predictions = train_evaluate_model(nb_model, X_train, y_train, X_test, y_test, 'Naive Bayes')


# # plot

# In[13]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_confusion_matrix(model_name, y_true, y_pred, model_classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model_classes, yticklabels=model_classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

model_classes = ['Positive', 'Negative']

plot_confusion_matrix('Linear SVC', y_test, svc_predictions, model_classes)

plot_confusion_matrix('Naive Bayes', y_test, nb_predictions, model_classes)


# # training with other models

# In[11]:


file_path = "/Users/Zahra's computer/Desktop/textmining/twitter_training.csv"
df = pd.read_csv(file_path, encoding='latin-1')

# Drop missing values
df = df.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Tweet content'], df['sentiment'], test_size=0.2, random_state=42)

# Create a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf_train = tfidf_vectorizer.fit_transform(X_train)
X_tfidf_test = tfidf_vectorizer.transform(X_test)

# Train and evaluate Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_tfidf_train, y_train)
logreg_predictions = logreg_model.predict(X_tfidf_test)

print("\nEvaluation metrics for Logistic Regression:\n")
print(f"Accuracy: {accuracy_score(y_test, logreg_predictions)}")
print(f"Classification Report:\n{classification_report(y_test, logreg_predictions)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, logreg_predictions)}")

# Train and evaluate Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_tfidf_train, y_train)
rf_predictions = rf_model.predict(X_tfidf_test)

print("\nEvaluation metrics for Random Forest:\n")
print(f"Accuracy: {accuracy_score(y_test, rf_predictions)}")
print(f"Classification Report:\n{classification_report(y_test, rf_predictions)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, rf_predictions)}")


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(model_name, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

X = df_tfidf
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)
logreg_predictions = logreg_model.predict(X_test)

plot_confusion_matrix('Logistic Regression', y_test, logreg_predictions, logreg_model.classes_)


# # model evaluation with twiter test
# 
# 

# In[18]:


documents_train = df['Tweet content'].astype(str)
labels_train = df['sentiment']
test_df = pd.read_csv("C:/Users/Zahra's computer/Desktop/textmining/twitter_test.csv",encoding='unicode-escape')
documents_test = test_df['Tweet content'].astype(str)
labels_test = test_df['sentiment']

tfidf_vectorizer = TfidfVectorizer(max_features=5000)  

X_tfidf_train = tfidf_vectorizer.fit_transform(documents_train)

X_tfidf_test = tfidf_vectorizer.transform(documents_test)

svc_model = LinearSVC()
svc_model.fit(X_tfidf_train, labels_train)

svc_predictions_test = svc_model.predict(X_tfidf_test)

print("\nEvaluation metrics for LinearSVC on Test Set:\n")
print(f"Accuracy: {accuracy_score(labels_test, svc_predictions_test)}")
print(f"Classification Report:\n{classification_report(labels_test, svc_predictions_test)}")
print(f"Confusion Matrix:\n{confusion_matrix(labels_test, svc_predictions_test)}")


# # validation

# In[19]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

documents = df['Tweet content'].astype(str)
labels = df['sentiment']
X_train, X_temp, y_train, y_temp = train_test_split(documents, labels, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  

X_tfidf_train = tfidf_vectorizer.fit_transform(X_train)
X_tfidf_validation = tfidf_vectorizer.transform(X_validation)
X_tfidf_test = tfidf_vectorizer.transform(X_test)
svc_model = LinearSVC()
svc_model.fit(X_tfidf_train, y_train)
svc_predictions_validation = svc_model.predict(X_tfidf_validation)
svc_predictions_test = svc_model.predict(X_tfidf_test)
X_validation_df = pd.DataFrame(X_validation, columns=['Tweet content'])
df_predictions_validation = pd.DataFrame({
    'Tweet content': X_validation_df['Tweet content'],
    'Actual sentiment': y_validation,
    'LinearSVC Predicted sentiment': svc_predictions_validation
})

save_predictions_and_metrics(svc_model, X_validation_df['Tweet content'], y_validation, svc_predictions_validation, 'LinearSVC', 'validation_predictions_svc.xlsx')

def save_predictions_and_metrics(model, X, y, predictions, model_name, file_name):
    df_predictions = pd.DataFrame({'Tweet content': X, 'Actual sentiment': y, f'{model_name} Predicted sentiment': predictions})
    df_metrics = pd.DataFrame({
        'Metric': ['Accuracy', 'Classification Report'],
        'Value': [accuracy_score(y, predictions), classification_report(y, predictions)]
    })

    with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
        df_predictions.to_excel(writer, sheet_name=f'{model_name} Predictions', index=False)
        df_metrics.to_excel(writer, sheet_name=f'{model_name} Metrics', index=False)






# # save predictions and metrics in column

# In[18]:


save_predictions_and_metrics(svc_model, X_tfidf_validation, y_validation, svc_predictions_validation, 'LinearSVC', 'validation_predictions_svc.xlsx')


# In[ ]:




