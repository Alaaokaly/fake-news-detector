from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import joblib

import pandas as pd 
import re 

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def load_prepare_data(file_path):
    data = pd.DataFrame(pd.read_csv("data/fake_or_real_news.csv"))
    data['all_text'] = data['text'] + data['title']
    data['text_cleaned'] = data['all_text'].apply(clean_text)
    data = data[data['text_cleaned'].str.len() > 10]
    return data 

# print(data.head(10))
data =  load_prepare_data("data/fake_or_real_news.csv")
y = data['label']
x =  data['text_cleaned']
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
X_train, X_test, y_train, y_test = train_test_split(x,y, 
                                                    random_state = 53, test_size=0.33)

count_vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Reduced from (1,3) for better performance
        max_df=0.85,
        min_df=5,  # Added min_df to filter rare terms
        max_features=10000,  # Limit features for performance
        lowercase=True,
        strip_accents='unicode'
    )
tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=5,
        max_features=10000,
        lowercase=True,
        strip_accents='unicode',
        sublinear_tf=True  # Use sublinear tf scaling
    )


tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

classifiers = {
    "nb": MultinomialNB(),
    "lr": LogisticRegression(max_iter=1000)
}
vectorizers = { "count": count_vectorizer,
               "tfidf" : tfidf_vectorizer}
evaluation_matrices = []
for vec_name , vec in vectorizers.items():
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    for clf_name, clf in classifiers.items():
        clf.fit(X_train_vec, y_train)
        preds  = clf.predict(X_test_vec)
        accuracy = metrics.accuracy_score(y_test, preds)
        precision = metrics.precision_score(y_test, preds, pos_label='FAKE')
        recall = metrics.recall_score(y_test, preds, pos_label='FAKE')
        f1 = metrics.f1_score(y_test, preds, pos_label='FAKE')
        score = metrics.accuracy_score(y_test,preds)
        cm =  metrics.confusion_matrix(y_test, preds, labels=['FAKE', 'REAL'])
        joblib.dump(clf, f"model/{clf_name}_model_{vec_name}.pkl")
        joblib.dump(vec, f"model/{vec_name}_vectorizer.pkl")
        evaluation_matrices.append({
         'vectorizer_name' :vec_name,
         'classifer':clf_name,  
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    })
df = pd.DataFrame(evaluation_matrices)
df.to_csv('model/evaluation.csv',index=False)


