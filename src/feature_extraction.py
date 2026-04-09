from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,3),

        min_df=2,
        max_df=0.9
        )

def transform_data(vectorizer, X_train, X_test):
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec