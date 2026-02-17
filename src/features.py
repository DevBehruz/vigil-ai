from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack

def build_tfidf(X_train, X_test):
    tfidf = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 3),
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf

def build_hybrid_features(X_train, X_test):
    word_vectorizer = CountVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2
    )

    char_vectorizer = CountVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=5000
    )

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_test_word = word_vectorizer.transform(X_test)

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_test_char = char_vectorizer.transform(X_test)

    X_train_combined = hstack([X_train_word, X_train_char])
    X_test_combined = hstack([X_test_word, X_test_char])

    return X_train_combined, X_test_combined
