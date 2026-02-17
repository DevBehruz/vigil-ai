from src.data_loader import load_and_split_data
from src.preprocessing import preprocess_text_aggressive
from src.features import build_tfidf
from src.train_models import get_models
from src.evaluate import evaluate

def main():
    X_train, X_test, y_train, y_test = load_and_split_data(
        "edos_labelled_data.csv"
    )

    X_train = X_train.apply(preprocess_text_aggressive)
    X_test = X_test.apply(preprocess_text_aggressive)

    X_train_feat, X_test_feat = build_tfidf(X_train, X_test)

    models = get_models()

    for name, model in models.items():
        print(f"\nTraining {name}")
        report = evaluate(model, X_train_feat, y_train, X_test_feat, y_test)
        print(report)

if __name__ == "__main__":
    main()
