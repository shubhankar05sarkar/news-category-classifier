import joblib

from src.data_preprocessing import load_data
from src.feature_extraction import get_vectorizer, transform_data
from src.train_model import train_models
from src.evaluate import evaluate_model

print("Step 1: Starting...")

print("Step 2: Loading Data...")
train_df, test_df = load_data("data/train.csv", "data/test.csv")

print("Step 3: Data Loaded...")

print("Step 4: Extracting features...")
X_train = train_df['clean_text']
y_train = train_df['Class Index']

X_test = test_df['clean_text']
y_test = test_df['Class Index']

print("Step 5: Vectorizing...")
vectorizer = get_vectorizer()
X_train_vec, X_test_vec = transform_data(vectorizer, X_train, X_test)

print("Step 6: Training...")
models = train_models(X_train_vec, y_train)

print("Step 7: Evaluating...")
for name, model in models.items():
    acc, f1 = evaluate_model(model, X_test_vec, y_test)
    print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

    joblib.dump(model, f"models/{name}.pkl")

joblib.dump(vectorizer, "models/vectorizer.pkl")