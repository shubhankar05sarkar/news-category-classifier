import joblib
vectorizer = joblib.load('models/vectorizer.pkl')


def predict(text, model_name):
    model = joblib.load(f"models/{model_name}.pkl")
    
    from src.data_preprocessing import clean_text
    text = clean_text(text)

    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    mapping = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech"
    }

    return mapping[pred]