from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

from data_preprocessing import load_data, preprocess_data

df = load_data("data/Advertisements-Data.csv")

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

joblib.dump(model, "models/ctr_model.pkl")

print("Model saved successfully")