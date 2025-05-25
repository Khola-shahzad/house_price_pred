from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-form")
def predict_form():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract and structure input data
        data = {
            "Area": float(request.form["area"]),
            "Bedrooms": int(request.form["bedrooms"]),
            "Bathrooms": int(request.form["bathrooms"]),
            "Floors": int(request.form["floors"]),
            "YearBuilt": int(request.form["yearbuilt"]),
            "Location": request.form["location"],
            "Condition": request.form["condition"],
            "Garage": request.form["garage"]
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # One-hot encode categorical features
        df = pd.get_dummies(df)

        # Load model columns used during training
        model_columns = model.feature_names_in_  # requires scikit-learn >=1.0

        # Add any missing columns with 0
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns to match training
        df = df[model_columns]

        # Make prediction
        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)

        return render_template("predict.html", prediction=prediction)
    
    except Exception as e:
        return render_template("predict.html", error=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True)
