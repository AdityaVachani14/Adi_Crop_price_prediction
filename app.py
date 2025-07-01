import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

# Load model and encoders
with open("label_encoded_cost_model.pkl", "rb") as f:
    bundle = pickle.load(f)
    model = bundle["model"]
    label_encoders = bundle["label_encoders"]

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            crop = request.form["crop"]
            state = request.form["state"]
            region = request.form["region"]
            soil = request.form["soil"]
            land_area = float(request.form["land_area"])
            temperature = float(request.form["temperature"])
            rainfall = float(request.form["rainfall"])
            budget = float(request.form["budget"])

            # Prepare input
            raw_input = pd.DataFrame([{
                "Crop": crop,
                "State": state,
                "Region": region,
                "Soil Type": soil,
                "Land Area (acres)": land_area,
                "Temperature (°C)": temperature,
                "Rainfall (mm)": rainfall,
                "Budget (₹)": budget
            }])

            # Encode categorical columns
            for col in label_encoders:
                val = raw_input[col].values[0]
                le = label_encoders[col]

                if val not in le.classes_:
                    return f"Unknown category '{val}' in column '{col}'", 400
                raw_input[col] = le.transform([val])[0]

            # Predict
            prediction = model.predict(raw_input)[0]
            output = {
                "Seed Cost": round(prediction[0], 2),
                "Fertilizer Cost": round(prediction[1], 2),
                "Irrigation Cost": round(prediction[2], 2)
            }

            return render_template("index.html", result=output)

        except Exception as e:
            return render_template("index.html", result={"error": str(e)})

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
