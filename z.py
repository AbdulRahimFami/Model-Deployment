from flask import Flask,render_template,request
import joblib



app = Flask(__name__)
model = joblib.load(r"model\iris_logistic_model.joblib")
scalar = joblib.load(r"model\iris_scaler.joblib")

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=["POST"])
def predict():
    try:
        # Read values from the form and convert to float
        f1 = float(request.form.get('feature1', 0))
        f2 = float(request.form.get('feature2', 0))
        f3 = float(request.form.get('feature3', 0))
        f4 = float(request.form.get('feature4', 0))

        features = [[f1, f2, f3, f4]]
        features_scaled = scalar.transform(features)
        prediction = model.predict(features_scaled)
        class_name = ['setosa', 'versicolor', 'virginica'][prediction[0]]

        # Always return a string
        return f"<h2>Prediction: {class_name}</h2>"

    except Exception as e:
        # Return error as string instead of None
        return f"<h2>Error: {str(e)}</h2>"


if __name__ == "__main__":
    app.run(debug=True)