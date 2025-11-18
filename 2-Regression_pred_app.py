from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/linear_regressor.joblib")

# Initialize Flask app
app = Flask(__name__)

# HTML Template for Home Page
HTML_TEMPLATE = """
<!doctype html>
<html>
    <head>
        <title>Emp Salary Prediction</title>
    </head>
    <body>
        <h1>Emp Salary Prediction</h1>
        <form action="/predict" method="post">
            <label for="experience">Enter Years of Experience (comma-separated for multiple):</label>
            <br><br>
            <input type="text" id="YOE" name="YOE" placeholder="e.g., 1, 2, 3">
            <br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """
    Home page with a simple form for input.
    """
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint that accepts input via:
    1. Form submission (UI)
    2. JSON API (curl or Postman)
    """
    try:
        # If request is from a form submission
        if request.form:
            experience = request.form.get("YOE")
            if not experience:
                return "Please provide years of experience values.", 400

            # Parse comma-separated values
            experience = [float(x) for x in experience.split(",")]

        # If request is JSON (from API)
        elif request.is_json:
            data = request.get_json()
            experience = data.get("YOE")
            if experience is None:
                return jsonify({"error": "Missing 'YOE' in request"}), 400
            if not isinstance(experience, list):
                experience = [experience]

        else:
            return "Unsupported request type. Use form submission or JSON.", 400

        # Convert input to numpy array and make predictions
        X_input = np.array(experience).reshape(-1, 1)
        predictions = model.predict(X_input).tolist()

        # Return JSON response
        if request.is_json:
            return jsonify({"predictions": predictions})

        # Return predictions for form submission
        return f"Predicted Salaries: {predictions}"

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)