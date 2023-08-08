from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("6-dalis\\model_top5.joblib")


@app.route("/", methods=["GET"])
def index():
    return "ML model for GDP per capita prediction is running"


@app.route("/predict_gdp", methods=["POST"])
def predict_gdp():
    try:
        # Get the JSON data from the request
        data = request.json
        current_account_balance = data.get("BCA")
        general_government_gross_debt = data.get("GGXWDG")
        population = data.get("LP")
        unemployment_rate = data.get("LUR")
        implied_PPP_conversion_rate = data.get("PPPEX")

        # Prepare input for your prediction model
        # Create a dictionary
        input_data = {}

        # Add the features to the dictionary
        input_data["BCA"] = [current_account_balance]
        input_data["GGXWDG"] = [general_government_gross_debt]
        input_data["LP"] = [population]
        input_data["LUR"] = [unemployment_rate]
        input_data["PPPEX"] = [implied_PPP_conversion_rate]

        # Make input data into dataframe
        input_data = pd.DataFrame(input_data)

        # Make the prediction using the model
        predicted_gdp_per_capita = model.predict(input_data).astype(float)

        # Print the predicted GDP per capita in the terminal
        print("Predicted GDP per capita:", predicted_gdp_per_capita)

        # Return the predicted GDP per capita in the response
        response = {
            "predicted_gdp_per_capita": predicted_gdp_per_capita.tolist(),
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 200


if __name__ == "__main__":
    app.run(debug=True)
