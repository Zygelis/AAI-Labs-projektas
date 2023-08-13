import unittest
import json
from api import *


class Unittest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_gdp(self):
        # Read JSON data from the file
        with open("6-API\\test_data.json") as json_file:
            json_data = json.load(json_file)

        # Get the JSON data
        data = json_data[0]["features"]

        # Send a POST request to the /predict_gdp endpoint
        response = self.app.post(
            "/predict_gdp",
            data=json.dumps(json_data),
            content_type="application/json",
        )

        response = self.app.post("/predict_gdp", json=data)
        data = response.get_json()

        # Check the response
        self.assertEqual(response.status_code, 200)

        self.assertTrue("predicted_gdp_per_capita" in data)

        # Check if the predicted_gdp_per_capita value is a float
        self.assertIsInstance(data["predicted_gdp_per_capita"][0], float)


if __name__ == "__main__":
    unittest.main()
