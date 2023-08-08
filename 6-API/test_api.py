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

        # Iterate over the JSON data
        for country in json_data:
            # Get the JSON data for each country
            json_data = country["features"]
            target_GPD = country["target"]

            # Send a POST request to the /predict_gdp endpoint
            response = self.app.post(
                "/predict_gdp",
                data=json.dumps(json_data),
                content_type="application/json",
            )

            # Check if the response status code is 200 (OK)
            self.assertEqual(response.status_code, 200)

            # Parse the JSON response data
            data = json.loads(response.data)

            # Check if the expected keys are present in the response
            for key in target_GPD:
                self.assertIn(key, data)

            # Check if the response contains the predicted_gdp_per_capita
            self.assertIn("predicted_gdp_per_capita", data)

            # Check if the predicted_gdp_per_capita value is a float
            self.assertIsInstance(data["predicted_gdp_per_capita"][0], float)


if __name__ == "__main__":
    unittest.main()
