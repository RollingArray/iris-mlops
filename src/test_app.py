import unittest
import requests
import requests_mock


class TestApp(unittest.TestCase):

    def test_predict(self):
        # Use requests-mock to mock the API endpoint
        with requests_mock.Mocker() as mock:
            # Mock the API response
            mock.post(
                "http://0.0.0.0:5000/predict",
                json={"prediction": "Iris-setosa"},
                status_code=200
            )

            url = "http://0.0.0.0:5000/predict"
            data = {"features": [5.2, 3.7, 1.5, 0.2]}  # Example input

            # Send the request (it will use the mocked response)
            response = requests.post(url, json=data)

            # Assert that the mock response is returned
            self.assertEqual(response.status_code, 200)
            self.assertIn("prediction", response.json())
            self.assertEqual(response.json()["prediction"], "Iris-setosa")


if __name__ == "__main__":
    unittest.main()
