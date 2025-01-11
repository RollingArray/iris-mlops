import unittest
import requests

class TestApp(unittest.TestCase):

    def test_predict(self):
        url = "http://127.0.0.1:5000/predict"
        data = {"features": [15.1, 3.5, 1.4, 0.2]}  # Example input
        
        print("Sending POST request to:", url)
        print("With data:", data)
        
        # Send the POST request
        response = requests.post(url, json=data)
        
        # Print actual response details for debugging
        print("Response Status Code:", response.status_code)
        print("Response JSON:", response.json())

        # Check if status code is 200 (success)
        self.assertEqual(response.status_code, 200, f"Expected status code 200, but got {response.status_code}")

        # Extract and check the response JSON
        response_json = response.json()
        self.assertIn("prediction", response_json, "Expected 'prediction' key in response JSON.")
        
        print("Test passed. Response contains the 'prediction' key and status code is 200.")

if __name__ == "__main__":
    unittest.main()
