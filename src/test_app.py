import unittest
from app import app


class TestApp(unittest.TestCase):

    def test_predict(self):
        tester = app.test_client()
        response = tester.post(
            "/predict",
            json={"features": [5.1, 3.5, 1.4, 0.2]},  # Example input
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.json)


if __name__ == "__main__":
    unittest.main()
