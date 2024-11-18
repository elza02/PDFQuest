import unittest
from fastapi.testclient import TestClient
from app.main import app  # Adjust import if the app structure changes

client = TestClient(app)

class TestFastAPI(unittest.TestCase):
    def test_root_endpoint(self):
        """Test if the root endpoint serves the HTML file correctly."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<html", response.text)  # Assuming the HTML file starts with <html>

    def test_chat_endpoint(self):
        """Test the /chat endpoint with a valid question."""
        payload = {"question": "quelles sont les droites d'un étudiant?"}
        response = client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("response", response.json())  # Ensure response contains the expected key

    def test_chat_invalid_payload(self):
        """Test the /chat endpoint with an invalid payload."""
        payload = {"wrong_key": "quelles sont les droites d'un étudiant?"}
        response = client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 422)  # 422 Unprocessable Entity

if __name__ == "__main__":
    unittest.main()
