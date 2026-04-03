"""Tests for GeminiClient — written first (TDD).

All google.generativeai calls are patched so no real API key is needed.
"""
import json
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_KEY = "AIzaSy_test_key_123"
FAKE_MODEL = "gemini-2.5-flash"


def _make_mock_response(text: str) -> MagicMock:
    """Return a mock object that looks like a Gemini GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGeminiClientInit(unittest.TestCase):
    """Initialisation and configuration tests."""

    @patch("src.gemini.client.genai")
    def test_client_initializes_with_defaults(self, mock_genai):
        """Client stores config and configures the genai library."""
        from src.gemini.client import GeminiClient

        client = GeminiClient(api_key=FAKE_KEY, model_name=FAKE_MODEL)

        # genai.configure should have been called with the key
        mock_genai.configure.assert_called_once_with(api_key=FAKE_KEY)

        # Defaults
        self.assertEqual(client.model_name, FAKE_MODEL)
        self.assertEqual(client.rpm_limit, 15)
        self.assertEqual(client.daily_limit, 1500)

    @patch("src.gemini.client.genai")
    def test_no_api_key_raises(self, mock_genai):
        """Empty api_key must raise ValueError immediately."""
        from src.gemini.client import GeminiClient

        with self.assertRaises(ValueError):
            GeminiClient(api_key="", model_name=FAKE_MODEL)

        with self.assertRaises(ValueError):
            GeminiClient(api_key="   ", model_name=FAKE_MODEL)


class TestGeminiClientGenerate(unittest.TestCase):
    """generate_text and generate_json tests."""

    def setUp(self):
        patcher = patch("src.gemini.client.genai")
        self.mock_genai = patcher.start()
        self.addCleanup(patcher.stop)

        # Set up a mock model instance returned by genai.GenerativeModel(...)
        self.mock_model = MagicMock()
        self.mock_genai.GenerativeModel.return_value = self.mock_model

        from src.gemini.client import GeminiClient
        self.client = GeminiClient(api_key=FAKE_KEY, model_name=FAKE_MODEL)

    # ------------------------------------------------------------------
    # generate_text
    # ------------------------------------------------------------------

    def test_generate_text_returns_response(self):
        """generate_text returns the .text of the API response."""
        self.mock_model.generate_content.return_value = _make_mock_response("Hello world")

        result = self.client.generate_text("Say hello")

        self.assertEqual(result, "Hello world")
        self.mock_model.generate_content.assert_called_once()

    def test_generate_text_with_image(self):
        """generate_text passes images alongside the prompt."""
        self.mock_model.generate_content.return_value = _make_mock_response("Image desc")
        fake_image = MagicMock(name="PIL.Image")

        result = self.client.generate_text("Describe this", images=[fake_image])

        self.assertEqual(result, "Image desc")
        call_args = self.mock_model.generate_content.call_args
        # First positional argument is the content list
        content_list = call_args[0][0]
        self.assertIn(fake_image, content_list)

    def test_generate_text_tracks_request_count(self):
        """Each successful call increments requests_today."""
        self.mock_model.generate_content.return_value = _make_mock_response("ok")

        before = self.client.requests_today
        self.client.generate_text("ping")
        self.assertEqual(self.client.requests_today, before + 1)

    # ------------------------------------------------------------------
    # generate_json
    # ------------------------------------------------------------------

    def test_generate_json_parses_response(self):
        """generate_json returns a dict when the model outputs valid JSON."""
        payload = {"answer": "42", "confidence": 0.99}
        self.mock_model.generate_content.return_value = _make_mock_response(
            json.dumps(payload)
        )

        result = self.client.generate_json("Give me JSON")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["answer"], "42")

    def test_generate_json_handles_code_fence(self):
        """generate_json extracts JSON wrapped in markdown code fences."""
        payload = {"key": "value"}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        self.mock_model.generate_content.return_value = _make_mock_response(fenced)

        result = self.client.generate_json("Give me fenced JSON")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["key"], "value")

    def test_generate_json_handles_malformed_json(self):
        """generate_json falls back to raw_response when JSON cannot be parsed."""
        self.mock_model.generate_content.return_value = _make_mock_response(
            "This is not JSON at all."
        )

        result = self.client.generate_json("Give me bad JSON")

        self.assertIsInstance(result, dict)
        self.assertIn("raw_response", result)
        self.assertEqual(result["raw_response"], "This is not JSON at all.")

    def test_generate_json_extracts_inline_object(self):
        """generate_json finds a JSON object embedded in surrounding text."""
        payload = {"status": "ok"}
        text = f'Here is the answer: {json.dumps(payload)} — done.'
        self.mock_model.generate_content.return_value = _make_mock_response(text)

        result = self.client.generate_json("Inline object")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], "ok")


class TestGeminiClientQuota(unittest.TestCase):
    """Quota tracking tests."""

    def setUp(self):
        patcher = patch("src.gemini.client.genai")
        self.mock_genai = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_model = MagicMock()
        self.mock_genai.GenerativeModel.return_value = self.mock_model

        from src.gemini.client import GeminiClient
        self.client = GeminiClient(
            api_key=FAKE_KEY,
            model_name=FAKE_MODEL,
            daily_limit=100,
            rpm_limit=10,
        )

    def test_quota_remaining(self):
        """quota_remaining returns correct dict structure and values."""
        self.mock_model.generate_content.return_value = _make_mock_response("ok")

        # Make 3 requests
        for _ in range(3):
            self.client.generate_text("test")

        quota = self.client.quota_remaining()

        self.assertIn("daily_remaining", quota)
        self.assertIn("rpm_remaining", quota)
        self.assertIn("daily_used", quota)
        self.assertEqual(quota["daily_used"], 3)
        self.assertEqual(quota["daily_remaining"], 97)   # 100 - 3
        self.assertEqual(quota["rpm_remaining"], 7)      # 10 - 3

    def test_client_tracks_request_count(self):
        """requests_today and requests_this_minute reflect live usage."""
        self.mock_model.generate_content.return_value = _make_mock_response("ok")

        self.assertEqual(self.client.requests_today, 0)
        self.assertEqual(self.client.requests_this_minute, 0)

        self.client.generate_text("one")
        self.client.generate_text("two")

        self.assertEqual(self.client.requests_today, 2)
        self.assertEqual(self.client.requests_this_minute, 2)


if __name__ == "__main__":
    unittest.main()
