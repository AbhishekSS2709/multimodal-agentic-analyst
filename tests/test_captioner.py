"""Tests for ImageCaptioner — written TDD style; GeminiClient is fully mocked."""

import unittest
from unittest.mock import MagicMock, patch


FAKE_KEY = "AIzaSy_test_key_captioner"


def _make_image_mock(width: int = 100, height: int = 80, mode: str = "RGB") -> MagicMock:
    """Return a mock that looks like a PIL.Image."""
    img = MagicMock()
    img.size = (width, height)
    img.mode = mode
    return img


class TestImageCaptioner(unittest.TestCase):

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_captioner(self, mock_client_cls):
        """Instantiate ImageCaptioner with the provided mock GeminiClient class."""
        from src.gemini.captioner import ImageCaptioner
        return ImageCaptioner(api_key=FAKE_KEY, rpm_limit=8), mock_client_cls

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @patch("src.gemini.captioner.GeminiClient")
    def test_caption_image(self, mock_client_cls):
        """caption() returns the text produced by GeminiClient.generate_text."""
        mock_instance = MagicMock()
        mock_instance.generate_text.return_value = "A red square image"
        mock_client_cls.return_value = mock_instance

        from src.gemini.captioner import ImageCaptioner
        captioner = ImageCaptioner(api_key=FAKE_KEY)

        img = _make_image_mock()
        result = captioner.caption(img)

        self.assertEqual(result, "A red square image")
        mock_instance.generate_text.assert_called_once()
        # Verify the image was passed along
        call_kwargs = mock_instance.generate_text.call_args
        self.assertIn(img, call_kwargs.kwargs.get("images", []))

    @patch("src.gemini.captioner.GeminiClient")
    def test_caption_batch(self, mock_client_cls):
        """caption_batch() returns one caption per image."""
        mock_instance = MagicMock()
        mock_instance.generate_text.return_value = "An image"
        mock_client_cls.return_value = mock_instance

        from src.gemini.captioner import ImageCaptioner
        captioner = ImageCaptioner(api_key=FAKE_KEY)

        images = [_make_image_mock(), _make_image_mock(200, 150)]
        results = captioner.caption_batch(images, delay_between=0)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r == "An image" for r in results))
        self.assertEqual(mock_instance.generate_text.call_count, 2)

    @patch("src.gemini.captioner.GeminiClient")
    def test_fallback_when_quota_exhausted(self, mock_client_cls):
        """caption() returns a fallback string when GeminiClient raises RuntimeError."""
        mock_instance = MagicMock()
        mock_instance.generate_text.side_effect = RuntimeError(
            "Daily request limit (8) reached."
        )
        mock_client_cls.return_value = mock_instance

        from src.gemini.captioner import ImageCaptioner
        captioner = ImageCaptioner(api_key=FAKE_KEY)

        img = _make_image_mock(320, 240, "RGBA")
        result = captioner.caption(img)

        # Must not raise; must include size info and unavailability note
        self.assertIn("320x240", result)
        self.assertIn("RGBA", result)
        self.assertIn("caption unavailable", result)

    @patch("src.gemini.captioner.GeminiClient")
    def test_fallback_caption_format(self, mock_client_cls):
        """_fallback_caption() produces the expected format string."""
        mock_client_cls.return_value = MagicMock()

        from src.gemini.captioner import ImageCaptioner
        captioner = ImageCaptioner(api_key=FAKE_KEY)

        img = _make_image_mock(640, 480, "RGB")
        result = captioner._fallback_caption(img)

        self.assertEqual(
            result,
            "Image (640x480, RGB). Visual content — detailed caption unavailable.",
        )


if __name__ == "__main__":
    unittest.main()
