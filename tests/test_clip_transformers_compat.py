"""CLIP feature extraction must work on transformers 4.x and 5.x.

transformers 5.x changed `get_image_features` / `get_text_features` to return a
`BaseModelOutputWithPooling` instead of a bare tensor, so the old
`features.detach()` raised:

    'BaseModelOutputWithPooling' object has no attribute 'detach'

Every image failed to embed, the visual store stayed empty and `modality_match`
could not rise above 0.200. The unit tests missed it because they mock the model
and the mock returned a tensor -- the shape of the real API was never exercised.
"""

import unittest

import numpy as np


class _Pooled:
    """Stands in for transformers 5.x BaseModelOutputWithPooling."""

    def __init__(self, tensor):
        self.pooler_output = tensor
        self.last_hidden_state = tensor


class _Tensor:
    """Stands in for the transformers 4.x bare tensor."""

    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class TestFeatureExtraction(unittest.TestCase):

    def _extract(self, value):
        from src.embedding.clip_engine import _feature_tensor
        return _feature_tensor(value)

    def test_bare_tensor_passes_through(self):
        tensor = _Tensor(np.zeros((1, 512), dtype=np.float32))
        self.assertIs(self._extract(tensor), tensor)

    def test_pooled_output_is_unwrapped(self):
        tensor = _Tensor(np.zeros((1, 512), dtype=np.float32))
        self.assertIs(self._extract(_Pooled(tensor)), tensor)

    def test_unwrapped_result_is_detachable(self):
        """Whatever comes back must support the .detach().cpu().numpy() chain."""
        tensor = _Tensor(np.ones((1, 512), dtype=np.float32))
        out = self._extract(_Pooled(tensor))
        self.assertEqual(out.detach().cpu().numpy().shape, (1, 512))


if __name__ == "__main__":
    unittest.main()
