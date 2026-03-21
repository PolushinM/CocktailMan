"""Tests for main.py business logic (ImageProcessor is mocked)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Must be set before config.py is imported
os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-exactly-32ch')

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

# Mock ImageProcessor BEFORE main.py is imported so models are never loaded
_mock_proc = MagicMock()
_mock_proc.ingredients_text = [
    {'nominative': 'Водка', 'genitive': 'водки'},
    {'nominative': 'Ром', 'genitive': 'рома'},
    {'nominative': 'Лайм', 'genitive': 'лайма'},
]
_mock_proc.generator.latent_size = 4
_mock_proc.predict_blur_save.return_value = ([], 0.0)
_mock_proc.generate_to_file.return_value = None

_mock_models = MagicMock()
_mock_models.ImageProcessor = MagicMock(return_value=_mock_proc)
sys.modules.setdefault('models', MagicMock())
sys.modules['models.models'] = _mock_models

from main import generate_recipe, get_confidence_text, predict, generate_image


class TestGenerateRecipe(unittest.TestCase):
    def test_empty_list_returns_default(self):
        result = generate_recipe([])
        self.assertIn('Не могу найти', result)

    def test_single_ingredient(self):
        result = generate_recipe(['водки'])
        self.assertIn('водки', result)
        self.assertTrue(result.startswith('Добавим только'))
        self.assertTrue(result.endswith('.'))

    def test_two_ingredients(self):
        result = generate_recipe(['водки', 'рома'])
        self.assertIn('водки', result)
        self.assertIn('рома', result)
        self.assertIn(' и ', result)
        self.assertTrue(result.startswith('Попробуем добавить'))

    def test_many_ingredients(self):
        result = generate_recipe(['водки', 'рома', 'лайма'])
        self.assertIn('водки', result)
        self.assertIn('рома', result)
        self.assertIn('лайма', result)
        self.assertIn(', ', result)
        self.assertIn(' и ', result)

    def test_returns_string(self):
        self.assertIsInstance(generate_recipe([]), str)
        self.assertIsInstance(generate_recipe(['водки']), str)


class TestGetConfidenceText(unittest.TestCase):
    def test_zero_confidence_returns_empty(self):
        self.assertEqual(get_confidence_text(0.0), '')

    def test_high_confidence_returns_text(self):
        result = get_confidence_text(1.0)
        self.assertIn('Я уверен на', result)
        self.assertIn('%', result)

    def test_medium_confidence_returns_text(self):
        result = get_confidence_text(0.5)
        self.assertIn('%', result)

    def test_format_contains_percent_sign(self):
        result = get_confidence_text(0.8)
        self.assertRegex(result, r'\d+%')

    def test_high_confidence_shows_high_percent(self):
        result = get_confidence_text(1.0)
        self.assertIn('100%', result)


class TestPredict(unittest.TestCase):
    def setUp(self):
        os.makedirs('cache', exist_ok=True)
        _mock_proc.predict_blur_save.return_value = (['водки'], 0.8)
        _mock_proc.predict_blur_save.reset_mock()

    def test_file_src_saves_file(self):
        mock_file = MagicMock()
        recipe, confidence, filename = predict(mock_file, src_type='file')
        mock_file.save.assert_called_once()

    def test_file_src_returns_tuple(self):
        mock_file = MagicMock()
        result = predict(mock_file, src_type='file')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_file_src_filename_is_jpg(self):
        mock_file = MagicMock()
        _, _, filename = predict(mock_file, src_type='file')
        self.assertTrue(filename.endswith('.jpg'))

    def test_file_src_recipe_is_string(self):
        mock_file = MagicMock()
        recipe, _, _ = predict(mock_file, src_type='file')
        self.assertIsInstance(recipe, str)

    def test_file_src_confidence_is_float(self):
        mock_file = MagicMock()
        _, confidence, _ = predict(mock_file, src_type='file')
        self.assertIsInstance(confidence, float)

    def test_file_src_calls_predict_blur_save(self):
        mock_file = MagicMock()
        predict(mock_file, src_type='file')
        _mock_proc.predict_blur_save.assert_called_once()

    @patch('urllib.request.urlopen')
    def test_url_src_returns_tuple(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = b''
        result = predict('http://example.com/img.jpg', src_type='url')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    @patch('urllib.request.urlopen')
    def test_url_src_opens_with_timeout(self, mock_urlopen):
        mock_urlopen.return_value.read.return_value = b''
        predict('http://example.com/img.jpg', src_type='url')
        _, kwargs = mock_urlopen.call_args if mock_urlopen.call_args else ((), {})
        args = mock_urlopen.call_args[0] if mock_urlopen.call_args else ()
        # urlopen called with timeout kwarg
        call_kwargs = mock_urlopen.call_args[1] if mock_urlopen.call_args else {}
        self.assertIn('timeout', call_kwargs)
        self.assertEqual(call_kwargs['timeout'], 10)

    @patch('urllib.request.urlopen')
    def test_url_src_handles_network_error_gracefully(self, mock_urlopen):
        mock_urlopen.side_effect = OSError('Network error')
        try:
            predict('http://example.com/img.jpg', src_type='url')
        except Exception as e:
            self.fail(f"predict raised an exception on network error: {e}")


class TestGenerateImage(unittest.TestCase):
    def setUp(self):
        os.makedirs('cache', exist_ok=True)
        _mock_proc.generate_to_file.reset_mock()

    def test_returns_string_path(self):
        import numpy as np
        result = generate_image(
            latent=np.zeros(4),
            background=np.array([0, 0, 0]),
            ingr_list=[]
        )
        self.assertIsInstance(result, str)

    def test_path_ends_with_jpg(self):
        import numpy as np
        result = generate_image(
            latent=np.zeros(4),
            background=np.array([0, 0, 0]),
            ingr_list=[]
        )
        self.assertTrue(result.endswith('.jpg'))

    def test_calls_generate_to_file(self):
        import numpy as np
        generate_image(
            latent=np.zeros(4),
            background=np.array([128, 128, 128]),
            ingr_list=[0, 1]
        )
        _mock_proc.generate_to_file.assert_called_once()

    def test_condition_vector_reflects_ingr_list(self):
        import numpy as np
        captured_condition = {}

        def capture_call(latent, background, condition, path):
            captured_condition['value'] = condition.copy()

        _mock_proc.generate_to_file.side_effect = capture_call
        generate_image(
            latent=np.zeros(4),
            background=np.array([0, 0, 0]),
            ingr_list=[0, 2]
        )
        condition = captured_condition['value']
        self.assertEqual(condition[0], 1.0)
        self.assertEqual(condition[2], 1.0)
        self.assertEqual(condition[1], 0.0)
        _mock_proc.generate_to_file.side_effect = None


if __name__ == '__main__':
    unittest.main()
