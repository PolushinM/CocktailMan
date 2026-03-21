"""Tests for Flask routes in views.py (ImageProcessor is mocked)."""

import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Must be set before config.py is imported
os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-exactly-32ch')

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

# Mock ImageProcessor BEFORE any project module is imported
_mock_proc = MagicMock()
_mock_proc.ingredients_text = [
    {'nominative': 'Водка', 'genitive': 'водки'},
    {'nominative': 'Ром', 'genitive': 'рома'},
]
_mock_proc.generator.latent_size = 4
_mock_proc.predict_blur_save.return_value = ([], 0.0)
_mock_proc.generate_to_file.return_value = None

_mock_models = MagicMock()
_mock_models.ImageProcessor = MagicMock(return_value=_mock_proc)
sys.modules.setdefault('models', MagicMock())
sys.modules['models.models'] = _mock_models

from views import app


class _ViewsTestBase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        self.ctx = app.test_request_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexGet(_ViewsTestBase):
    def test_returns_200(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)

    def test_contains_project_name(self):
        resp = self.client.get('/')
        self.assertIn(b'CocktailMan', resp.data)

    def test_contains_ingredients_section(self):
        resp = self.client.get('/')
        self.assertIn('Доступные ингредиенты'.encode('utf-8'), resp.data)

    def test_contains_upload_prompt(self):
        resp = self.client.get('/')
        self.assertIn('Выбери файл'.encode('utf-8'), resp.data)


# ---------------------------------------------------------------------------
# POST / — no input
# ---------------------------------------------------------------------------

class TestIndexPostEmpty(_ViewsTestBase):
    # WTF FileSize passes with an empty FileStorage (size=0), so form.validate_on_submit()
    # returns True and the code reaches request.files['input_file'] — must always include
    # the field (even empty) to avoid a 400 KeyError from Werkzeug.
    def _post_empty(self):
        return self.client.post('/', data={
            'image_url': '',
            'input_file': (io.BytesIO(b''), ''),
        }, content_type='multipart/form-data')

    def test_no_input_returns_200(self):
        resp = self._post_empty()
        self.assertEqual(resp.status_code, 200)

    @patch('views.flash')
    def test_no_input_calls_flash(self, mock_flash):
        # Flash messages are stored in session and rendered via JS (const messages=[...]),
        # not as raw UTF-8 in HTML — assert the call directly.
        self._post_empty()
        mock_flash.assert_called_once()


# ---------------------------------------------------------------------------
# POST / — URL input
# ---------------------------------------------------------------------------

class TestIndexPostUrl(_ViewsTestBase):
    def _post_url(self, url='http://example.com/img.jpg'):
        return self.client.post('/', data={
            'image_url': url,
            'input_file': (io.BytesIO(b''), ''),
        }, content_type='multipart/form-data')

    @patch('views.predict')
    @patch('views.uri_validator', return_value=True)
    def test_valid_url_calls_predict(self, mock_validator, mock_predict):
        mock_predict.return_value = ('Добавим водки.', 0.8, 'abcde12345.jpg')
        resp = self._post_url()
        self.assertEqual(resp.status_code, 200)
        mock_predict.assert_called_once_with('http://example.com/img.jpg', src_type='url')

    @patch('views.predict')
    @patch('views.uri_validator', return_value=True)
    def test_recipe_call_returns_200(self, mock_validator, mock_predict):
        # Recipe text is JSON-escaped in JS (const messages=[...]), not raw UTF-8 in HTML.
        # Verify predict was called and response is OK.
        mock_predict.return_value = ('Добавим водки.', 0.9, 'abcde12345.jpg')
        resp = self._post_url()
        self.assertEqual(resp.status_code, 200)
        mock_predict.assert_called_once()

    @patch('views.flash')
    @patch('views.predict', side_effect=Exception('download error'))
    @patch('views.uri_validator', return_value=True)
    def test_predict_exception_calls_flash(self, mock_validator, mock_predict, mock_flash):
        self._post_url()
        mock_flash.assert_called_once()


# ---------------------------------------------------------------------------
# POST / — file input
# ---------------------------------------------------------------------------

class TestIndexPostFile(_ViewsTestBase):
    @patch('views.predict')
    def test_file_upload_calls_predict(self, mock_predict):
        mock_predict.return_value = ('Добавим рома.', 0.7, 'abcde12345.jpg')
        data = {
            'input_file': (io.BytesIO(b'fake image data'), 'test.jpg'),
            'image_url': 'http://example.com/img.jpg',
        }
        resp = self.client.post('/', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)

    @patch('views.flash')
    @patch('views.predict', side_effect=Exception('read error'))
    def test_file_predict_exception_calls_flash(self, mock_predict, mock_flash):
        data = {
            'input_file': (io.BytesIO(b'bad data'), 'test.jpg'),
            'image_url': 'http://example.com/img.jpg',
        }
        self.client.post('/', data=data, content_type='multipart/form-data')
        mock_flash.assert_called_once()


# ---------------------------------------------------------------------------
# GET /generative_model
# ---------------------------------------------------------------------------

class TestGenerativeModelGet(_ViewsTestBase):
    def test_returns_200(self):
        resp = self.client.get('/generative_model')
        self.assertEqual(resp.status_code, 200)

    def test_contains_ingredients(self):
        resp = self.client.get('/generative_model')
        self.assertIn('Водка'.encode('utf-8'), resp.data)


# ---------------------------------------------------------------------------
# POST /generative_model
# ---------------------------------------------------------------------------

class TestGenerativeModelPost(_ViewsTestBase):
    @patch('views.generate_image')
    def test_valid_params_returns_json(self, mock_gen):
        mock_gen.return_value = 'cache/abcde12345.jpg'
        resp = self.client.post('/generative_model', data={
            'Range_1': '50',
            'Color': '#ff0000',
        })
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIn('image_path', data)

    @patch('views.generate_image')
    def test_valid_ingredient_included(self, mock_gen):
        mock_gen.return_value = 'cache/abcde12345.jpg'
        resp = self.client.post('/generative_model', data={
            'Ingr_0': 'on',
            'Color': '#000000',
        })
        self.assertEqual(resp.status_code, 200)
        _, _, kwargs = mock_gen.mock_calls[0]
        self.assertIn(0, mock_gen.call_args[1].get('ingr_list', []))

    @patch('views.generate_image')
    def test_invalid_color_does_not_crash(self, mock_gen):
        mock_gen.return_value = 'cache/abcde12345.jpg'
        resp = self.client.post('/generative_model', data={
            'Color': 'not-a-color',
        })
        self.assertEqual(resp.status_code, 200)

    @patch('views.generate_image')
    def test_out_of_bounds_range_does_not_crash(self, mock_gen):
        mock_gen.return_value = 'cache/abcde12345.jpg'
        resp = self.client.post('/generative_model', data={
            'Range_9999': '50',
            'Color': '#000000',
        })
        self.assertEqual(resp.status_code, 200)

    @patch('views.generate_image')
    def test_out_of_bounds_ingr_does_not_crash(self, mock_gen):
        mock_gen.return_value = 'cache/abcde12345.jpg'
        resp = self.client.post('/generative_model', data={
            'Ingr_9999': 'on',
            'Color': '#000000',
        })
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# GET /cache/<filename>
# ---------------------------------------------------------------------------

class TestDownloadFile(_ViewsTestBase):
    def test_valid_filename_not_blocked(self):
        # Valid pattern — will get 404 from filesystem (no actual file), not from our guard
        resp = self.client.get('/cache/abcde12345.jpg')
        self.assertNotEqual(resp.status_code, 400)

    def test_path_traversal_blocked(self):
        resp = self.client.get('/cache/../../etc/passwd')
        self.assertEqual(resp.status_code, 404)

    def test_dotdot_jpg_blocked(self):
        resp = self.client.get('/cache/../config.py')
        self.assertEqual(resp.status_code, 404)

    def test_invalid_chars_blocked(self):
        resp = self.client.get('/cache/invalid!name.jpg')
        self.assertEqual(resp.status_code, 404)

    def test_subdirectory_blocked(self):
        resp = self.client.get('/cache/subdir/file.jpg')
        self.assertEqual(resp.status_code, 404)

    def test_empty_filename_blocked(self):
        resp = self.client.get('/cache/')
        # Flask won't even route this — acceptable 404 or 308
        self.assertIn(resp.status_code, [404, 308, 301])

    def test_too_long_filename_blocked(self):
        long_name = 'a' * 50 + '.jpg'
        resp = self.client.get(f'/cache/{long_name}')
        self.assertEqual(resp.status_code, 404)

    def test_no_extension_blocked(self):
        resp = self.client.get('/cache/abcde12345')
        self.assertEqual(resp.status_code, 404)

    def test_wrong_extension_blocked(self):
        resp = self.client.get('/cache/abcde12345.png')
        self.assertEqual(resp.status_code, 404)


if __name__ == '__main__':
    unittest.main()
