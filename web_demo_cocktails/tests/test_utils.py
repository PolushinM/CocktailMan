"""Tests for utils.py."""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('SECRET_KEY', 'test-only-secret-key-exactly-32ch')
sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

from utils import calibrate_confidence, clear_cache, clip, get_random_filename, uri_validator


class TestGetRandomFilename(unittest.TestCase):
    def test_ends_with_jpg(self):
        self.assertTrue(get_random_filename().endswith('.jpg'))

    def test_basename_length(self):
        name = get_random_filename()
        self.assertEqual(len(name), 14)  # 10 chars + '.jpg'

    def test_only_alphanumeric(self):
        name = get_random_filename().removesuffix('.jpg')
        self.assertRegex(name, r'^[A-Za-z0-9]+$')

    def test_uniqueness(self):
        names = {get_random_filename() for _ in range(50)}
        self.assertEqual(len(names), 50)


class TestClearCache(unittest.TestCase):
    def test_deletes_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        files = [path]
        clear_cache(files)
        self.assertFalse(os.path.exists(path))

    def test_clears_list(self):
        files = ['nonexistent_file_1.jpg', 'nonexistent_file_2.jpg']
        clear_cache(files)
        self.assertEqual(files, [])

    def test_handles_missing_file_silently(self):
        files = ['/tmp/surely_does_not_exist_abc123.jpg']
        try:
            clear_cache(files)
        except Exception as e:
            self.fail(f"clear_cache raised an exception on missing file: {e}")

    def test_deletes_multiple_files(self):
        paths = []
        for _ in range(3):
            f = tempfile.NamedTemporaryFile(delete=False)
            f.close()
            paths.append(f.name)
        clear_cache(paths)
        for p in paths:
            self.assertFalse(os.path.exists(p))


class TestCalibrateConfidence(unittest.TestCase):
    def test_zero_returns_zero(self):
        self.assertAlmostEqual(calibrate_confidence(0.0), 0.0)

    def test_one_returns_one(self):
        self.assertAlmostEqual(calibrate_confidence(1.0), 1.0, places=5)

    def test_output_between_zero_and_one(self):
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = calibrate_confidence(v)
            self.assertGreater(result, 0.0)
            self.assertLessEqual(result, 1.0)

    def test_monotonically_increasing(self):
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = [calibrate_confidence(v) for v in values]
        self.assertEqual(results, sorted(results))

    def test_returns_float(self):
        self.assertIsInstance(calibrate_confidence(0.5), float)


class TestUriValidator(unittest.TestCase):
    def _make_getaddrinfo(self, ip):
        return [(None, None, None, None, (ip, 0))]

    @patch('utils.socket.getaddrinfo')
    def test_valid_http(self, mock_dns):
        mock_dns.return_value = self._make_getaddrinfo('93.184.216.34')
        self.assertTrue(uri_validator('http://example.com/image.jpg'))

    @patch('utils.socket.getaddrinfo')
    def test_valid_https(self, mock_dns):
        mock_dns.return_value = self._make_getaddrinfo('93.184.216.34')
        self.assertTrue(uri_validator('https://example.com/image.jpg'))

    def test_ftp_rejected(self):
        self.assertFalse(uri_validator('ftp://example.com/image.jpg'))

    def test_file_scheme_rejected(self):
        self.assertFalse(uri_validator('file:///etc/passwd'))

    def test_empty_string(self):
        self.assertFalse(uri_validator(''))

    def test_no_scheme(self):
        self.assertFalse(uri_validator('example.com/image.jpg'))

    def test_no_netloc(self):
        self.assertFalse(uri_validator('http:///path'))

    @patch('utils.socket.getaddrinfo')
    def test_loopback_blocked(self, mock_dns):
        mock_dns.return_value = self._make_getaddrinfo('127.0.0.1')
        self.assertFalse(uri_validator('http://localhost/'))

    @patch('utils.socket.getaddrinfo')
    def test_private_10_blocked(self, mock_dns):
        mock_dns.return_value = self._make_getaddrinfo('10.0.0.1')
        self.assertFalse(uri_validator('http://internal.corp/'))

    @patch('utils.socket.getaddrinfo')
    def test_private_192_blocked(self, mock_dns):
        mock_dns.return_value = self._make_getaddrinfo('192.168.1.1')
        self.assertFalse(uri_validator('http://router.local/'))

    @patch('utils.socket.getaddrinfo')
    def test_link_local_blocked(self, mock_dns):
        mock_dns.return_value = self._make_getaddrinfo('169.254.0.1')
        self.assertFalse(uri_validator('http://link-local.test/'))

    @patch('utils.socket.getaddrinfo')
    def test_ipv6_loopback_blocked(self, mock_dns):
        mock_dns.return_value = self._make_getaddrinfo('::1')
        self.assertFalse(uri_validator('http://[::1]/'))

    @patch('utils.socket.getaddrinfo')
    def test_dns_failure_returns_false(self, mock_dns):
        mock_dns.side_effect = OSError('DNS failure')
        self.assertFalse(uri_validator('http://nonexistent.invalid/'))

    def test_malformed_url(self):
        self.assertFalse(uri_validator('not_a_url_at_all'))


class TestClip(unittest.TestCase):
    def test_below_min(self):
        self.assertEqual(clip(-1), 0)

    def test_above_max(self):
        self.assertEqual(clip(2), 1)

    def test_within_range(self):
        self.assertEqual(clip(0.5), 0.5)

    def test_at_min_boundary(self):
        self.assertEqual(clip(0), 0)

    def test_at_max_boundary(self):
        self.assertEqual(clip(1), 1)

    def test_custom_min_max(self):
        self.assertEqual(clip(5, min_=0, max_=10), 5)
        self.assertEqual(clip(-5, min_=0, max_=10), 0)
        self.assertEqual(clip(15, min_=0, max_=10), 10)


if __name__ == '__main__':
    unittest.main()
