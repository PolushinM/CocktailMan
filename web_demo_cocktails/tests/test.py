import unittest
import os
import sys
from pathlib import Path

from flask import url_for

sys.path.append(str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))
from views import app


TEST_TEXT = ["CocktailMan",
             "Выбери файл с изображением или вставь ссылку:",
             "Доступные ингредиенты",
             "github.com/PolushinM/CocktailMan"]


class HomePageDisplayTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.ctx = app.test_request_context()
        self.ctx.push()

    def tearDown(self):
        self.ctx.pop()

    def test_main_text_displayed(self):
        self.assertGreater(2.0, 1.0)
        response = self.app.get(url_for('index'))
        response_data = response.data.decode('utf-8')
        for string in TEST_TEXT:
            assert string in response_data


if __name__ == '__main__':
    unittest.main()
