"""
Test suite entry point — collects all test modules.
Run via: pytest ./web_demo_cocktails/tests/
"""

# ruff: noqa: F401  (imports used by pytest via collection)
from test_utils import (
    TestGetRandomFilename,
    TestClearCache,
    TestCalibrateConfidence,
    TestUriValidator,
    TestClip,
)
from test_main import (
    TestGenerateRecipe,
    TestGetConfidenceText,
    TestPredict,
    TestGenerateImage,
)
from test_views import (
    TestIndexGet,
    TestIndexPostEmpty,
    TestIndexPostUrl,
    TestIndexPostFile,
    TestGenerativeModelGet,
    TestGenerativeModelPost,
    TestDownloadFile,
)
# Note: test_no_input_shows_flash → test_no_input_calls_flash
#       test_recipe_appears_in_response → test_recipe_call_returns_200
#       test_predict_exception_shows_flash → test_predict_exception_calls_flash
#       test_file_predict_exception_shows_flash → test_file_predict_exception_calls_flash
