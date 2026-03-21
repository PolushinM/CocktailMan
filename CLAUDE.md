# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `web_demo_cocktails/` unless noted otherwise.

**Run all tests:**
```bash
cd web_demo_cocktails
SECRET_KEY=any-32-char-string python3 -m pytest tests/ -v
```

**Run a single test file:**
```bash
SECRET_KEY=any-32-char-string python3 -m pytest tests/test_utils.py -v
```

**Run a single test:**
```bash
SECRET_KEY=any-32-char-string python3 -m pytest tests/test_views.py::TestDownloadFile::test_path_traversal_blocked -v
```

**Lint (flake8):**
```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

**Pylint (CI threshold ≥ 7):**
```bash
pylint --fail-under=7 $(git ls-files '*.py')
```

**Run locally (dev server):**
```bash
cd web_demo_cocktails
SECRET_KEY=<key> python3 wsgi.py
```

**Build Docker image:**
```bash
cd web_demo_cocktails
docker build . --file Dockerfile --tag cocktailman
```

**Run Docker container:**
```bash
docker run -e SECRET_KEY=<key> -p 8000:8000 cocktailman
```

## Architecture

The app is a Flask web service that runs three ONNX/OpenVINO ML models on images of cocktails.

### Inference pipeline (`models/models.py`)

`ImageProcessor` orchestrates the full pipeline:
1. **Detector** (YOLOv5m) — finds the cocktail bounding box in the image
2. **BlurModel** — generates a gradient mask around the bounding box for visual blurring
3. **Classifier** (EfficientNetV2-S) — multilabel classification of 45 ingredients
4. **Generator** (C-IDVAE) — conditional image generation from a latent vector + ingredient vector

Models are loaded from `.onnx` files at startup via OpenVINO. Config (thresholds, paths, sizes) lives in `models/config/*.json` and `config.py`.

### Web layer

- `app.py` — Flask factory (`get_app()`)
- `main.py` — initialises `ImageProcessor` at module level (i.e. at import time), exposes `predict()` and `generate_image()`
- `views.py` — Flask routes: `GET/POST /`, `GET/POST /generative_model`, `GET /cache/<filename>`
- `utils.py` — `uri_validator` (SSRF protection), `get_random_filename`, `clear_cache`, `calibrate_confidence`, `clip`
- `config.py` — all tuneable constants; `SECRET_KEY` **must** be set via environment variable or the app raises `RuntimeError` on startup

### Frontend

AJAX-only: form submissions and generative model calls go to Flask endpoints that return JSON. The page never does a full reload after the initial GET. Recipe text and flash messages are injected into the page via `const messages = [...]` in the rendered HTML, then read by `static/js/indexMain.js`.

### Key data flow for `/` POST

```
request → uri_validator / FileStorage → predict() → ImageProcessor.predict_blur_save()
       → (ingredients[], confidence) → generate_recipe() → render_index()
```

Processed images are saved to `cache/` with a 10-char random name and cleaned up on the next request via `files_to_delete`.

## Testing approach

Tests mock `models.models.ImageProcessor` in `sys.modules` **before** any project module is imported (because `main.py` instantiates it at module level). All test files set `os.environ['SECRET_KEY']` at the top for the same reason.

Flash messages and recipe text are **not** findable as raw UTF-8 in response HTML — they are JSON-escaped inside a JS variable. Tests check flash by mocking `views.flash` and asserting it was called.

All POST requests to `/` must include an `input_file` field (even empty) in `multipart/form-data`, otherwise WTForms considers the form valid and the code reaches `request.files['input_file']` → Werkzeug raises `BadRequestKeyError` → 400.

## Branch conventions

- `main` — production
- `develop` — integration branch, merge target before `main`
- `security/*` — security fixes
- `feature/*` — new features
