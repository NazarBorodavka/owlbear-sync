# RUNE-Tag Tracking Engine - Deployment Manifest

To ensure your Docker container builds correctly on your server, make sure to push the following files/directories to your GitHub repository:

## Core Package
- `runetag/` - The main Python package containing detection logic.
  - `runetag/__init__.py`
  - `runetag/coding.py`
  - `runetag/detector.py`
  - `runetag/generator.py`
  - `runetag/utils.py`
  - `runetag/pose.py`

## Application & Infrastructure
- `app.py` - The main entry point for the Flask/SocketIO server.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - Docker build configuration.
- `templates/` - Web dashboard templates.
- `codebooks/` - (Optional) Precomputed codebooks if used.

## Static Assets
- `assets/runetags/` - High-resolution RUNE-129 marker PNGs (IDs 0-10).

## Exclude (add to .gitignore)
- `venv/`
- `__pycache__/`
- `*.png` (except those in `assets/runetags/`)
