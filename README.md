# Long-video-generator (powered by SVI)

Small local webpage to configure and launch `Stable-Video-Infinity/scripts/test/*.sh` equivalents.

Acknowledgment: This project is powered by Stable-Video-Infinity (SVI) from VITA-Group: https://github.com/vita-epfl/Stable-Video-Infinity

## Features

- Script chooser for all test shell scripts under `../Stable-Video-Infinity/scripts/test/`
- Primary run inputs for each run:
  - script template
  - output path
  - image input (default from template or browser upload)
  - prompt input (default from template, browser upload, or manual line-by-line scenes)
- Live preview before run:
  - selected/final image
  - finalized `prompts = [...]` list
- Editable defaults for advanced script arguments (model roots, cfg, steps, etc.) in collapsible advanced section
- Configurable `CUDA_VISIBLE_DEVICES`
- Runs inference through:
  1. `conda activate svi_wan22`
  2. `cd ../Stable-Video-Infinity/scripts/test`
  3. `python <test_script>.py ...`
- Live status and log panel

## Server-only Config (not pushed)

Server connection values are stored in local-only file `server_upload_config.local.json`.

- Template file: `server_upload_config.example.json`
- Git-ignored local file: `server_upload_config.local.json`

## Image API Modes

The image generator endpoint supports two modes through `server_upload_config.local.json`:

- `image_api_mode: "openai"` (default)
  - Uses OpenAI-style `/v1/chat/completions` format.
- `image_api_mode: "simple"`
  - Sends a plain JSON `POST` request to `image_api_endpoint`.
  - Payload includes at least:
    - `prompt`
    - `model`
    - optional `image_base64` and `image_mime` when a reference image is uploaded.

Example local/simple setup:

```json
{
  "image_api_mode": "simple",
  "image_api_endpoint": "http://127.0.0.1:8000/generate",
  "image_api_model": "your-local-model",
  "image_api_key": "",
  "image_api_auth_header": "Authorization",
  "image_api_auth_scheme": "Bearer",
  "image_api_extra_json": "{\"temperature\":0.7}"
}
```

Notes:

- If your local endpoint does not require auth, keep `image_api_key` empty.
- If auth uses a custom header, set `image_api_auth_header` and `image_api_auth_scheme`.
- `image_api_extra_json` must be a JSON object encoded as a string.

## Quick Start

```bash
cd /your/server/path/Long_Video_Generation_SVI
python -m pip install -r requirements.txt
python app.py
```

Open in browser:

- http://127.0.0.1:8888

## Prompt Format for Manual Mode

Use Python-like prompt files, for example:

```python
prompts = [
    "A cat in a hat.",
    "The cat jumps onto the table."
]
```

