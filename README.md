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

## Local In-Process Qwen Image (Diffusers + LoRA)

The image generator also supports running Diffusers directly inside this Flask app for local runs.

Enable in-process mode by setting either local field to `inprocess` in the Image Generator page:

- Local Base URL: `inprocess`
- or Local Simple Endpoint: `inprocess`

Behavior:

- `inprocess` means Flask runs Diffusers directly in this process for each request.
- The model is loaded lazily on first matching request and cached in memory for reuse.
- If you set a real HTTP URL instead of `inprocess`, Flask forwards each request to that API endpoint.
- Local in-process image generation is restricted to conda env `jaden`.
- Video generation continues to use SVI env resolution (`svi_wan22` preferred, then `svi`).

Recommended local model value:

- `Qwen/Qwen-Image-Edit-2511`

Optional LoRA:

- Check `Use LoRA safetensors`
- Set `LoRA Safetensors Path` (local filesystem path or supported Hugging Face repo path)
- Install PEFT backend in your app environment: `pip install peft`

Optional save-to-server:

- Enable `Save generated image on server (optional)` in the page.
- Optionally set `Server Save Directory`; if empty, default is `uploads/images/generated`.
- API response includes `saved_image_path` when a file is written.

GPU selection:

- Set `Local GPU Index(es)` in the page (for example `0` or `0,1`).
- Or set `local_image_cuda_devices` in `server_upload_config.local.json` for default local runs.
- Backward-compatible key `local_image_cuda_device` is still supported.

Notes:

- In-process mode requires a CUDA GPU.
- In-process mode expects a reference image upload for Qwen edit models.
- If `torch`/`diffusers` are missing in the runtime environment, the API returns a clear installation error.
- Stopping the Flask process releases its CUDA context and frees GPU memory used by in-process model inference.

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

