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
  1. `conda activate svi`
  2. `cd ../Stable-Video-Infinity/scripts/test`
  3. `python <test_script>.py ...`
- Live status and log panel

## Server-only Config (not pushed)

Server connection values are stored in local-only file `server_upload_config.local.json`.

- Template file: `server_upload_config.example.json`
- Git-ignored local file: `server_upload_config.local.json`

## Quick Start

```bash
cd /your/server/path/Long_Video_Generation_SVI
python -m pip install -r requirements.txt
python app.py
```

Open in browser:

- http://127.0.0.1:7861

## Prompt Format for Manual Mode

Use Python-like prompt files, for example:

```python
prompts = [
    "A cat in a hat.",
    "The cat jumps onto the table."
]
```

