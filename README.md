# SVI Local Web Runner

Small local webpage to configure and launch `Stable-Video-Infinity/scripts/test/*.sh` equivalents.

## Features

- Script chooser for all test shell scripts under `../Stable-Video-Infinity/scripts/test/`
- Editable defaults for script arguments (output path, model roots, cfg, steps, etc.)
- Configurable `CUDA_VISIBLE_DEVICES`
- File inputs support:
  - Use server path (good for files uploaded via SCP)
  - Upload via browser
  - Manual prompt text for `prompt_path` with format validation
- Runs inference through:
  1. `conda activate svi`
  2. `cd ../Stable-Video-Infinity/scripts/test`
  3. `python <test_script>.py ...`
- Live status and log panel

## Quick Start

```bash
cd /home/usnmp/jaden/Long_Video_Generation_SVI
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

## SCP Example

Upload an image from your laptop to this machine:

```bash
scp ./my_image.png user@server:/home/usnmp/jaden/Long_Video_Generation_SVI/uploads/images/
```

Then choose path mode in the form and set:

```text
/home/usnmp/jaden/Long_Video_Generation_SVI/uploads/images/my_image.png
```
