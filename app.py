import ast
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from flask import after_this_request, Flask, abort, jsonify, redirect, render_template, request, send_file, url_for
from PIL import Image
from werkzeug.utils import secure_filename

APP_ROOT = Path(__file__).resolve().parent
ENGINE_ROOT = (APP_ROOT / "../Stable-Video-Infinity").resolve()
SCRIPT_DIR = ENGINE_ROOT / "scripts" / "test"
UPLOAD_ROOT = APP_ROOT / "uploads"
SERVER_CONFIG_PATH = APP_ROOT / "server_upload_config.local.json"

FILE_ARGS = {"ref_image_path", "image_path", "prompt_path", "pose_path", "audio_path"}
IMAGE_ARGS = {"ref_image_path", "image_path"}
PROMPT_ARG = "prompt_path"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB


@dataclass
class JobState:
    status: str
    logs: List[str]
    command: str
    script_name: str
    output_arg: str
    output_path: str
    created_ts: float


jobs: Dict[str, JobState] = {}
jobs_lock = threading.Lock()
CUDA_VISIBLE_DEVICE_TOKEN = re.compile(r"^-?\d+$")


def _load_server_upload_config() -> Dict[str, str]:
    if not SERVER_CONFIG_PATH.exists():
        return {}

    try:
        data = json.loads(SERVER_CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}
    return {k: str(v) for k, v in data.items()}


SERVER_UPLOAD_CONFIG = _load_server_upload_config()


def _ensure_dirs() -> None:
    (UPLOAD_ROOT / "images").mkdir(parents=True, exist_ok=True)
    (UPLOAD_ROOT / "prompts").mkdir(parents=True, exist_ok=True)
    (UPLOAD_ROOT / "audio").mkdir(parents=True, exist_ok=True)
    (UPLOAD_ROOT / "pose").mkdir(parents=True, exist_ok=True)


def _build_prompt_file_from_lines(lines_text: str) -> Tuple[bool, str, str]:
    scenes = [line.strip() for line in lines_text.splitlines() if line.strip()]
    if not scenes:
        return False, "Manual prompt requires at least one non-empty line.", ""

    prompt_literal = json.dumps(scenes, ensure_ascii=False, indent=2)
    prompt_file_text = f"prompts = {prompt_literal}\n"
    return True, "", prompt_file_text


def _extract_prompt_scenes(prompt_text: str) -> List[str]:
    try:
        tree = ast.parse(prompt_text)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue

            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue

            if node.targets[0].id != "prompts":
                continue

            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue

            scenes: List[str] = []
            for item in node.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    scenes.append(item.value)
            if scenes:
                return scenes
    except SyntaxError:
        pass

    return [line.strip() for line in prompt_text.splitlines() if line.strip()]


def _resolve_input_path(raw_path: str) -> Path | None:
    candidate = Path(raw_path)
    possible: List[Path] = []

    if candidate.is_absolute():
        possible.append(candidate)
    else:
        possible.append((ENGINE_ROOT / candidate).resolve())
        possible.append((SCRIPT_DIR / candidate).resolve())

    allowed_roots = [ENGINE_ROOT.resolve(), UPLOAD_ROOT.resolve()]

    for path in possible:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            continue

        if not resolved.exists() or not resolved.is_file():
            continue

        if any(str(resolved).startswith(str(root)) for root in allowed_roots):
            return resolved

    return None


def _flatten_shell_command(script_text: str) -> str:
    parts: List[str] = []
    for raw in script_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            parts.append(line[:-1].strip())
        else:
            parts.append(line)
    return " ".join(parts)


def _extract_python_tokens(script_text: str, script_name: str) -> Tuple[List[str], str, int]:
    lines = script_text.splitlines()
    env_assignment = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")

    line_index = 0
    while line_index < len(lines):
        raw = lines[line_index].strip()
        if not raw or raw.startswith("#"):
            line_index += 1
            continue

        combined = raw
        next_index = line_index
        while combined.endswith("\\") and next_index + 1 < len(lines):
            next_index += 1
            combined = combined[:-1].strip() + " " + lines[next_index].strip()

        # Tolerate a dangling trailing backslash at EOF.
        while combined.endswith("\\"):
            combined = combined[:-1].rstrip()

        try:
            tokens = shlex.split(combined)
        except ValueError:
            line_index = next_index + 1
            continue

        if not tokens:
            line_index = next_index + 1
            continue

        cuda_device = ""
        token_index = 0
        while token_index < len(tokens):
            tok = tokens[token_index]
            if tok == "python":
                return tokens, cuda_device, token_index

            if env_assignment.fullmatch(tok):
                if tok.startswith("CUDA_VISIBLE_DEVICES="):
                    cuda_device = tok.split("=", 1)[1]
                token_index += 1
                continue

            break

        line_index = next_index + 1

    raise ValueError(f"Unsupported script format in {script_name}: expected python command")


def _parse_script(script_path: Path) -> Dict:
    text = script_path.read_text(encoding="utf-8")
    tokens, cuda_device, python_index = _extract_python_tokens(text, script_path.name)

    token_index = python_index + 1
    if token_index >= len(tokens):
        raise ValueError(f"Unsupported script format in {script_path.name}: missing python script")

    py_script = tokens[token_index]
    token_index += 1

    args_order: List[str] = []
    args_map: Dict[str, object] = {}

    while token_index < len(tokens):
        tok = tokens[token_index]
        if not tok.startswith("--"):
            token_index += 1
            continue

        key = tok[2:]
        args_order.append(key)
        has_value = token_index + 1 < len(tokens) and not tokens[token_index + 1].startswith("--")

        if has_value:
            args_map[key] = tokens[token_index + 1]
            token_index += 2
        else:
            args_map[key] = True
            token_index += 1

    return {
        "name": script_path.name,
        "python_script": py_script,
        "cuda_device": cuda_device,
        "args_order": args_order,
        "args": args_map,
    }


def _load_script_configs() -> Dict[str, Dict]:
    configs: Dict[str, Dict] = {}
    for script in sorted(SCRIPT_DIR.glob("*.sh")):
        configs[script.name] = _parse_script(script)
    return configs


def _save_uploaded_file(file_storage, folder_name: str, force_rgb: bool = False) -> str:
    _ensure_dirs()
    safe_name = secure_filename(file_storage.filename or "uploaded.dat")
    file_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_dir = UPLOAD_ROOT / folder_name
    save_path = save_dir / file_name
    file_storage.save(str(save_path))

    if force_rgb:
        with Image.open(save_path) as image:
            if image.mode != "RGB":
                rgb_image = image.convert("RGB")
                rgb_image.save(save_path)

    return str(save_path)


def _resolve_svi_python_executable() -> str:
    # Prefer explicit CONDA_EXE from process environment when available.
    conda_exe = os.environ.get("CONDA_EXE", "conda")
    preferred_envs = ("svi_wan22", "svi")

    try:
        result = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        envs = data.get("envs", []) if isinstance(data, dict) else []

        for env_name in preferred_envs:
            for env_path in envs:
                p = Path(str(env_path))
                if p.name == env_name:
                    python_path = p / "bin" / "python"
                    if python_path.exists():
                        return str(python_path)
    except Exception:
        pass

    # Fallback for common miniforge/anaconda layout inferred from CONDA_EXE.
    conda_path = Path(conda_exe)
    if conda_path.name == "conda":
        base = conda_path.parent.parent
        for env_name in preferred_envs:
            candidate = base / "envs" / env_name / "bin" / "python"
            if candidate.exists():
                return str(candidate)

    # Last fallback keeps command behavior if conda env lookup fails.
    return "python"


def _validate_prompt_count(final_args: Dict[str, object]) -> Tuple[bool, str]:
    prompt_path_value = final_args.get(PROMPT_ARG)
    if not isinstance(prompt_path_value, str) or not prompt_path_value.strip():
        return True, ""

    num_clips_value = final_args.get("num_clips")
    if num_clips_value is None:
        return True, ""

    try:
        num_clips = int(str(num_clips_value).strip())
    except ValueError:
        return False, "num_clips must be an integer."

    if num_clips <= 0:
        return False, "num_clips must be greater than 0."

    prompt_path = _resolve_input_path(prompt_path_value)
    if prompt_path is None:
        return False, f"Prompt file is not accessible: {prompt_path_value}"

    prompt_text = prompt_path.read_text(encoding="utf-8")
    scenes = _extract_prompt_scenes(prompt_text)
    if not scenes:
        return False, "Prompt file must contain at least one non-empty prompt scene."

    if len(scenes) < num_clips:
        return (
            False,
            f"Prompt scene count ({len(scenes)}) is smaller than num_clips ({num_clips}). "
            "Add more prompt lines or reduce num_clips.",
        )

    return True, ""


def _build_launch_command(config: Dict, final_args: Dict[str, object], cuda_device: str) -> str:
    python_exe = _resolve_svi_python_executable()
    cli_parts: List[str] = [shlex.quote(python_exe), shlex.quote(config["python_script"])]

    for key in config["args_order"]:
        value = final_args[key]
        cli_parts.append(f"--{key}")
        if isinstance(value, bool):
            if not value:
                cli_parts.pop()
            continue
        cli_parts.append(shlex.quote(str(value)))

    env_prefix = ""
    if cuda_device.strip():
        env_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_device.strip())} "

    script_command = env_prefix + " ".join(cli_parts)
    engine_cd = shlex.quote(str(ENGINE_ROOT))
    return f"cd {engine_cd} && {script_command}"


def _normalize_cuda_visible_devices(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""

    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        return ""

    if not all(CUDA_VISIBLE_DEVICE_TOKEN.fullmatch(item) for item in devices):
        raise ValueError("Use comma-separated GPU IDs, for example: 0 or 0,1,2")

    return ",".join(devices)


def _normalize_output_path(raw_output: object) -> str:
    if not isinstance(raw_output, str) or not raw_output.strip():
        return ""

    output_path = Path(raw_output.strip())
    if not output_path.is_absolute():
        output_path = (ENGINE_ROOT / output_path).resolve()
    else:
        output_path = output_path.resolve()
    return str(output_path)


def _is_path_under_allowed_roots(path: Path) -> bool:
    allowed_roots = [ENGINE_ROOT.resolve(), UPLOAD_ROOT.resolve()]
    return any(path.is_relative_to(root) for root in allowed_roots)


def _job_can_download(state: JobState) -> bool:
    if state.status != "completed" or not state.output_path:
        return False

    path = Path(state.output_path)
    if not path.exists():
        return False

    resolved = path.resolve()
    return _is_path_under_allowed_roots(resolved) and (resolved.is_file() or resolved.is_dir())


def _get_job_state(job_id: str) -> JobState | None:
    with jobs_lock:
        return jobs.get(job_id)


def _run_job(job_id: str, shell_command: str) -> None:
    with jobs_lock:
        if job_id not in jobs:
            return
        jobs[job_id].status = "running"

    process = subprocess.Popen(
        ["bash", "-lc", shell_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        with jobs_lock:
            if job_id in jobs:
                jobs[job_id].logs.append(line.rstrip("\n"))

    process.wait()
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].status = "completed" if process.returncode == 0 else "failed"
            jobs[job_id].logs.append(f"\n[exit code] {process.returncode}")


@app.route("/", methods=["GET"])
def index():
    script_configs = _load_script_configs()
    selected_script = request.args.get("script")
    if selected_script not in script_configs:
        selected_script = next(iter(script_configs), "")

    selected_job = request.args.get("job", "")
    return render_template(
        "index.html",
        scripts=sorted(script_configs.keys()),
        selected_script=selected_script,
        script_configs_json=json.dumps(script_configs),
        selected_job=selected_job,
        engine_root=str(ENGINE_ROOT),
    )


@app.route("/image-generator", methods=["GET"])
def image_generator_page():
    return render_template("image_generator.html")


def _read_config_value(config: Dict[str, str], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _extract_partial_text(partial: object) -> str:
    text_attr = getattr(partial, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    response_attr = getattr(partial, "response", None)
    if isinstance(response_attr, str):
        return response_attr

    return str(partial)


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://api.openai.com/v1"

    if normalized.endswith("/chat/completions"):
        return normalized

    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return f"{normalized}/chat/completions"


def _extract_text_from_openai_response(response_json: Dict[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        return "\n".join(text_parts).strip()

    return ""


def _extract_text_from_simple_response(response_json: object) -> str:
    if isinstance(response_json, str):
        return response_json

    if not isinstance(response_json, dict):
        return ""

    for key in ["text", "result", "response", "output_text", "message"]:
        value = response_json.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            maybe_text = first.get("text")
            if isinstance(maybe_text, str) and maybe_text.strip():
                return maybe_text

    return ""


def _extract_json_obj_or_empty(raw: str) -> Dict[str, object]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return parsed
    return {}


def _extract_image_url_from_text(text: str) -> str:
    if not text:
        return ""

    # Inline markdown image: ![alt](https://...)
    inline_match = re.search(r"!\[[^\]]*\]\((https?://[^)\s]+)", text, flags=re.IGNORECASE)
    if inline_match:
        return inline_match.group(1).rstrip("),.;")

    # Reference-style markdown image:
    # ![alt][ref]\n\n[ref]: https://...
    ref_defs: Dict[str, str] = {}
    for key, url in re.findall(r"^\s*\[([^\]]+)\]:\s*(https?://\S+)", text, flags=re.IGNORECASE | re.MULTILINE):
        ref_defs[key.strip().lower()] = url.rstrip("),.;")

    ref_match = re.search(r"!\[[^\]]*\]\[([^\]]+)\]", text, flags=re.IGNORECASE)
    if ref_match:
        ref_key = ref_match.group(1).strip().lower()
        if ref_key in ref_defs:
            return ref_defs[ref_key]

    # Fallback: first direct URL in text.
    direct_match = re.search(r"https?://\S+", text, flags=re.IGNORECASE)
    if not direct_match:
        return ""

    return direct_match.group(0).rstrip("),.;")


def _extract_image_url_from_partial(partial: object) -> str:
    seen_ids = set()

    def find_url(value: object, depth: int = 0) -> str:
        if depth > 4 or value is None:
            return ""

        obj_id = id(value)
        if obj_id in seen_ids:
            return ""
        seen_ids.add(obj_id)

        if isinstance(value, str):
            match = re.search(r"https?://\S+", value, flags=re.IGNORECASE)
            if not match:
                return ""
            return match.group(0).rstrip("),.;")

        if isinstance(value, dict):
            for item in value.values():
                found = find_url(item, depth + 1)
                if found:
                    return found
            return ""

        if isinstance(value, (list, tuple, set)):
            for item in value:
                found = find_url(item, depth + 1)
                if found:
                    return found
            return ""

        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
            except Exception:
                dumped = None
            found = find_url(dumped, depth + 1)
            if found:
                return found

        if hasattr(value, "__dict__"):
            found = find_url(vars(value), depth + 1)
            if found:
                return found

        return ""

    return find_url(partial)


@app.route("/api/image-generator/run", methods=["POST"])
def run_image_generator():
    prompt = request.form.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    config = _load_server_upload_config()
    api_mode = _read_config_value(config, ["image_api_mode"], "openai").lower()
    api_key = _read_config_value(config, ["image_api_key", "openai_api_key", "poe_api_key"])
    model = _read_config_value(config, ["image_api_model", "openai_model", "poe_model", "poe_bot_name"], "gpt-4o")
    raw_base_url = _read_config_value(
        config,
        ["image_api_base_url", "openai_base_url", "poe_base_url"],
        "https://api.openai.com/v1",
    )
    chat_completions_url = _normalize_openai_base_url(raw_base_url)
    simple_endpoint = _read_config_value(config, ["image_api_endpoint"], raw_base_url)

    if api_mode not in {"openai", "simple"}:
        return (
            jsonify(
                {
                    "error": "Invalid image_api_mode. Use 'openai' or 'simple'."
                }
            ),
            400,
        )

    if api_mode == "openai" and not api_key:
        return (
            jsonify(
                {
                    "error": "Missing API key in server_upload_config.local.json (image_api_key or openai_api_key)."
                }
            ),
            400,
        )

    image_b64 = ""
    image_mime = ""
    message_content: object = prompt

    uploaded_image = request.files.get("image")
    if uploaded_image and uploaded_image.filename:
        image_bytes = uploaded_image.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded image is empty."}), 400

        # Keep uploaded image available for both openai and simple payload styles.
        image_mime = uploaded_image.mimetype or "image/png"
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{image_mime};base64,{image_b64}"
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

    target_url = chat_completions_url if api_mode == "openai" else simple_endpoint
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        auth_header_name = _read_config_value(config, ["image_api_auth_header"], "Authorization")
        auth_scheme = _read_config_value(config, ["image_api_auth_scheme"], "Bearer")
        if auth_scheme:
            headers[auth_header_name] = f"{auth_scheme} {api_key}"
        else:
            headers[auth_header_name] = api_key

    if api_mode == "openai":
        payload: Dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
            "stream": False,
        }
    else:
        payload = {
            "prompt": prompt,
            "model": model,
        }
        if image_b64:
            payload["image_base64"] = image_b64
            payload["image_mime"] = image_mime

        extra_payload = _extract_json_obj_or_empty(_read_config_value(config, ["image_api_extra_json"], ""))
        if extra_payload:
            payload.update(extra_payload)

    body_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        target_url,
        data=body_bytes,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8", errors="replace")
            response_json = json.loads(raw)
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        return jsonify({"error": f"Image API request failed ({exc.code}): {err_body}"}), 502
    except urllib.error.URLError as exc:
        return jsonify({"error": f"Image API request failed: {exc.reason}"}), 502
    except Exception as exc:
        return jsonify({"error": f"Image API request failed: {exc}"}), 502

    if api_mode == "openai":
        response_text = _extract_text_from_openai_response(response_json).strip()
    else:
        response_text = _extract_text_from_simple_response(response_json).strip()
    if not response_text:
        response_text = "(No textual content returned by model.)"

    image_url = _extract_image_url_from_text(response_text)
    if not image_url:
        image_url = _extract_image_url_from_partial(response_json)

    return jsonify(
        {
            "ok": True,
            "model": model,
            "base_url": target_url,
            "result": response_text,
            "image_url": image_url,
        }
    )


@app.route("/run", methods=["POST"])
def run_script():
    script_configs = _load_script_configs()
    script_name = request.form.get("script_name", "")
    if script_name not in script_configs:
        return "Unknown script selected.", 400

    config = script_configs[script_name]
    final_args = dict(config["args"])

    for key in config["args_order"]:
        default_value = config["args"][key]

        if isinstance(default_value, bool):
            final_args[key] = request.form.get(f"flag__{key}") == "on"
            continue

        if key in FILE_ARGS:
            mode = request.form.get(f"file_mode__{key}", "default")
            if mode == "default":
                final_args[key] = default_value
                continue

            if mode == "path":
                server_path = request.form.get(f"path__{key}", "").strip()
                if not server_path:
                    return f"Path is required for {key} in path mode.", 400
                final_args[key] = server_path
                continue

            if key == PROMPT_ARG and mode == "manual":
                manual_prompt_lines = request.form.get("manual_prompt", "")
                ok, reason, prompt_file_content = _build_prompt_file_from_lines(manual_prompt_lines)
                if not ok:
                    return f"Manual prompt format error: {reason}", 400

                prompt_file = UPLOAD_ROOT / "prompts" / f"manual_prompt_{uuid.uuid4().hex}.txt"
                prompt_file.parent.mkdir(parents=True, exist_ok=True)
                prompt_file.write_text(prompt_file_content, encoding="utf-8")
                final_args[key] = str(prompt_file)
                continue

            if mode == "upload":
                uploaded = request.files.get(f"upload__{key}")
                if not uploaded or not uploaded.filename:
                    return f"No file uploaded for {key}.", 400

                folder_name = "images"
                if key == PROMPT_ARG:
                    folder_name = "prompts"
                elif key == "audio_path":
                    folder_name = "audio"
                elif key == "pose_path":
                    folder_name = "pose"

                final_args[key] = _save_uploaded_file(uploaded, folder_name, force_rgb=(key in IMAGE_ARGS))
            else:
                return f"Unsupported mode '{mode}' for {key}.", 400
            continue

        field_value = request.form.get(f"param__{key}", "").strip()
        if field_value:
            final_args[key] = field_value

    raw_cuda_devices = request.form.get("cuda_device", config.get("cuda_device", ""))
    try:
        cuda_device = _normalize_cuda_visible_devices(raw_cuda_devices)
    except ValueError as exc:
        return f"Invalid CUDA_VISIBLE_DEVICES value: {exc}", 400

    ok, reason = _validate_prompt_count(final_args)
    if not ok:
        return f"Prompt/clip validation error: {reason}", 400

    shell_command = _build_launch_command(config, final_args, cuda_device)

    job_id = uuid.uuid4().hex
    output_arg = str(final_args.get("output", ""))
    output_path = _normalize_output_path(output_arg)

    with jobs_lock:
        jobs[job_id] = JobState(
            status="queued",
            logs=[],
            command=shell_command,
            script_name=script_name,
            output_arg=output_arg,
            output_path=output_path,
            created_ts=time.time(),
        )

    thread = threading.Thread(target=_run_job, args=(job_id, shell_command), daemon=True)
    thread.start()

    return redirect(url_for("index", script=script_name, job=job_id))


@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    state = _get_job_state(job_id)
    if state is None:
        return jsonify({"error": "job not found"}), 404

    return jsonify(
        {
            "job_id": job_id,
            "status": state.status,
            "logs": state.logs[-400:],
            "command": state.command,
            "script_name": state.script_name,
            "output_arg": state.output_arg,
            "output_path": state.output_path,
            "created_ts": state.created_ts,
            "can_download": _job_can_download(state),
        }
    )


@app.route("/jobs", methods=["GET"])
def list_jobs():
    with jobs_lock:
        rows = [
            {
                "job_id": job_id,
                "status": state.status,
                "script_name": state.script_name,
                "output_arg": state.output_arg,
                "output_path": state.output_path,
                "created_ts": state.created_ts,
                "can_download": _job_can_download(state),
            }
            for job_id, state in jobs.items()
        ]

    rows.sort(key=lambda row: row["created_ts"], reverse=True)
    return jsonify({"jobs": rows})


@app.route("/job/<job_id>/download-output", methods=["GET"])
def download_job_output(job_id: str):
    state = _get_job_state(job_id)
    if state is None:
        return jsonify({"error": "job not found"}), 404

    if state.status != "completed":
        return jsonify({"error": "job is not completed yet"}), 409

    if not state.output_path:
        return jsonify({"error": "job has no output path configured"}), 404

    output_path = Path(state.output_path)
    if not output_path.exists():
        return jsonify({"error": "output path does not exist"}), 404

    resolved = output_path.resolve()
    if not _is_path_under_allowed_roots(resolved):
        return jsonify({"error": "output path is outside allowed directories"}), 403

    if resolved.is_file():
        return send_file(str(resolved), as_attachment=True, download_name=resolved.name)

    if not resolved.is_dir():
        return jsonify({"error": "output path is neither a file nor a directory"}), 404

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    temp_file.close()
    zip_path = Path(temp_file.name)
    archive_base = str(zip_path.with_suffix(""))
    shutil.make_archive(archive_base, "zip", root_dir=str(resolved))
    final_zip = zip_path.with_suffix(".zip")
    if final_zip != zip_path:
        zip_path.unlink(missing_ok=True)

    @after_this_request
    def cleanup_zip(response):
        final_zip.unlink(missing_ok=True)
        return response

    zip_name = f"{resolved.name or 'output'}_{job_id[:8]}.zip"
    return send_file(str(final_zip), as_attachment=True, download_name=zip_name)


@app.route("/api/default-image", methods=["GET"])
def default_image_preview():
    script_name = request.args.get("script", "")
    arg_name = request.args.get("arg", "")

    script_configs = _load_script_configs()
    if script_name not in script_configs or arg_name not in IMAGE_ARGS:
        abort(404)

    value = script_configs[script_name]["args"].get(arg_name)
    if not isinstance(value, str):
        abort(404)

    resolved = _resolve_input_path(value)
    if resolved is None:
        abort(404)

    return send_file(str(resolved))


@app.route("/api/default-prompt-scenes", methods=["GET"])
def default_prompt_scenes():
    script_name = request.args.get("script", "")
    arg_name = request.args.get("arg", "")

    script_configs = _load_script_configs()
    if script_name not in script_configs or arg_name != PROMPT_ARG:
        return jsonify({"scenes": []})

    value = script_configs[script_name]["args"].get(arg_name)
    if not isinstance(value, str):
        return jsonify({"scenes": []})

    resolved = _resolve_input_path(value)
    if resolved is None:
        return jsonify({"scenes": []})

    prompt_text = resolved.read_text(encoding="utf-8")
    scenes = _extract_prompt_scenes(prompt_text)
    return jsonify({"scenes": scenes})


@app.route("/api/preview-image-path", methods=["GET"])
def preview_image_path():
    raw_path = request.args.get("path", "").strip()
    if not raw_path:
        abort(400)

    resolved = _resolve_input_path(raw_path)
    if resolved is None:
        abort(404)

    return send_file(str(resolved))


@app.route("/api/preview-prompt-path", methods=["GET"])
def preview_prompt_path():
    raw_path = request.args.get("path", "").strip()
    if not raw_path:
        return jsonify({"scenes": []})

    resolved = _resolve_input_path(raw_path)
    if resolved is None:
        return jsonify({"scenes": []})

    prompt_text = resolved.read_text(encoding="utf-8")
    scenes = _extract_prompt_scenes(prompt_text)
    return jsonify({"scenes": scenes})


if __name__ == "__main__":
    _ensure_dirs()
    app.run(host="0.0.0.0", port=8888, debug=True)
