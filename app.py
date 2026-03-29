import json
import os
import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

APP_ROOT = Path(__file__).resolve().parent
ENGINE_ROOT = (APP_ROOT / "../Stable-Video-Infinity").resolve()
SCRIPT_DIR = ENGINE_ROOT / "scripts" / "test"
UPLOAD_ROOT = APP_ROOT / "uploads"

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


jobs: Dict[str, JobState] = {}


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


def _parse_script(script_path: Path) -> Dict:
    text = script_path.read_text(encoding="utf-8")
    cmd = _flatten_shell_command(text)
    tokens = shlex.split(cmd)

    cuda_device = ""
    token_index = 0
    if tokens and tokens[0].startswith("CUDA_VISIBLE_DEVICES="):
        cuda_device = tokens[0].split("=", 1)[1]
        token_index += 1

    if token_index >= len(tokens) or tokens[token_index] != "python":
        raise ValueError(f"Unsupported script format in {script_path.name}: expected python command")

    token_index += 1
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


def _save_uploaded_file(file_storage, folder_name: str) -> str:
    _ensure_dirs()
    safe_name = secure_filename(file_storage.filename or "uploaded.dat")
    file_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_dir = UPLOAD_ROOT / folder_name
    save_path = save_dir / file_name
    file_storage.save(str(save_path))
    return str(save_path)


def _build_launch_command(config: Dict, final_args: Dict[str, object], cuda_device: str) -> str:
    cli_parts: List[str] = ["python", shlex.quote(config["python_script"])]

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
    engine_cd = shlex.quote(str(SCRIPT_DIR))
    return (
        "eval \"$(conda shell.bash hook)\" && "
        "conda activate svi && "
        f"cd {engine_cd} && "
        f"{script_command}"
    )


def _run_job(job_id: str, shell_command: str) -> None:
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
        jobs[job_id].logs.append(line.rstrip("\n"))

    process.wait()
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
            mode = request.form.get(f"file_mode__{key}", "path")
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

                final_args[key] = _save_uploaded_file(uploaded, folder_name)
            else:
                server_path = request.form.get(f"path__{key}", "").strip()
                if not server_path:
                    return f"Path is required for {key} in path mode.", 400
                final_args[key] = server_path
            continue

        field_value = request.form.get(f"param__{key}", "").strip()
        if field_value:
            final_args[key] = field_value

    cuda_device = request.form.get("cuda_device", config.get("cuda_device", "")).strip()
    shell_command = _build_launch_command(config, final_args, cuda_device)

    job_id = uuid.uuid4().hex
    jobs[job_id] = JobState(status="queued", logs=[], command=shell_command)

    thread = threading.Thread(target=_run_job, args=(job_id, shell_command), daemon=True)
    thread.start()

    return redirect(url_for("index", script=script_name, job=job_id))


@app.route("/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "job not found"}), 404

    state = jobs[job_id]
    return jsonify(
        {
            "status": state.status,
            "logs": state.logs[-400:],
            "command": state.command,
        }
    )


if __name__ == "__main__":
    _ensure_dirs()
    app.run(host="0.0.0.0", port=7861, debug=True)
