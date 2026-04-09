const scriptSelect = document.getElementById("scriptSelect");
const primaryFields = document.getElementById("primaryFields");
const advancedFields = document.getElementById("advancedFields");
const promptHelp = document.getElementById("promptHelp");
const previewImage = document.getElementById("previewImage");
const previewImageHint = document.getElementById("previewImageHint");
const previewPrompts = document.getElementById("previewPrompts");

const logPanel = document.getElementById("logPanel");
const runsList = document.getElementById("runsList");
const selectedRunLabel = document.getElementById("selectedRunLabel");
const runStatus = document.getElementById("runStatus");
const runCommand = document.getElementById("runCommand");
const runOutput = document.getElementById("runOutput");
const downloadOutputBtn = document.getElementById("downloadOutputBtn");
const videoPreviewWrap = document.getElementById("videoPreviewWrap");
const previewVideo = document.getElementById("previewVideo");
const previewVideoHint = document.getElementById("previewVideoHint");
const runLogs = document.getElementById("runLogs");

const configsNode = document.getElementById("script-configs");
const selectedScriptNode = document.getElementById("selected-script");
const configs = configsNode ? JSON.parse(configsNode.textContent) : {};
const selectedScriptFromServer = selectedScriptNode ? JSON.parse(selectedScriptNode.textContent) : "";
const PRIMARY_KEYS = new Set(["output", "ref_image_path", "image_path", "prompt_path"]);
const OUTPUT_KEYS = ["output", "output_root", "output_dir", "output_path", "save_dir"];
let selectedJobId = logPanel ? (logPanel.dataset.selectedJob || "") : "";
let jobsSnapshot = [];

function createLabeledInput(labelText, inputEl) {
  const wrap = document.createElement("div");
  wrap.className = "field";

  const label = document.createElement("label");
  label.className = "label";
  label.textContent = labelText;

  wrap.appendChild(label);
  wrap.appendChild(inputEl);
  return wrap;
}

function createTextInput(name, value, placeholder = "") {
  const input = document.createElement("input");
  input.type = "text";
  input.name = name;
  input.value = value || "";
  input.placeholder = placeholder;
  return input;
}

function createFileSourceField(key, defaultValue, options = {}) {
  const allowManualPrompt = Boolean(options.allowManualPrompt);
  const allowDefault = options.allowDefault !== false;
  const allowPath = options.allowPath !== false;
  const defaultMode = options.defaultMode || (allowDefault ? "default" : "upload");

  const wrap = document.createElement("div");
  wrap.className = "field file-field";

  const label = document.createElement("label");
  label.className = "label";
  label.textContent = key;

  const mode = document.createElement("select");
  mode.name = `file_mode__${key}`;
  mode.dataset.key = key;

  if (allowDefault) {
    const defaultOpt = document.createElement("option");
    defaultOpt.value = "default";
    defaultOpt.textContent = "Use script default";
    mode.appendChild(defaultOpt);
  }

  if (allowPath) {
    const pathOpt = document.createElement("option");
    pathOpt.value = "path";
    pathOpt.textContent = "Use server path";
    mode.appendChild(pathOpt);
  }

  const uploadOpt = document.createElement("option");
  uploadOpt.value = "upload";
  uploadOpt.textContent = "Upload via browser";
  mode.appendChild(uploadOpt);

  if (allowManualPrompt) {
    const manualOpt = document.createElement("option");
    manualOpt.value = "manual";
    manualOpt.textContent = "Manual scenes (one line per scene)";
    mode.appendChild(manualOpt);
  }

  const defaultPath = document.createElement("div");
  defaultPath.className = "default-path";
  defaultPath.textContent = `Default: ${defaultValue || "(none)"}`;

  const uploadInput = document.createElement("input");
  uploadInput.type = "file";
  uploadInput.name = `upload__${key}`;
  uploadInput.accept = key.includes("image") ? "image/*" : "";
    const pathInput = document.createElement("input");
    pathInput.type = "text";
    pathInput.name = `path__${key}`;
    pathInput.value = defaultValue || "";
    pathInput.placeholder = "/server/path/to/file";
    pathInput.style.display = "none";

  uploadInput.style.display = "none";

  const manualPrompt = document.createElement("textarea");
  manualPrompt.name = "manual_prompt";
  manualPrompt.placeholder = "Scene 1 description\nScene 2 description\nScene 3 description";
  manualPrompt.rows = 8;
  manualPrompt.style.display = "none";

  mode.addEventListener("change", () => {
    if (mode.value === "default") {
      defaultPath.style.display = "block";
      pathInput.style.display = "none";
      uploadInput.style.display = "none";
      manualPrompt.style.display = "none";
    } else if (mode.value === "path") {
      defaultPath.style.display = "none";
      pathInput.style.display = "block";
      uploadInput.style.display = "none";
      manualPrompt.style.display = "none";
    } else if (mode.value === "upload") {
      defaultPath.style.display = "none";
      pathInput.style.display = "none";
      uploadInput.style.display = "block";
      manualPrompt.style.display = "none";
    } else {
      defaultPath.style.display = "none";
      pathInput.style.display = "none";
      uploadInput.style.display = "none";
      manualPrompt.style.display = "block";
    }
  });

  if (Array.from(mode.options).some((opt) => opt.value === defaultMode)) {
    mode.value = defaultMode;
  }
  mode.dispatchEvent(new Event("change"));

  wrap.appendChild(label);
  wrap.appendChild(mode);
  wrap.appendChild(defaultPath);
  wrap.appendChild(pathInput);
  wrap.appendChild(uploadInput);
  if (allowManualPrompt) {
    wrap.appendChild(manualPrompt);
  }
  return wrap;
}

function getField(name) {
  return document.querySelector(`[name="${name}"]`);
}

function clearImagePreview(message) {
  previewImage.removeAttribute("src");
  previewImage.style.display = "none";
  previewImageHint.textContent = message;
}

function showImagePreview(src, hint) {
  previewImage.src = src;
  previewImage.style.display = "block";
  previewImageHint.textContent = hint;
}

function renderPromptList(scenes) {
  if (!Array.isArray(scenes) || scenes.length === 0) {
    previewPrompts.textContent = "No prompt scenes yet.";
    return;
  }

  const readable = scenes
    .map((scene, index) => `Scene ${index + 1}: ${scene}`)
    .join("\n\n");
  previewPrompts.textContent = readable;
}

function parseScenesFromPromptText(promptText) {
  const quoted = [];
  const regex = /["']([^"'\\]*(?:\\.[^"'\\]*)*)["']/g;
  let match = regex.exec(promptText);
  while (match) {
    quoted.push(match[1].replace(/\\n/g, "\n"));
    match = regex.exec(promptText);
  }

  if (quoted.length > 0) {
    return quoted;
  }

  return promptText
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

async function refreshImagePreview(scriptName, imageArg) {
  if (!imageArg) {
    clearImagePreview("No image argument in this template.");
    return;
  }

  const mode = getField(`file_mode__${imageArg}`);
  if (!mode) {
    clearImagePreview("No image selected.");
    return;
  }

  if (mode.value === "default") {
    const url = `/api/default-image?script=${encodeURIComponent(scriptName)}&arg=${encodeURIComponent(imageArg)}&t=${Date.now()}`;
    showImagePreview(url, "Using script default image.");
    return;
  }

  if (mode.value === "upload") {
    const upload = getField(`upload__${imageArg}`);
    if (!upload || !upload.files || upload.files.length === 0) {
      clearImagePreview("Upload an image to preview.");
      return;
    }

    const file = upload.files[0];
    const dataUrl = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(file);
    });
    showImagePreview(dataUrl, `Using uploaded image: ${file.name}`);
    return;
  }

  if (mode.value === "path") {
    const pathField = getField(`path__${imageArg}`);
    const pathText = pathField ? pathField.value.trim() : "";
    if (!pathText) {
      clearImagePreview("Enter a server image path.");
      return;
    }

    const url = `/api/preview-image-path?path=${encodeURIComponent(pathText)}&t=${Date.now()}`;
    try {
      const response = await fetch(url, { method: "HEAD" });
      if (!response.ok) {
        clearImagePreview("Cannot preview this server image path.");
        return;
      }
      showImagePreview(url, `Using server path: ${pathText}`);
    } catch {
      clearImagePreview("Cannot preview this server image path.");
    }
    return;
  }

  clearImagePreview("No image selected.");
}

async function refreshPromptPreview(scriptName, promptArg) {
  if (!promptArg) {
    renderPromptList([]);
    return;
  }

  const mode = getField(`file_mode__${promptArg}`);
  if (!mode) {
    renderPromptList([]);
    return;
  }

  if (mode.value === "default") {
    const response = await fetch(
      `/api/default-prompt-scenes?script=${encodeURIComponent(scriptName)}&arg=${encodeURIComponent(promptArg)}`
    );
    if (!response.ok) {
      renderPromptList([]);
      return;
    }
    const data = await response.json();
    renderPromptList(Array.isArray(data.scenes) ? data.scenes : []);
    return;
  }

  if (mode.value === "upload") {
    const upload = getField(`upload__${promptArg}`);
    if (!upload || !upload.files || upload.files.length === 0) {
      renderPromptList([]);
      return;
    }
    const promptText = await upload.files[0].text();
    renderPromptList(parseScenesFromPromptText(promptText));
    return;
  }

  if (mode.value === "path") {
    const pathField = getField(`path__${promptArg}`);
    const pathText = pathField ? pathField.value.trim() : "";
    if (!pathText) {
      previewPrompts.textContent = "Enter a server prompt path.";
      return;
    }

    try {
      const response = await fetch(`/api/preview-prompt-path?path=${encodeURIComponent(pathText)}`);
      if (!response.ok) {
        previewPrompts.textContent = "Cannot preview this server prompt path.";
        return;
      }
      const data = await response.json();
      const scenes = Array.isArray(data.scenes) ? data.scenes : [];
      if (scenes.length === 0) {
        previewPrompts.textContent = "No prompt scenes found at this path.";
      } else {
        renderPromptList(scenes);
      }
    } catch {
      previewPrompts.textContent = "Cannot preview this server prompt path.";
    }
    return;
  }

  const manualPrompt = getField("manual_prompt");
  const scenes = manualPrompt
    ? manualPrompt.value
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0)
    : [];
  renderPromptList(scenes);
}

function bindPreviewHandlers(scriptName, imageArg, promptArg) {
  const modeKeys = [];
  if (imageArg) {
    modeKeys.push(`file_mode__${imageArg}`);
    modeKeys.push(`upload__${imageArg}`);
    modeKeys.push(`path__${imageArg}`);
  }
  if (promptArg) {
    modeKeys.push(`file_mode__${promptArg}`);
    modeKeys.push(`upload__${promptArg}`);
    modeKeys.push(`path__${promptArg}`);
    modeKeys.push("manual_prompt");
  }

  for (const key of modeKeys) {
    const field = getField(key);
    if (!field) {
      continue;
    }

    const eventName = key.startsWith("upload__") ? "change" : "input";
    field.addEventListener(eventName, () => {
      refreshImagePreview(scriptName, imageArg);
      refreshPromptPreview(scriptName, promptArg);
    });

    if (eventName !== "change") {
      field.addEventListener("change", () => {
        refreshImagePreview(scriptName, imageArg);
        refreshPromptPreview(scriptName, promptArg);
      });
    }
  }
}

function getImageArg(config) {
  if (Object.prototype.hasOwnProperty.call(config.args, "ref_image_path")) {
    return "ref_image_path";
  }
  if (Object.prototype.hasOwnProperty.call(config.args, "image_path")) {
    return "image_path";
  }
  return "";
}

function getPromptArg(config) {
  if (Object.prototype.hasOwnProperty.call(config.args, "prompt_path")) {
    return "prompt_path";
  }
  return "";
}

function getOutputArg(config) {
  for (const key of OUTPUT_KEYS) {
    if (Object.prototype.hasOwnProperty.call(config.args, key)) {
      return key;
    }
  }
  return "";
}

function renderFields(scriptName) {
  const config = configs[scriptName];
  primaryFields.innerHTML = "";
  advancedFields.innerHTML = "";

  if (!config) {
    return;
  }

  const scriptInfo = document.createElement("div");
  scriptInfo.className = "script-info";
  scriptInfo.textContent = `Template entry script: ${config.python_script}`;
  primaryFields.appendChild(scriptInfo);

  const outputArg = getOutputArg(config);
  if (outputArg) {
    primaryFields.appendChild(
      createLabeledInput(
        outputArg,
        createTextInput(`param__${outputArg}`, config.args[outputArg], "videos/my_run/")
      )
    );
  }

  const imageArg = getImageArg(config);
  if (imageArg) {
    primaryFields.appendChild(
      createFileSourceField(imageArg, config.args[imageArg], {
        allowManualPrompt: false,
        allowDefault: false,
        allowPath: true,
        defaultMode: "upload",
      })
    );
  }

  const promptArg = getPromptArg(config);
  if (promptArg) {
    primaryFields.appendChild(
      createFileSourceField(promptArg, config.args[promptArg], {
        allowManualPrompt: true,
        allowDefault: false,
        allowPath: true,
        defaultMode: "upload",
      })
    );
    promptHelp.style.display = "block";
  } else {
    promptHelp.style.display = "none";
  }

  const cudaInput = createTextInput("cuda_device", config.cuda_device || "", "e.g. 0,1");
  cudaInput.title = "Use comma-separated GPU IDs, for example: 0 or 0,1,2";
  advancedFields.appendChild(createLabeledInput("CUDA_VISIBLE_DEVICES", cudaInput));

  for (const key of config.args_order) {
    const value = config.args[key];

    if (PRIMARY_KEYS.has(key)) {
      continue;
    }

    if (["ref_image_path", "image_path", "prompt_path", "pose_path", "audio_path"].includes(key)) {
      const allowManualPrompt = key === "prompt_path";
      advancedFields.appendChild(
        createFileSourceField(key, typeof value === "string" ? value : "", {
          allowManualPrompt,
          allowDefault: true,
          allowPath: true,
          defaultMode: "default",
        })
      );
      continue;
    }

    if (typeof value === "boolean") {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = `flag__${key}`;
      checkbox.checked = Boolean(value);
      advancedFields.appendChild(createLabeledInput(key, checkbox));
      continue;
    }

    advancedFields.appendChild(createLabeledInput(key, createTextInput(`param__${key}`, value)));
  }

  bindPreviewHandlers(scriptName, imageArg, promptArg);
  refreshImagePreview(scriptName, imageArg);
  refreshPromptPreview(scriptName, promptArg);
}

function initScriptSelect() {
  const names = Object.keys(configs);
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    scriptSelect.appendChild(option);
  }

  const initial = names.includes(selectedScriptFromServer) ? selectedScriptFromServer : names[0];
  if (initial) {
    scriptSelect.value = initial;
    renderFields(initial);
  }

  scriptSelect.addEventListener("change", () => renderFields(scriptSelect.value));
}

async function fetchJob(jobId) {
  const response = await fetch(`/job/${jobId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch job info.");
  }
  return response.json();
}

async function fetchJobs() {
  const response = await fetch("/jobs");
  if (!response.ok) {
    throw new Error("Failed to fetch jobs list.");
  }
  const data = await response.json();
  return Array.isArray(data.jobs) ? data.jobs : [];
}

function statusLabel(status) {
  if (status === "running") {
    return "Running";
  }
  if (status === "queued") {
    return "Queued";
  }
  if (status === "completed") {
    return "Completed";
  }
  if (status === "failed") {
    return "Failed";
  }
  return status || "Unknown";
}

function formatTimestamp(ts) {
  if (!ts) {
    return "";
  }
  const date = new Date(ts * 1000);
  return date.toLocaleString();
}

function renderRunList(jobs) {
  if (!runsList) {
    return;
  }

  runsList.innerHTML = "";
  if (!Array.isArray(jobs) || jobs.length === 0) {
    runsList.textContent = "No runs yet.";
    return;
  }

  for (const job of jobs) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `run-item${job.job_id === selectedJobId ? " selected" : ""}`;

    const title = document.createElement("p");
    title.className = "run-item-title";
    title.textContent = `${job.script_name || "script"} (${statusLabel(job.status)})`;

    const meta = document.createElement("p");
    meta.className = "run-item-meta";
    meta.textContent = `${job.job_id.slice(0, 8)} • ${formatTimestamp(job.created_ts)}`;

    item.appendChild(title);
    item.appendChild(meta);

    item.addEventListener("click", () => {
      selectedJobId = job.job_id;
      renderRunList(jobsSnapshot);
      refreshSelectedJob();
    });

    runsList.appendChild(item);
  }
}

async function refreshSelectedJob() {
  if (!selectedJobId) {
    selectedRunLabel.textContent = "Selected run: none";
    runStatus.textContent = "No active job.";
    runCommand.textContent = "";
    runOutput.textContent = "";
    runLogs.textContent = "(logs will appear here)";
    downloadOutputBtn.style.display = "none";
    downloadOutputBtn.removeAttribute("href");
    videoPreviewWrap.style.display = "none";
    previewVideo.removeAttribute("src");
    previewVideoHint.textContent = "";
    return;
  }

  try {
    const data = await fetchJob(selectedJobId);
    selectedRunLabel.textContent = `Selected run: ${selectedJobId}`;
    runStatus.textContent = `Status: ${statusLabel(data.status)}`;
    runCommand.textContent = `Command: ${data.command || ""}`;
    runOutput.textContent = data.output_path ? `Output path: ${data.output_path}` : "Output path: (not set)";
    runLogs.textContent = (data.logs || []).join("\n");

    if (data.can_download) {
      downloadOutputBtn.href = `/job/${encodeURIComponent(selectedJobId)}/download-output`;
      downloadOutputBtn.style.display = "inline-block";
    } else {
      downloadOutputBtn.style.display = "none";
      downloadOutputBtn.removeAttribute("href");
    }

    if (data.can_preview_video && data.preview_video_url) {
      const previewUrl = `${data.preview_video_url}?t=${Date.now()}`;
      if (previewVideo.src !== previewUrl) {
        previewVideo.src = previewUrl;
      }
      videoPreviewWrap.style.display = "block";
      previewVideoHint.textContent = "Previewing latest generated video from output path.";
    } else {
      videoPreviewWrap.style.display = "none";
      previewVideo.removeAttribute("src");
      previewVideoHint.textContent = "";
    }
  } catch (err) {
    runStatus.textContent = `Error: ${err.message}`;
  }
}

async function refreshMonitor() {
  try {
    jobsSnapshot = await fetchJobs();

    if (!selectedJobId && jobsSnapshot.length > 0) {
      selectedJobId = jobsSnapshot[0].job_id;
    }

    if (selectedJobId && !jobsSnapshot.some((job) => job.job_id === selectedJobId)) {
      selectedJobId = jobsSnapshot.length > 0 ? jobsSnapshot[0].job_id : "";
    }

    renderRunList(jobsSnapshot);
    await refreshSelectedJob();
  } catch (err) {
    runStatus.textContent = `Error: ${err.message}`;
  }
}

function startJobPolling() {
  refreshMonitor();
  setInterval(refreshMonitor, 1500);
}

initScriptSelect();
startJobPolling();
