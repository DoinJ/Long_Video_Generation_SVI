const scriptSelect = document.getElementById("scriptSelect");
const dynamicFields = document.getElementById("dynamicFields");
const cudaField = document.getElementById("cudaField");

const logPanel = document.getElementById("logPanel");
const runStatus = document.getElementById("runStatus");
const runCommand = document.getElementById("runCommand");
const runLogs = document.getElementById("runLogs");

const configsNode = document.getElementById("script-configs");
const selectedScriptNode = document.getElementById("selected-script");
const configs = configsNode ? JSON.parse(configsNode.textContent) : {};
const selectedScriptFromServer = selectedScriptNode ? JSON.parse(selectedScriptNode.textContent) : "";

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

function createPathUploadField(key, defaultValue) {
  const wrap = document.createElement("div");
  wrap.className = "field file-field";

  const label = document.createElement("label");
  label.className = "label";
  label.textContent = key;

  const mode = document.createElement("select");
  mode.name = `file_mode__${key}`;
  mode.dataset.key = key;

  const pathOpt = document.createElement("option");
  pathOpt.value = "path";
  pathOpt.textContent = "Use server path (supports SCP files)";
  mode.appendChild(pathOpt);

  const uploadOpt = document.createElement("option");
  uploadOpt.value = "upload";
  uploadOpt.textContent = "Upload via browser";
  mode.appendChild(uploadOpt);

  if (key === "prompt_path") {
    const manualOpt = document.createElement("option");
    manualOpt.value = "manual";
    manualOpt.textContent = "Manual prompt text";
    mode.appendChild(manualOpt);
  }

  const pathInput = document.createElement("input");
  pathInput.type = "text";
  pathInput.name = `path__${key}`;
  pathInput.value = defaultValue || "";
  pathInput.placeholder = "/path/to/file";

  const uploadInput = document.createElement("input");
  uploadInput.type = "file";
  uploadInput.name = `upload__${key}`;
  uploadInput.style.display = "none";

  const manualPrompt = document.createElement("textarea");
  manualPrompt.name = "manual_prompt";
  manualPrompt.placeholder = 'prompts = ["Scene one", "Scene two"]';
  manualPrompt.rows = 8;
  manualPrompt.style.display = "none";

  mode.addEventListener("change", () => {
    if (mode.value === "path") {
      pathInput.style.display = "block";
      uploadInput.style.display = "none";
      manualPrompt.style.display = "none";
    } else if (mode.value === "upload") {
      pathInput.style.display = "none";
      uploadInput.style.display = "block";
      manualPrompt.style.display = "none";
    } else {
      pathInput.style.display = "none";
      uploadInput.style.display = "none";
      manualPrompt.style.display = "block";
    }
  });

  wrap.appendChild(label);
  wrap.appendChild(mode);
  wrap.appendChild(pathInput);
  wrap.appendChild(uploadInput);
  wrap.appendChild(manualPrompt);
  return wrap;
}

function renderFields(scriptName) {
  const config = configs[scriptName];
  dynamicFields.innerHTML = "";

  if (!config) {
    return;
  }

  cudaField.value = config.cuda_device || "";

  const scriptInfo = document.createElement("div");
  scriptInfo.className = "script-info";
  scriptInfo.textContent = `Python entry: ${config.python_script}`;
  dynamicFields.appendChild(scriptInfo);

  for (const key of config.args_order) {
    const value = config.args[key];

    if (["ref_image_path", "image_path", "prompt_path", "pose_path", "audio_path"].includes(key)) {
      dynamicFields.appendChild(createPathUploadField(key, typeof value === "string" ? value : ""));
      continue;
    }

    if (typeof value === "boolean") {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.name = `flag__${key}`;
      checkbox.checked = Boolean(value);
      dynamicFields.appendChild(createLabeledInput(key, checkbox));
      continue;
    }

    const input = document.createElement("input");
    input.type = "text";
    input.name = `param__${key}`;
    input.value = value;
    dynamicFields.appendChild(createLabeledInput(key, input));
  }
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

function startJobPolling() {
  const jobId = logPanel.dataset.jobId;
  if (!jobId) {
    return;
  }

  const timer = setInterval(async () => {
    try {
      const data = await fetchJob(jobId);
      runStatus.textContent = `Status: ${data.status}`;
      runCommand.textContent = `Command: ${data.command}`;
      runLogs.textContent = (data.logs || []).join("\n");

      if (data.status === "completed" || data.status === "failed") {
        clearInterval(timer);
      }
    } catch (err) {
      runStatus.textContent = `Error: ${err.message}`;
      clearInterval(timer);
    }
  }, 1500);
}

initScriptSelect();
startJobPolling();
