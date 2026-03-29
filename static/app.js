const scriptSelect = document.getElementById("scriptSelect");
const primaryFields = document.getElementById("primaryFields");
const advancedFields = document.getElementById("advancedFields");
const promptHelp = document.getElementById("promptHelp");

const logPanel = document.getElementById("logPanel");
const runStatus = document.getElementById("runStatus");
const runCommand = document.getElementById("runCommand");
const runLogs = document.getElementById("runLogs");

const configsNode = document.getElementById("script-configs");
const selectedScriptNode = document.getElementById("selected-script");
const configs = configsNode ? JSON.parse(configsNode.textContent) : {};
const selectedScriptFromServer = selectedScriptNode ? JSON.parse(selectedScriptNode.textContent) : "";
const PRIMARY_KEYS = new Set(["output", "ref_image_path", "image_path", "prompt_path"]);

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
  manualPrompt.placeholder = "Scene 1 description\nScene 2 description\nScene 3 description";
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

  if (Object.prototype.hasOwnProperty.call(config.args, "output")) {
    primaryFields.appendChild(
      createLabeledInput(
        "output",
        createTextInput("param__output", config.args.output, "videos/my_run/")
      )
    );
  }

  const imageArg = getImageArg(config);
  if (imageArg) {
    primaryFields.appendChild(createPathUploadField(imageArg, config.args[imageArg]));
  }

  const promptArg = getPromptArg(config);
  if (promptArg) {
    primaryFields.appendChild(createPathUploadField(promptArg, config.args[promptArg]));
    promptHelp.style.display = "block";
  } else {
    promptHelp.style.display = "none";
  }

  advancedFields.appendChild(
    createLabeledInput(
      "CUDA_VISIBLE_DEVICES",
      createTextInput("cuda_device", config.cuda_device || "", "e.g. 0")
    )
  );

  for (const key of config.args_order) {
    const value = config.args[key];

    if (PRIMARY_KEYS.has(key)) {
      continue;
    }

    if (["ref_image_path", "image_path", "prompt_path", "pose_path", "audio_path"].includes(key)) {
      advancedFields.appendChild(createPathUploadField(key, typeof value === "string" ? value : ""));
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
