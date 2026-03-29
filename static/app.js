const scriptSelect = document.getElementById("scriptSelect");
const primaryFields = document.getElementById("primaryFields");
const advancedFields = document.getElementById("advancedFields");
const promptHelp = document.getElementById("promptHelp");
const previewImage = document.getElementById("previewImage");
const previewImageHint = document.getElementById("previewImageHint");
const previewPrompts = document.getElementById("previewPrompts");

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

function createFileSourceField(key, defaultValue, allowManualPrompt) {
  const wrap = document.createElement("div");
  wrap.className = "field file-field";

  const label = document.createElement("label");
  label.className = "label";
  label.textContent = key;

  const mode = document.createElement("select");
  mode.name = `file_mode__${key}`;
  mode.dataset.key = key;

  const defaultOpt = document.createElement("option");
  defaultOpt.value = "default";
  defaultOpt.textContent = "Use script default";
  mode.appendChild(defaultOpt);

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
  uploadInput.style.display = "none";

  const manualPrompt = document.createElement("textarea");
  manualPrompt.name = "manual_prompt";
  manualPrompt.placeholder = "Scene 1 description\nScene 2 description\nScene 3 description";
  manualPrompt.rows = 8;
  manualPrompt.style.display = "none";

  mode.addEventListener("change", () => {
    if (mode.value === "default") {
      defaultPath.style.display = "block";
      uploadInput.style.display = "none";
      manualPrompt.style.display = "none";
    } else if (mode.value === "upload") {
      defaultPath.style.display = "none";
      uploadInput.style.display = "block";
      manualPrompt.style.display = "none";
    } else {
      defaultPath.style.display = "none";
      uploadInput.style.display = "none";
      manualPrompt.style.display = "block";
    }
  });

  wrap.appendChild(label);
  wrap.appendChild(mode);
  wrap.appendChild(defaultPath);
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
  }
  if (promptArg) {
    modeKeys.push(`file_mode__${promptArg}`);
    modeKeys.push(`upload__${promptArg}`);
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
    primaryFields.appendChild(createFileSourceField(imageArg, config.args[imageArg], false));
  }

  const promptArg = getPromptArg(config);
  if (promptArg) {
    primaryFields.appendChild(createFileSourceField(promptArg, config.args[promptArg], true));
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
      const allowManualPrompt = key === "prompt_path";
      advancedFields.appendChild(createFileSourceField(key, typeof value === "string" ? value : "", allowManualPrompt));
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
