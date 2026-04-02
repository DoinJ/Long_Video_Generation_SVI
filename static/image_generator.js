const imageGenForm = document.getElementById("imageGenForm");
const promptInput = document.getElementById("promptInput");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const imagePreviewHint = document.getElementById("imagePreviewHint");
const promptPreview = document.getElementById("promptPreview");
const imageApiMeta = document.getElementById("imageApiMeta");
const imageApiOutput = document.getElementById("imageApiOutput");
const generatedImage = document.getElementById("generatedImage");
const generatedImageLink = document.getElementById("generatedImageLink");
const providerPresetSelect = document.getElementById("providerPresetSelect");
const sourceSelect = document.getElementById("sourceSelect");
const apiFormatSelect = document.getElementById("apiFormatSelect");
const cloudSection = document.getElementById("cloudSection");
const localSection = document.getElementById("localSection");
const cloudSimpleEndpointWrap = document.getElementById("cloudSimpleEndpointWrap");
const localSimpleEndpointWrap = document.getElementById("localSimpleEndpointWrap");
const localUseLoraInput = document.getElementById("localUseLoraInput");
const localLoraPathWrap = document.getElementById("localLoraPathWrap");
const localLoraPathInput = document.getElementById("localLoraPathInput");
const apiExtraJsonWrap = document.getElementById("apiExtraJsonWrap");
const cloudModelInput = document.getElementById("cloudModelInput");
const cloudApiKeyInput = document.getElementById("cloudApiKeyInput");
const cloudBaseUrlInput = document.getElementById("cloudBaseUrlInput");
const cloudSimpleEndpointInput = document.getElementById("cloudSimpleEndpointInput");
const localModelInput = document.getElementById("localModelInput");
const localBaseUrlInput = document.getElementById("localBaseUrlInput");
const localApiKeyInput = document.getElementById("localApiKeyInput");
const localCudaDeviceInput = document.getElementById("localCudaDeviceInput");
const saveGeneratedImageInput = document.getElementById("saveGeneratedImageInput");
const saveGeneratedDirWrap = document.getElementById("saveGeneratedDirWrap");
const saveGeneratedDirInput = document.getElementById("saveGeneratedDirInput");
const localSimpleEndpointInput = document.getElementById("localSimpleEndpointInput");
const apiExtraJsonInput = document.getElementById("apiExtraJsonInput");

function setIfEmpty(inputEl, value) {
  if (!inputEl) {
    return;
  }
  if ((inputEl.value || "").trim()) {
    return;
  }
  inputEl.value = value;
}

function applyProviderPreset() {
  const preset = providerPresetSelect.value;
  if (preset === "custom") {
    return;
  }

  if (preset === "openai-cloud") {
    sourceSelect.value = "cloud";
    apiFormatSelect.value = "openai";
    setIfEmpty(cloudModelInput, "gpt-4o");
    setIfEmpty(cloudBaseUrlInput, "https://api.openai.com/v1");
    if (apiExtraJsonInput) {
      apiExtraJsonInput.value = "";
    }
  }

  if (preset === "minimax-image-01") {
    sourceSelect.value = "cloud";
    apiFormatSelect.value = "simple";
    setIfEmpty(cloudModelInput, "image-01");
    setIfEmpty(cloudSimpleEndpointInput, "https://api.minimax.io/v1/image_generation");
    if (cloudBaseUrlInput && !(cloudBaseUrlInput.value || "").trim()) {
      cloudBaseUrlInput.value = "https://api.minimax.io";
    }
    if (apiExtraJsonInput && !(apiExtraJsonInput.value || "").trim()) {
      apiExtraJsonInput.value = JSON.stringify(
        {
          aspect_ratio: "16:9",
          response_format: "base64",
          subject_reference: [
            {
              type: "character",
              image_file: "https://example.com/character_reference.jpg",
            },
          ],
        },
        null,
        2,
      );
    }
  }

  if (preset === "local-qwen") {
    sourceSelect.value = "local";
    apiFormatSelect.value = "openai";
    setIfEmpty(localModelInput, "Qwen/Qwen-Image-Edit-2511");
    setIfEmpty(localBaseUrlInput, "inprocess");
    setIfEmpty(localSimpleEndpointInput, "inprocess");
    if (localApiKeyInput && !(localApiKeyInput.value || "").trim()) {
      localApiKeyInput.value = "";
    }
    if (localCudaDeviceInput && !(localCudaDeviceInput.value || "").trim()) {
      localCudaDeviceInput.value = "0";
    }
    if (saveGeneratedImageInput) {
      saveGeneratedImageInput.checked = false;
    }
    if (saveGeneratedDirInput && !(saveGeneratedDirInput.value || "").trim()) {
      saveGeneratedDirInput.value = "uploads/images/generated";
    }
    if (localLoraPathInput && !(localLoraPathInput.value || "").trim()) {
      localLoraPathInput.value = "/home/usnmp/jaden/your_lora_dir/model.safetensors";
    }
  }

  refreshSourceSections();
  refreshLoraFields();
  refreshSaveGeneratedFields();
}

function refreshPromptPreview() {
  const text = (promptInput.value || "").trim();
  promptPreview.textContent = text || "No prompt yet.";
}

function clearImagePreview() {
  imagePreview.removeAttribute("src");
  imagePreview.style.display = "none";
  imagePreviewHint.textContent = "No image selected.";
}

function refreshImagePreview() {
  if (!imageInput.files || imageInput.files.length === 0) {
    clearImagePreview();
    return;
  }

  const file = imageInput.files[0];
  const reader = new FileReader();
  reader.onload = () => {
    imagePreview.src = String(reader.result);
    imagePreview.style.display = "block";
    imagePreviewHint.textContent = `Selected: ${file.name}`;
  };
  reader.readAsDataURL(file);
}

function tryExtractImageUrl(text) {
  if (!text) {
    return "";
  }

  const markdown = text.match(/!\[[^\]]*\]\((https?:\/\/[^)\s]+)\)/i);
  if (markdown && markdown[1]) {
    return markdown[1].replace(/[),.;]+$/, "");
  }

  const refDefs = new Map();
  const refDefRegex = /^\s*\[([^\]]+)\]:\s*(https?:\/\/\S+)/gim;
  let refDefMatch;
  while ((refDefMatch = refDefRegex.exec(text)) !== null) {
    const key = String(refDefMatch[1] || "").trim().toLowerCase();
    const url = String(refDefMatch[2] || "").replace(/[),.;]+$/, "");
    if (key && url) {
      refDefs.set(key, url);
    }
  }

  const markdownRef = text.match(/!\[[^\]]*\]\[([^\]]+)\]/i);
  if (markdownRef && markdownRef[1]) {
    const refKey = String(markdownRef[1]).trim().toLowerCase();
    const refUrl = refDefs.get(refKey);
    if (refUrl) {
      return refUrl;
    }
  }

  const direct = text.match(/https?:\/\/\S+/i);
  if (!direct || !direct[0]) {
    return "";
  }

  return direct[0].replace(/[),.;]+$/, "");
}

function showGeneratedImage(url) {
  if (!url) {
    generatedImage.removeAttribute("src");
    generatedImage.style.display = "none";
    generatedImageLink.removeAttribute("href");
    generatedImageLink.removeAttribute("download");
    generatedImageLink.style.display = "none";
    return;
  }

  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  const fileName = `generated_${ts}.png`;
  generatedImage.src = url;
  generatedImage.style.display = "block";
  generatedImageLink.href = url;
  generatedImageLink.setAttribute("download", fileName);
  generatedImageLink.style.display = "inline-block";
}

async function forceDownloadImage(url, fileName) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  const tempLink = document.createElement("a");
  tempLink.href = blobUrl;
  tempLink.download = fileName;
  document.body.appendChild(tempLink);
  tempLink.click();
  tempLink.remove();
  URL.revokeObjectURL(blobUrl);
}

function refreshSourceSections() {
  const source = sourceSelect.value;
  const apiFormat = apiFormatSelect.value;

  cloudSection.style.display = source === "cloud" ? "block" : "none";
  localSection.style.display = source === "local" ? "block" : "none";

  cloudSimpleEndpointWrap.style.display = source === "cloud" && apiFormat === "simple" ? "block" : "none";
  localSimpleEndpointWrap.style.display = source === "local" && apiFormat === "simple" ? "block" : "none";
  apiExtraJsonWrap.style.display = apiFormat === "simple" ? "block" : "none";
}

function refreshLoraFields() {
  const isLocal = sourceSelect.value === "local";
  const enabled = Boolean(localUseLoraInput.checked) && isLocal;
  localLoraPathWrap.style.display = enabled ? "block" : "none";
  localLoraPathInput.required = enabled;
}

function refreshSaveGeneratedFields() {
  const isLocal = sourceSelect.value === "local";
  const enabled = Boolean(saveGeneratedImageInput.checked) && isLocal;
  saveGeneratedDirWrap.style.display = enabled ? "block" : "none";
}

async function submitImageGeneration(event) {
  event.preventDefault();

  const prompt = (promptInput.value || "").trim();
  if (!prompt) {
    imageApiOutput.textContent = "Prompt is required.";
    return;
  }

  if (sourceSelect.value === "local" && localUseLoraInput.checked) {
    const loraPath = (localLoraPathInput.value || "").trim();
    if (!loraPath) {
      imageApiOutput.textContent = "LoRA path is required when 'Use LoRA safetensors' is enabled.";
      return;
    }
  }

  if (sourceSelect.value === "local") {
    const gpuText = (localCudaDeviceInput.value || "").trim();
    if (gpuText && !/^\d+(\s*,\s*\d+)*$/.test(gpuText)) {
      imageApiOutput.textContent = "Local GPU index must be a non-negative integer list, e.g. 0 or 0,1.";
      return;
    }
  }

  const formData = new FormData(imageGenForm);

  imageApiMeta.style.display = "none";
  showGeneratedImage("");
  imageApiOutput.textContent = "Requesting model...";

  try {
    const response = await fetch("/api/image-generator/run", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      imageApiOutput.textContent = data.error || "Request failed.";
      return;
    }

    imageApiMeta.style.display = "block";
    const loraText = data.lora_used ? ` | LoRA: ON (${data.lora_path || "path not provided"})` : "";
    const gpuValue = data.cuda_devices || data.cuda_device;
    const gpuText = gpuValue !== undefined && gpuValue !== null ? ` | GPU(s): ${gpuValue}` : "";
    const saveText = data.saved_image_path ? ` | Saved: ${data.saved_image_path}` : "";
    imageApiMeta.textContent = `Source: ${data.source || "unknown"} | API: ${data.api_format || "unknown"} | Model: ${data.model} | Base URL: ${data.base_url}${gpuText}${loraText}${saveText}`;
    const resultText = data.result || "(no content)";
    imageApiOutput.textContent = resultText;
    showGeneratedImage(data.image_url || tryExtractImageUrl(resultText));
  } catch (error) {
    imageApiOutput.textContent = `Request error: ${error}`;
    showGeneratedImage("");
  }
}

promptInput.addEventListener("input", refreshPromptPreview);
imageInput.addEventListener("change", refreshImagePreview);
imageGenForm.addEventListener("submit", submitImageGeneration);
sourceSelect.addEventListener("change", () => {
  refreshSourceSections();
  refreshLoraFields();
  refreshSaveGeneratedFields();
});
apiFormatSelect.addEventListener("change", refreshSourceSections);
localUseLoraInput.addEventListener("change", refreshLoraFields);
saveGeneratedImageInput.addEventListener("change", refreshSaveGeneratedFields);
providerPresetSelect.addEventListener("change", applyProviderPreset);
generatedImageLink.addEventListener("click", async (event) => {
  event.preventDefault();
  const href = generatedImageLink.getAttribute("href") || "";
  if (!href) {
    return;
  }

  const downloadName = generatedImageLink.getAttribute("download") || "generated.png";
  try {
    await forceDownloadImage(href, downloadName);
  } catch (error) {
    window.open(href, "_blank", "noopener");
  }
});

refreshPromptPreview();
clearImagePreview();
refreshSourceSections();
refreshLoraFields();
refreshSaveGeneratedFields();
