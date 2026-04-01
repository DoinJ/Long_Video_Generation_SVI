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
    generatedImageLink.style.display = "none";
    return;
  }

  generatedImage.src = url;
  generatedImage.style.display = "block";
  generatedImageLink.href = url;
  generatedImageLink.style.display = "inline-block";
}

async function submitImageGeneration(event) {
  event.preventDefault();

  const prompt = (promptInput.value || "").trim();
  if (!prompt) {
    imageApiOutput.textContent = "Prompt is required.";
    return;
  }

  const formData = new FormData();
  formData.append("prompt", prompt);
  if (imageInput.files && imageInput.files.length > 0) {
    formData.append("image", imageInput.files[0]);
  }

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
    imageApiMeta.textContent = `Model: ${data.model} | Base URL: ${data.base_url}`;
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

refreshPromptPreview();
clearImagePreview();
