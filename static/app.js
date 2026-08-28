/**
 * Desk Review — client-side form handling and results rendering.
 */

const form = document.getElementById("analyze-form");
const submitBtn = document.getElementById("submit-btn");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");
const formError = document.getElementById("form-error");
const resultsSection = document.getElementById("results");

const fields = {
  targetRole: {
    input: document.getElementById("target-role"),
    error: document.getElementById("target-role-error"),
    min: 2,
    max: 200,
    label: "Target role",
    required: true,
    needsLetters: true,
  },
  jobDescription: {
    input: document.getElementById("job-description"),
    error: document.getElementById("job-description-error"),
    max: 5000,
    label: "Job description",
    required: false,
  },
  resume: {
    input: document.getElementById("resume"),
    error: document.getElementById("resume-error"),
    min: 50,
    max: 15000,
    label: "Resume",
    required: true,
    minAlpha: 30,
  },
};

const REQUIRED_RESPONSE_KEYS = [
  "score",
  "strengths",
  "weaknesses",
  "missing_keywords",
  "bullet_rewrites",
  "next_steps",
];

function setFieldError(fieldConfig, message) {
  const { input, error } = fieldConfig;
  if (message) {
    error.textContent = message;
    error.hidden = false;
    input.setAttribute("aria-invalid", "true");
  } else {
    error.textContent = "";
    error.hidden = true;
    input.removeAttribute("aria-invalid");
  }
}

function clearAllErrors() {
  Object.values(fields).forEach((field) => setFieldError(field, ""));
  formError.hidden = true;
  formError.textContent = "";
}

function validateField(fieldConfig) {
  const value = fieldConfig.input.value.trim();
  const { label, required, min, max, needsLetters, minAlpha } = fieldConfig;

  if (required && !value) {
    return `${label} is required.`;
  }

  if (!value) {
    return "";
  }

  if (min !== undefined && value.length < min) {
    return `${label} must be at least ${min} characters.`;
  }

  if (max !== undefined && value.length > max) {
    return `${label} must not exceed ${max} characters.`;
  }

  if (needsLetters && !/[A-Za-z]/.test(value)) {
    return `${label} must contain letters.`;
  }

  if (minAlpha !== undefined) {
    const alphaCount = (value.match(/[A-Za-z]/g) || []).length;
    if (alphaCount < minAlpha) {
      return `${label} must contain meaningful text content.`;
    }
  }

  return "";
}

function validateForm() {
  clearAllErrors();
  let firstInvalid = null;
  let isValid = true;

  Object.values(fields).forEach((fieldConfig) => {
    const message = validateField(fieldConfig);
    if (message) {
      setFieldError(fieldConfig, message);
      if (!firstInvalid) {
        firstInvalid = fieldConfig.input;
      }
      isValid = false;
    }
  });

  if (firstInvalid) {
    firstInvalid.focus();
  }

  return isValid;
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle("is-loading", isLoading);
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "Analyzing…" : "Analyze Resume";
  submitBtn.setAttribute("aria-busy", String(isLoading));
}

function showFormError(message) {
  formError.textContent = message;
  formError.hidden = false;
  formError.focus();
}

function isValidResponse(data) {
  if (!data || typeof data !== "object") {
    return false;
  }

  for (const key of REQUIRED_RESPONSE_KEYS) {
    if (!(key in data)) {
      return false;
    }
  }

  if (typeof data.score !== "number" || data.score < 0 || data.score > 100) {
    return false;
  }

  if (!Array.isArray(data.strengths) || data.strengths.length === 0) {
    return false;
  }

  if (!Array.isArray(data.weaknesses) || data.weaknesses.length === 0) {
    return false;
  }

  if (!Array.isArray(data.missing_keywords)) {
    return false;
  }

  if (!Array.isArray(data.bullet_rewrites)) {
    return false;
  }

  if (!Array.isArray(data.next_steps) || data.next_steps.length === 0) {
    return false;
  }

  return data.bullet_rewrites.every(
    (item) =>
      item &&
      typeof item.original === "string" &&
      typeof item.improved === "string" &&
      item.original.trim() &&
      item.improved.trim()
  );
}

function fillList(elementId, items, ordered = false) {
  const list = document.getElementById(elementId);
  list.innerHTML = "";
  const tag = ordered ? "ol" : "ul";

  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "None identified.";
    list.replaceWith(empty);
    empty.id = elementId;
    return;
  }

  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    list.appendChild(li);
  });
}

function renderResults(data) {
  document.getElementById("score-value").textContent = String(data.score);

  fillList("strengths-list", data.strengths);
  fillList("weaknesses-list", data.weaknesses);

  const keywordsList = document.getElementById("keywords-list");
  keywordsList.innerHTML = "";
  if (data.missing_keywords.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No major gaps";
    keywordsList.appendChild(li);
  } else {
    data.missing_keywords.forEach((keyword) => {
      const li = document.createElement("li");
      li.textContent = keyword;
      keywordsList.appendChild(li);
    });
  }

  const rewritesList = document.getElementById("rewrites-list");
  rewritesList.innerHTML = "";
  if (data.bullet_rewrites.length === 0) {
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = "No bullet rewrites suggested.";
    rewritesList.appendChild(p);
  } else {
    data.bullet_rewrites.forEach((rewrite) => {
      const dl = document.createElement("dl");
      dl.className = "rewrite-item";

      const originalDt = document.createElement("dt");
      originalDt.textContent = "Original";
      const originalDd = document.createElement("dd");
      originalDd.textContent = rewrite.original;

      const improvedDt = document.createElement("dt");
      improvedDt.textContent = "Improved";
      const improvedDd = document.createElement("dd");
      improvedDd.className = "improved";
      improvedDd.textContent = rewrite.improved;

      dl.append(originalDt, originalDd, improvedDt, improvedDd);
      rewritesList.appendChild(dl);
    });
  }

  fillList("next-steps-list", data.next_steps, true);

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function formatServerValidation(details) {
  if (!Array.isArray(details) || !details.length) {
    return "Please check your input and try again.";
  }
  return details
    .map((item) => {
      const field = item.loc?.slice(-1)[0] || "field";
      return `${field}: ${item.msg}`;
    })
    .join(" ");
}

async function handleSubmit(event) {
  event.preventDefault();

  if (!validateForm()) {
    return;
  }

  setLoading(true);
  clearAllErrors();
  resultsSection.hidden = true;

  const payload = {
    target_role: fields.targetRole.input.value.trim(),
    resume: fields.resume.input.value.trim(),
  };

  const jobDesc = fields.jobDescription.input.value.trim();
  if (jobDesc) {
    payload.job_description = jobDesc;
  }

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }

    if (!response.ok) {
      if (body?.code === "INVALID_MODEL_OUTPUT") {
        showFormError(
          "The AI returned unexpected data. Please try again in a moment."
        );
        return;
      }

      if (body?.error === "Validation failed" && body.details) {
        showFormError(formatServerValidation(body.details));
        return;
      }

      const message =
        body?.error ||
        (response.status === 429
          ? "Too many requests. Please wait and try again."
          : "Something went wrong. Please try again.");
      showFormError(message);
      return;
    }

    if (!isValidResponse(body)) {
      showFormError(
        "Received incomplete feedback from the server. Please try again."
      );
      return;
    }

    renderResults(body);
  } catch {
    showFormError(
      "Could not reach the server. Check your connection and try again."
    );
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", handleSubmit);

Object.values(fields).forEach((fieldConfig) => {
  fieldConfig.input.addEventListener("blur", () => {
    const message = validateField(fieldConfig);
    setFieldError(fieldConfig, message);
  });
});
