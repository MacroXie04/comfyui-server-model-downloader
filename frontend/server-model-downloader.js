import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  ACTIVE_JOB_STATUSES,
  CANCELLABLE_JOB_STATUSES,
  PARTIAL_DISCARD_STATUSES,
  TERMINAL_SUCCESS_STATUSES,
  RequestCoordinator,
  asBoolean,
  asFiniteNumber,
  asString,
  formatBytes,
  historyPath,
  isAbortError,
  mergeJobs,
  normaliseJobs,
  phaseLabel,
  safeExternalUrl,
} from "./client-state.mjs";
import { scanCurrentGraphModels } from "./model-scan.mjs";

const EXTENSION_NAME = "server-model-downloader.ui";
const SIDEBAR_ID = "server-model-downloader";
const OPEN_COMMAND_ID = "ServerModelDownloader.Open";
const API_ROOT = "/server-model-downloader";
const POLL_INTERVAL_MS = 2_000;
const PAGE_SIZE = 50;

const scanRequests = new RequestCoordinator();
const jobRequests = new RequestCoordinator();
const createRequests = new RequestCoordinator();
const jobActionRequests = new RequestCoordinator();
const sessionRequests = new RequestCoordinator();

const state = {
  container: null,
  session: null,
  models: [],
  scannedModelSignature: "",
  jobs: [],
  nextCursor: null,
  selectedTokens: new Set(),
  licenseConfirmed: false,
  scanning: false,
  creatingJobs: false,
  loadingJobs: false,
  loadingMoreJobs: false,
  statusMessage: "",
  errorMessage: "",
  pollTimer: null,
  destroyed: false,
  refreshedCompletedJobs: new Set(),
  refreshingCombos: false,
  comboRefreshPending: false,
};

function ensureStylesheet() {
  const id = "server-model-downloader-styles";
  if (document.getElementById(id)) return;
  const link = document.createElement("link");
  link.id = id;
  link.rel = "stylesheet";
  link.href = new URL("./server-model-downloader.css", import.meta.url).href;
  document.head.appendChild(link);
}

function modelKey(model) {
  return `${model.directory}\u0000${model.name}\u0000${model.url}`;
}

function modelSetSignature(models) {
  return models.map(modelKey).sort().join("\u0001");
}

function clearScannedModels() {
  state.models = [];
  state.scannedModelSignature = "";
  state.selectedTokens.clear();
  state.licenseConfirmed = false;
}

function describeValue(value) {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.map(describeValue).filter(Boolean).join(", ");
  if (typeof value === "object") {
    return asString(value.name) || asString(value.id) || asString(value.title) || "";
  }
  return String(value);
}

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function createLink(url, text) {
  const safeUrl = safeExternalUrl(url);
  if (!safeUrl) return createElement("span", "smd-muted", text || "—");
  const link = createElement("a", "smd-link", text || "Open");
  link.href = safeUrl;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

function setMessage(message, { error = false } = {}) {
  if (error) {
    state.errorMessage = message;
    state.statusMessage = "";
  } else {
    state.statusMessage = message;
    state.errorMessage = "";
  }
  render();
}

async function readError(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const body = await response.json().catch(() => null);
    const message =
      asString(body?.detail) ||
      asString(body?.error) ||
      asString(body?.message) ||
      response.statusText;
    const requestId = asString(body?.request_id);
    return requestId ? `${message} (request ${requestId})` : message;
  }
  return asString(await response.text().catch(() => "")) || response.statusText;
}

async function requestJson(path, options = {}) {
  const response = await api.fetchApi(path, options);
  if (!response.ok) {
    const message = await readError(response);
    const error = new Error(`${response.status}: ${message}`);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) return null;
  return response.json();
}

async function loadSession(force = false, signal = undefined) {
  if (state.session && !force) return state.session;
  const payload = await requestJson(`${API_ROOT}/session`, { signal });
  const csrfToken = asString(payload?.csrf_token);
  if (!csrfToken) throw new Error("The server did not return a CSRF token.");
  state.session = {
    csrfToken,
    csrfExpiresAt: asFiniteNumber(payload?.csrf_expires_at),
    apiVersion: asString(payload?.api_version) || "1",
    extensionVersion: asString(payload?.extension_version) || "unknown",
    allowedDirectories: Array.isArray(payload?.allowed_directories)
      ? payload.allowed_directories.map(asString).filter(Boolean)
      : [],
    safeExtensions: Array.isArray(payload?.safe_extensions)
      ? payload.safe_extensions.map(asString).filter(Boolean)
      : [],
    capabilities:
      payload?.capabilities && typeof payload.capabilities === "object"
        ? payload.capabilities
        : {},
    identity: {
      email: asString(payload?.identity?.email),
      authMode: asString(payload?.identity?.auth_mode),
    },
  };
  return state.session;
}

async function mutateJson(path, { method = "POST", body, signal, retryCsrf = true } = {}) {
  const session = await loadSession(false, signal);
  const headers = { "X-SMD-CSRF": session.csrfToken };
  const options = { method, headers, signal };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  try {
    return await requestJson(path, options);
  } catch (error) {
    if (retryCsrf && error?.status === 403 && !signal?.aborted) {
      await loadSession(true, signal);
      return mutateJson(path, { method, body, signal, retryCsrf: false });
    }
    throw error;
  }
}

function normaliseInspectedModels(payload, scannedModels) {
  const items = Array.isArray(payload?.models) ? payload.models : [];
  const sourcesByKey = new Map(scannedModels.map((model) => [modelKey(model), model.sources]));
  return items.map((item, index) => {
    const name = asString(item?.name) || asString(item?.filename) || `Model ${index + 1}`;
    const filename = asString(item?.filename) || name;
    const url = asString(item?.url);
    const directory = asString(item?.directory);
    const key = modelKey({ name, url, directory });
    return {
      id: asString(item?.id) || key || String(index),
      name,
      filename,
      url,
      directory,
      installed: asBoolean(item?.installed),
      eligible: asBoolean(item?.eligible),
      reason: asString(item?.reason),
      source: describeValue(item?.source),
      graphSources: sourcesByKey.get(key) ?? [],
      size: asFiniteNumber(item?.size),
      sha256: asString(item?.sha256),
      license: describeValue(item?.license) || "Not specified",
      licenseUrl: asString(item?.license_url),
      revision: asString(item?.revision),
      downloadToken: asString(item?.download_token),
    };
  });
}

async function inspectModels() {
  const request = scanRequests.begin();
  state.scanning = true;
  state.errorMessage = "";
  state.statusMessage = "Scanning the current workflow…";
  render();
  try {
    const scannedModels = scanCurrentGraphModels(app);
    const scannedModelSignature = modelSetSignature(scannedModels);
    if (!request.isCurrent()) return;
    clearScannedModels();
    if (scannedModels.length === 0) {
      state.statusMessage = "The current workflow does not expose downloadable model metadata.";
      return;
    }
    const response = await mutateJson(`${API_ROOT}/inspect`, {
      body: { models: scannedModels.map(({ name, url, directory }) => ({ name, url, directory })) },
      signal: request.signal,
    });
    if (!request.isCurrent()) return;
    if (modelSetSignature(scanCurrentGraphModels(app)) !== scannedModelSignature) {
      clearScannedModels();
      state.errorMessage = "The active workflow changed during the scan. Scan it again before downloading.";
      state.statusMessage = "";
      return;
    }
    state.models = normaliseInspectedModels(response, scannedModels);
    state.scannedModelSignature = scannedModelSignature;
    const missing = state.models.filter((model) => !model.installed).length;
    const eligible = state.models.filter((model) => !model.installed && model.eligible && model.downloadToken).length;
    state.statusMessage = `${state.models.length} model(s) found: ${missing} missing, ${eligible} eligible for server download.`;
  } catch (error) {
    if (!request.isCurrent() || isAbortError(error)) return;
    state.models = [];
    state.errorMessage = `Scan failed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    if (request.isCurrent()) {
      state.scanning = false;
      render();
    }
  }
}

function selectedModels() {
  return state.models.filter((model) => model.downloadToken && state.selectedTokens.has(model.downloadToken));
}

async function createJobs() {
  if (state.creatingJobs) return;
  if (
    !state.scannedModelSignature ||
    modelSetSignature(scanCurrentGraphModels(app)) !== state.scannedModelSignature
  ) {
    clearScannedModels();
    setMessage("The active workflow changed. Scan it again before downloading.", { error: true });
    return;
  }
  const models = selectedModels();
  if (models.length === 0) {
    setMessage("Select at least one eligible model.", { error: true });
    return;
  }
  if (!state.licenseConfirmed) {
    setMessage("Confirm the licenses for every selected model before downloading.", { error: true });
    return;
  }
  const request = createRequests.begin();
  state.creatingJobs = true;
  state.errorMessage = "";
  state.statusMessage = "Submitting download jobs…";
  render();
  try {
    const response = await mutateJson(`${API_ROOT}/jobs`, {
      body: {
        download_tokens: models.map((model) => model.downloadToken),
        license_confirmed: true,
      },
      signal: request.signal,
    });
    if (!request.isCurrent()) return;
    const returned = normaliseJobs(response);
    if (returned.jobs.length) state.jobs = mergeJobs(state.jobs, returned.jobs);
    else await refreshJobs({ silent: true });
    state.statusMessage = `${models.length} download job(s) submitted.`;
    state.selectedTokens.clear();
    state.licenseConfirmed = false;
    schedulePolling();
  } catch (error) {
    if (!request.isCurrent() || isAbortError(error)) return;
    state.errorMessage = `Submission failed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    if (request.isCurrent()) {
      state.creatingJobs = false;
      render();
    }
  }
}

async function refreshJobs({ silent = false, append = false } = {}) {
  const request = jobRequests.begin();
  const cursor = append ? state.nextCursor : null;
  if (append) state.loadingMoreJobs = true;
  else state.loadingJobs = true;
  if (!silent) {
    state.errorMessage = "";
    state.statusMessage = append ? "Loading older jobs…" : "Refreshing download jobs…";
    render();
  }
  try {
    const response = await requestJson(historyPath(API_ROOT, cursor, PAGE_SIZE), { signal: request.signal });
    if (!request.isCurrent()) return;
    const page = normaliseJobs(response);
    state.jobs = append ? mergeJobs(state.jobs, page.jobs) : page.jobs;
    state.nextCursor = page.nextCursor;
    if (!silent) state.statusMessage = `${state.jobs.length} download job(s) loaded.`;
    await refreshNodeDefinitionsForCompletedJobs();
  } catch (error) {
    if (!request.isCurrent() || isAbortError(error)) return;
    state.errorMessage = `Could not load jobs: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    if (request.isCurrent()) {
      state.loadingJobs = false;
      state.loadingMoreJobs = false;
      render();
    }
  }
}

async function refreshNodeDefinitionsForCompletedJobs() {
  const newlyCompleted = state.jobs.filter(
    (job) => TERMINAL_SUCCESS_STATUSES.has(job.status) && !state.refreshedCompletedJobs.has(job.id),
  );
  if (!newlyCompleted.length) return;
  if (state.refreshingCombos) {
    state.comboRefreshPending = true;
    return;
  }
  state.refreshingCombos = true;
  try {
    await app.refreshComboInNodes();
    for (const job of newlyCompleted) state.refreshedCompletedJobs.add(job.id);
  } catch (error) {
    state.errorMessage = `Models were downloaded, but loader lists could not be refreshed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    state.refreshingCombos = false;
    if (state.comboRefreshPending) {
      state.comboRefreshPending = false;
      void refreshNodeDefinitionsForCompletedJobs();
    }
  }
}

async function cancelJob(jobId) {
  if (!jobId) return;
  const request = jobActionRequests.begin();
  try {
    await mutateJson(`${API_ROOT}/jobs/${encodeURIComponent(jobId)}/cancel`, {
      body: {},
      signal: request.signal,
    });
    if (!request.isCurrent()) return;
    setMessage("Cancellation requested.");
    await refreshJobs({ silent: true });
    schedulePolling();
  } catch (error) {
    if (!request.isCurrent() || isAbortError(error)) return;
    setMessage(`Cancellation failed: ${error instanceof Error ? error.message : String(error)}`, { error: true });
  }
}

async function discardPartial(job) {
  if (!job?.id || !PARTIAL_DISCARD_STATUSES.has(job.status)) return;
  if (!window.confirm(`Discard the resumable partial file for ${job.name}?`)) return;
  const request = jobActionRequests.begin();
  try {
    const response = await mutateJson(`${API_ROOT}/jobs/${encodeURIComponent(job.id)}/partial`, {
      method: "DELETE",
      signal: request.signal,
    });
    if (!request.isCurrent()) return;
    const updated = normaliseJobs(response).jobs;
    if (updated.length) state.jobs = mergeJobs(state.jobs, updated);
    else await refreshJobs({ silent: true });
    setMessage("Partial download discarded.");
  } catch (error) {
    if (!request.isCurrent() || isAbortError(error)) return;
    setMessage(`Could not discard the partial file: ${error instanceof Error ? error.message : String(error)}`, { error: true });
  }
}

function hasActiveJobs() {
  return state.jobs.some((job) => ACTIVE_JOB_STATUSES.has(job.status) || job.cancelRequested || job.phase === "retrying");
}

function schedulePolling() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = null;
  if (state.destroyed || !hasActiveJobs()) return;
  state.pollTimer = setTimeout(async () => {
    state.pollTimer = null;
    await refreshJobs({ silent: true });
    schedulePolling();
  }, POLL_INTERVAL_MS);
}

function createHeader() {
  const header = createElement("header", "smd-header");
  const titleWrap = createElement("div", "smd-title-wrap");
  titleWrap.append(
    createElement("h2", "smd-title", "Server Model Downloader"),
    createElement("p", "smd-subtitle", "Download verified workflow models directly to this ComfyUI server."),
  );
  const actions = createElement("div", "smd-header-actions");
  const scanButton = createElement("button", "smd-button smd-button-primary", state.scanning ? "Scanning…" : "Scan workflow");
  scanButton.type = "button";
  scanButton.disabled = state.scanning;
  scanButton.addEventListener("click", () => void inspectModels());
  const refreshButton = createElement("button", "smd-button", "Refresh jobs");
  refreshButton.type = "button";
  refreshButton.disabled = state.loadingJobs;
  refreshButton.addEventListener("click", () => void refreshJobs());
  actions.append(scanButton, refreshButton);
  header.append(titleWrap, actions);
  return header;
}

function createSessionNote() {
  const note = createElement("section", "smd-security-note");
  const directories = state.session?.allowedDirectories?.join(", ") || "server allowlist";
  const extensions = state.session?.safeExtensions?.join(", ") || "server allowlist";
  const identity = state.session?.identity?.email;
  note.append(
    createElement("strong", "", "Server-enforced restrictions"),
    createElement("span", "", ` — destinations: ${directories}; formats: ${extensions}.`),
  );
  if (identity) note.append(createElement("span", "smd-session-identity", ` Signed in as ${identity}.`));
  return note;
}

function createMessageArea() {
  const wrapper = createElement("div", "smd-messages");
  wrapper.setAttribute("aria-live", "polite");
  if (state.errorMessage) wrapper.append(createElement("div", "smd-alert smd-alert-error", state.errorMessage));
  if (state.statusMessage) wrapper.append(createElement("div", "smd-alert smd-alert-info", state.statusMessage));
  return wrapper;
}

function createBadge(text, kind) {
  return createElement("span", `smd-badge smd-badge-${kind}`, text);
}

function createMetadataRow(label, valueNode) {
  const row = createElement("div", "smd-meta-row");
  row.append(createElement("span", "smd-meta-label", label), valueNode);
  return row;
}

function createModelCard(model) {
  const selectable = !model.installed && model.eligible && Boolean(model.downloadToken);
  const card = createElement("article", `smd-card${selectable ? "" : " smd-card-disabled"}`);
  const heading = createElement("div", "smd-card-heading");
  const check = document.createElement("input");
  check.type = "checkbox";
  check.className = "smd-checkbox";
  check.disabled = !selectable || state.creatingJobs;
  check.checked = selectable && state.selectedTokens.has(model.downloadToken);
  check.setAttribute("aria-label", `Select ${model.name}`);
  check.addEventListener("change", () => {
    if (check.checked) state.selectedTokens.add(model.downloadToken);
    else state.selectedTokens.delete(model.downloadToken);
    state.licenseConfirmed = false;
    render();
  });
  const nameWrap = createElement("div", "smd-model-name-wrap");
  nameWrap.append(createElement("div", "smd-model-name", model.name), createElement("div", "smd-model-filename", model.filename));
  const badge = model.installed
    ? createBadge("Installed", "success")
    : model.eligible && model.downloadToken
      ? createBadge("Eligible", "ready")
      : createBadge("Unavailable", "muted");
  heading.append(check, nameWrap, badge);
  card.append(heading);

  const metadata = createElement("div", "smd-metadata");
  metadata.append(
    createMetadataRow("Folder", createElement("code", "smd-code", model.directory || "—")),
    createMetadataRow("Size", createElement("span", "", formatBytes(model.size))),
    createMetadataRow("Source", createElement("span", "", model.source || "Unverified (link hidden)")),
    createMetadataRow("License", createLink(model.licenseUrl, model.license)),
  );
  if (model.revision) metadata.append(createMetadataRow("Revision", createElement("code", "smd-code", model.revision)));
  if (model.sha256) metadata.append(createMetadataRow("SHA-256", createElement("code", "smd-code smd-hash", model.sha256)));
  if (model.graphSources.length) {
    const sources = createElement("span", "smd-source-path", model.graphSources.join("; "));
    sources.title = model.graphSources.join("\n");
    metadata.append(createMetadataRow("Nodes", sources));
  }
  card.append(metadata);
  if (model.reason) card.append(createElement("p", "smd-reason", model.reason));
  return card;
}

function createModelsSection() {
  const section = createElement("section", "smd-section");
  const heading = createElement("div", "smd-section-heading");
  heading.append(createElement("h3", "smd-section-title", `Workflow models (${state.models.length})`));
  const eligible = state.models.filter((model) => !model.installed && model.eligible && model.downloadToken);
  const selectAll = createElement("button", "smd-button smd-button-small", "Select eligible");
  selectAll.type = "button";
  selectAll.disabled = eligible.length === 0;
  selectAll.addEventListener("click", () => {
    const allSelected = eligible.every((model) => state.selectedTokens.has(model.downloadToken));
    state.selectedTokens.clear();
    if (!allSelected) for (const model of eligible) state.selectedTokens.add(model.downloadToken);
    state.licenseConfirmed = false;
    render();
  });
  heading.append(selectAll);
  section.append(heading);

  const list = createElement("div", "smd-card-list");
  if (!state.models.length) {
    list.append(createElement("div", "smd-empty", state.scanning ? "Scanning…" : "Choose Scan workflow to inspect the active workflow. No provider is contacted until you scan."));
  } else {
    for (const model of state.models) list.append(createModelCard(model));
  }
  section.append(list);

  const selected = selectedModels();
  const consent = createElement("label", "smd-consent");
  const consentCheck = document.createElement("input");
  consentCheck.type = "checkbox";
  consentCheck.checked = state.licenseConfirmed;
  consentCheck.disabled = selected.length === 0 || state.creatingJobs;
  consentCheck.addEventListener("change", () => {
    state.licenseConfirmed = consentCheck.checked;
    render();
  });
  consent.append(consentCheck, createElement("span", "", "I reviewed and accept each selected model license and am authorized to store it on this server."));
  section.append(consent);

  const downloadButton = createElement("button", "smd-button smd-button-primary smd-download-button", state.creatingJobs ? "Submitting…" : `Download selected to server (${selected.length})`);
  downloadButton.type = "button";
  downloadButton.disabled = selected.length === 0 || !state.licenseConfirmed || state.creatingJobs || state.scanning;
  downloadButton.addEventListener("click", () => void createJobs());
  section.append(downloadButton);
  return section;
}

function createJobCard(job) {
  const card = createElement("article", "smd-job");
  const heading = createElement("div", "smd-job-heading");
  const nameWrap = createElement("div", "smd-model-name-wrap");
  nameWrap.append(createElement("div", "smd-model-name", job.name), createElement("div", "smd-model-filename", job.directory || "—"));
  const badgeKind = TERMINAL_SUCCESS_STATUSES.has(job.status)
    ? "success"
    : job.status === "failed"
      ? "error"
      : ACTIVE_JOB_STATUSES.has(job.status) || job.cancelRequested
        ? "ready"
        : "muted";
  const displayedPhase = job.cancelRequested ? "cancelling" : job.phase;
  heading.append(nameWrap, createBadge(phaseLabel(displayedPhase, job.status), badgeKind));
  card.append(heading);

  const progress = job.progress ?? 0;
  const progressTrack = createElement("div", "smd-progress");
  progressTrack.setAttribute("role", "progressbar");
  progressTrack.setAttribute("aria-label", `${job.name} download progress`);
  progressTrack.setAttribute("aria-valuemin", "0");
  progressTrack.setAttribute("aria-valuemax", "100");
  progressTrack.setAttribute("aria-valuenow", String(progress));
  const progressBar = createElement("div", "smd-progress-bar");
  progressBar.style.width = `${progress}%`;
  progressTrack.append(progressBar);
  card.append(progressTrack);

  const details = createElement("div", "smd-job-details");
  const byteText = job.size === null ? formatBytes(job.bytesDownloaded) : `${formatBytes(job.bytesDownloaded)} / ${formatBytes(job.size)}`;
  const attemptText = job.attempt > 0 ? `Attempt ${job.attempt}/${job.maxAttempts}` : "";
  details.append(createElement("span", "", byteText), createElement("span", "", job.progress === null ? attemptText : `${job.progress.toFixed(1)}%${attemptText ? ` · ${attemptText}` : ""}`));
  card.append(details);
  if (job.error) card.append(createElement("p", "smd-reason smd-job-error", `${job.errorCode ? `${job.errorCode}: ` : ""}${job.error}`));

  const actions = createElement("div", "smd-job-actions");
  if (CANCELLABLE_JOB_STATUSES.has(job.status) && !job.cancelRequested) {
    const cancel = createElement("button", "smd-button smd-button-danger smd-button-small", "Cancel");
    cancel.type = "button";
    cancel.addEventListener("click", () => void cancelJob(job.id));
    actions.append(cancel);
  }
  if (PARTIAL_DISCARD_STATUSES.has(job.status) && job.partialAvailable) {
    const discard = createElement("button", "smd-button smd-button-small", "Discard partial");
    discard.type = "button";
    discard.addEventListener("click", () => void discardPartial(job));
    actions.append(discard);
  }
  if (actions.childElementCount) card.append(actions);
  return card;
}

function createJobsSection() {
  const section = createElement("section", "smd-section");
  section.append(createElement("h3", "smd-section-title", `Download history (${state.jobs.length})`));
  const list = createElement("div", "smd-job-list");
  if (!state.jobs.length) list.append(createElement("div", "smd-empty", "No download jobs yet."));
  else for (const job of state.jobs) list.append(createJobCard(job));
  section.append(list);
  if (state.nextCursor) {
    const more = createElement("button", "smd-button smd-load-more", state.loadingMoreJobs ? "Loading…" : "Load older jobs");
    more.type = "button";
    more.disabled = state.loadingMoreJobs;
    more.addEventListener("click", () => void refreshJobs({ append: true }));
    section.append(more);
  }
  return section;
}

function render() {
  const container = state.container;
  if (!container || state.destroyed) return;
  const root = createElement("div", "smd-root");
  root.append(createHeader(), createSessionNote(), createMessageArea(), createModelsSection(), createJobsSection());
  container.replaceChildren(root);
}

async function initialise() {
  const request = sessionRequests.begin();
  try {
    await loadSession(true, request.signal);
    if (!request.isCurrent()) return;
    render();
    await refreshJobs({ silent: true });
    schedulePolling();
  } catch (error) {
    if (!request.isCurrent() || isAbortError(error)) return;
    state.errorMessage = `Initialisation failed: ${error instanceof Error ? error.message : String(error)}`;
    render();
  }
}

function renderSidebar(container) {
  state.container = container;
  state.destroyed = false;
  render();
  void initialise();
}

function destroySidebar() {
  state.destroyed = true;
  state.container = null;
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = null;
  scanRequests.cancel();
  jobRequests.cancel();
  createRequests.cancel();
  jobActionRequests.cancel();
  sessionRequests.cancel();
  state.scanning = false;
  state.loadingJobs = false;
  state.loadingMoreJobs = false;
  state.creatingJobs = false;
  state.comboRefreshPending = false;
  clearScannedModels();
}

ensureStylesheet();

app.registerExtension({
  name: EXTENSION_NAME,
  commands: [
    {
      id: OPEN_COMMAND_ID,
      label: "Server Model Downloader",
      icon: "mdi mdi-cloud-download",
      function: () => app.extensionManager.command.execute(`Workspace.ToggleSidebarTab.${SIDEBAR_ID}`),
    },
  ],
  menuCommands: [{ path: ["Extensions", "Server Model Downloader"], commands: [OPEN_COMMAND_ID] }],
  async setup() {
    app.extensionManager.registerSidebarTab({
      id: SIDEBAR_ID,
      icon: "mdi mdi-cloud-download",
      title: "Server Model Downloader",
      tooltip: "Download verified workflow models directly to this server",
      type: "custom",
      render: renderSidebar,
      destroy: destroySidebar,
    });
  },
});
