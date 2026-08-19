import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { scanCurrentGraphModels } from "./model-scan.mjs";

const EXTENSION_NAME = "hongzhe.ServerModelDownloader";
const SIDEBAR_ID = "server-model-downloader";
const OPEN_COMMAND_ID = "ServerModelDownloader.Open";
const API_ROOT = "/server-model-downloader";
const POLL_INTERVAL_MS = 2_000;
const ACTIVE_JOB_STATUSES = new Set([
  "pending",
  "queued",
  "starting",
  "running",
  "in-progress",
  "downloading",
  "verifying",
  "cancelling",
]);
const TERMINAL_SUCCESS_STATUSES = new Set(["completed", "complete", "success", "done"]);
const CANCELLABLE_JOB_STATUSES = new Set([
  "pending",
  "queued",
  "starting",
  "running",
  "in-progress",
  "downloading",
  "verifying",
]);

/** @typedef {{name: string, url: string, directory: string, sources: string[]}} ScannedModel */

const state = {
  container: null,
  session: null,
  models: [],
  jobs: [],
  selectedTokens: new Set(),
  licenseConfirmed: false,
  scanning: false,
  creatingJobs: false,
  loadingJobs: false,
  statusMessage: "",
  errorMessage: "",
  pollTimer: null,
  destroyed: false,
  refreshedCompletedJobs: new Set(),
  refreshingCombos: false,
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

function asString(value) {
  return typeof value === "string" ? value.trim() : "";
}

function asBoolean(value) {
  return value === true || value === "true" || value === 1;
}

function asFiniteNumber(value) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

function normaliseStatus(value) {
  return asString(value).toLowerCase().replaceAll("_", "-") || "unknown";
}

function modelKey(model) {
  return `${model.directory}\u0000${model.name}\u0000${model.url}`;
}

function safeExternalUrl(value) {
  const raw = asString(value);
  if (!raw) return null;
  try {
    const parsed = new URL(raw);
    return parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
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

function formatBytes(value) {
  const bytes = asFiniteNumber(value);
  if (bytes === null) return "未知";
  if (bytes === 0) return "0 B";

  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const amount = bytes / 1024 ** exponent;
  return `${amount.toFixed(amount >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

function clampProgress(value) {
  const number = asFiniteNumber(value);
  if (number === null) return null;
  const percent = number <= 1 ? number * 100 : number;
  return Math.max(0, Math.min(100, percent));
}

function statusLabel(status) {
  const labels = {
    pending: "等待中",
    queued: "已排队",
    starting: "正在启动",
    running: "进行中",
    "in-progress": "进行中",
    downloading: "下载中",
    verifying: "正在校验",
    completed: "已完成",
    complete: "已完成",
    success: "已完成",
    done: "已完成",
    failed: "失败",
    error: "失败",
    cancelling: "正在取消",
    cancelled: "已取消",
    canceled: "已取消",
    unknown: "未知",
  };
  return labels[status] ?? status;
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

  const link = createElement("a", "smd-link", text || "打开");
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
    return (
      asString(body?.detail) ||
      asString(body?.error) ||
      asString(body?.message) ||
      response.statusText
    );
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

async function loadSession(force = false) {
  if (state.session && !force) return state.session;
  const payload = await requestJson(`${API_ROOT}/session`);
  const csrfToken = asString(payload?.csrf_token);
  if (!csrfToken) throw new Error("服务器没有返回 CSRF token");

  state.session = {
    csrfToken,
    allowedDirectories: Array.isArray(payload?.allowed_directories)
      ? payload.allowed_directories.map(asString).filter(Boolean)
      : [],
    safeExtensions: Array.isArray(payload?.safe_extensions)
      ? payload.safe_extensions.map(asString).filter(Boolean)
      : [],
  };
  return state.session;
}

async function postJson(path, body, { retryCsrf = true } = {}) {
  const session = await loadSession();
  try {
    return await requestJson(path, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-SMD-CSRF": session.csrfToken,
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    if (retryCsrf && error?.status === 403) {
      await loadSession(true);
      return postJson(path, body, { retryCsrf: false });
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
    const backendSource = describeValue(item?.source);
    const localSources = sourcesByKey.get(key) ?? [];

    return {
      id: asString(item?.id) || key || String(index),
      name,
      filename,
      url,
      directory,
      installed: asBoolean(item?.installed),
      eligible: asBoolean(item?.eligible),
      reason: asString(item?.reason),
      source: backendSource,
      graphSources: localSources,
      size: asFiniteNumber(item?.size),
      sha256: asString(item?.sha256),
      license: describeValue(item?.license) || "未标注",
      licenseUrl: asString(item?.license_url),
      revision: asString(item?.revision),
      downloadToken: asString(item?.download_token),
    };
  });
}

function normaliseJobs(payload) {
  const items = Array.isArray(payload) ? payload : Array.isArray(payload?.jobs) ? payload.jobs : [];
  return items.map((item, index) => {
    const status = normaliseStatus(item?.status);
    const bytesDownloaded = asFiniteNumber(item?.bytes_downloaded) ?? 0;
    const size = asFiniteNumber(item?.size);
    const explicitProgress = clampProgress(item?.progress);
    const progress =
      explicitProgress ?? (size && size > 0 ? Math.min(100, (bytesDownloaded / size) * 100) : null);
    return {
      id: asString(item?.id) || String(index),
      name: asString(item?.name) || `Job ${index + 1}`,
      directory: asString(item?.directory),
      status,
      bytesDownloaded,
      size,
      progress,
      error: asString(item?.error),
      createdAt: asString(item?.created_at),
      updatedAt: asString(item?.updated_at),
    };
  });
}

async function inspectModels({ silent = false } = {}) {
  if (state.scanning) return;
  state.scanning = true;
  state.errorMessage = "";
  if (!silent) state.statusMessage = "正在扫描当前工作流…";
  render();

  try {
    const scannedModels = scanCurrentGraphModels(app);
    state.selectedTokens.clear();
    state.licenseConfirmed = false;

    if (scannedModels.length === 0) {
      state.models = [];
      state.statusMessage = "当前工作流没有可下载的模型元数据。";
      return;
    }

    const response = await postJson(`${API_ROOT}/inspect`, {
      models: scannedModels.map(({ name, url, directory }) => ({ name, url, directory })),
    });
    state.models = normaliseInspectedModels(response, scannedModels);
    const missing = state.models.filter((model) => !model.installed).length;
    const eligible = state.models.filter(
      (model) => !model.installed && model.eligible && model.downloadToken,
    ).length;
    state.statusMessage = `发现 ${state.models.length} 个模型：${missing} 个缺失，${eligible} 个可由服务器下载。`;
  } catch (error) {
    state.models = [];
    state.errorMessage = `扫描失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    state.scanning = false;
    render();
  }
}

function selectedModels() {
  return state.models.filter(
    (model) => model.downloadToken && state.selectedTokens.has(model.downloadToken),
  );
}

async function createJobs() {
  if (state.creatingJobs) return;
  const models = selectedModels();
  if (models.length === 0) {
    setMessage("请先选择至少一个可下载模型。", { error: true });
    return;
  }
  if (!state.licenseConfirmed) {
    setMessage("请先确认你已阅读并接受所选模型的许可条款。", { error: true });
    return;
  }

  state.creatingJobs = true;
  state.errorMessage = "";
  state.statusMessage = "正在提交下载任务…";
  render();
  try {
    const response = await postJson(`${API_ROOT}/jobs`, {
      download_tokens: models.map((model) => model.downloadToken),
      license_confirmed: true,
    });
    const returnedJobs = normaliseJobs(response);
    if (returnedJobs.length) {
      state.jobs = returnedJobs;
      await refreshNodeDefinitionsForCompletedJobs();
    } else {
      await refreshJobs({ silent: true });
    }
    state.statusMessage = `已向 AWS 提交 ${models.length} 个下载任务。`;
    state.selectedTokens.clear();
    state.licenseConfirmed = false;
    schedulePolling();
  } catch (error) {
    state.errorMessage = `提交失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    state.creatingJobs = false;
    render();
  }
}

async function refreshJobs({ silent = false } = {}) {
  if (state.loadingJobs) return;
  state.loadingJobs = true;
  if (!silent) {
    state.errorMessage = "";
    state.statusMessage = "正在刷新下载任务…";
    render();
  }

  try {
    const response = await requestJson(`${API_ROOT}/jobs`);
    state.jobs = normaliseJobs(response);
    if (!silent) state.statusMessage = `已刷新 ${state.jobs.length} 个下载任务。`;
    await refreshNodeDefinitionsForCompletedJobs();
  } catch (error) {
    state.errorMessage = `任务刷新失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    state.loadingJobs = false;
    render();
  }
}

async function refreshNodeDefinitionsForCompletedJobs() {
  const newlyCompleted = state.jobs.filter(
    (job) => TERMINAL_SUCCESS_STATUSES.has(job.status) && !state.refreshedCompletedJobs.has(job.id),
  );
  if (!newlyCompleted.length || state.refreshingCombos) return;

  for (const job of newlyCompleted) state.refreshedCompletedJobs.add(job.id);
  state.refreshingCombos = true;
  try {
    await app.refreshComboInNodes();
    await inspectModels({ silent: true });
  } catch (error) {
    state.errorMessage = `模型已下载，但刷新节点列表失败：${
      error instanceof Error ? error.message : String(error)
    }`;
  } finally {
    state.refreshingCombos = false;
  }
}

async function cancelJob(jobId) {
  if (!jobId) return;
  try {
    await postJson(`${API_ROOT}/jobs/${encodeURIComponent(jobId)}/cancel`, {});
    state.statusMessage = "已发送取消请求。";
    state.errorMessage = "";
    await refreshJobs({ silent: true });
    schedulePolling();
  } catch (error) {
    setMessage(`取消失败：${error instanceof Error ? error.message : String(error)}`, {
      error: true,
    });
  }
}

function hasActiveJobs() {
  return state.jobs.some((job) => ACTIVE_JOB_STATUSES.has(job.status));
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
    createElement("h2", "smd-title", "下载到 AWS"),
    createElement("p", "smd-subtitle", "模型由服务器直接下载，不经过当前浏览器。"),
  );

  const actions = createElement("div", "smd-header-actions");
  const scanButton = createElement("button", "smd-button smd-button-primary", "扫描工作流");
  scanButton.type = "button";
  scanButton.disabled = state.scanning;
  scanButton.addEventListener("click", () => void inspectModels());
  const refreshButton = createElement("button", "smd-button", "刷新任务");
  refreshButton.type = "button";
  refreshButton.disabled = state.loadingJobs;
  refreshButton.addEventListener("click", () => void refreshJobs());
  actions.append(scanButton, refreshButton);
  header.append(titleWrap, actions);
  return header;
}

function createSessionNote() {
  const note = createElement("section", "smd-security-note");
  const title = createElement("strong", "", "安全限制");
  const directories = state.session?.allowedDirectories?.join("、") || "由服务器决定";
  const extensions = state.session?.safeExtensions?.join("、") || "由服务器决定";
  const body = createElement(
    "span",
    "",
    ` 服务器仅接受签名下载令牌；目录：${directories}；格式：${extensions}。`,
  );
  note.append(title, body);
  return note;
}

function createMessageArea() {
  const wrapper = createElement("div", "smd-messages");
  wrapper.setAttribute("aria-live", "polite");
  if (state.errorMessage) wrapper.append(createElement("div", "smd-alert smd-alert-error", state.errorMessage));
  if (state.statusMessage)
    wrapper.append(createElement("div", "smd-alert smd-alert-info", state.statusMessage));
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
  check.setAttribute("aria-label", `选择 ${model.name}`);
  check.addEventListener("change", () => {
    if (check.checked) state.selectedTokens.add(model.downloadToken);
    else state.selectedTokens.delete(model.downloadToken);
    state.licenseConfirmed = false;
    render();
  });

  const nameWrap = createElement("div", "smd-model-name-wrap");
  nameWrap.append(
    createElement("div", "smd-model-name", model.name),
    createElement("div", "smd-model-filename", model.filename),
  );
  const badge = model.installed
    ? createBadge("已安装", "success")
    : model.eligible && model.downloadToken
      ? createBadge("可下载", "ready")
      : createBadge("不可下载", "muted");
  heading.append(check, nameWrap, badge);
  card.append(heading);

  const metadata = createElement("div", "smd-metadata");
  metadata.append(
    createMetadataRow("目录", createElement("code", "smd-code", model.directory || "—")),
    createMetadataRow("大小", createElement("span", "", formatBytes(model.size))),
    createMetadataRow(
      "来源",
      createElement("span", "", model.source || "未验证（链接已隐藏）"),
    ),
    createMetadataRow("许可", createLink(model.licenseUrl, model.license)),
  );
  if (model.revision)
    metadata.append(createMetadataRow("版本", createElement("code", "smd-code", model.revision)));
  if (model.sha256)
    metadata.append(
      createMetadataRow("SHA-256", createElement("code", "smd-code smd-hash", model.sha256)),
    );
  if (model.graphSources.length) {
    const sources = createElement("span", "smd-source-path", model.graphSources.join("；"));
    sources.title = model.graphSources.join("\n");
    metadata.append(createMetadataRow("节点", sources));
  }
  card.append(metadata);

  if (model.reason) card.append(createElement("p", "smd-reason", model.reason));
  return card;
}

function createModelsSection() {
  const section = createElement("section", "smd-section");
  const heading = createElement("div", "smd-section-heading");
  heading.append(createElement("h3", "smd-section-title", `工作流模型 (${state.models.length})`));

  const eligible = state.models.filter(
    (model) => !model.installed && model.eligible && model.downloadToken,
  );
  const selectAll = createElement("button", "smd-button smd-button-small", "全选可下载");
  selectAll.type = "button";
  selectAll.disabled = eligible.length === 0;
  selectAll.addEventListener("click", () => {
    const allSelected = eligible.every((model) => state.selectedTokens.has(model.downloadToken));
    state.selectedTokens.clear();
    if (!allSelected) {
      for (const model of eligible) state.selectedTokens.add(model.downloadToken);
    }
    state.licenseConfirmed = false;
    render();
  });
  heading.append(selectAll);
  section.append(heading);

  const list = createElement("div", "smd-card-list");
  if (!state.models.length) {
    list.append(
      createElement(
        "div",
        "smd-empty",
        state.scanning ? "正在扫描…" : "点击“扫描工作流”检查当前工作流。",
      ),
    );
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
  consent.append(
    consentCheck,
    createElement(
      "span",
      "",
      "我已逐项阅读并接受所选模型的许可条款，并确认有权在此 AWS 服务器上使用。",
    ),
  );
  section.append(consent);

  const downloadButton = createElement(
    "button",
    "smd-button smd-button-primary smd-download-button",
    state.creatingJobs ? "正在提交…" : `下载所选模型到 AWS (${selected.length})`,
  );
  downloadButton.type = "button";
  downloadButton.disabled =
    selected.length === 0 || !state.licenseConfirmed || state.creatingJobs || state.scanning;
  downloadButton.addEventListener("click", () => void createJobs());
  section.append(downloadButton);
  return section;
}

function createJobCard(job) {
  const card = createElement("article", "smd-job");
  const heading = createElement("div", "smd-job-heading");
  const nameWrap = createElement("div", "smd-model-name-wrap");
  nameWrap.append(
    createElement("div", "smd-model-name", job.name),
    createElement("div", "smd-model-filename", job.directory || "—"),
  );
  const badgeKind = TERMINAL_SUCCESS_STATUSES.has(job.status)
    ? "success"
    : job.status === "failed" || job.status === "error"
      ? "error"
      : ACTIVE_JOB_STATUSES.has(job.status)
        ? "ready"
        : "muted";
  heading.append(nameWrap, createBadge(statusLabel(job.status), badgeKind));
  card.append(heading);

  const progress = job.progress ?? 0;
  const progressTrack = createElement("div", "smd-progress");
  const progressBar = createElement("div", "smd-progress-bar");
  progressBar.style.width = `${progress}%`;
  progressTrack.append(progressBar);
  card.append(progressTrack);

  const details = createElement("div", "smd-job-details");
  const byteText = job.size === null
    ? formatBytes(job.bytesDownloaded)
    : `${formatBytes(job.bytesDownloaded)} / ${formatBytes(job.size)}`;
  details.append(
    createElement("span", "", byteText),
    createElement("span", "", job.progress === null ? "" : `${job.progress.toFixed(1)}%`),
  );
  card.append(details);
  if (job.error) card.append(createElement("p", "smd-reason smd-job-error", job.error));

  if (CANCELLABLE_JOB_STATUSES.has(job.status)) {
    const cancel = createElement("button", "smd-button smd-button-danger smd-button-small", "取消");
    cancel.type = "button";
    cancel.addEventListener("click", () => void cancelJob(job.id));
    card.append(cancel);
  }
  return card;
}

function createJobsSection() {
  const section = createElement("section", "smd-section");
  section.append(createElement("h3", "smd-section-title", `下载任务 (${state.jobs.length})`));
  const list = createElement("div", "smd-job-list");
  if (!state.jobs.length) {
    list.append(createElement("div", "smd-empty", "目前没有下载任务。"));
  } else {
    for (const job of state.jobs) list.append(createJobCard(job));
  }
  section.append(list);
  return section;
}

function render() {
  const container = state.container;
  if (!container || state.destroyed) return;

  const root = createElement("div", "smd-root");
  root.append(
    createHeader(),
    createSessionNote(),
    createMessageArea(),
    createModelsSection(),
    createJobsSection(),
  );
  container.replaceChildren(root);
}

async function initialise() {
  try {
    await loadSession();
    await Promise.all([inspectModels({ silent: true }), refreshJobs({ silent: true })]);
    schedulePolling();
  } catch (error) {
    state.errorMessage = `初始化失败：${error instanceof Error ? error.message : String(error)}`;
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
}

ensureStylesheet();

app.registerExtension({
  name: EXTENSION_NAME,
  commands: [
    {
      id: OPEN_COMMAND_ID,
      label: "下载到 AWS",
      icon: "mdi mdi-cloud-download",
      function: () => {
        app.extensionManager.command.execute(`Workspace.ToggleSidebarTab.${SIDEBAR_ID}`);
      },
    },
  ],
  menuCommands: [
    {
      path: ["Extensions", "AWS 模型下载"],
      commands: [OPEN_COMMAND_ID],
    },
  ],
  async setup() {
    app.extensionManager.registerSidebarTab({
      id: SIDEBAR_ID,
      icon: "mdi mdi-cloud-download",
      title: "下载到 AWS",
      tooltip: "把工作流缺失模型直接下载到 AWS 服务器",
      type: "custom",
      render: renderSidebar,
      destroy: destroySidebar,
    });
  },
});
