export const ACTIVE_JOB_STATUSES = new Set(["queued", "downloading"]);
export const TERMINAL_SUCCESS_STATUSES = new Set(["completed"]);
export const CANCELLABLE_JOB_STATUSES = new Set(["queued", "downloading"]);
export const PARTIAL_DISCARD_STATUSES = new Set(["failed", "cancelled"]);

const PHASE_LABELS = Object.freeze({
  queued: "Queued",
  connecting: "Connecting",
  downloading: "Downloading",
  retrying: "Waiting to retry",
  hashing: "Verifying SHA-256",
  validating: "Validating Safetensors",
  publishing: "Publishing",
  cancelling: "Cancelling",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
});

export function asString(value) {
  return typeof value === "string" ? value.trim() : "";
}

export function asBoolean(value) {
  return value === true || value === "true" || value === 1;
}

export function asFiniteNumber(value) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) && number >= 0 ? number : null;
}

export function normaliseStatus(value) {
  return asString(value).toLowerCase().replaceAll("_", "-") || "unknown";
}

export function clampProgress(value) {
  const number = asFiniteNumber(value);
  if (number === null) return null;
  const percent = number <= 1 ? number * 100 : number;
  return Math.max(0, Math.min(100, percent));
}

export function formatBytes(value) {
  const bytes = asFiniteNumber(value);
  if (bytes === null) return "Unknown";
  if (bytes === 0) return "0 B";

  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const amount = bytes / 1024 ** exponent;
  return `${amount.toFixed(amount >= 10 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

export function phaseLabel(phase, status = "unknown") {
  const normalisedPhase = normaliseStatus(phase);
  const normalisedStatus = normaliseStatus(status);
  return PHASE_LABELS[normalisedPhase] ?? PHASE_LABELS[normalisedStatus] ?? normalisedPhase;
}

export function safeExternalUrl(value) {
  const raw = asString(value);
  if (!raw) return null;
  try {
    const parsed = new URL(raw);
    return parsed.protocol === "https:" ? parsed.href : null;
  } catch {
    return null;
  }
}

export function normaliseJobs(payload) {
  const items = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.jobs)
      ? payload.jobs
      : payload?.job && typeof payload.job === "object"
        ? [payload.job]
      : [];

  const jobs = items.map((item, index) => {
    const status = normaliseStatus(item?.status);
    const phase = normaliseStatus(item?.phase || status);
    const bytesDownloaded = asFiniteNumber(item?.bytes_downloaded) ?? 0;
    const size = asFiniteNumber(item?.size);
    const explicitProgress = clampProgress(item?.progress);
    const progress =
      explicitProgress ??
      (size && size > 0 ? Math.min(100, (bytesDownloaded / size) * 100) : null);
    const attempt = asFiniteNumber(item?.attempt) ?? 0;
    const maxAttempts = asFiniteNumber(item?.max_attempts) ?? 5;

    return {
      id: asString(item?.id) || String(index),
      name: asString(item?.name) || `Job ${index + 1}`,
      directory: asString(item?.directory),
      status,
      phase,
      bytesDownloaded,
      size,
      progress,
      error: asString(item?.error),
      errorCode: asString(item?.error_code),
      attempt,
      maxAttempts,
      cancelRequested: asBoolean(item?.cancel_requested),
      partialAvailable: asBoolean(item?.partial_available) || bytesDownloaded > 0,
      createdAt: asFiniteNumber(item?.created_at) ?? asString(item?.created_at),
      updatedAt: asFiniteNumber(item?.updated_at) ?? asString(item?.updated_at),
    };
  });

  return {
    jobs,
    nextCursor: asString(payload?.next_cursor) || null,
  };
}

export function mergeJobs(current, incoming) {
  const merged = new Map();
  for (const job of [...current, ...incoming]) merged.set(job.id, job);
  return [...merged.values()].sort((left, right) => {
    const leftRaw = left.createdAt || left.updatedAt || "";
    const rightRaw = right.createdAt || right.updatedAt || "";
    const leftTime =
      typeof leftRaw === "number" ? leftRaw : Date.parse(leftRaw) || 0;
    const rightTime =
      typeof rightRaw === "number" ? rightRaw : Date.parse(rightRaw) || 0;
    return rightTime - leftTime;
  });
}

export function historyPath(apiRoot, cursor = null, limit = 50) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cursor) params.set("cursor", cursor);
  return `${apiRoot}/jobs?${params}`;
}

/**
 * Cancels an older request and gives every new request an identity. Callers must
 * check `isCurrent()` before applying a result, including after awaited cleanup.
 */
export class RequestCoordinator {
  #generation = 0;
  #controller = null;

  begin() {
    this.#controller?.abort();
    this.#controller = new AbortController();
    const generation = ++this.#generation;
    const controller = this.#controller;
    return {
      signal: controller.signal,
      isCurrent: () =>
        generation === this.#generation &&
        controller === this.#controller &&
        !controller.signal.aborted,
    };
  }

  cancel() {
    this.#controller?.abort();
    this.#controller = null;
    this.#generation += 1;
  }
}

export function isAbortError(error) {
  return error instanceof Error && error.name === "AbortError";
}
