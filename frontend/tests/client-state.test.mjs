import assert from "node:assert/strict";
import test from "node:test";

import {
  RequestCoordinator,
  clampProgress,
  formatBytes,
  historyPath,
  mergeJobs,
  normaliseJobs,
  phaseLabel,
  safeExternalUrl,
} from "../client-state.mjs";

test("a newer request aborts and invalidates the older generation", () => {
  const coordinator = new RequestCoordinator();
  const first = coordinator.begin();
  const second = coordinator.begin();

  assert.equal(first.signal.aborted, true);
  assert.equal(first.isCurrent(), false);
  assert.equal(second.signal.aborted, false);
  assert.equal(second.isCurrent(), true);

  coordinator.cancel();
  assert.equal(second.signal.aborted, true);
  assert.equal(second.isCurrent(), false);
});

test("job payloads expose additive v1 state without losing progress compatibility", () => {
  const { jobs, nextCursor } = normaliseJobs({
    jobs: [
      {
        id: "job-1",
        name: "model.safetensors",
        directory: "diffusion_models",
        status: "DOWNLOADING",
        phase: "retrying",
        bytes_downloaded: 25,
        size: 100,
        progress: 0.25,
        attempt: 2,
        max_attempts: 5,
        cancel_requested: true,
        partial_available: true,
        error_code: "upstream_rate_limited",
      },
    ],
    next_cursor: "opaque-token",
  });

  assert.equal(nextCursor, "opaque-token");
  assert.deepEqual(jobs[0], {
    id: "job-1",
    name: "model.safetensors",
    directory: "diffusion_models",
    status: "downloading",
    phase: "retrying",
    bytesDownloaded: 25,
    size: 100,
    progress: 25,
    error: "",
    errorCode: "upstream_rate_limited",
    attempt: 2,
    maxAttempts: 5,
    cancelRequested: true,
    partialAvailable: true,
    createdAt: "",
    updatedAt: "",
  });
});

test("legacy array job responses remain supported", () => {
  const { jobs, nextCursor } = normaliseJobs([
    { id: "legacy", status: "queued", bytes_downloaded: 50, size: 200 },
  ]);
  assert.equal(jobs[0].progress, 25);
  assert.equal(jobs[0].maxAttempts, 5);
  assert.equal(nextCursor, null);
});

test("single-job envelopes and numeric timestamps remain compatible", () => {
  const { jobs } = normaliseJobs({
    job: {
      id: "single",
      status: "cancelled",
      created_at: 1_723_000_000,
      updated_at: 1_723_000_001,
    },
  });
  assert.equal(jobs.length, 1);
  assert.equal(jobs[0].createdAt, 1_723_000_000);
  assert.equal(jobs[0].updatedAt, 1_723_000_001);
});

test("history cursors are encoded and limits are explicit", () => {
  assert.equal(
    historyPath("/server-model-downloader", "cursor/with spaces", 25),
    "/server-model-downloader/jobs?limit=25&cursor=cursor%2Fwith+spaces",
  );
});

test("job pages merge by id and preserve newest order", () => {
  const current = [
    { id: "a", createdAt: "2026-01-01T00:00:00Z", status: "queued" },
    { id: "b", createdAt: "2026-01-02T00:00:00Z", status: "failed" },
  ];
  const incoming = [
    { id: "a", createdAt: "2026-01-01T00:00:00Z", status: "completed" },
  ];
  assert.deepEqual(
    mergeJobs(current, incoming).map(({ id, status }) => ({ id, status })),
    [
      { id: "b", status: "failed" },
      { id: "a", status: "completed" },
    ],
  );
  assert.deepEqual(
    mergeJobs(
      [{ id: "old", createdAt: 1, status: "queued" }],
      [{ id: "new", createdAt: 2, status: "queued" }],
    ).map(({ id }) => id),
    ["new", "old"],
  );
});

test("only HTTPS links are presented as external links", () => {
  assert.equal(safeExternalUrl("https://huggingface.co/org/repo"), "https://huggingface.co/org/repo");
  assert.equal(safeExternalUrl("http://example.com/model"), null);
  assert.equal(safeExternalUrl("javascript:alert(1)"), null);
  assert.equal(safeExternalUrl("not a URL"), null);
});

test("progress, sizes, and public phases are rendered consistently", () => {
  assert.equal(clampProgress(0.5), 50);
  assert.equal(clampProgress(101), 100);
  assert.equal(formatBytes(1024 ** 3), "1.0 GiB");
  assert.equal(phaseLabel("validating"), "Validating Safetensors");
  assert.equal(phaseLabel("", "completed"), "Completed");
});
