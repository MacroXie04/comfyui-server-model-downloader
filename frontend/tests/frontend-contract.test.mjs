import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../server-model-downloader.js", import.meta.url), "utf8");

test("public UI uses generic English branding", () => {
  assert.match(source, /Server Model Downloader/);
  assert.doesNotMatch(source, /[\u3400-\u9fff]/u);
});

test("opening the sidebar does not inspect providers", () => {
  const initialise = source.match(/async function initialise\(\) \{([\s\S]*?)\n\}/)?.[1] ?? "";
  assert.match(initialise, /loadSession/);
  assert.match(initialise, /refreshJobs/);
  assert.doesNotMatch(initialise, /inspectModels/);
});

test("v1 job controls and stale-request protection are wired", () => {
  assert.match(source, /Discard partial/);
  assert.match(source, /cancel_requested|cancelRequested/);
  assert.match(source, /RequestCoordinator/);
  assert.match(source, /aria-valuenow/);
});

test("workflow changes invalidate inspected download tokens", () => {
  const inspect = source.match(/async function inspectModels\(\) \{([\s\S]*?)\n\}/)?.[1] ?? "";
  const create = source.match(/async function createJobs\(\) \{([\s\S]*?)\n\}/)?.[1] ?? "";
  const destroy = source.match(/function destroySidebar\(\) \{([\s\S]*?)\n\}/)?.[1] ?? "";

  assert.ok((inspect.match(/scanCurrentGraphModels\(app\)/g) ?? []).length >= 2);
  assert.match(inspect, /active workflow changed during the scan/i);
  assert.match(create, /scanCurrentGraphModels\(app\)/);
  assert.match(create, /active workflow changed/i);
  assert.match(destroy, /clearScannedModels/);
});
