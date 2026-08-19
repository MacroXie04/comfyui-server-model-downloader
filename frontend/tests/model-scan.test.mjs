import assert from "node:assert/strict";

import { scanCurrentGraphModels } from "../model-scan.mjs";

const rootModel = {
  name: "model.safetensors",
  directory: "diffusion_models",
  url: "https://huggingface.co/org/repo/resolve/main/model.safetensors?download=true",
};

function appWith(nodes, models = [rootModel]) {
  return {
    rootGraph: { nodes },
    extensionManager: {
      workflow: { activeWorkflow: { activeState: { models } } },
    },
  };
}

{
  const [found] = scanCurrentGraphModels(
    appWith([{ id: 1, type: "UNETLoader", widgets_values: ["model.safetensors"] }]),
  );
  assert.equal(found.name, "model.safetensors");
  assert.equal(found.directory, "diffusion_models");
  assert.deepEqual(found.sources, ["UNETLoader (#1)"]);
}

assert.deepEqual(
  scanCurrentGraphModels(
    appWith([{ id: 2, type: "UNETLoader", widgets_values: ["other.safetensors"] }]),
  ),
  [],
);

for (const mode of [2, 4]) {
  assert.deepEqual(
    scanCurrentGraphModels(
      appWith([
        { id: 3, type: "Inactive", mode, widgets_values: ["model.safetensors"] },
      ]),
    ),
    [],
  );
}

assert.deepEqual(
  scanCurrentGraphModels(
    appWith([
      {
        id: 4,
        type: "InactiveSubgraph",
        mode: 4,
        subgraph: {
          nodes: [
            { id: 5, type: "Interior", widgets_values: ["model.safetensors"] },
          ],
        },
      },
    ]),
  ),
  [],
);

{
  const [found] = scanCurrentGraphModels(
    appWith([
      {
        id: 6,
        type: "ActiveSubgraph",
        subgraph: {
          nodes: [
            {
              id: 7,
              type: "Interior",
              widgets: [{ value: "folder/model.safetensors" }],
            },
          ],
        },
      },
    ]),
  );
  assert.equal(found.name, "model.safetensors");
  assert.deepEqual(found.sources, ["ActiveSubgraph (#6) → Interior (#7)"]);
}

{
  const embeddedOnly = {
    name: "vae.safetensors",
    directory: "vae",
    url: "https://huggingface.co/org/repo/resolve/main/vae.safetensors",
  };
  const [found] = scanCurrentGraphModels(
    appWith(
      [
        {
          id: 8,
          type: "VAELoader",
          widgets_values: ["vae.safetensors"],
          properties: { models: [embeddedOnly] },
        },
      ],
      [],
    ),
  );
  assert.equal(found.directory, "vae");
}

console.log("model scan contract: ok");
