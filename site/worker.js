// Runs the picklabel Python package in Pyodide, off the main thread so the page stays responsive.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js");

const progress = (text) => self.postMessage({ type: "progress", text });

const ready = (async () => {
  progress("Starting Python (WebAssembly)…");
  const py = await loadPyodide();
  progress("Loading NumPy and OpenCV…");
  await py.loadPackage(["numpy", "opencv-python", "pyyaml"]);
  progress("Loading PickLabel…");
  const files = await (await fetch("py/manifest.json")).json();
  await Promise.all(files.map(async (f) => {
    const src = await (await fetch("py/" + f)).text();
    const path = "/home/pyodide/" + f;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    py.FS.writeFile(path, src);
  }));
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  return py.pyimport("demo_bridge");
})();

ready.then(() => self.postMessage({ type: "ready" }),
           (e) => self.postMessage({ type: "fatal", text: String(e) }));

self.onmessage = async ({ data }) => {
  const bridge = await ready;
  try {
    if (data.type === "load") {
      const info = bridge.load(data.key, data.before, data.after, data.gt ?? null);
      self.postMessage({ type: "loaded", key: data.key, info: JSON.parse(info) });
    } else if (data.type === "run") {
      const out = bridge.run(data.key, data.gx, data.gy, data.graspSuccess ?? null);
      self.postMessage({ type: "result", id: data.id, key: data.key, result: JSON.parse(out) });
    }
  } catch (e) {
    self.postMessage({ type: "error", id: data.id, text: String(e) });
  }
};
