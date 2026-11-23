// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseArgs } from "node:util";
import { rollup } from "@rollup/wasm-node";
import zopfli from "@gfx/zopfli";
import CleanCSS from "clean-css";
import { minify as minifyHTML } from "html-minifier-terser";
import { minify as minifyJS } from "terser";

const list = { type: "string", multiple: true, default: [] };
const { values } = parseArgs({
  options: { entry: list, page: list, style: list, output: list, etags: { type: "string" } },
});
const assets = new Map();
const bundle = await rollup({
  input: values.entry.map(file => path.resolve(file)),
  preserveEntrySignatures: "allow-extension",
});
try {
  const { output } = await bundle.generate({
    format: "es",
    entryFileNames: "[name].js",
    chunkFileNames: "[name].js",
    minifyInternalExports: true,
    hoistTransitiveImports: false,
  });
  for (const { fileName, code } of output) {
    const result = await minifyJS(code, {
      ecma: 2022, module: true, compress: { passes: 3 },
      format: { comments: /@license|@preserve|^!/ },
    });
    assets.set(fileName, result.code);
  }
} finally {
  await bundle.close();
}

for (const file of values.page) {
  const html = await minifyHTML(await readFile(file, "utf8"), {
    collapseWhitespace: true, collapseBooleanAttributes: true, sortAttributes: true,
    removeAttributeQuotes: true, removeOptionalTags: true, removeComments: true,
    ignoreCustomComments: [/copyright|SPDX|@license/i],
  });
  assets.set(path.basename(file), html);
}
const styles = await Promise.all(values.style.map(file => readFile(file, "utf8")));
const css = new CleanCSS({ level: 2, rebase: false }).minify(styles.join("\n"));
if (css.errors.length || css.warnings.length) {
  throw new Error([...css.errors, ...css.warnings].join("\n"));
}
assets.set("style.css", css.styles);

// The output list is also the native server's resource table. A new or
// missing module must change the declared inputs to the embed action.
const etags = [];
let total = 0;
for (const file of values.output) {
  const name = path.basename(file);
  if (!assets.has(name)) throw new Error("Missing dashboard resource: " + name);
  const encoded = await zopfli.gzipAsync(Buffer.from(assets.get(name)), { numiterations: 15 });
  encoded[9] = 255; // OS-independent gzip header.
  total += encoded.length;
  const digest = createHash("sha256").update(encoded).digest().subarray(0, 16);
  etags.push(`  "${digest.toString("base64url")}",`);
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, encoded);
  assets.delete(name);
}
if (assets.size) throw new Error("Undeclared dashboard resources: " + [...assets.keys()].join(", "));
if (total >= 50000) throw new Error(`Dashboard is ${total} gzip bytes. The budget is below 50,000.`);
// The embed-data generator consumes the same ordered list of files.
await writeFile(values.etags, [
  "static const char loom_serve_dashboard_etags[][23] = {", ...etags, "};", "",
].join("\n"));
