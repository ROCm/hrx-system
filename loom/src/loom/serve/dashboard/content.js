// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

export const imageTypes = ["image/png", "image/jpeg", "image/webp", "image/gif"];

export async function attachment(file, images) {
  if (imageTypes.includes(file.type)) {
    if (!images) throw new Error("Select a model with image input to attach " + file.name);
    return { type: "image", file };
  }
  if (/^(image|audio|video)\//.test(file.type) || file.type === "application/pdf")
    throw new Error(file.name + " · choose UTF-8 text, source code, or a supported image");
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(await file.arrayBuffer());
    if (/[\x00-\x08\x0e-\x1f]/.test(text)) throw new Error("Binary file");
    return { type: "document", name: file.name, text };
  } catch (error) {
    throw new Error(file.name + " · could not read as UTF-8 text (" + error.message + ")");
  }
}
export const partName = (part) => part.name ?? part.file?.name ?? "Image";
export const documentText = (part) => "[File: " + part.name + "]\n" + part.text;
export const entryContent = (entry) => entry.message?.content ?? entry.text;
export function entryText(entry) {
  const content = entryContent(entry);
  if (Array.isArray(content))
    return content
      .map((part) =>
        part.type === "document" ? documentText(part) : (part.text ?? partName(part)),
      )
      .join("\n\n");
  return content || entry.imagePrompt || "";
}

// Compaction keeps its own prompt immediately before the summary. A tool
// availability notice can also occupy a user message with no actual input.
export function isUserTurn(entry, index, entries) {
  return (
    (entry.message?.role ?? entry.role) === "user" &&
    !entry.automatic &&
    !entries[index + 1]?.compaction &&
    entryContent(entry)?.length > 0
  );
}

// Older exports stored only the expanded user text. Remove the exact template
// framing once when reopening one of those turns in the composer.
export function editable(entry, values) {
  const content = entryContent(entry);
  const files = [];
  let text = content;
  if (Array.isArray(content)) {
    const lines = [];
    for (const part of content) {
      if (part.type === "text") lines.push(part.text);
      else files.push(part);
    }
    text = lines.join("\n\n");
  }
  if (entry.draft !== undefined) text = entry.draft;
  else if (entry.message && values.template) {
    const [head, tail] = values.template.split("{{input}}");
    if (
      tail !== undefined &&
      text.startsWith(head) &&
      text.endsWith(tail) &&
      text.length >= head.length + tail.length
    )
      text = text.slice(head.length, text.length - tail.length);
  }
  return { text, files };
}
