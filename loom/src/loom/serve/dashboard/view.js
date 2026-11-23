// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { imageTypes } from "./content.js";

export const byId = (id) => document.getElementById(id);

export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function disclosure(label, content) {
  const box = element("details");
  box.append(element("summary", label), content);
  return box;
}

export function button(label, action) {
  const node = element("button", label);
  node.type = "button";
  node.onclick = action;
  return node;
}

export function copyButton(label, text) {
  const copy = button(label, async () => {
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = "Copied";
    } catch (error) {
      copy.textContent = "Copy failed";
      copy.title = error.message;
    }
  });
  return copy;
}

// Keep the complete tool message in the transcript. Only its presentation folds.
export function toolOutput(text) {
  const full = element("pre", text);
  if (text.length <= 420 && text.split(/\r\n?|\n/, 6).length <= 5) return full;
  const excerpt = text
    .trimStart()
    .slice(0, 420)
    .split(/\r\n?|\n/, 5)
    .join("\n")
    .trimEnd()
    .replace(/[\uD800-\uDBFF]$/, "");
  const more = disclosure("Full result", full);
  more.className = "tool-output";
  more.firstChild.append(element("pre", excerpt + "…"));
  return more;
}

// Disclosure state belongs to this view. Opening a fenced document starts its scripts.
const openFences = new WeakMap();

function resizeGrip(box, layout) {
  box.tabIndex = 0;
  box.setAttribute("role", "group");
  box.setAttribute("aria-label", "Resize preview with arrow keys. Home resets the size.");
  let previous = layout.size || {};
  const axes = ["width", "height"];
  for (const axis of axes) if (previous[axis]) box.style[axis] = previous[axis] + "px";
  function commitSize() {
    const bounds = box.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const size = {};
    for (const axis of axes)
      if (box.style[axis]) {
        // An untouched axis keeps its preference even on a narrower display.
        size[axis] =
          parseFloat(box.style[axis]) === previous[axis]
            ? previous[axis]
            : Math.round(bounds[axis]);
        box.style[axis] = size[axis] + "px";
      }
    if (size.width === previous.width && size.height === previous.height) return;
    previous = size;
    layout.save(size.width || size.height ? size : undefined);
  }
  box.onkeydown = (event) => {
    if (event.target !== box || event.ctrlKey || event.altKey || event.metaKey) return;
    const { key } = event;
    if (key === "Home") box.style.cssText = "";
    else if (key === "ArrowLeft" || key === "ArrowRight")
      box.style.width = Math.max(0, box.offsetWidth + (key === "ArrowLeft" ? -32 : 32)) + "px";
    else if (key === "ArrowUp" || key === "ArrowDown")
      box.style.height = Math.max(0, box.offsetHeight + (key === "ArrowUp" ? -32 : 32)) + "px";
    else return;
    commitSize();
    event.preventDefault();
  };
  const grip = element("span", undefined, "preview-grip");
  grip.setAttribute("aria-hidden", "true");
  grip.onpointerdown = (down) => {
    if (down.button !== 0 || !down.isPrimary || grip.hasAttribute("data-dragging")) return;
    down.preventDefault();
    box.focus({ preventScroll: true });
    const { pointerId, clientX, clientY } = down;
    const width = box.offsetWidth,
      height = box.offsetHeight;
    const original = { width: box.style.width, height: box.style.height };
    const drag = new AbortController(),
      options = { signal: drag.signal, capture: true };
    const end = () => {
      drag.abort();
      grip.removeAttribute("data-dragging");
      if (grip.hasPointerCapture(pointerId)) grip.releasePointerCapture(pointerId);
      commitSize();
    };
    const track = (event) => {
      if (event.pointerId !== pointerId) return;
      const released = event.type === "pointerup";
      if (grip.isConnected && (released || (event.type === "pointermove" && event.buttons & 1))) {
        const dx = event.clientX - clientX,
          dy = event.clientY - clientY;
        box.style.width = dx ? Math.max(0, width + dx) + "px" : original.width;
        box.style.height = dy ? Math.max(0, height + dy) + "px" : original.height;
        if (!released) return;
      }
      end();
    };
    grip.setPointerCapture(pointerId);
    grip.setAttribute("data-dragging", "");
    // A process-isolated frame can interrupt pointer capture while the
    // button is still down. The host shield and document own the gesture.
    for (const type of ["pointermove", "pointerup", "pointercancel"])
      document.addEventListener(type, track, options);
    window.addEventListener("blur", end, { signal: drag.signal });
  };
  return grip;
}

export function preview(html, layout) {
  const box = element("div", undefined, "preview-box");
  const frame = element("iframe", undefined, "preview");
  frame.setAttribute("aria-label", "HTML preview");
  frame.referrerPolicy = "no-referrer";
  frame.setAttribute("aria-busy", "true");
  frame.onload = () => {
    const doc = frame.contentDocument;
    if (doc?.URL !== frame.src) return;
    // The static shell supplies the document policy. Only its opaque child
    // receives authored content, parsed by the browser as a complete document.
    frame.onload = null;
    const app = doc.querySelector("iframe");
    app.onload = () => frame.setAttribute("aria-busy", "false");
    app.srcdoc = html;
  };
  frame.src = "preview.html";
  // A download-only MIME type keeps the generated HTML outside the app origin.
  // Materialize its bytes once, on the first save, and release them with the row.
  let file;
  const save = button("Download HTML", () => {
    if (!file) {
      file = element("a");
      file.download = "preview.html";
      file.href = URL.createObjectURL(
        new Blob(["\uFEFF", html], { type: "application/octet-stream" }),
      );
      file.hidden = true;
      box.append(file);
    }
    file.click();
  });
  save.className = "save-html";
  box.append(frame, save, resizeGrip(box, layout));
  return box;
}

function inline(node, line) {
  const tokens = /\x60([^\x60]+)\x60|\*\*([^*]+)\*\*|\[([^\]]+)\]\((https?:\/\/[^\s()]+)\)/gi;
  let start = 0;
  for (const match of line.matchAll(tokens)) {
    node.append(line.slice(start, match.index));
    if (match[1]) node.append(element("code", match[1]));
    else if (match[2]) node.append(element("strong", match[2]));
    else {
      const link = element("a", match[3]);
      link.href = match[4];
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      node.append(link);
    }
    start = match.index + match[0].length;
  }
  node.append(line.slice(start));
  return node;
}
const cells = (line) =>
  line
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((value) => value.trim());
function paragraphs(parent, text) {
  const lines = text.split(/\r?\n/);
  let list;
  for (let i = 0; i < lines.length; ++i) {
    const line = lines[i],
      item = /^\s*(?:([-*+])|(\d+)\.)\s+(.+)/.exec(line);
    if (item) {
      const tag = item[1] ? "ul" : "ol";
      if (list?.localName !== tag) {
        list = element(tag);
        if (item[2]) list.start = Number(item[2]);
        parent.append(list);
      }
      list.append(inline(element("li"), item[3]));
      continue;
    }
    list = undefined;
    if (line.includes("|") && lines[i + 1]?.includes("|")) {
      const header = cells(line),
        rule = cells(lines[i + 1]);
      if (rule.length === header.length && rule.every((cell) => /^:?-{3,}:?$/.test(cell))) {
        const table = element("table");
        const row = (section, parts, tag) => {
          const tr = section.insertRow();
          for (const value of parts) tr.append(inline(element(tag), value));
        };
        row(table.createTHead(), header, "th");
        const body = table.createTBody();
        ++i;
        while (lines[i + 1]?.includes("|")) row(body, cells(lines[++i]), "td");
        parent.append(table);
        continue;
      }
    }
    const heading = /^(#{1,6}) (.+)/.exec(line);
    const node = heading ? element("h" + heading[1].length) : element("div", undefined, "prose");
    if (line) inline(node, heading?.[2] || line);
    else node.append(element("br"));
    parent.append(node);
  }
}

export function formatted(text, owner, nextPreview) {
  const result = document.createDocumentFragment();
  const fences = /^(\x60{3,})([\w-]*)[^\S\n]*\n([\s\S]*?)^\1[^\S\n]*(?:\n|$)/gm;
  let start = 0;
  for (const match of text.matchAll(fences)) {
    if (match.index > start) paragraphs(result, text.slice(start, match.index).trimEnd());
    const code = element("pre");
    code.append(copyButton("Copy code", match[3]), element("code", match[3]));
    result.append(code);
    if (match[2].toLowerCase() === "html") {
      const box = element("details");
      const layout = nextPreview();
      let open = openFences.get(owner);
      if (!open) openFences.set(owner, (open = new Set()));
      box.append(element("summary", "Interactive preview"));
      box.open = open.has(match.index);
      box.ontoggle = () => {
        if (box.open) {
          open.add(match.index);
          if (box.childElementCount === 1) box.append(preview(match[3], layout));
        } else open.delete(match.index);
      };
      result.append(box);
    }
    start = match.index + match[0].length;
  }
  if (start < text.length) paragraphs(result, text.slice(start));
  return result;
}

export function releaseURLs(row) {
  for (const node of row.querySelectorAll("img,a[download]"))
    URL.revokeObjectURL(node.src || node.href);
}
export function imagePart(file, description = "Conversation image") {
  if (!imageTypes.includes(file.type)) throw new Error("Unsupported display image");
  const image = element("img");
  image.alt = description;
  image.src = URL.createObjectURL(file);
  return image;
}
