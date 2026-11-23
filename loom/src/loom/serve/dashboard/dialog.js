// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { byId, element } from "./view.js";

const dialog = byId("action-dialog");
const form = dialog.querySelector("form"),
  fields = byId("action-fields");
const apply = byId("apply");
byId("dismiss").onclick = () => dialog.close();

export function field(label, control) {
  const node = element("label", label);
  node.append(control);
  return node;
}
export function ask(title, inputs, verb, accept) {
  byId("action-title").textContent = title;
  byId("dismiss").textContent = verb ? "Cancel" : "Close";
  fields.replaceChildren(...inputs);
  apply.textContent = verb || "";
  apply.hidden = !verb;
  form.onsubmit = (event) => {
    event.preventDefault();
    if (!verb) return;
    dialog.close();
    accept();
  };
  dialog.showModal();
  fields.querySelector("input,select")?.focus();
}
export function askName(title, value, verb, accept) {
  const name = element("input");
  name.value = value;
  name.required = true;
  name.maxLength = 100;
  name.pattern = ".*\\S.*";
  ask(title, [field("Name", name)], verb, () => accept(name.value.trim()));
  name.select();
}
export function selectChoices(choices) {
  const select = element("select");
  for (const [label, value] of choices) select.add(new Option(label, value));
  return select;
}
export function findChats(search, select, title) {
  const input = element("input"),
    list = element("div", undefined, "search-results");
  input.type = "search";
  input.placeholder = "Title or message text";
  list.setAttribute("aria-live", "polite");
  let generation = 0,
    timer;
  async function refresh(version) {
    try {
      const rows = await search(input.value.trim().toLowerCase());
      if (!dialog.open || version !== generation) return;
      list.replaceChildren(
        ...rows.map((row) => {
          const node = element("button", (row.pinned ? "★ " : "") + title(row));
          node.type = "button";
          node.onclick = () => {
            dialog.close();
            select(row.id);
          };
          return node;
        }),
      );
      if (!rows.length) list.textContent = "No matching chats";
    } catch (error) {
      if (version === generation) list.textContent = error.message;
    }
  }
  input.oninput = () => {
    const version = ++generation;
    clearTimeout(timer);
    list.textContent = "Searching…";
    timer = setTimeout(() => refresh(version), 160);
  };
  input.onkeydown = (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      list.querySelector("button")?.click();
    }
  };
  ask("Find chat in this project", [field("Search", input), list]);
  dialog.addEventListener(
    "close",
    () => {
      clearTimeout(timer);
      ++generation;
    },
    { once: true },
  );
  refresh(generation);
}
