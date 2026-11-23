// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native control names are the persisted keys for projects, presets and exports.
const fields = document.querySelectorAll("aside [name]");
const numbers = [
  "max_tokens",
  "temperature",
  "top_p",
  "top_k",
  "frequency_penalty",
  "presence_penalty",
];
export const samplingKeys = [...numbers, "seed", "stop", "reasoning_effort"];
export function importSettings(input = {}) {
  if (!input || typeof input !== "object" || Array.isArray(input))
    throw new Error("Invalid conversation settings");
  const values = {};
  for (const { name: key, type, defaultChecked } of fields) {
    if (type === "checkbox") {
      if (Object.hasOwn(input, key) && typeof input[key] !== "boolean")
        throw new Error("Invalid conversation setting · " + key);
      values[key] = input[key] ?? defaultChecked;
    } else if (Object.hasOwn(input, key)) {
      if (typeof input[key] !== "string") throw new Error("Invalid conversation setting · " + key);
      values[key] = input[key];
    }
  }
  return values;
}
export function settings() {
  const values = {};
  for (const field of fields) {
    if (!field.checkValidity())
      throw new Error(field.labels[0].textContent.trim() + " · " + field.validationMessage);
    if (field.type === "checkbox") values[field.name] = field.checked;
    else if (field.value !== "") values[field.name] = field.value;
  }
  if (values.template && values.template.split("{{input}}").length !== 2) {
    throw new Error("The user template needs one {{input}} placeholder");
  }
  if (values.seed && !/^(0|[1-9]\d*)$/.test(values.seed))
    throw new Error("Seed must be an unsigned decimal integer");
  if (values.seed && BigInt(values.seed) > 18446744073709551615n)
    throw new Error("Seed exceeds uint64");
  return values;
}
export function restoreSettings(values) {
  for (const field of fields) {
    if (field.type === "checkbox") field.checked = values[field.name] ?? field.defaultChecked;
    else field.value = values[field.name] ?? "";
    if (["reader", "recognizer", "imageModel"].includes(field.name) && field.selectedIndex < 0)
      field.selectedIndex = 0;
  }
}
export function chatSettings(values) {
  const options = {};
  for (const key of numbers) {
    if (values[key] !== undefined) options[key] = Number(values[key]);
  }
  if (values.seed !== undefined) {
    const seed = Number(values.seed);
    if (!Number.isSafeInteger(seed) && !JSON.rawJSON)
      throw new Error("This browser cannot encode a large exact integer for chat");
    options.seed = Number.isSafeInteger(seed) ? seed : JSON.rawJSON(values.seed);
  }
  if (values.reasoning_effort) options.reasoning_effort = values.reasoning_effort;
  if (values.stop) options.stop = values.stop.split(/\r?\n/).filter(Boolean);
  return options;
}
export function userText(input, values) {
  return values.template ? values.template.replace("{{input}}", () => input) : input;
}
