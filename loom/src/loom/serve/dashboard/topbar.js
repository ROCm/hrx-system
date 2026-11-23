// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const telemetryLinks = document.querySelectorAll('.topbar a[href="./"]');
const returnLink = document.querySelector('.topbar nav a[href="chat.html"]');
const privacy = document.getElementById("privacy");
export const incognito = new URLSearchParams(location.search).has("private");

// Canvas gradients need literal colors. Resolve the shared CSS palette once.
const style = getComputedStyle(document.documentElement);
export const chartColors = Object.fromEntries(
  ["input", "output", "memory", "active", "errors", "rule", "dim"].map((name) => [
    name,
    style.getPropertyValue("--" + name).trim(),
  ]),
);

document.getElementById("endpoint").textContent = location.host;
if (incognito) {
  for (const link of document.querySelectorAll(".topbar a")) link.search = "?private";
  privacy.textContent = "Exit incognito";
  privacy.href = "chat.html";
} else if (returnLink) {
  // The chat fragment stays in the browser. Keep its return address across
  // telemetry's own device and memory-pool anchors, including a reload.
  const fragment = /^#(?:session|project)\//.test(location.hash)
    ? location.hash
    : history.state?.conversation;
  if (fragment) {
    returnLink.hash = fragment;
    for (const link of telemetryLinks) link.hash = fragment;
    const remember = () => history.replaceState({ conversation: fragment }, "");
    remember();
    addEventListener("hashchange", remember);
  }
}

export function setConversationLocation(fragment) {
  history.replaceState(null, "", fragment);
  for (const link of telemetryLinks) link.hash = fragment;
}
