// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import {
  appendCalls,
  chatChunks,
  checked,
  checkpoint,
  codeTool,
  compactionTool,
  endpoint,
  generateImage,
  htmlTool,
  imageEndpoint as images,
  imageTool,
  messages,
  offeredTools,
  post,
  summaryPrompt,
  toolArgument,
  toolChange,
} from "./wire.js";
import { runJavaScript } from "./code.js";
import { conversationDownload, conversationImport } from "./export.js";
import { callMcp, loadMcp, manageMcp, mcpSettings, offeredMcp, restoreMcp } from "./mcp.js";
import { drawTurns, metrics, savedReplyStats, watchReplyStats } from "./reply-stats.js";
import {
  byId,
  button,
  copyButton,
  disclosure,
  element,
  formatted,
  imagePart,
  preview,
  releaseURLs,
  toolOutput,
} from "./view.js";
import { microphone, speak, record, transcribe } from "./audio.js";
import { incognito, setConversationLocation } from "./topbar.js";
import { attachment, editable, entryText, partName } from "./content.js";
import { ask, askName, field, findChats, selectChoices } from "./dialog.js";
import {
  createProject,
  deleteFromWorkspace,
  emptySession,
  finishRelease,
  forkSession,
  openWorkspace,
  pendingReleases,
  previewPrefix,
  projects,
  readSession,
  recordOrphanPin,
  savePreviewSize,
  updateProject,
  saveSession,
  searchSessions,
  sessions,
} from "./storage.js";
import { chatSettings, restoreSettings, samplingKeys, settings, userText } from "./settings.js";

const transcript = byId("transcript"),
  prompt = byId("prompt"),
  models = byId("model");
const inspectorToggle = byId("inspector-toggle");
const chat = "/v1/chat/completions";
const compactTool = compactionTool({});
const speech = "/v1/audio/speech",
  recognition = "/v1/audio/transcriptions";
const modes = new Map([
  [chat, "Chat"],
  [images, "Image"],
  [speech, "Speech"],
]);
let catalog = new Map(),
  active,
  endRecording,
  attachments = [],
  database,
  thread,
  project;
const privateChats = new Map();
const drafts = new Map();
const alternatives = new Map();
let chatLabels = new Map();
const workspaceBusy = Symbol();
let waitingForTools = false;

function notice(text = "", failure = false) {
  const status = byId("notice");
  const download = status.lastElementChild;
  if (download?.download) URL.revokeObjectURL(download.href);
  status.textContent = text;
  status.dataset.error = String(failure);
}
function guarded(work) {
  return (...args) =>
    Promise.resolve()
      .then(() => work(...args))
      .catch((error) => notice(error.message, true));
}
function newThread() {
  return emptySession(project?.id, project?.settings);
}
async function remember() {
  if (incognito) {
    thread.updated = Date.now();
    privateChats.set(thread.id, thread);
    return;
  }
  const selected = thread;
  await saveSession(database, selected);
  if (thread === selected) setConversationLocation("#session/" + selected.id);
}
function chatTitle(row) {
  const tag = chatLabels.get(row.id);
  return tag ? row.title + " · #" + tag.suffix : row.title;
}
function indexChatTitles(rows) {
  const groups = new Map();
  for (const row of rows) {
    const ids = groups.get(row.title);
    if (ids) ids.push(row.id);
    else groups.set(row.title, [row.id]);
  }
  const previous = chatLabels;
  chatLabels = new Map();
  for (const ids of groups.values()) {
    // A surviving choice keeps its visible ID when its namesake is removed.
    if (ids.length === 1 && !previous.has(ids[0])) continue;
    const unique = new Set(ids.map((id) => id.slice(0, 6))).size === ids.length;
    for (const id of ids)
      chatLabels.set(id, {
        suffix: unique ? (previous.get(id)?.suffix ?? id.slice(0, 6)) : id,
        shared: ids.length > 1,
      });
  }
}
async function historyMenu() {
  const selected = thread;
  const rows = incognito
    ? [...privateChats.values()].reverse()
    : await sessions(database, project.id);
  if (thread !== selected) return;
  rows.sort(chatOrder);
  indexChatTitles(rows);
  alternatives.clear();
  const pinned = byId("pinned-chats"),
    recent = byId("recent-chats");
  pinned.replaceChildren();
  recent.replaceChildren();
  const draft = byId("draft-chat");
  draft.hidden = rows.some((row) => row.id === thread.id);
  draft.value = draft.hidden ? "" : thread.id;
  for (const row of rows) {
    if (row.parent?.id === thread.id && row.parent.turn !== undefined) {
      const turn = row.parent.turn;
      if (!alternatives.has(turn)) alternatives.set(turn, []);
      alternatives.get(turn).push(row);
    }
    const suffix = row.checkpoint && !row.checkpoint.error ? " · KV checkpoint" : "";
    (row.pinned ? pinned : recent).append(
      new Option((row.pinned ? "★ " : "") + chatTitle(row) + suffix, row.id),
    );
  }
  pinned.hidden = !pinned.childElementCount;
  recent.hidden = !recent.childElementCount;
  byId("pin-chat").textContent = thread.pinned ? "Unpin chat" : "Pin chat";
  byId("delete-chat").textContent = draft.hidden ? "Delete chat…" : "Discard draft…";
  updateDeleteAction();
  byId("threads").value = thread.id;
  if (thread.parent) drawBranch();
}
function chatOrder(a, b) {
  return Number(!!b.pinned) - Number(!!a.pinned) || b.updated - a.updated;
}
async function projectMenu(selected) {
  const options = await projects(database);
  project =
    options.find((item) => item.id === selected) ||
    options.find((item) => item.id === "local") ||
    options[0];
  options.sort((a, b) => a.name.localeCompare(b.name));
  byId("project-list").replaceChildren(...options.map((item) => new Option(item.name, item.id)));
  byId("projects").value = project.id;
}
function unfinished() {
  const pending = new Map();
  for (const { message, partial } of thread.entries) {
    if (!message || partial) continue;
    if (message.role === "assistant") {
      for (const call of message.tool_calls || []) pending.set(call.id, call);
    } else if (message.role === "tool") pending.delete(message.tool_call_id);
  }
  return pending;
}
function controls() {
  const model = catalog.get(models.value),
    route = model?.loom.endpoint;
  const busy = Boolean(active),
    isChat = route === chat;
  waitingForTools = unfinished().size !== 0;
  for (const field of document.querySelectorAll(
    "aside :is(input,textarea,select,button), #branch button, " +
      "#transcript :is(button:not(.save-html),select.turn-picker), #files button, #model, .chat-actions button, #threads, #projects, #upload",
  ))
    field.disabled = busy;
  models.disabled ||= waitingForTools || Boolean(thread.checkpoint && !thread.checkpoint.release);
  byId("stop").hidden = !busy || active === workspaceBusy;
  const canSpeak = byId("reader").length > 0,
    canHear = byId("recognizer").length > 0;
  for (const id of ["attach", "token-setting", "tools-setting", "compact-settings", "sampling"])
    byId(id).hidden = !isChat;
  byId("compact").disabled = busy || !canCompact(model);
  byId("compact-at").disabled = busy || !model?.loom.input_tokens || !model.context_length;
  draftChanged();
  byId("effort-setting").hidden = !isChat || !model.supported_reasoning_levels?.length;
  byId("image-tool-setting").hidden = byId("image-tool").length <= 1;
  byId("size-setting").hidden = route !== images;
  byId("audio-settings").hidden = !canSpeak && !canHear;
  byId("reader-setting").hidden = !isChat || !canSpeak;
  byId("speed-setting").hidden = byId("voice-setting").hidden = !(
    route === speech ||
    (isChat && canSpeak)
  );
  for (const id of ["recognizer-setting", "transcribe-file", "mic"]) byId(id).hidden = !canHear;
  byId("voice-chat").hidden = !isChat || !canSpeak || !canHear;
  byId("voice-chat").disabled = busy || waitingForTools || !!thread.checkpoint?.error;
  byId("mic").disabled = busy && !endRecording;
  byId("mic").textContent = endRecording ? "Finish dictation" : "Dictate";
  byId("remove-preset").disabled = busy || !byId("presets").value;
}
function selectModel() {
  const model = catalog.get(models.value);
  const capacity = model?.context_length?.toLocaleString() || "";
  byId("context-window").textContent = capacity ? " / " + capacity : "";
  document.documentElement.style.setProperty("--count-width", Math.max(5, capacity.length) + "ch");
  const effort = byId("effort");
  effort.replaceChildren(new Option("Model default", ""));
  for (const level of model?.supported_reasoning_levels || []) effort.add(new Option(level, level));
  const limit = byId("limit"),
    preset = model?.limits?.default_output_tokens;
  limit.max = model?.limits?.max_output_tokens || 2147483647;
  limit.placeholder = preset ? "Auto · " + preset.toLocaleString() : "Model default";
  controls();
}
async function discover() {
  const response = await checked(await fetch(endpoint("/v1/models")));
  const { data } = await response.json();
  catalog = new Map(data.map((model) => [model.id, model]));
  const previous = models.value;
  models.replaceChildren(
    ...data
      .filter((model) => modes.has(model.loom?.endpoint))
      .map((model) => new Option(model.id + " · " + modes.get(model.loom.endpoint), model.id)),
  );
  const preferred = [
    previous,
    thread.model,
    data.find((model) => model.loom?.endpoint === chat)?.id,
  ];
  for (const id of preferred)
    if (modes.has(catalog.get(id)?.loom?.endpoint)) {
      models.value = id;
      break;
    }
  for (const [id, route] of [
    ["recognizer", recognition],
    ["reader", speech],
    ["image-tool", images],
  ]) {
    const picker = byId(id),
      selected = picker.value;
    const options = data
      .filter((model) => model.loom?.endpoint === route)
      .map((model) => new Option(model.id, model.id));
    if (route === images) options.unshift(new Option("Off", ""));
    picker.replaceChildren(...options);
    if (catalog.get(selected)?.loom?.endpoint === route) picker.value = selected;
  }
  if (!models.length && !byId("recognizer").length)
    notice("No conversation models are prepared. Refresh after registration.");
  selectModel();
  if (unfinished().size) redraw();
}
function card(role) {
  const row = element("article");
  row.dataset.role = role;
  row.append(element("span", role.toUpperCase(), "stamp"));
  return row;
}
function drawEntry(entry, pending, position) {
  const session = thread;
  let slot = 0;
  // Assign the slot as each preview is parsed, independent of which fence opens first.
  const nextPreview = () => {
    const key = position + "/" + slot++;
    return {
      size: session.previewSizes?.[key],
      save: (size) =>
        savePreviewSize(database, session, key, size).catch((error) => notice(error.message, true)),
    };
  };
  const message = entry.message;
  const row = card(
    entry.compaction
      ? "summary"
      : entry.toolUpdate && !message.content
        ? "tools"
        : message?.role || entry.role,
  );
  if (entry.model) row.firstChild.textContent += "  /  " + entry.model;
  if (entry.toolUpdate) row.append(element("div", entry.toolUpdate, "caption"));
  if (message?.reasoning_content)
    row.append(disclosure("Reasoning", element("div", message.reasoning_content, "prose")));
  const content = message?.content ?? entry.text;
  if (
    typeof content === "string" &&
    !(entry.html !== undefined && content === "Displayed HTML preview.")
  )
    row.append(entry.verbatim ? toolOutput(content) : formatted(content, entry, nextPreview));
  if (Array.isArray(content))
    for (const part of content) {
      if (part.type === "text") row.append(formatted(part.text, part, nextPreview));
      if (part.type === "image") row.append(imagePart(part.file));
      if (part.type === "document") row.append(disclosure(part.name, element("pre", part.text)));
    }
  if (entry.image) row.append(imagePart(entry.image, entry.imagePrompt));
  if (entry.html !== undefined) row.append(preview(entry.html, nextPreview()));
  if (entry.problem) row.append(element("p", entry.problem, "failed"));
  else if (entry.stats?.finish === "length")
    row.append(element("p", "Output limit reached", "caption"));
  if (entry.failed) row.classList.add("failed");
  drawToolCalls(row, entry, pending);
  if (entry.stats && (message?.role || entry.role) === "assistant")
    row.append(savedReplyStats(entry.stats));
  return row;
}
function executableTool(entry, call) {
  const name = call.function.name;
  if (
    name === compactTool.function.name &&
    entry.settings?.compactTool &&
    byId("compact-tool").checked
  )
    return compactTool;
  if (name === codeTool.function.name && entry.settings?.codeTool === true) return codeTool;
  if (name === htmlTool.function.name && entry.offeredHTML) return htmlTool;
  if (name === imageTool.function.name && catalog.get(entry.imageModel)?.loom.endpoint === images)
    return imageTool;
  return offeredMcp(entry.settings, name);
}
function drawToolCalls(row, entry, pending) {
  for (const call of entry.message?.tool_calls || []) {
    const box = element("div", undefined, "tool");
    const tool = executableTool(entry, call);
    let source = call.function.arguments;
    if (tool === codeTool) {
      try {
        source = toolArgument(call, "code");
      } catch (error) {
        box.append(element("div", error.message, "failed"));
      }
    }
    const details = disclosure(call.function.name + " · " + call.id, element("pre", source));
    box.append(details);
    if (pending.get(call.id) === call) {
      if (tool) {
        details.open = tool === codeTool;
        const label = tool.server
          ? "Run MCP · " + tool.name
          : tool === codeTool
            ? "Run JavaScript"
            : tool === htmlTool
              ? "Show preview"
              : tool === compactTool
                ? "Compact chat"
                : "Generate image · " + entry.imageModel;
        box.append(
          button(
            label,
            toolAction(call, (signal) => runTool(entry, call, tool, signal)),
          ),
        );
      }
      const manual = element("textarea");
      manual.rows = 2;
      manual.setAttribute("aria-label", "Tool result");
      const submit = button(
        "Return result",
        toolAction(call, () => {
          if (!manual.value.trim()) throw new Error("Enter a tool result");
          return toolResult(call, manual.value);
        }),
      );
      const manualBox = element("div");
      manualBox.append(manual, submit);
      box.append(disclosure("Supply result", manualBox));
    }
    row.insertBefore(box, row.querySelector(":scope > .reply-stats"));
  }
}
function followingLatest() {
  return transcript.scrollHeight - transcript.clientHeight - transcript.scrollTop < 36;
}
function comparisonRow(answer, stats, original, index) {
  if (!original) return answer;
  const pair = element("div", undefined, "comparison");
  for (const [name, row, data] of [
    ["Original", drawEntry(original, new Map(), index + "/original"), original.stats],
    ["Comparison", answer, stats],
  ]) {
    const section = element("section"),
      usage = data?.usage;
    const cached = usage?.prompt_tokens_details?.cached_tokens;
    const timing = data?.firstText == null ? "—" : Math.round(data.firstText) + " ms";
    const label = element("header", name + " · " + (data?.model || "—"));
    const caption = element(
      "div",
      timing +
        " first text · " +
        (cached == null ? "—" : usage.prompt_tokens - cached) +
        " fresh · " +
        (cached ?? "—") +
        " cached · " +
        (usage?.completion_tokens ?? "—") +
        " output",
      "caption",
    );
    section.append(label, caption, row);
    pair.append(section);
  }
  return pair;
}
function turnActions(entry, index) {
  const actions = element("div", undefined, "turn-actions");
  const text = entry.html ?? entryText(entry);
  if (text) actions.append(copyButton("Copy", text));
  if (entry.message?.role === "user" && entry.message.content)
    actions.append(
      button(
        "Edit in fork",
        workspaceAction(() => editTurn(index)),
      ),
    );
  if (entry.message?.role === "assistant") {
    if (catalog.get(entry.model)?.loom.endpoint === chat)
      actions.append(button("Retry", replayAction(index, entry.model, entry.settings, "retry")));
    if ([...catalog.values()].some((model) => model.loom.endpoint === chat))
      actions.append(button("Compare…", () => compareTurn(index)));
    if (text && byId("reader").length)
      actions.append(
        button(
          "Read aloud",
          guarded(() => readAloud(text)),
        ),
      );
  }
  const versions = alternatives.get(index);
  if (versions?.length) {
    const select = selectChoices([
      ["Alternatives (" + versions.length + ")", ""],
      ...versions.map((row) => [chatTitle(row), row.id]),
    ]);
    select.className = "turn-picker";
    select.setAttribute("aria-label", "Open an alternative");
    select.disabled = !!active;
    select.onchange = workspaceAction(() => openChat(select.value));
    actions.append(select);
  }
  return actions;
}
function showLatest() {
  transcript.scrollTop = transcript.scrollHeight;
}
function appendReply(row, follow = followingLatest()) {
  transcript.append(row);
  if (follow) showLatest();
}
function peerReceipt(peer) {
  // One immutable text answer survives parent deletion and export. A nested
  // comparison retains its direct peer, never a recursive chain of answers.
  return (
    peer && { message: peer.message, model: peer.model, stats: peer.stats, problem: peer.problem }
  );
}
// Completed message bodies own live preview documents. Update their chat
// actions in place so later turns cannot restart the user's running UI.
const renderedTurns = new WeakMap();
function redraw(follow = followingLatest()) {
  const pending = unfinished();
  const heads = new Map();
  for (let i = thread.parent?.count || thread.importedCount || 0; i < thread.entries.length; ++i) {
    const entry = thread.entries[i];
    if (entry.message?.role === "assistant" && !entry.compaction) heads.set(entry.model, i);
  }
  transcript.querySelector(".continue-tools")?.remove();
  thread.entries.forEach((entry, i) => {
    const current = transcript.children[i],
      saved = renderedTurns.get(current);
    let row;
    if (saved?.entry === entry && saved.session === thread) {
      row = saved.row;
      if (entry.message?.tool_calls?.length) {
        for (const tool of row.querySelectorAll(":scope > .tool")) tool.remove();
        drawToolCalls(row, entry, pending);
      }
    } else {
      row = drawEntry(entry, pending, i);
      const heading = element("div", undefined, "turn-heading"),
        stamp = row.firstChild;
      stamp.replaceWith(heading);
      heading.append(stamp);
      const next = comparisonRow(row, entry.stats, entry.against, i);
      if (current) {
        releaseURLs(current);
        current.replaceWith(next);
      } else transcript.append(next);
      if (!entry.partial) renderedTurns.set(next, { entry, row, session: thread });
    }
    const actions = turnActions(entry, i);
    if ((entry.message?.role || entry.role) === "assistant" && !entry.partial && !entry.problem) {
      let count = i + 1;
      while (thread.entries[count]?.message?.role === "tool") ++count;
      const live =
        !incognito &&
        heads.get(entry.model) === i &&
        catalog.get(entry.model)?.loom.endpoint === chat;
      const fork = button(
        live ? "Fork here" : "Fork from history",
        workspaceAction(() => startFork(count, entry, live)),
      );
      fork.className = "fork-turn";
      fork.title = live
        ? "Fork this reply using its retained KV cache"
        : "Start a chat with the history through this reply";
      actions.append(fork);
    }
    const heading = row.firstElementChild;
    heading.children[1]?.remove();
    heading.append(actions);
  });
  while (transcript.children.length > thread.entries.length) {
    const stale = transcript.lastElementChild;
    releaseURLs(stale);
    stale.remove();
  }
  drawBranch();
  if (thread.entries.at(-1)?.message?.role === "tool" && !pending.size) {
    const resume = button(
      "Continue with results",
      guarded(() => submitTurn("continue")),
    );
    resume.className = "continue-tools";
    transcript.lastChild.append(resume);
  }
  const stats = thread.entries.findLast((entry) => entry.stats)?.stats;
  metrics(stats);
  drawTurns(thread.entries);
  controls();
  if (follow) showLatest();
}
function workspaceAction(work, after) {
  return guarded(async (...args) => {
    if (active) return;
    active = workspaceBusy;
    controls();
    try {
      await work(...args);
    } finally {
      active = undefined;
      redraw();
    }
    if (after) await after();
  });
}
async function openChat(id) {
  const selected = incognito ? privateChats.get(id) : await readSession(database, id);
  if (!selected) {
    await historyMenu();
    throw new Error("This chat no longer exists.");
  }
  selectThread(selected);
  await restoreChat();
}
async function restoreChat() {
  if (thread.checkpoint?.release) {
    try {
      await releaseCheckpoint();
    } catch (error) {
      notice("Checkpoint release pending · " + error.message, true);
    }
    drawBranch();
  }
  await historyMenu();
  redraw();
}
async function startFork(count, entry, live, action, turn) {
  const parent = thread;
  if (incognito || !parent.revision || parent.saved < count) await remember();
  const child = forkSession(parent, count, entry.model, entry.settings || parent.settings);
  if (action) {
    Object.assign(child.parent, { action, turn });
    child.title =
      (action === "compare" ? "Compare · " + entry.model : action === "edit" ? "Edit" : "Retry") +
      " · " +
      parent.title;
    if (action === "compare") child.pendingPeer = peerReceipt(parent.entries[turn]);
  }
  if (live) {
    child.checkpoint = { model: entry.model };
    return checkpointWork(child.id, () => openFork(parent, child));
  }
  return openFork(parent, child);
}
async function editTurn(index) {
  const entry = thread.entries[index];
  const answer = thread.entries[index + 1],
    model = answer?.model || thread.model || models.value;
  if (catalog.get(model)?.loom.endpoint !== chat)
    throw new Error("This turn's chat model is unavailable");
  const values = answer?.settings || thread.settings;
  const edit = editable(entry, values);
  await startFork(index, { model, settings: values }, false, "edit", index);
  prompt.value = edit.text;
  attachments = edit.files;
  showFiles();
  prompt.focus();
}
function replayAction(index, model, values, action) {
  return workspaceAction(
    () => startFork(index, { model, settings: values }, false, action, index),
    () => submitTurn("replay"),
  );
}
function compareTurn(index) {
  const original = thread.entries[index],
    values = original.settings || thread.settings;
  const imagesInHistory = thread.entries
    .slice(0, index)
    .some(
      (entry) =>
        Array.isArray(entry.message?.content) &&
        entry.message.content.some((part) => part.type === "image"),
    );
  const peers = [...catalog.values()].filter(
    (model) =>
      model.loom.endpoint === chat &&
      (!imagesInHistory || model.input_modalities.includes("image")),
  );
  if (!peers.length) {
    notice("No prepared chat model accepts this history", true);
    return;
  }
  const model = selectChoices(peers.map((peer) => [peer.id, peer.id]));
  model.value = (peers.find((peer) => peer.id !== original.model) || peers[0]).id;
  const presets = project.presets || [];
  const selection = selectChoices([
    ["Original answer", "original"],
    ["Current controls", "current"],
    ...presets.map((preset, i) => [preset.name, String(i)]),
  ]);
  ask(
    "Compare this answer",
    [field("Model", model), field("Sampling settings", selection)],
    "Run comparison",
    guarded(() => {
      const source =
        selection.value === "original"
          ? values
          : selection.value === "current"
            ? settings()
            : presets[Number(selection.value)].settings;
      // Preserve the original system message and tool offers. Only sampling
      // fields from the other configuration apply to the identical messages.
      const candidate = { ...values };
      for (const key of samplingKeys) {
        if (source[key] === undefined) delete candidate[key];
        else candidate[key] = source[key];
      }
      return replayAction(index, model.value, candidate, "compare")();
    }),
  );
}
function checkpointWork(id, work) {
  return navigator.locks
    ? navigator.locks.request(database.name + "/checkpoint/" + id, work)
    : work();
}
async function openFork(parent, child) {
  const pin = child.checkpoint;
  // Journal the unique pin name (the child ID) before sending the control
  // request. A reload can still consume or release it if the reply was lost.
  if (incognito) privateChats.set(child.id, child);
  else await saveSession(database, child);
  const draft = prompt.value,
    files = attachments;
  selectThread(child);
  if (!child.parent.action) {
    prompt.value = draft;
    attachments = files;
    showFiles();
  }
  await historyMenu();
  if (pin) {
    try {
      const receipt = await checkpoint("POST", pin.model, child.id, parent.id);
      pin.position = receipt.position;
    } catch (error) {
      if (error.status) {
        pin.error = error.message;
        pin.absent = [400, 404, 503].includes(error.status);
      } else notice("Checkpoint unconfirmed · " + error.message, true);
    }
    try {
      await remember();
    } catch (error) {
      // Deletion can commit in another tab during the native pin request.
      if (await recordOrphanPin(database, child.id, pin.model)) {
        await checkpoint("DELETE", pin.model, child.id);
        await finishRelease(database, child.id);
      }
      throw error;
    }
    await historyMenu();
  }
  prompt.focus();
}
async function releaseCheckpoint() {
  const pin = thread.checkpoint;
  pin.release = true;
  await remember();
  if (!pin.absent) await checkpoint("DELETE", pin.model, thread.id);
  delete thread.checkpoint;
  try {
    await remember();
  } catch (error) {
    thread.checkpoint = pin;
    throw error;
  }
}
function drawBranch() {
  const box = byId("branch"),
    parent = thread.parent,
    pin = thread.checkpoint;
  box.replaceChildren();
  box.hidden = !parent;
  if (!parent) return;
  if ([...byId("threads").options].some((option) => option.value === parent.id)) {
    const origin = button(
      parent.title,
      workspaceAction(() => openChat(parent.id)),
    );
    origin.title = "Open parent chat";
    box.append("Forked from ", origin);
  } else box.append("Forked from a deleted chat");
  const first = thread.entries.find(
    (e, i) =>
      i >= parent.count &&
      !e.compaction &&
      e.message?.role === "assistant" &&
      (!e.partial || e.stats?.usage),
  );
  if (first) {
    const usage = first.stats?.usage,
      cached = usage?.prompt_tokens_details?.cached_tokens;
    box.append(
      cached == null
        ? "Cache reuse unreported"
        : cached.toLocaleString() +
            " of " +
            usage.prompt_tokens.toLocaleString() +
            " input tokens reused",
    );
  } else if (!pin) box.append("From history");
  if (thread.entries.length === parent.count && ["retry", "compare"].includes(parent.action)) {
    box.append(
      button(
        parent.action === "compare" ? "Run comparison" : "Retry original prompt",
        guarded(() => submitTurn("replay")),
      ),
    );
  }
  if (!pin) return;
  if (pin.error && !pin.release) box.append("KV unavailable · " + pin.error);
  else if (!pin.release)
    box.append(
      pin.position == null
        ? "KV checkpoint unconfirmed"
        : "KV checkpoint · " +
            pin.position.toLocaleString() +
            (pin.position === 1 ? " token" : " tokens"),
    );
  const release = button(
    pin.release ? "Release checkpoint" : "Fork from history",
    workspaceAction(async () => {
      await releaseCheckpoint();
      await historyMenu();
      notice();
    }),
  );
  release.title = "Release the reserved KV cache and keep the conversation history";
  box.append(release);
}
function toolAction(call, work) {
  return guarded(async () => {
    if (active || unfinished().get(call.id) !== call) return;
    const controller = new AbortController();
    active = controller;
    controls();
    try {
      await work(controller.signal);
      controller.signal.throwIfAborted();
      notice();
    } catch (error) {
      if (!controller.signal.aborted) throw error;
      notice("Stopped waiting");
      return;
    } finally {
      active = undefined;
      redraw();
    }
    if (byId("auto-tools").checked && !unfinished().size) await submitTurn("continue");
  });
}
async function toolResult(call, content, display = {}) {
  const message = { role: "tool", tool_call_id: call.id, content };
  thread.entries.push({ message, ...display });
  redraw();
  await remember();
  return message;
}
async function runTool(entry, call, tool, signal) {
  signal.throwIfAborted();
  if (tool.server) {
    notice("Calling MCP · " + tool.name + "…");
    let result;
    try {
      result = await callMcp(tool, call.function.arguments, signal);
    } catch (error) {
      result = {
        failed: true,
        text: signal.aborted ? "Stopped waiting for MCP · " + tool.name : "MCP · " + error.message,
      };
    }
    return toolResult(call, result.text, { failed: result.failed, verbatim: true });
  }
  const display =
    tool === codeTool ? { verbatim: true } : tool === imageTool ? { model: entry.imageModel } : {};
  let content, summary;
  try {
    const argument = toolArgument(call, tool.function.parameters.required[0]);
    if (tool === compactTool) {
      if (entry.message.tool_calls.length !== 1) throw Error("Call compact_chat alone");
      summary = argument;
      content = "Summary received.";
    } else if (tool === codeTool) {
      notice("Running JavaScript…");
      const result = await runJavaScript(argument, signal);
      content = result.text;
      display.failed = result.failed;
    } else if (tool === htmlTool) {
      display.html = argument;
      content = "";
    } else {
      notice("Generating image · " + display.model);
      display.image = await generateImage(display.model, argument, signal);
      signal.throwIfAborted();
      display.imagePrompt = argument;
      content = "Generated image with " + display.model + ".";
    }
  } catch (error) {
    signal.throwIfAborted();
    content = tool === imageTool ? "Image generation failed: " + error.message : String(error);
    display.failed = true;
  }
  const result = await toolResult(call, content, display);
  if (summary) await compactConversation(catalog.get(models.value), settings(), signal, summary);
  return result;
}
function showFiles() {
  byId("files").replaceChildren(
    ...attachments.map((file) => {
      const remove = button(partName(file) + " ×", () => {
        attachments = attachments.filter((other) => other !== file);
        showFiles();
      });
      remove.disabled = Boolean(active);
      return remove;
    }),
  );
  if (!acceptsFiles(catalog.get(models.value)))
    byId("files").append("Select a model that accepts these files, or remove them");
  draftChanged();
}
function acceptsFiles(model) {
  return (
    !attachments.length ||
    (model?.loom.endpoint === chat &&
      (model.input_modalities.includes("image") ||
        attachments.every((part) => part.type !== "image")))
  );
}
function selectThread(value) {
  inspectorToggle.ariaExpanded = "false";
  const text = commands.has(prompt.value.trim()) ? "" : prompt.value;
  if (thread && thread !== value && (text || attachments.length))
    drafts.set(thread.id, { text, files: attachments, project: thread.project });
  else if (thread) drafts.delete(thread.id);
  thread = value;
  if (!incognito) {
    setConversationLocation(value.revision ? "#session/" + value.id : "#project/" + project.id);
  }
  if (catalog.has(value.model)) models.value = value.model;
  else if (
    value.importedCount &&
    value.entries.at(-1)?.message &&
    catalog.get(models.value)?.loom.endpoint !== chat
  ) {
    const replacement = [...catalog.values()].find((model) => model.loom.endpoint === chat);
    if (replacement) models.value = replacement.id;
  }
  attachments = drafts.get(value.id)?.files || [];
  prompt.value = drafts.get(value.id)?.text || "";
  showFiles();
  selectModel();
  restoreSettings(value.settings);
  presetMenu();
  notice();
  redraw(true);
}
function userEntry(content) {
  return { message: { role: "user", content } };
}
async function chatBody(model, entries, user, values, signal, replay = false) {
  const tools = offeredTools(values);
  // Replays already contain the original input. A fresh tool-result continuation
  // can carry its own input note without changing any tool's returned value.
  const previous = entries.findLast((entry) => entry.message?.role === "assistant")?.settings;
  const update = !replay && toolChange(previous, tools);
  const input = update ? { ...(user || userEntry("")), toolUpdate: update } : user;
  const history = await messages(input ? [...entries, input] : entries, signal);
  if (values.system) history.unshift({ role: "system", content: values.system });
  const body = {
    model: model.id,
    messages: history,
    stream: true,
    stream_options: { include_usage: true },
    ...chatSettings(values),
  };
  if (tools.length) body.tools = tools;
  return { body, input };
}
function composerMessage(text) {
  if (!attachments.length) return userEntry(text);
  const parts = text ? [{ type: "text", text }] : [];
  return userEntry(parts.concat(attachments));
}
function tokenReadout(id, measure) {
  const output = byId(id);
  let owner, previous, timer, controller;
  return (model, input) => {
    if (model === owner && input === previous) return;
    if (model !== owner) output.parentElement.hidden = !model;
    owner = model;
    previous = input;
    clearTimeout(timer);
    controller?.abort();
    const placeholder = input == null ? "—" : "…";
    if (output.value !== placeholder) output.value = placeholder;
    if (input == null) return;
    controller = new AbortController();
    const { signal } = controller;
    timer = setTimeout(async () => {
      try {
        const count = await measure(model, input, signal);
        if (!signal.aborted) output.value = count.toLocaleString();
      } catch (error) {
        if (!signal.aborted) {
          const title = "Token count unavailable";
          const detail = button("n/a", () => ask(title, [element("p", error.message)]));
          detail.title = detail.ariaLabel = title + " · " + error.message;
          output.replaceChildren(detail);
        }
      }
    }, 350);
  };
}
async function countText(model, content, signal) {
  const result = await (await post("/tokenize", { model: model.id, content }, signal)).json();
  return result.tokens.length;
}
const systemTokens = tokenReadout("system-tokens", countText);
const templateTokens = tokenReadout("template-tokens", countText);
const promptTokens = tokenReadout("prompt-tokens", async (model, draft, signal) => {
  const values = settings();
  const entry = composerMessage(userText(draft.text, values));
  const { body } = await chatBody(model, thread.entries, entry, values, signal);
  signal.throwIfAborted();
  return inputTokens(body, signal);
});
async function inputTokens(body, signal) {
  return (await (await post(chat + "/input_tokens", body, signal)).json()).input_tokens;
}
function canCompact(model) {
  return (
    model?.loom.endpoint === chat &&
    !waitingForTools &&
    thread.entries.some((e) => e.message && !e.partial)
  );
}
async function compactConversation(model, values, signal, summary, automatic = false) {
  const source = thread;
  const handoff = userEntry(summaryPrompt(values));
  const { body } = await chatBody(model, source.entries, handoff, values, signal, true);
  notice("Compacting conversation…");
  if (summary === undefined) {
    const { stream_options, max_tokens, stop, ...request } = body;
    request.stream = false;
    if (request.tools) request.tool_choice = "none";
    // A separate request leaves the original chat's named KV endpoint intact.
    const result = await (await post(chat, request, signal)).json();
    const choice = result.choices?.[0];
    summary = choice?.message?.content;
    if (choice?.finish_reason !== "stop" || choice.message.tool_calls?.length || !summary?.trim())
      throw Error("Compaction needs a complete text summary");
  }
  const child = forkSession(source, source.entries.length, model.id, values);
  child.title = "Compact · " + source.title;
  child.named = true;
  const reply = {
    compaction: true,
    message: { role: "assistant", content: summary },
    model: model.id,
    settings: values,
  };
  child.entries.push(handoff, reply);
  let receipt = "Compacted";
  if (model.loom.input_tokens) {
    const next = await chatBody(model, [handoff, reply], handoff, values, signal, true);
    const [before, after] = await Promise.all([
      inputTokens(body, signal),
      inputTokens(next.body, signal),
    ]);
    if (after >= before) {
      if (automatic) return false;
      throw Error("Summary would not shorten this chat");
    }
    receipt += " · " + before.toLocaleString() + " → " + after.toLocaleString() + " input tokens";
  }
  signal.throwIfAborted();
  await openFork(source, child);
  notice(receipt);
  return true;
}
async function autoCompact(model, values, body, signal) {
  if (
    !values.compactAt ||
    !model.loom.input_tokens ||
    !model.context_length ||
    !canCompact(model) ||
    thread.entries.at(-1)?.compaction
  )
    return false;
  if ((await inputTokens(body, signal)) < (model.context_length * Number(values.compactAt)) / 100)
    return false;
  return compactConversation(model, values, signal, undefined, true);
}
function updateDeleteAction() {
  byId("delete-chat").disabled = !byId("draft-chat").hidden && !prompt.value && !attachments.length;
}
function draftChanged() {
  const text = prompt.value.trim(),
    command = commands.get(text);
  updateDeleteAction();
  const model = catalog.get(models.value),
    route = model?.loom.endpoint;
  const ready =
    !waitingForTools &&
    !thread?.checkpoint?.error &&
    acceptsFiles(model) &&
    route &&
    (text || attachments.length);
  byId("send").disabled = Boolean(active) || (command ? command.disabled : !ready);
  byId("send").textContent = command
    ? command.firstChild.textContent
    : route === images
      ? "Generate"
      : route === speech
        ? "Speak"
        : "Send";
  // A local command and its prefixes never enter a tokenizer request.
  const chatModel = route === chat ? model : null;
  const counting = !active && !(text && [...commands.keys()].some((key) => key.startsWith(text)));
  // Any edit may change history, attachments, tools, or framing. The raw
  // fields reuse a completed or pending count when their model and text match.
  const prepared = chatModel?.loom.input_tokens ? model : null;
  promptTokens(prepared, prepared && counting && ready ? { text } : null);
  const raw = chatModel?.loom.tokenize ? model : null;
  const fields = raw && counting && byId("prompt-options").open;
  systemTokens(raw, (fields && byId("system").value) || null);
  templateTokens(raw, (fields && userText(text, { template: byId("template").value })) || null);
}
function beginReply(model) {
  const start = performance.now(),
    stats = { model: model.id, arrivals: [] };
  const kind =
    model.loom.endpoint === chat ? "text" : model.loom.endpoint === images ? "image" : "audio";
  const activity = watchReplyStats(stats, start, kind, metrics);
  const row = card("assistant");
  row.firstChild.textContent += "  /  " + model.id;
  row.append(activity.node);
  appendReply(row);
  return { stats, start, activity, row };
}
async function generateText(model, signal, values, response, turn) {
  const { stats, start, activity, row } = turn;
  const follow = followingLatest();
  const message = { role: "assistant", content: "" };
  const entry = {
    message,
    model: model.id,
    settings: values,
    stats,
    offeredHTML: values.htmlTool,
    imageModel: values.imageModel,
    partial: true,
  };
  if (thread.pendingPeer) {
    entry.against = thread.pendingPeer;
    delete thread.pendingPeer;
  }
  const index = thread.entries.push(entry) - 1;
  const content = element("div", undefined, "prose");
  const live = document.createTextNode("");
  content.append(live);
  const reasonText = document.createTextNode(""),
    reason = disclosure("Reasoning", reasonText);
  reason.hidden = true;
  row.append(reason, content, activity.node);
  appendReply(comparisonRow(row, stats, entry.against, index), follow);
  let paint = 0,
    textDelta = "",
    reasonDelta = "",
    finish;
  const calls = [];
  function flush() {
    paint = 0;
    const pinned = followingLatest();
    if (textDelta) live.appendData(textDelta);
    if (reasonDelta) {
      reason.hidden = false;
      reasonText.appendData(reasonDelta);
    }
    textDelta = reasonDelta = "";
    if (pinned) showLatest();
  }
  try {
    for await (const chunk of chatChunks(response)) {
      const elapsed = performance.now() - start;
      stats.elapsed = elapsed;
      if (stats.arrivals.length === 128) stats.arrivals.shift();
      stats.arrivals.push(elapsed);
      if (chunk.usage) stats.usage = chunk.usage;
      const choice = chunk.choices[0];
      const delta = choice?.delta;
      if (delta?.content || delta?.reasoning_content || delta?.tool_calls?.length)
        stats.firstDelta ??= elapsed;
      if (delta?.content) {
        stats.firstText ??= elapsed;
        textDelta += delta.content;
        message.content += delta.content;
      }
      if (delta?.reasoning_content) {
        reasonDelta += delta.reasoning_content;
        message.reasoning_content = (message.reasoning_content || "") + delta.reasoning_content;
      }
      if (delta?.tool_calls) appendCalls(calls, delta.tool_calls);
      activity.stream(chunk.id, delta, calls.at(-1)?.function.name);
      if (choice?.finish_reason) finish = choice.finish_reason;
      if (!paint) paint = requestAnimationFrame(flush);
    }
    if (!finish) throw new Error("Chat stream had no finish reason");
    if (calls.length) {
      if (
        calls.some((call) => !call.id || !call.function.name) ||
        new Set(calls.map((call) => call.id)).size !== calls.length
      )
        throw new Error("Incomplete tool call identity");
      message.tool_calls = calls;
    }
    stats.finish = finish;
    entry.partial = calls.length > 0 && finish !== "tool_calls";
    if (entry.partial)
      entry.problem =
        "Generation ended at " + finish + ". Tool actions are unavailable for this partial turn.";
  } catch (error) {
    entry.problem = signal.aborted ? "Stopped · partial answer" : error.message;
    throw error;
  } finally {
    cancelAnimationFrame(paint);
    activity.end();
  }
  return entry;
}
async function submitTurn(mode = "send", spoken) {
  const voiceTurn = mode === "voice";
  if ((active && !voiceTurn) || unfinished().size) return;
  const continuation = mode === "continue" || mode === "replay";
  const pin = thread.checkpoint?.release ? undefined : thread.checkpoint;
  if (pin?.error) throw new Error("Choose Fork from history to continue this chat");
  if (mode === "continue") {
    const index = thread.entries.findLastIndex(
      (entry) => !entry.partial && entry.message?.tool_calls?.length,
    );
    const owner = index < (thread.importedCount || 0) ? models.value : thread.entries[index]?.model;
    if (catalog.get(owner)?.loom.endpoint !== chat)
      throw new Error("Select an available chat model to continue with tool results");
    if (models.value !== owner) {
      models.value = owner;
      selectModel();
    }
  }
  const model = catalog.get(models.value),
    route = model?.loom.endpoint;
  if (!modes.has(route)) throw new Error("Select a prepared model");
  if (pin && pin.model !== model.id)
    throw new Error("The pinned model is unavailable. Choose Fork from history to change models.");
  const draft = voiceTurn ? spoken : prompt.value,
    input = draft.trim(),
    files = voiceTurn ? [] : attachments;
  if (!continuation && !input && !files.length) return;
  if (mode === "send" && !acceptsFiles(model))
    throw new Error("The selected model cannot accept these files");
  const values = mode === "replay" ? { ...thread.settings, mcp: mcpSettings().mcp } : settings();
  const text = route === chat ? userText(input, values) : input;
  thread.settings = values;
  if (route === speech && !byId("voice").value.trim())
    throw new Error("Enter a voice ID from this deployment");
  const controller = voiceTurn ? active : new AbortController();
  inspectorToggle.ariaExpanded = "false";
  active = controller;
  controls();
  showFiles();
  notice("Working…");
  showLatest();
  let turn = beginReply(model),
    admitted = false;
  const before = thread.entries.length;
  try {
    let inputEntry = continuation
      ? mode === "continue" && thread.entries.at(-1)?.compaction
        ? userEntry("Continue.")
        : null
      : voiceTurn
        ? userEntry(text)
        : route === chat
          ? composerMessage(text)
          : { role: "user", text };
    let body, headers, response;
    if (route === chat) {
      let request = await chatBody(
        model,
        thread.entries,
        inputEntry,
        values,
        controller.signal,
        mode === "replay",
      );
      if (
        !pin &&
        mode !== "replay" &&
        (await autoCompact(model, values, request.body, controller.signal))
      ) {
        inputEntry ??= userEntry("Continue.");
        request = await chatBody(model, thread.entries, inputEntry, values, controller.signal);
      }
      body = request.body;
      headers = incognito ? {} : { "X-Loom-Session": thread.id };
      response = await post(
        chat,
        body,
        controller.signal,
        pin ? { ...headers, "X-Loom-Checkpoint": thread.id } : headers,
      );
      inputEntry = request.input;
      admitted = true;
    }
    if (inputEntry) {
      controller.signal.throwIfAborted();
      thread.entries.push(inputEntry);
      if (!continuation) {
        if (inputEntry.message && input !== text) inputEntry.draft = input;
        delete thread.pendingPeer;
        const title = (input || (files.length ? partName(files[0]) : "")).slice(0, 64);
        if (!thread.named && (!before || thread.parent?.count === before))
          thread.title = (thread.parent ? "Fork · " : "") + title;
        thread.model = model.id;
        if (!voiceTurn) {
          if (prompt.value === draft) prompt.value = "";
          attachments = [];
          showFiles();
        }
      }
      redraw();
      if (route !== chat) appendReply(turn.row);
    }
    if (route === chat) {
      const automatic =
        byId("auto-tools").checked && !(mode === "replay" && thread.parent?.action === "compare");
      for (;;) {
        const entry = await generateText(model, controller.signal, values, response, turn);
        if (pin) pin.release = true;
        const calls = entry.message.tool_calls;
        if (!automatic || entry.partial || !calls?.length) break;
        // Persist the complete call before any side effect. Restoring history
        // exposes unresolved calls without replaying a browser or model action.
        if (thread.checkpoint?.release) await releaseCheckpoint();
        else await remember();
        redraw();
        const results = [];
        const source = thread;
        for (const call of calls) {
          controller.signal.throwIfAborted();
          const tool = executableTool(entry, call);
          if (tool && (!tool.server || tool.auto))
            results.push(await runTool(entry, call, tool, controller.signal));
        }
        controller.signal.throwIfAborted();
        if (results.length !== calls.length) break;
        // Only these messages are new. Retain the prepared history, including
        // any encoded attachments, for every request in this user action.
        body.messages.push(entry.message, ...results);
        if (thread !== source || (await autoCompact(model, values, body, controller.signal))) {
          const request = await chatBody(
            model,
            thread.entries,
            userEntry("Continue."),
            values,
            controller.signal,
          );
          body = request.body;
          headers = incognito ? {} : { "X-Loom-Session": thread.id };
          thread.entries.push(request.input);
          await remember();
          redraw();
        }
        turn = beginReply(model);
        notice("Continuing with tool results…");
        response = await post(chat, body, controller.signal, headers);
      }
    } else if (route === images) {
      const image = await generateImage(model.id, text, controller.signal, values);
      thread.entries.push({
        role: "assistant",
        model: model.id,
        settings: values,
        image,
        imagePrompt: text,
        stats: turn.stats,
      });
      turn.stats.finish = "image";
    } else {
      await speak(model.id, text, values, controller.signal, (frames) => {
        turn.stats.frames = frames;
      });
      turn.stats.finish = "played";
      thread.entries.push({
        role: "assistant",
        model: model.id,
        settings: values,
        text: "Played " + turn.stats.frames + " PCM frames",
        stats: turn.stats,
      });
    }
    notice();
  } catch (error) {
    let problem = controller.signal.aborted ? "Stopped waiting" : error.message;
    if (pin && !pin.release) {
      if (!admitted && error.message.includes("does not extend the named checkpoint"))
        problem =
          "This input changed the cached prefix. Restore the previous settings or choose Fork from history.";
      if (!admitted && error.status === 404) pin.error = problem;
      if (admitted) {
        thread.entries.length = before;
        thread.previewSizes = previewPrefix(thread, before);
        if (!prompt.value) prompt.value = draft;
        if (!voiceTurn) attachments = files;
      }
    }
    if (route !== chat)
      thread.entries.push({
        role: "assistant",
        model: model.id,
        settings: values,
        problem,
        stats: turn.stats,
      });
    if (mode === "replay" && thread.entries.length === before) {
      thread.entries.push({
        message: { role: "assistant", content: "" },
        model: model.id,
        settings: values,
        partial: true,
        problem,
        stats: turn.stats,
        against: thread.pendingPeer,
      });
      delete thread.pendingPeer;
    }
    notice(problem, !controller.signal.aborted);
    if (voiceTurn) {
      if (!admitted && !prompt.value) prompt.value = draft;
      throw error;
    }
  } finally {
    turn.activity.end();
    try {
      if (thread.checkpoint?.release) await releaseCheckpoint();
      else await remember();
      await historyMenu();
    } finally {
      if (!voiceTurn) active = undefined;
      redraw();
      showFiles();
    }
  }
}
// The Stop button owns one foreground operation. A voice turn borrows this
// controller while the call owns its microphone and audio context.
async function withStop(starting, stopped, work) {
  if (active) return;
  const controller = new AbortController();
  active = controller;
  controls();
  notice(starting);
  try {
    await work(controller.signal);
  } catch (error) {
    if (!controller.signal.aborted) throw error;
    notice(stopped);
  } finally {
    active = undefined;
    controls();
  }
}
async function voiceChat() {
  if (active) return;
  const values = settings();
  if (!values.voice?.trim()) {
    byId("voice").focus();
    throw new Error("Enter a voice ID before starting a call");
  }
  return withStop("Opening microphone…", "Voice chat ended", async (signal) => {
    inspectorToggle.ariaExpanded = "false";
    const context = new AudioContext();
    let stream;
    try {
      await context.resume();
      stream = await microphone(signal);
      const [track] = stream.getAudioTracks();
      for (;;) {
        signal.throwIfAborted();
        track.enabled = true;
        notice("Voice chat · listening");
        const recording = await record(null, signal, () => {}, { stream, context });
        track.enabled = false;
        notice("Voice chat · transcribing");
        const text = await transcribe(values.recognizer, recording, signal);
        if (!text.trim()) continue;
        await submitTurn("voice", text);
        const reply = thread.entries.findLast((entry) => entry.message?.role === "assistant");
        if (reply?.message.tool_calls?.length || !reply?.message.content) {
          notice("Voice chat paused · continue in the transcript");
          break;
        }
        notice("Voice chat · speaking");
        await speak(values.reader, reply.message.content, values, signal, () => {}, context);
      }
    } finally {
      stream?.getTracks().forEach((track) => track.stop());
      await context.close();
    }
  });
}
async function readAloud(text) {
  if (active) return;
  if (!byId("voice").value.trim()) {
    byId("audio-settings").open = true;
    if (inspectorToggle.offsetWidth) inspectorToggle.ariaExpanded = "true";
    byId("voice").focus();
    throw new Error("Enter a voice ID, then choose Read aloud");
  }
  const values = settings();
  return withStop("Reading aloud…", "Playback stopped", async (signal) => {
    const frames = await speak(values.reader, text, values, signal, () => {});
    notice("Read aloud · " + frames.toLocaleString() + " frames played");
  });
}
async function dictate(file) {
  if (endRecording) {
    if (!file) endRecording();
    return;
  }
  if (active) return;
  const finish = file
    ? null
    : new Promise((resolve) => {
        endRecording = resolve;
      });
  const model = byId("recognizer").value;
  return withStop(
    file ? "Transcribing…" : "Opening microphone…",
    "Dictation cancelled",
    async (signal) => {
      try {
        const audio =
          file ||
          (await record(finish, signal, (frames) => notice("Recording · " + frames + " samples")));
        endRecording = undefined;
        controls();
        notice("Transcribing…");
        const text = await transcribe(model, audio, signal);
        prompt.setRangeText(text, prompt.selectionStart, prompt.selectionEnd, "end");
        prompt.focus();
        notice("Dictation inserted · " + model);
      } finally {
        endRecording = undefined;
      }
    },
  );
}
function exportChat() {
  return withStop("Preparing download…", "Export cancelled", async (signal) => {
    const link = await conversationDownload(thread, signal);
    notice("Conversation export · ");
    byId("notice").append(link);
    link.click();
  });
}
function compactChat() {
  const model = catalog.get(models.value);
  return withStop("Compacting conversation…", "Compaction stopped", (signal) =>
    compactConversation(model, settings(), signal),
  );
}
const commands = new Map();
for (const [name, id, run] of [
  ["/export", "export-chat", exportChat],
  ["/compact", "compact", compactChat],
]) {
  const control = byId(id);
  commands.set(name, control);
  control.onclick = guarded(async () => {
    await run();
    if (prompt.value.trim() === name) {
      prompt.value = "";
      draftChanged();
    }
  });
}
byId("compose").onsubmit = (event) => {
  event.preventDefault();
  const command = commands.get(prompt.value.trim());
  if (command) command.click();
  else guarded(submitTurn)();
};
byId("compact-tool").onchange = () => redraw();
byId("mcp").onclick = workspaceAction(async () => {
  const draft = await manageMcp();
  if (!draft) return;
  restoreMcp({ mcp: draft });
  thread.settings = { ...thread.settings, mcp: mcpSettings().mcp };
  await remember();
  await historyMenu();
  notice("MCP tools saved to chat");
});
byId("import-chat").onclick = () => byId("import-file").click();
inspectorToggle.onclick = () => {
  const open = inspectorToggle.ariaExpanded !== "true";
  inspectorToggle.ariaExpanded = String(open);
  if (open) byId("inspector").focus({ preventScroll: true });
};
prompt.oninput = draftChanged;
document.querySelector("aside").addEventListener("input", draftChanged);
byId("prompt-options").ontoggle = draftChanged;
async function startChat() {
  selectThread(newThread());
  await historyMenu();
  prompt.focus();
}
byId("new").onclick = workspaceAction(startChat);
const switchChat = workspaceAction((id) => (id === "+" ? startChat() : openChat(id)));
const rename = workspaceAction(async (name) => {
  thread.title = name;
  thread.named = true;
  await remember();
  await historyMenu();
});
function findHistory() {
  if (active) return;
  findChats(
    async (needle) =>
      (incognito
        ? [...privateChats.values()].filter(
            (row) =>
              row.title.toLowerCase().includes(needle) ||
              row.entries.some((entry) => entryText(entry).toLowerCase().includes(needle)),
          )
        : await searchSessions(database, project.id, needle)
      ).sort(chatOrder),
    switchChat,
    chatTitle,
  );
}
byId("threads").onchange = (event) => {
  const choice = event.target.value;
  event.target.value = thread.id;
  if (choice === "-") askDelete(false);
  else if (choice === "find") findHistory();
  else if (choice === "rename") askName("Rename chat", thread.title, "Rename", rename);
  else if (choice === "pin")
    workspaceAction(async () => {
      thread.pinned = !thread.pinned;
      await remember();
      await historyMenu();
    })();
  else if (choice === "import") byId("import-chat").click();
  else switchChat(choice);
};
function chooseFiles(id, accept) {
  const picker = byId(id);
  picker.onchange = () => {
    const files = [...picker.files];
    picker.value = "";
    if (files.length) accept(files);
  };
}
chooseFiles(
  "import-file",
  workspaceAction(async ([file]) => {
    notice("Reading " + file.name + "…");
    const session = await conversationImport(file, project.id);
    if (incognito) privateChats.set(session.id, session);
    else await saveSession(database, session);
    selectThread(session);
    await historyMenu();
    notice("Imported · " + session.title);
  }),
);
async function selectProject(id) {
  await projectMenu(id);
  selectThread(newThread());
  await historyMenu();
}
const switchProject = workspaceAction(selectProject);
byId("projects").onchange = (event) => {
  const id = event.target.value;
  event.target.value = project.id;
  if (id === "-") askDelete(true);
  else if (id === "+") askName("New project", "", "Create", newProject, 90);
  else switchProject(id);
};
const newProject = workspaceAction(async (name) => {
  await selectProject((await createProject(database, name)).id);
  prompt.focus();
});
async function releaseDeletedPins() {
  const pending = await pendingReleases(database);
  const failures = [];
  for (const pin of pending) {
    try {
      await checkpointWork(pin.id, async () => {
        await checkpoint("DELETE", pin.model, pin.id);
        await finishRelease(database, pin.id);
      });
    } catch (error) {
      failures.push(error.message);
    }
  }
  if (failures.length) {
    notice("KV checkpoint release pending · " + failures[0] + " ", true);
    byId("notice").append(
      button(
        "Retry release",
        workspaceAction(async () => {
          notice();
          await releaseDeletedPins();
        }),
      ),
    );
  }
}
const deleteSelected = workspaceAction(async (projectId, chatId, message) => {
  const listed = [
    ...byId("threads").querySelectorAll("#pinned-chats option, #recent-chats option"),
  ].map((option) => option.value);
  const index = listed.indexOf(chatId);
  const adjacent = chatId && (listed[index + 1] || listed[index - 1]);
  if (incognito) privateChats.delete(chatId);
  else {
    await deleteFromWorkspace(database, projectId, chatId);
    await projectMenu(projectId);
  }
  const next =
    adjacent && (incognito ? privateChats.get(adjacent) : await readSession(database, adjacent));
  if (next && next.title === thread.title && chatLabels.has(next.id))
    message += " · Now showing " + chatTitle(next);
  selectThread(next || newThread());
  for (const [id, draft] of drafts)
    if (chatId ? id === chatId : draft.project === projectId) drafts.delete(id);
  notice(message);
  await restoreChat();
  prompt.focus();
  if (!incognito) await releaseDeletedPins();
});
function askDelete(wholeProject) {
  const projectId = project?.id,
    chatId = wholeProject ? undefined : thread.id;
  const draft = !wholeProject && !byId("draft-chat").hidden;
  const name = wholeProject ? project.name : chatTitle(thread);
  const action = wholeProject ? "Delete project" : draft ? "Discard draft" : "Delete chat";
  const title = draft ? "Discard this draft?" : "Delete “" + name + "”?";
  let detail = wholeProject
    ? "All chats and forks in this project will be removed from this browser."
    : draft
      ? "The unsent message and its attachments will be cleared."
      : "Forks keep the messages they inherited from this chat.";
  if (chatLabels.get(chatId)?.shared)
    detail = "Several chats share this title. " + detail;
  const message = draft
    ? "Draft discarded"
    : (wholeProject ? "Project deleted · " : "Chat deleted · ") + name;
  ask(title, [element("p", detail)], action, () => deleteSelected(projectId, chatId, message), true);
}
byId("save-settings").onclick = guarded(async () => {
  const selected = thread;
  thread.settings = settings();
  thread.model = models.value;
  await remember();
  if (thread === selected) {
    await historyMenu();
    notice(incognito ? "Settings applied in memory" : "Session settings saved");
  }
});
async function changeProject(change) {
  if (incognito) change(project);
  else project = await updateProject(database, project.id, change);
}
byId("defaults").onclick = workspaceAction(async () => {
  const values = settings();
  await changeProject((row) => {
    row.settings = values;
  });
  notice("Project defaults saved");
});
function presetMenu(selected = "") {
  const select = byId("presets");
  select.replaceChildren(
    new Option("Choose a preset…", ""),
    ...(project.presets || []).map((preset) => new Option(preset.name, preset.name)),
  );
  select.value = selected;
}
byId("presets").onchange = () => {
  const preset = project.presets?.find((item) => item.name === byId("presets").value);
  if (preset) {
    restoreSettings(preset.settings);
    notice("Applied · " + preset.name);
  }
  controls();
};
byId("new-preset").onclick = guarded(() => {
  const values = settings();
  askName(
    "Save named preset",
    "",
    "Save",
    workspaceAction(async (name) => {
      await changeProject((row) => {
        const presets = (row.presets ??= []);
        if (presets.some((item) => item.name === name))
          throw new Error("This preset name is already in use");
        presets.push({ name, settings: values });
      });
      presetMenu(name);
      notice("Preset saved · " + name);
    }),
  );
});
byId("remove-preset").onclick = () => {
  const name = byId("presets").value;
  ask(
    "Remove preset “" + name + "”?",
    [],
    "Remove",
    workspaceAction(async () => {
      await changeProject((row) => {
        row.presets = (row.presets || []).filter((preset) => preset.name !== name);
      });
      presetMenu();
      notice("Preset removed");
    }),
  );
};
const attachFiles = workspaceAction(async (files) => {
  const model = catalog.get(models.value);
  if (model?.loom.endpoint !== chat) throw new Error("Select a chat model to attach files");
  const parts = await Promise.all(
    files.map((file) => attachment(file, model.input_modalities.includes("image"))),
  );
  attachments.push(...parts);
  showFiles();
  notice();
});
chooseFiles("upload", attachFiles);
for (const type of ["dragover", "drop"])
  byId("compose").addEventListener(type, (event) => {
    if (!event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    if (type === "drop") attachFiles([...event.dataTransfer.files]);
  });
byId("mic").onclick = guarded(() => dictate());
chooseFiles(
  "recording",
  guarded(([file]) => dictate(file)),
);
byId("voice-chat").onclick = guarded(voiceChat);
byId("refresh").onclick = guarded(discover);
byId("stop").onclick = () => active?.abort?.();
models.onchange = () => {
  selectModel();
  showFiles();
};
document.addEventListener("keydown", (event) => {
  if (document.querySelector("dialog[open]")) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    findHistory();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !byId("send").disabled) {
    event.preventDefault();
    byId("compose").requestSubmit();
  } else if (event.key === "Escape") {
    if (active) active.abort?.();
    else if (inspectorToggle.ariaExpanded === "true" && inspectorToggle.offsetWidth) {
      inspectorToggle.ariaExpanded = "false";
      inspectorToggle.focus();
    }
  }
});

guarded(async () => {
  let missing;
  if (incognito) {
    project = { id: "private", name: "Incognito", settings: {} };
    document.querySelector("main").spellcheck = false;
    for (const field of document.querySelectorAll("input, textarea")) field.autocomplete = "off";
    for (const key of ["project-picker", "defaults"]) byId(key).hidden = true;
    thread = newThread();
  } else {
    database = await openWorkspace();
    const [kind, id] = location.hash.slice(1).split("/");
    if (kind === "session" && id) {
      thread = await readSession(database, id);
      if (!thread) missing = "chat";
    }
    await projectMenu(thread?.project ?? (kind === "project" ? id : "local"));
    if (kind === "project" && id && id !== project.id) missing = "project";
    thread ??= newThread();
  }
  await loadMcp(database);
  await discover();
  selectThread(thread);
  await restoreChat();
  if (incognito) notice("Incognito · this tab keeps the conversation in memory");
  if (missing) notice("This " + missing + " no longer exists.");
  const main = document.querySelector("main");
  main.inert = false;
  main.setAttribute("aria-busy", "false");
  if (!incognito) await releaseDeletedPins();
})();
window.addEventListener("pagehide", () => {
  active?.abort?.();
  for (const cancel of [promptTokens, systemTokens, templateTokens]) cancel(null, null);
});
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  if (incognito) location.reload();
  else draftChanged();
});
