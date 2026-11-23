// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { entryText } from "./content.js";

// The session index only reads headers. Image blobs live in append-only message
// records and do not move when a later turn or a project setting is saved.
export function identity() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}
export function emptySession(project, settings = {}) {
  return {
    id: identity(),
    title: "Untitled chat",
    project,
    settings: { ...settings },
    entries: [],
    saved: 0,
    revision: 0,
    updated: Date.now(),
  };
}
export function previewPrefix(session, count) {
  return (
    session.previewSizes &&
    Object.fromEntries(
      Object.entries(session.previewSizes).filter(([key]) => parseInt(key, 10) < count),
    )
  );
}
export function forkSession(parent, count, model, settings) {
  const child = emptySession(parent.project, settings);
  child.parent = { id: parent.id, title: parent.title, count };
  child.title = "Fork · " + parent.title;
  child.model = model;
  child.entries = parent.entries.slice(0, count);
  child.saved = count;
  child.previewSizes = previewPrefix(parent, count);
  return child;
}
function result(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}
function committed(tx) {
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onabort = () => reject(tx.error || new Error("Storage transaction aborted"));
  });
}
const readStore = (db, name) => db.transaction(name).objectStore(name);
function stores(tx) {
  return Object.fromEntries(
    Array.from(tx.objectStoreNames, (name) => [name, tx.objectStore(name)]),
  );
}
async function write(db, names, update) {
  const tx = db.transaction(names, "readwrite");
  const done = committed(tx);
  const operation = (async () => {
    try {
      return await update(stores(tx), tx);
    } catch (error) {
      // A request error may already have aborted the transaction.
      try {
        tx.abort();
      } catch (state) {
        if (state.name !== "InvalidStateError") throw state;
      }
      throw error;
    }
  })();
  const [value] = await Promise.all([operation, done]);
  return value;
}
const entryRange = (id, start = 0, end = Infinity) =>
  IDBKeyRange.bound([id, start], [id, end], false, true);
// Every preview key is a string, which sorts before an array in IndexedDB.
const previewRange = (id) => IDBKeyRange.bound([id], [id, []], false, true);
const localProject = (id) => ({ id, name: "Local", settings: {} });
export async function openWorkspace() {
  const prefix = new URL("../", document.baseURI).pathname;
  const req = indexedDB.open("loom-conversations@" + prefix, 3);
  req.onupgradeneeded = (event) => {
    const db = req.result;
    if (event.oldVersion === 0) {
      db.createObjectStore("projects", { keyPath: "id" }).add(localProject("local"));
      db.createObjectStore("sessions", { keyPath: "id" }).createIndex("project", "project");
      db.createObjectStore("entries", { keyPath: ["session", "position"] });
    }
    if (event.oldVersion < 2) {
      req.transaction.objectStore("sessions").createIndex("parent", "parent.id");
      db.createObjectStore("releases", { keyPath: "id" });
    }
    if (event.oldVersion < 3)
      for (const name of ["previewSizes", "previewEdits"])
        db.createObjectStore(name, { keyPath: ["session", "key"] });
  };
  const db = await result(req);
  db.onversionchange = () => db.close();
  await applyPreviewEdits(db);
  return db;
}
export const projects = (db) => result(readStore(db, "projects").getAll());
export function createProject(db, name) {
  const project = { id: identity(), name, settings: {} };
  return write(db, "projects", ({ projects }) => {
    projects.add(project);
    return project;
  });
}
export function updateProject(db, id, change) {
  return write(db, "projects", async ({ projects: store }) => {
    const project = await result(store.get(id));
    if (!project) throw new Error("This project was deleted in another tab.");
    change(project);
    store.put(project);
    return project;
  });
}
export async function sessions(db, project) {
  const rows = await result(readStore(db, "sessions").index("project").getAll(project));
  return rows.filter((row) => !row.deleted);
}
export async function readSession(db, id) {
  await applyPreviewEdits(db, previewRange(id));
  const {
    sessions,
    entries: records,
    previewSizes,
  } = stores(db.transaction(["sessions", "entries", "previewSizes"]));
  const parts = [];
  let selected,
    limit = Infinity;
  // Each branch stores only its suffix, indexed at absolute transcript
  // positions. An ancestor may itself have been forked before its own suffix.
  while (id) {
    const [row, portion, sizes] = await Promise.all([
      result(sessions.get(id)),
      result(records.getAll(entryRange(id, 0, limit))),
      selected ? undefined : result(previewSizes.getAll(previewRange(id))),
    ]);
    if (!selected && (!row || row.deleted)) return;
    if (!row) throw new Error("Fork history is missing");
    if (!selected) {
      selected = row;
      if (sizes.length)
        selected.previewSizes = Object.fromEntries(sizes.map(({ key, size }) => [key, size]));
    }
    parts.push(portion.map((record) => record.value));
    if (!row.parent?.count) break;
    limit = Math.min(limit, row.parent.count);
    id = row.parent.id;
  }
  const entries = parts.reverse().flat();
  return { ...selected, entries, saved: entries.length };
}

// Scan each owned suffix once on an explicit history search. A child inherits
// a match only when its immutable prefix still includes that position.
export async function searchSessions(db, project, needle) {
  const { sessions, entries } = stores(db.transaction(["sessions", "entries"]));
  const headers = await result(sessions.index("project").getAll(project));
  const rows = new Map(headers.map((row) => [row.id, row]));
  const hits = new Map();
  if (needle) {
    await Promise.all(
      headers.map(
        (row) =>
          new Promise((resolve, reject) => {
            const request = entries.openCursor(entryRange(row.id));
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
              const cursor = request.result;
              if (!cursor) {
                resolve();
                return;
              }
              if (entryText(cursor.value.value).toLowerCase().includes(needle)) {
                hits.set(row.id, cursor.value.position);
                resolve();
              } else cursor.continue();
            };
          }),
      ),
    );
  }
  return headers.filter((row) => {
    if (row.deleted) return false;
    if (row.title.toLowerCase().includes(needle)) return true;
    let limit = Infinity;
    for (let part = row; part; part = rows.get(part.parent?.id)) {
      if (hits.get(part.id) < limit) return true;
      if (!part.parent?.count) break;
      limit = Math.min(limit, part.parent.count);
    }
    return false;
  });
}
const publishing = new WeakMap();
export async function saveSession(db, current) {
  const { entries, saved, revision, previewSizes, ...fields } = current;
  const header = { ...fields, revision: revision + 1, updated: Date.now() };
  const end = entries.length;
  const appended = write(
    db,
    ["projects", "sessions", "entries", "previewSizes"],
    async ({ projects, sessions: summaries, entries: records, previewSizes }, tx) => {
      const stored = await result(summaries.get(current.id));
      if (stored?.deleted || (stored?.revision ?? 0) !== revision) {
        throw new Error(
          "This chat changed in another tab. Use /export to keep this version before reopening it.",
        );
      }
      if (!revision) {
        if (!(await result(projects.getKey(current.project))))
          throw new Error("This project was deleted in another tab.");
        if (current.parent) {
          const parent = await result(summaries.get(current.parent.id));
          if (!parent || parent.deleted)
            throw new Error("The chat you are forking was deleted in another tab.");
        }
      }
      for (let position = saved; position < end; ++position) {
        records.add({ session: current.id, position, value: entries[position] });
      }
      if (current.previewSizes && (!revision || saved < end)) {
        for (const [key, size] of Object.entries(current.previewSizes)) {
          const position = parseInt(key, 10);
          if ((!revision || position >= saved) && position < end)
            previewSizes.put({ session: current.id, key, size });
        }
      }
      summaries.put(header);
      tx.commit();
      publishing.set(current, end);
    },
  );
  try {
    await appended;
  } finally {
    publishing.delete(current);
  }
  current.saved = end;
  current.revision = header.revision;
  current.updated = header.updated;
}

// Another tab can delete a chat while its preview is open. Consume the edit
// and update the live owner's layout in a single transaction. An interrupted
// consumer leaves the edit for the next open, including an explicit reset.
function applyPreviewEdits(db, address) {
  return write(
    db,
    ["sessions", "previewSizes", "previewEdits", "entries"],
    async ({ sessions, entries, previewEdits: edits, previewSizes: sizes }) => {
      for (const { session, key, size, position } of await result(edits.getAll(address))) {
        const row = await result(sessions.get(session));
        const target = [session, key];
        if (
          row &&
          !row.deleted &&
          (position === undefined || (await result(entries.getKey([session, position]))))
        ) {
          if (size) sizes.put({ session, key, size });
          else sizes.delete(target);
        }
        edits.delete(target);
      }
    },
  );
}
// Submit the completed gesture before returning to the browser. Navigation
// can destroy the document before even one IndexedDB read callback runs.
export async function savePreviewSize(db, session, key, size) {
  const sizes = (session.previewSizes ??= {});
  if (size) sizes[key] = size;
  else delete sizes[key];
  if (!db) return;
  const position = parseInt(key, 10);
  // Streaming entries carry their in-memory sizes into the message commit.
  // Once it is submitted, the edit must follow it even during navigation.
  if (position >= (publishing.get(session) ?? session.saved)) return;
  const tx = db.transaction("previewEdits", "readwrite"),
    done = committed(tx);
  tx.objectStore("previewEdits").put({
    session: session.id,
    key,
    size,
    position: position >= session.saved ? position : undefined,
  });
  tx.commit();
  await done;
  await applyPreviewEdits(db, [session.id, key]);
}

// A deleted ancestor owns only the prefix its direct children still reference.
// Shrinking that prefix can release entries from a previously deleted ancestor.
async function removeChat(headers, entries, row) {
  while (row) {
    const children = await result(headers.index("parent").getAll(row.id));
    const keep = children.reduce((end, child) => Math.max(end, child.parent.count), 0);
    entries.delete(entryRange(row.id, keep));
    const parent = row.parent;
    if (keep) {
      const prefix = { id: row.id, project: row.project, deleted: true };
      if (parent) prefix.parent = { id: parent.id, count: Math.min(keep, parent.count) };
      headers.put(prefix);
    } else headers.delete(row.id);
    row = parent && (await result(headers.get(parent.id)));
    if (!row?.deleted) break;
  }
}
// Local removal and the receipt for any native pin commit together. A lost
// connection can leave only an idempotent DELETE to replay on the next visit.
export function deleteFromWorkspace(db, project, chat) {
  return write(
    db,
    ["projects", "sessions", "entries", "releases", "previewSizes", "previewEdits"],
    async ({ projects, sessions: headers, entries, releases, previewSizes, previewEdits }) => {
      const rows = chat
        ? [await result(headers.get(chat))]
        : await result(headers.index("project").getAll(project));
      for (const row of rows) {
        if (!row) continue; // An unsaved draft has no storage to reclaim.
        previewSizes.delete(previewRange(row.id));
        previewEdits.delete(previewRange(row.id));
        if (row.checkpoint && !row.checkpoint.absent)
          releases.put({ id: row.id, model: row.checkpoint.model });
        if (chat) await removeChat(headers, entries, row);
        else {
          entries.delete(entryRange(row.id));
          headers.delete(row.id);
        }
      }
      if (!chat) {
        projects.delete(project);
        if (!(await result(projects.count()))) projects.add(localProject(identity()));
      }
    },
  );
}
export const pendingReleases = (db) => result(readStore(db, "releases").getAll());
export const finishRelease = (db, id) =>
  write(db, "releases", ({ releases }) => {
    releases.delete(id);
  });
export function recordOrphanPin(db, id, model) {
  return write(db, ["sessions", "releases"], async ({ sessions, releases }) => {
    const owner = await result(sessions.get(id));
    if (owner && !owner.deleted) return false;
    releases.put({ id, model });
    return true;
  });
}
