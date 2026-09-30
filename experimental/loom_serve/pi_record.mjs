// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import { closeSync, openSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";

// Attaches before the first prompt. The caller retains pi's native session file
// and calls finish after its last prompt and waitForIdle (not just agent_end).
// Both ordinary turns and compaction use the public Agent.streamFunction.
export function recordPiSession(session, path, piVersion) {
  if (!session.isIdle || !session.model) {
    throw new Error("Recording requires an idle session with a model");
  }
  if (session.settingsManager.getRetrySettings().enabled ||
      session.settingsManager.getProviderRetrySettings().maxRetries !== 0) {
    throw new Error("Disable agent and provider retries before recording");
  }
  const model = session.model;
  const compaction = session.settingsManager.getCompactionSettings();
  const configuration = {
    pi_version: piVersion,
    context_window_tokens: model.contextWindow,
    max_output_tokens: model.maxTokens,
    thinking_level: session.thinkingLevel,
    compaction_enabled: compaction.enabled,
    reserve_tokens: compaction.reserveTokens,
    keep_recent_tokens: compaction.keepRecentTokens,
  };
  const file = openSync(path, "wx", 0o600);
  const write = (entry) => writeFileSync(file, JSON.stringify(entry) + "\n");
  // A monotonic process clock anchored to Unix time permits cross-client
  // alignment without making request durations depend on wall-clock steps.
  const now = () => Math.floor((performance.timeOrigin + performance.now()) * 1000);
  try {
    write({ format: "loom-pi-calls-v1", provider: model.provider,
      model: model.id, configuration });
  } catch (error) {
    closeSync(file);
    throw error;
  }
  let purpose = "agent";
  let count = 0;
  let active = false;
  let failed = false;
  let closed = false;
  const original = session.agent.streamFunction;
  const recordError = (index, error) => {
    failed = true;
    active = false;
    try {
      write({ type: "request_error", index, end_us: now() });
    } catch (recordingError) {
      throw new AggregateError([error, recordingError], "Provider and recording failed");
    }
    throw error;
  };
  const unsubscribe = session.subscribe((event) => {
    if (event.type === "compaction_start") purpose = "compaction";
    if (event.type === "compaction_end") {
      failed ||= !!event.errorMessage || event.aborted;
      purpose = "agent";
    }
  });
  session.agent.streamFunction = async (requestModel, context, options) => {
    if (active || requestModel.provider !== model.provider ||
        requestModel.id !== model.id || requestModel.contextWindow !== model.contextWindow) {
      failed = true;
      throw new Error("Recording requires sequential calls and a fixed client model");
    }
    active = true;
    const index = count++;
    try {
      write({ type: "request_start", index, start_us: now(), purpose,
        api: requestModel.api });
      const stream = await original(requestModel, context, options);
      // Observe completion without consuming the iterator used by pi's agent
      // loop. Compaction consumes result() directly; both paths share this one
      // promise and therefore one completion record.
      const result = stream.result().then((message) => {
        failed ||= message.stopReason === "error" || message.stopReason === "aborted";
        const usage = message.usage;
        write({ type: "request_end", index, end_us: now(),
          stop_reason: message.stopReason,
          usage: { input: usage.input, output: usage.output,
            cacheRead: usage.cacheRead, cacheWrite: usage.cacheWrite,
            totalTokens: usage.totalTokens } });
        active = false;
        return message;
      }).catch((error) => recordError(index, error));
      return {
        async *[Symbol.asyncIterator]() {
          yield* stream;
          await result;
        },
        result: () => result,
      };
    } catch (error) {
      return recordError(index, error);
    }
  };
  return {
    finish() {
      if (closed || active || !session.isIdle || purpose !== "agent" || failed || !count) {
        throw new Error("Recording is failed, unfinished, empty, or already closed");
      }
      write({ type: "recording_end", requests: count });
      this.close();
    },
    // Closes an unsuccessful recording without a successful end marker.
    // Abort/wait for pi first so no provider callback can write to a closed file.
    close() {
      if (active || !session.isIdle || purpose !== "agent") {
        throw new Error("Wait for the session before closing its recording");
      }
      if (!closed) {
        closed = true;
        session.agent.streamFunction = original;
        unsubscribe();
        closeSync(file);
      }
    },
  };
}
