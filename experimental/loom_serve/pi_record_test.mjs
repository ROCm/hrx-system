// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exercises the real pi SDK with a scripted provider, without network or GPU.
// Arguments: installed pi package directory and a new private output directory.
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { recordPiSession } from './pi_record.mjs';

if (process.argv.length !== 4) {
  throw new Error('Usage: node pi_record_test.mjs PI_PACKAGE NEW_OUTPUT_DIRECTORY');
}
const packagePath = resolve(process.argv[2]);
const outputPath = resolve(process.argv[3]);
const pi = await import(pathToFileURL(join(packagePath, 'dist/index.js')));
const { AssistantMessageEventStream } = await import(pathToFileURL(join(
  packagePath, 'node_modules/@earendil-works/pi-ai/dist/utils/event-stream.js')));

mkdirSync(outputPath, { recursive: false });
const results = [];
for (const contextWindow of [8192, 16384]) {
  const agentDir = join(outputPath, String(contextWindow));
  mkdirSync(agentDir);
  const settingsManager = pi.SettingsManager.inMemory({
    compaction: { enabled: true, reserveTokens: 2048, keepRecentTokens: 256 },
    retry: { enabled: false, provider: { maxRetries: 0 } },
  });
  const resourceLoader = new pi.DefaultResourceLoader({
    cwd: process.cwd(), agentDir, settingsManager,
    noExtensions: true, noSkills: true, noPromptTemplates: true,
    noThemes: true, noContextFiles: true,
    systemPrompt: 'CPU fixture: no actual model or file tools.',
  });
  await resourceLoader.reload();
  const model = {
    id: 'fixture', name: 'CPU fixture', provider: 'fixture',
    api: 'openai-completions', reasoning: false, input: ['text'],
    contextWindow, maxTokens: 1024,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  };
  writeFileSync(join(agentDir, 'models.json'), JSON.stringify({ providers: {
    fixture: { api: model.api, baseUrl: 'http://127.0.0.1:1/v1',
      apiKey: 'cpu-fixture', models: [{ id: model.id, name: model.name,
        contextWindow, maxTokens: model.maxTokens,
        reasoning: false, input: ['text'], cost: model.cost }] },
  } }), { flag: 'wx' });
  const sessionManager = pi.SessionManager.create(process.cwd(), agentDir);
  const { session } = await pi.createAgentSession({
    cwd: process.cwd(), agentDir, model, settingsManager,
    resourceLoader, sessionManager, noTools: 'all', thinkingLevel: 'off',
  });
  let purpose = 'agent';
  let agentCalls = 0;
  let failureReason;
  const calls = [];
  const events = [];
  session.subscribe(event => {
    events.push(event.type);
    if (event.type === 'compaction_start') purpose = 'compaction';
    if (event.type === 'compaction_end') {
      assert.equal(event.errorMessage, undefined);
      purpose = 'agent';
    }
  });
  session.agent.streamFunction = (model, context, options) => {
    if (failureReason === 'throw') throw new Error('scripted provider exception');
    if (failureReason === 'reject') {
      return {
        async *[Symbol.asyncIterator]() {},
        result: () => Promise.reject(new Error('scripted result exception')),
      };
    }
    if (purpose === 'agent') ++agentCalls;
    const input = purpose === 'agent' && agentCalls === 2 ? 7000 : 200;
    const output = purpose === 'compaction' ? 20 : 100;
    const text = purpose === 'compaction' ? 'COMPACTED_EVIDENCE' :
      agentCalls === 2 ? 'recent evidence '.repeat(100) : 'agent response';
    calls.push({ purpose, contextWindow: model.contextWindow,
      context: structuredClone(context), options: {
        maxTokens: options?.maxTokens, cacheRetention: options?.cacheRetention,
      }, input, output });
    const message = {
      role: 'assistant', api: model.api, provider: model.provider, model: model.id,
      timestamp: Date.now(), stopReason: failureReason ?? 'stop',
      content: [{ type: 'text', text }],
      usage: { input, output, cacheRead: 0, cacheWrite: 0,
        totalTokens: input + output,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    };
    const stream = new AssistantMessageEventStream();
    if (failureReason) {
      message.errorMessage = 'Scripted provider failure';
      stream.push({ type: 'error', reason: failureReason, error: message });
    } else {
      stream.push({ type: 'done', reason: 'stop', message });
    }
    stream.end();
    return stream;
  };
  const recorder = recordPiSession(session, join(agentDir, 'provider-calls.jsonl'), pi.VERSION);
  assert.throws(() => recorder.finish(), /empty/);
  await session.prompt('First investigation: ORIGINAL_HISTORY');
  await session.waitForIdle();
  await session.prompt('Second investigation: prefix to preserve with the recent suffix.');
  await session.waitForIdle();
  const automaticCompactions = events.filter(event => event === 'compaction_end').length;
  assert.equal(automaticCompactions, contextWindow === 8192 ? 1 : 0);
  if (contextWindow === 16384) await session.compact();
  assert.equal(calls.filter(call => call.purpose === 'compaction').length, 2);
  await session.prompt('Continue after compaction using the checkpoint.');
  await session.waitForIdle();
  assert.match(JSON.stringify(calls.at(-1).context), /COMPACTED_EVIDENCE/);
  assert.doesNotMatch(JSON.stringify(calls.at(-1).context), /ORIGINAL_HISTORY/);
  recorder.finish();
  const recorded = readFileSync(join(agentDir, 'provider-calls.jsonl'), 'utf8');
  assert.doesNotMatch(recorded, /ORIGINAL_HISTORY|COMPACTED_EVIDENCE|recent evidence/);
  const entries = recorded.trim().split('\n').map(line => JSON.parse(line));
  assert.equal(entries[0].configuration.context_window_tokens, contextWindow);
  assert.equal(entries.at(-1).requests, 5);
  assert.deepEqual(entries.filter(entry => entry.type === 'request_start')
    .map(entry => entry.purpose), ['agent', 'agent', 'compaction', 'compaction', 'agent']);
  assert.equal(entries.filter(entry => entry.type === 'request_end').length, 5);
  assert.throws(() => recorder.finish(), /closed/);
  const summary = { contextWindow, automaticCompactions,
    purposes: calls.map(call => call.purpose),
    sessionFile: sessionManager.getSessionFile(), events };
  writeFileSync(join(agentDir, 'calls.json'), JSON.stringify(calls, null, 2), { flag: 'wx' });
  results.push(summary);
  for (const reason of ['error', 'aborted', 'throw', 'reject']) {
    failureReason = reason;
    const path = join(agentDir, `failed-${reason}.jsonl`);
    const failedRecorder = recordPiSession(session, path, pi.VERSION);
    await session.prompt('Exercise a failed provider call.');
    await session.waitForIdle();
    assert.throws(() => failedRecorder.finish(), /failed/);
    failedRecorder.close();
    const failureEntries = readFileSync(path, 'utf8').trim().split('\n')
      .map(line => JSON.parse(line));
    assert.equal(failureEntries.length, 3);
    assert.notEqual(failureEntries.at(-1).type, 'recording_end');
  }
  session.dispose();
}
writeFileSync(join(outputPath, 'result.json'), JSON.stringify(results, null, 2), { flag: 'wx' });
console.log(JSON.stringify(results, null, 2));
