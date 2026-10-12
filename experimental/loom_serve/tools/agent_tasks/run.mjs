// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bounded real-pi source-repair experiment. The model writes every repair.
import { copyFileSync, mkdirSync, readFileSync, writeFileSync, openSync, closeSync } from "node:fs";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
if (process.argv.length !== 8) {
  throw new Error("Expected: run.mjs PI_PACKAGE OUTPUT_DIR BASE_URL MODEL TASK_COUNT ENGINE");
}
const [packagePath, outputArgument, baseUrl, modelName, countArgument, engine] = process.argv.slice(2);
const count = Number(countArgument);
const tasks = JSON.parse(readFileSync(join(root, "tasks.json"), "utf8")).slice(0, count);
if (!Number.isInteger(count) || count < 1 || count > 8 || tasks.length !== count) {
  throw new Error("Expected: run.mjs PI_PACKAGE OUTPUT_DIR BASE_URL MODEL TASK_COUNT ENGINE (one to eight tasks)");
}
if (!["loom", "llama", "vllm"].includes(engine)) throw new Error("Need an explicit serving engine");
const output = resolve(outputArgument);
mkdirSync(output);
const pi = await import(pathToFileURL(join(resolve(packagePath), "dist/index.js")));
const piVersion = JSON.parse(readFileSync(join(resolve(packagePath), "package.json"), "utf8")).version;
const { recordPiSession } = await import(pathToFileURL(join(root, "../pi_record.mjs")));
const save = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2) + "\n", { flag: "wx", mode: 0o600 });
const emit = value => process.stdout.write(JSON.stringify({ unix_ms: Date.now(), ...value }) + "\n");
const equal = (actual, expected) => Array.isArray(expected) ? Array.isArray(actual) && actual.length === expected.length && actual.every((value, index) => equal(value, expected[index])) : typeof actual === "number" && Math.abs(actual - expected) <= 1e-6 * Math.max(1, Math.abs(expected));
const started = performance.now();
save(join(output, "manifest.json"), { protocol: "pi-numerical-repair-v1", pi_version: piVersion, node: process.version, baseUrl, modelName, count, engine, lease: process.env.BENCHMARK_LOCK_LEASE_ID ?? null,
  task_sha256: createHash("sha256").update(readFileSync(join(root, "tasks.json"))).digest("hex"),
  evaluator_sha256: createHash("sha256").update(readFileSync(join(root, "check.py"))).digest("hex"),
  script_sha256: createHash("sha256").update(readFileSync(fileURLToPath(import.meta.url))).digest("hex"),
  maximum_tools: 10, maximum_requests: 8, context_tokens: 8192, maximum_output_tokens: 384 });
copyFileSync(join(root, "tasks.json"), join(output, "tasks.json"));
const system = "You are repairing a small numerical utility using real file tools. Read task.md and candidate.py, make a focused repair to candidate.py, and run python3 check.py. Preserve the function signature. Only candidate.py may be modified. The only permitted shell command is python3 check.py. The math module is supplied to the function: no imports, files, network, classes or introspection in candidate code. Use at most ten tools. After the tests pass, reply only DONE. Keep intermediate prose minimal.";

async function run(task, index) {
  const directory = join(output, task.id);
  mkdirSync(directory);
  const agentDirectory = join(directory, "agent");
  mkdirSync(agentDirectory);
  writeFileSync(join(directory, "candidate.py"), task.candidate, { flag: "wx" });
  writeFileSync(join(directory, "task.md"), task.contract + "\n", { flag: "wx" });
  writeFileSync(join(directory, "check.py"), `import subprocess, sys\nsys.exit(subprocess.call([sys.executable, ${JSON.stringify(join(root, "check.py"))}, ${JSON.stringify(task.id)}, "candidate.py"]))\n`, { flag: "wx" });
  const initial = spawnSync("python3", [join(root, "check.py"), task.id, join(directory, "candidate.py")], { encoding: "utf8" });
  const initialResult = JSON.parse(initial.stdout);
  save(join(directory, "initial-evaluation.json"), initialResult);
  if (initial.status !== 1 || initialResult.error || initialResult.passed) throw new Error("Task seed must fail a numerical oracle, not the harness contract");
  const sessionKey = `${output.split("/").at(-1)}-${task.id}`;
  save(join(agentDirectory, "models.json"), { providers: { local: {
    api: "openai-completions", baseUrl, apiKey: "local",
    headers: { "X-Loom-Session": sessionKey },
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false, supportsStore: false, supportsStrictMode: false, maxTokensField: "max_tokens" },
    models: [{ id: modelName, name: "Comparison model", reasoning: false, input: ["text"], contextWindow: 8192, maxTokens: 384, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
  } } });
  const settingsManager = pi.SettingsManager.inMemory({ httpIdleTimeoutMs: 0, compaction: { enabled: false }, retry: { enabled: false, provider: { maxRetries: 0 } }, enableInstallTelemetry: false, enableAnalytics: false });
  const resourceLoader = new pi.DefaultResourceLoader({ cwd: directory, agentDir: agentDirectory, settingsManager, systemPrompt: system, noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true });
  await resourceLoader.reload();
  const modelRuntime = await pi.ModelRuntime.create({ authPath: join(agentDirectory, "auth.json"), modelsPath: join(agentDirectory, "models.json"), allowModelNetwork: false });
  const model = modelRuntime.getModel("local", modelName);
  if (!model) throw new Error("Configured model unavailable");
  const sessionManager = pi.SessionManager.create(directory, agentDirectory);
  const { session } = await pi.createAgentSession({ cwd: directory, agentDir: agentDirectory, modelRuntime, model, settingsManager, resourceLoader, sessionManager, tools: ["read", "write", "edit", "bash"], thinkingLevel: "off" });
  const eventFile = openSync(join(directory, "events.jsonl"), "wx", 0o600);
  const record = value => writeFileSync(eventFile, JSON.stringify({ unix_ms: Date.now(), ...value }) + "\n");
  const summary = { task: task.id, success: false, tools: 0, requests: 0, output_tokens: 0, cached_tokens: 0, replies: [], errors: [], tests: [], first_text_ms: [] };
  let phase = "repair";
  let requestStart;
  let firstText;
  const priorPayload = session.agent.onPayload;
  session.agent.onPayload = async (payload, requestModel) => {
    if (summary.requests >= 8) throw new Error("Experiment request budget exhausted");
    const adjusted = await priorPayload?.(payload, requestModel);
    const effective = { ...(adjusted ?? payload), temperature: 0,
      ...(engine === "loom" ? { enable_thinking: false } : { chat_template_kwargs: { enable_thinking: false } }) };
    if (effective.messages[0]?.role !== "system") throw new Error("Expected the experiment's system message");
    // Pi appends its physical cwd. Keep initial prompts identical across run
    // directories; all tasks explicitly use relative file paths.
    effective.messages = [{ ...effective.messages[0], content: system }, ...effective.messages.slice(1)];
    if (effective.max_tokens !== 384) throw new Error(`Unexpected pi output clamp: ${effective.max_tokens}`);
    save(join(directory, `request-${summary.requests++}.json`), effective);
    requestStart = performance.now();
    firstText = false;
    return effective;
  };
  session.agent.toolExecution = "sequential";
  const priorTool = session.agent.beforeToolCall;
  session.agent.beforeToolCall = async (context, signal) => {
    if (phase === "recall") {
      summary.errors.push("The retained recall turn must complete without tools");
      return { block: true, reason: "The recall turn does not permit tools" };
    }
    if (summary.tools++ >= 10) {
      session.agent.abort();
      return { block: true, reason: "Experiment tool budget exhausted" };
    }
    const name = context.toolCall.name;
    if (name === "bash") {
      if (context.args.command !== "python3 check.py") return { block: true, reason: "Only python3 check.py is permitted" };
    } else {
      const path = resolve(directory, context.args.path);
      const permitted = (name === "read" ? ["task.md", "candidate.py", "check.py"] : ["candidate.py"]).map(file => join(directory, file));
      if (!permitted.includes(path)) return { block: true, reason: "Path is outside this task's permitted files" };
    }
    return priorTool?.(context, signal);
  };
  const unsubscribe = session.subscribe(value => {
    if (value.type === "message_update") {
      const { partial, ...delta } = value.assistantMessageEvent;
      record({ type: value.type, assistantMessageEvent: delta });
      if (!firstText && delta.type === "text_delta" && delta.delta) {
        firstText = true;
        summary.first_text_ms.push(performance.now() - requestStart);
      }
    } else if (value.type !== "tool_execution_update") record(value);
    if (value.type === "message_end" && value.message.role === "assistant") {
      const message = value.message;
      summary.output_tokens += message.usage?.output ?? 0;
      summary.cached_tokens += message.usage?.cacheRead ?? 0;
      const text = message.content.filter(item => item.type === "text").map(item => item.text).join("");
      if (text) summary.replies.push(text);
      if (["error", "aborted", "length"].includes(message.stopReason)) summary.errors.push(message.errorMessage ?? message.stopReason);
      emit({ event: "assistant_complete", task: task.id, reason: message.stopReason, output_tokens: message.usage?.output });
    }
    if (value.type === "tool_execution_end" && value.toolName === "bash") summary.tests.push({ passed: !value.isError, result: value.result });
  });
  const recorder = recordPiSession(session, join(directory, "calls.jsonl"), piVersion);
  try {
    await new Promise(done => setTimeout(done, index * 75));
    summary.started_ms = performance.now() - started;
    emit({ event: "task_start", task: task.id });
    await session.prompt(`Repair ${task.id}. Read task.md and candidate.py, fix candidate.py and run python3 check.py. Finish with DONE after a passing test. A second turn will ask about the repaired function.`);
    await session.waitForIdle();
    const result = spawnSync("python3", [join(root, "check.py"), task.id, join(directory, "candidate.py")], { encoding: "utf8" });
    summary.evaluation = { exit_code: result.status, signal: result.signal, stdout: result.stdout, stderr: result.stderr };
    if (result.status === 0 && !summary.errors.length && summary.tests.some(test => test.passed)) {
      await new Promise(done => setTimeout(done, 100 + index * 150));
      phase = "recall";
      await session.prompt(`Without tools, what does solve return for these positional arguments: ${JSON.stringify(task.followup.args)}? Reply only with the JSON value, without Markdown or commentary.`);
      await session.waitForIdle();
      try { summary.followup_passed = equal(JSON.parse(summary.replies.at(-1)), task.followup.expected); }
      catch { summary.followup_passed = false; }
      summary.success = summary.followup_passed && !summary.errors.length;
    }
    if (summary.errors.length) recorder.close();
    else recorder.finish();
  } catch (error) {
    summary.success = false;
    summary.errors.push(error.stack ?? String(error));
    await session.abort();
    await session.waitForIdle();
    recorder.close();
  } finally {
    await session.waitForIdle();
    summary.completed_ms = performance.now() - started;
    summary.candidate_sha256 = createHash("sha256").update(readFileSync(join(directory, "candidate.py"))).digest("hex");
    save(join(directory, "summary.json"), summary);
    unsubscribe();
    session.dispose();
    closeSync(eventFile);
  }
  emit({ event: "task_complete", task: task.id, success: summary.success, requests: summary.requests, tools: summary.tools, output_tokens: summary.output_tokens, elapsed_ms: summary.completed_ms - summary.started_ms });
  return summary;
}

const settled = await Promise.allSettled(tasks.map(run));
const results = settled.map((result, index) => result.status === "fulfilled" ? result.value : { task: tasks[index].id, success: false, errors: [String(result.reason)] });
const summary = { elapsed_ms: performance.now() - started, planned_tasks: tasks.length, completed_tasks: results.filter(item => item.success).length, tasks: results };
save(join(output, "summary.json"), summary);
emit({ event: "agent_cohort_complete", elapsed_ms: summary.elapsed_ms, planned_tasks: summary.planned_tasks, completed_tasks: summary.completed_tasks });
if (summary.completed_tasks !== summary.planned_tasks) process.exitCode = 1;
