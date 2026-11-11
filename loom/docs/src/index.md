---
hide:
  - navigation
  - toc
---

<div class="loom-landing" markdown>

<div class="loom-intro" markdown>
<div class="loom-intro-copy" markdown>

<span class="loom-preview">Tech preview</span>

# Loom

Loom is a language and compiler for native programs on CPUs, GPUs, and NPUs.
The SDK includes a runtime and `loom-serve`, an inference engine that hosts
multiple models and modalities together. You can use the full stack, embed
parts of it, or take the compiler's output into your own runtime.

[Get the SDK](getting-started/acquiring-loom.md){ .md-button .md-button--primary }
[Run a model](serving/workflows/fetch-and-serve.md){ .md-button }

</div>
<aside class="loom-facts" aria-label="Loom at a glance">
<dl>
<div>
<dt>~8 MB</dt>
<dd>The full SDK, including the server, compressed with xz.</dd>
</div>
<div>
<dt>~1 second</dt>
<dd>All of DeepSeek's kernels and programs JIT from source text.</dd>
</div>
<div>
<dt>CPU to multi-GPU</dt>
<dd>From CPUs and integrated GPUs to DeepSeek on eight MI300X GPUs.</dd>
</div>
</dl>
</aside>

<div class="loom-intro-authoring" markdown>

Programs can be written in [a subset of C++](https://github.com/ROCm/hrx-system/blob/main/loom/src/loom/import/cxx/README.md)
or an isomorphic, agent-friendly Loom IR. The compiler provides
[structured diagnostics and reports](workflows/compile-reports.md)
for authoring and tuning, and emits ELF, HSACO, SPIR-V, WebAssembly, and other
[artifacts](workflows/compile-artifacts.md).

Those programs can be bundled into model packages: `.loom` source or `.loombc`
bytecode, weights, and a JSON descriptor. Model-specific code lives in those
files. You can load a new model, edit an implementation, or share your changes
without rebuilding the server.

It runs on **Windows and Linux**; native execution needs only the OS and hardware
drivers.

</div>
</div>

<p class="loom-authorship">This project has been engineered with the assistance of agents.</p>

<div class="loom-scope" markdown>

| Layer | What it provides |
| --- | --- |
| **Loom language and compiler** | Reusable programs, command schedules, and kernels, specialized and compiled ahead of time or just in time. |
| **Runtime and hardware abstraction** | Asynchronous execution, memory, and data movement on CPUs, AMD GPUs, AMD XDNA NPUs, and Vulkan devices. |
| **Model and kernel libraries** | A growing source library of model implementations, kernels, and shared math in [hrx-loom-library](https://github.com/ROCm/hrx-loom-library). |
| **Loom services and loom-serve** | Model hosting, compatible request batching, residency management, and HTTP APIs. |

Your application supplies its interface, tools, and behavior. These layers run
the computation underneath it.

</div>

<section class="loom-section" markdown>

## An application with several models

Say you're building a digital assistant. Start with speech recognition, a small
language model, and speech synthesis. Add image understanding so it can see what
you're working on, embeddings so it can search your documents, and an image
model when you want it to draw something. A larger language model can take on
deeper research.

Install the SDK, start `loom-serve`, and have your agent register the models
you want to use. Connect your application through the OpenAI-compatible
endpoints. Adding another capability means registering another model package
with the running server.

<figure class="loom-diagram loom-hosting" aria-labelledby="hosting-caption">
<div class="loom-clients">
<div class="loom-node">
<strong>Your application</strong>
<span>Interface, tools, agent behavior, and orchestration</span>
</div>
<div class="loom-node">
<strong>Other clients</strong>
<span>Your devices, friends, and family</span>
</div>
</div>
<div class="loom-request-path"><span aria-hidden="true">↓</span> HTTP requests</div>
<div class="loom-node loom-host-heading">
<strong>loom-serve</strong>
<span>HTTP endpoints · telemetry · serving policy</span>
</div>
<div class="loom-request-path"><span aria-hidden="true">↓</span> Embeddable C API (in development)</div>
<div class="loom-host">
<div class="loom-host-heading">
<strong>Loom services</strong>
<span>Models · scheduling · resource management</span>
</div>
<p>Each model brings its programs, weights, and descriptor.</p>
<ul class="loom-models">
<li>Speech recognition</li>
<li>Conversation LLM</li>
<li>Speech synthesis</li>
<li>Embeddings</li>
<li>Image understanding</li>
<li>Image generation</li>
<li>Research LLM</li>
<li class="loom-add-model">+ Another model</li>
</ul>
<div class="loom-shared">
<strong>Shared scheduling and execution</strong>
<span>Compatible batches · on-demand weights · retained state · managed memory</span>
</div>
</div>
<div class="loom-flow-arrow" aria-hidden="true">↓</div>
<ul class="loom-devices">
<li class="loom-cpu">x86 CPU</li>
<li class="loom-gpu">AMD GPU</li>
<li class="loom-npu">AMD XDNA NPU</li>
<li class="loom-vulkan">Vulkan GPU</li>
</ul>
<figcaption id="hosting-caption">Several model packages, one hosting system. Each model's programs determine which targets it supports.</figcaption>
</figure>

Model weights, state, and workspace can share a managed memory budget. Weights
can load on demand, useful state can remain between requests, and idle memory
can be reclaimed as the workload changes. Each modality batches compatible
requests. Several clients using a model share its weights and can share
execution; other computers, friends, or family can use the same service.

You can leave residency to the configured policies or control it from the
application. HTTP endpoints let you drain a model, resume it, keep its weights
resident, or trim idle backing while the server keeps running.

</section>

<section class="loom-section" markdown>

## From model source to execution

A model's structure, tensor layouts, parameter handling, and numerical
implementations live in its source. The compiler works with programs, types,
memory, and target instructions. The hardware abstraction handles execution and
data movement. Loom's services host text, image, speech, and other workloads;
`loom-serve` exposes them over HTTP. The model supplies the computation behind
those services.

That source covers more than individual kernels. Command programs describe
kernel launches, their buffers, and their dependencies. Model programs prepare
parameters and drive execution. Functions and templates can be shared across
those programs and their kernels. An author can change how a layer is computed
and how its work is launched in the same source package.

When model dimensions, data layouts, workload shapes, or hardware features
become known, Loom can specialize the program around them. The author chooses
what remains configurable and what becomes concrete. This can happen ahead of
time, when a model loads, or when a new workload needs a different specialization.
The same Qwen27B source files already run through two quite different paths:
Loom emits AMD GPU instructions for native execution, or SPIR-V for a Vulkan
driver to compile and run.

<figure class="loom-diagram loom-compilation" aria-labelledby="compilation-caption">
<div class="loom-compile-flow">
<div class="loom-sources">
<div class="loom-node">
<strong>C++ subset</strong>
<span>Loom authoring API</span>
</div>
<div class="loom-source-link"><span aria-hidden="true">↕</span> Import / export</div>
<div class="loom-node">
<strong>Loom IR</strong>
<span><code>.loom</code> text / <code>.loombc</code> bytecode</span>
</div>
<div class="loom-node loom-library">
<strong>Reusable libraries</strong>
<span>Programs, kernels, and motifs</span>
</div>
</div>
<div class="loom-flow-arrow loom-across" aria-hidden="true">→</div>
<div class="loom-compiler">
<strong>Loom compiler</strong>
<span>Link · specialize · optimize</span>
<span class="loom-compiler-programs">Programs · commands · kernels</span>
<div class="loom-codegen">Code generation</div>
<div class="loom-specialization">
<strong>Specialization inputs</strong>
<span>Shapes · layouts · target facts</span>
</div>
</div>
<div class="loom-flow-arrow loom-across" aria-hidden="true">→</div>
<dl class="loom-artifacts">
<div class="loom-cpu"><dt>x86 CPU</dt><dd>ELF</dd></div>
<div class="loom-gpu"><dt>AMD GPU</dt><dd>HSACO</dd></div>
<div class="loom-npu"><dt>AMD XDNA NPU</dt><dd>Tile and array programs</dd></div>
<div class="loom-vulkan"><dt>Vulkan</dt><dd>SPIR-V</dd></div>
<div class="loom-wasm"><dt>WebAssembly</dt><dd>Wasm modules</dd></div>
</dl>
</div>
<figcaption id="compilation-caption">AOT and JIT use the same source, linking, and specialization model. Use the artifacts with Loom's runtime or your own.</figcaption>
</figure>

This also makes experimentation part of normal use. A running server can compile
and select a compatible kernel or command-program variant while keeping the
model's weights and state resident. An author can test a change, inspect its
structured compile report, and compare it against the original on the same
workload. The improvement is something they can share as source files.

The small distribution follows from these boundaries. Model behavior lives in
model packages, specialization produces the code needed for a particular use,
and execution machinery is shared. A new model can add capabilities without
adding another runtime and its dependency tree. That matters even when the
weights occupy 100 GB: the software around them still has to be deployed,
updated, trusted, and kept compatible.

</section>

<section class="loom-section" markdown>

## Work that carries forward

The [loom-fun project](https://loom-fun.github.io/loom-skills/) gives an early
example of what people can build on this foundation. Over one weekend, a
community contributor used a Claude subscription and the released SDK to port
fifty models. They published [the model packages](https://huggingface.co/loom-fun),
source, validation results, and performance comparisons on Strix Halo, along
with the [skills used to do the work](https://github.com/loom-fun/loom-skills).
Those skills give the next person a starting point too.

Today, a practical way to begin is to find a working model similar to the one
you want, point your agent at its source and the loom-fun skills, and adapt it.
The existing model supplies a complete example of preparation, execution,
serving, and qualification. The result is another package that people can run,
inspect, and change with the same SDK.

We're building out `hrx-loom-library` so that more of this work becomes
assembling components. The aim is reusable service behavior, techniques such as
speculative decoding and cohort batching, common kernels for common targets,
and the math underneath them: quantization, reductions, matrix operations, and
other recurring pieces. Authors should be able to reuse a component, fork it,
or replace the part they want to explore, then concentrate optimization where
their application needs it.

A model port can then contribute more than one supported model. A better
decoder, a new quantization implementation, or a useful scheduling technique
can become something the next author builds on. The source, checks, and
measurements preserve what was learned while leaving room to try something
different.

The goal is to make models programmable components of an application. A model
brings its weights and the programs that implement it, assembled from pieces
you can reuse or change. **The model owns its implementation. Loom provides
the platform.**

</section>

<section class="loom-section" markdown>

## From CPUs to multi-GPU servers

Deployments already span CPU applications, integrated GPUs, and full DeepSeek serving
across eight MI300X GPUs. They use the same compiler and runtime foundation,
with model programs and deployment configuration describing how work is placed.
x86 execution covers AMD and Intel CPUs; Vulkan provides a path to other vendors'
GPUs alongside the native AMD GPU backend. AMD XDNA NPUs already run embedding
models and can work alongside GPU workloads.

Work can also span different kinds of processors. Our image-generation pipeline
experiments run on the GPU, the NPU, or a mixture of both from the same Loom
source files. The [Quake rendering demonstration](https://gist.github.com/benvanik/f6b8206990a1c8e24e74611d8d63334b)
uses CPU preparation, NPU rasterization, and GPU consumption, with Loom
compiling the NPU and GPU programs. These are general compute tools that can
support quite different applications.

Multi-node execution is under development. A pipeline-parallel DeepSeek
demonstration has already run across two Strix Halo all-in-one machines; that
work has not yet landed in the main project.

</section>

<section class="loom-section" markdown>

## A tech preview you can build on

Loom is an early experimental project. We're building out the reusable library
and refactoring model hosting into an embeddable C API, with HTTP and serving
policy layered above it. More scheduling and request handling will move into
Loom programs so that authors can experiment with those behaviors in libraries
and model packages.

Experiments in model behavior belong in those programs and libraries. Changes
to the core should support new hardware, programming models, and shared
capabilities that help those implementations.

Start by [running a model](serving/workflows/fetch-and-serve.md)
or [authoring a model package](serving/model-programs.md).
The [integration guides](integration/index.md)
cover using the compiler and its artifacts in your own software, and the
[programming guide](guide/index.md) explains the
language. The compiler and runtime are useful independently, and taking the
ideas elsewhere is welcome too.

</section>

</div>
