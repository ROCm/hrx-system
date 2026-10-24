// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_MODEL_CARD_H_
#define LOOM_MODEL_CARD_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Local runtime coordinates. Acquisition origins and integrity locks are not
// runtime inputs. Paths are relative to the directory containing the card.
typedef struct loom_model_file_t {
  // Normalized relative path, independent of the file's upstream name.
  iree_string_view_t path;
  // Explicit format, or empty when the consumer must infer it unambiguously.
  iree_string_view_t format;
} loom_model_file_t;

typedef struct loom_model_file_list_t {
  // Number of ordered files.
  iree_host_size_t count;
  // Borrowed immutable file descriptors.
  const loom_model_file_t* values;
} loom_model_file_list_t;

typedef struct loom_model_provider_t {
  // File providing linkable symbols.
  loom_model_file_t file;
  // Ordered importer options; their meaning belongs to the selected importer.
  iree_string_view_list_t options;
  // Auxiliary local files required by this provider, outside the link index.
  loom_model_file_list_t data;
} loom_model_provider_t;

typedef struct loom_model_provider_list_t {
  // Number of providers. Order does not imply execution order.
  iree_host_size_t count;
  // Borrowed immutable providers with a common linker role.
  const loom_model_provider_t* values;
} loom_model_provider_list_t;

typedef struct loom_model_parameter_t {
  // Code-facing parameter index scope, independent of file names.
  iree_string_view_t scope;
  // Ordered shards populating one parameter index.
  loom_model_file_list_t files;
} loom_model_parameter_t;

typedef struct loom_model_tokenizer_t {
  // Code-facing tokenizer identity, independent of file names.
  iree_string_view_t name;
  // Tokenizer file and its explicit or inferable encoding.
  loom_model_file_t file;
} loom_model_tokenizer_t;

// A caller may select a canonical preset and override its choices.
// Empty choices use that preset, or the sole available compatible
// choice. Ambiguity is an error; no hardware or payload inspection occurs.
typedef struct loom_model_selector_t {
  // Optional canonical preset name.
  iree_string_view_t preset;
  // Optional service identity override.
  iree_string_view_t service;
  // Optional parameter variant override.
  iree_string_view_t variant;
  // Optional provider set override; names imply no encoding.
  iree_string_view_t programs;
  // Optional executable partition; names imply no host or device assignment.
  iree_string_view_t partition;
} loom_model_selector_t;

// Unresolved local card reference. Resolution belongs to a cold I/O worker;
// retaining this descriptor never opens files or consults acquisition metadata.
typedef struct loom_model_reference_t {
  // Card locator in the application's local file or package namespace.
  iree_string_view_t path;
  // Explicit choices and optional canonical preset.
  loom_model_selector_t selector;
} loom_model_reference_t;

// A service-owned role sharing its parent's providers and parameter scopes.
// The host defines its lifecycle and checks its independently versioned ABI.
typedef struct loom_model_component_t {
  // Role within the parent service, such as text's image projector.
  iree_string_view_t name;
  // Selected symbol namespace, formed once from service kind and role.
  iree_string_view_t export_prefix;
  // Exact component ABI revision; zero denotes unversioned development.
  uint32_t version;
  // Canonical preparation options, or an empty JSON object.
  iree_string_view_t options;
  // Canonical execution settings, interpreted by the parent service host.
  iree_string_view_t execution;
} loom_model_component_t;

// Immutable selected model facts, with no device, HTTP, or acquisition policy.
// Every view is owned by the enclosing card and remains valid until release.
typedef struct loom_model_selection_t {
  // Base model identity from its original training source.
  iree_string_view_t model_id;
  // Human-readable model name.
  iree_string_view_t model_name;
  // Resolved selection identities.
  loom_model_selector_t selector;
  // Service contract family, interpreted and version-checked by its host.
  iree_string_view_t kind;
  // Exact service ABI revision; zero denotes unversioned development behavior.
  uint32_t version;
  // Number of independently versioned service-owned components.
  iree_host_size_t component_count;
  // Borrowed immutable role contracts, sharing this selection's file closure.
  const loom_model_component_t* components;
  // Execution context relative to the card; does not rebase declared files.
  iree_string_view_t directory;
  // Primary link providers, potentially in several different formats.
  loom_model_provider_list_t inputs;
  // Library providers whose symbols are linked on demand.
  loom_model_provider_list_t libraries;
  // Merged JIT configuration JSON: variant values override provider defaults.
  iree_string_view_t config;
  // Number of independent parameter indexes.
  iree_host_size_t parameter_count;
  // Borrowed parameter indexes keyed by their code-facing scope.
  const loom_model_parameter_t* parameters;
  // Number of independently named tokenizers.
  iree_host_size_t tokenizer_count;
  // Borrowed tokenizer declarations.
  const loom_model_tokenizer_t* tokenizers;
  // Canonical preparation options, or an empty JSON object.
  iree_string_view_t options;
  // Canonical execution settings, interpreted by the service host.
  iree_string_view_t execution;
  // Canonical request settings, interpreted by the service client.
  iree_string_view_t request;
} loom_model_selection_t;

// Owns one selected closure, independent of the input JSON's lifetime.
typedef struct loom_model_card_t loom_model_card_t;

// Parses and selects a versioned card without any file, network, or program
// access. Validates the selected runtime contract; acquisition-only metadata
// has no authority here. Unsupported schema revisions and ambiguous selections
// fail explicitly. Unselected service versions do not restrict the host.
// Failure leaves out_card NULL. Safe on an ordinary preparation worker.
iree_status_t loom_model_card_parse(iree_string_view_t json,
                                    const loom_model_selector_t* selector,
                                    iree_allocator_t host_allocator,
                                    loom_model_card_t** out_card);
void loom_model_card_retain(loom_model_card_t* card);
void loom_model_card_release(loom_model_card_t* card);

const loom_model_selection_t* loom_model_card_selection(
    const loom_model_card_t* card);

// Queries selected immutable declarations. Missing identities return NULL.
const loom_model_parameter_t* loom_model_selection_parameter(
    const loom_model_selection_t* selection, iree_string_view_t scope);
const loom_model_tokenizer_t* loom_model_selection_tokenizer(
    const loom_model_selection_t* selection, iree_string_view_t name);
const loom_model_component_t* loom_model_selection_component(
    const loom_model_selection_t* selection, iree_string_view_t name);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_MODEL_CARD_H_
