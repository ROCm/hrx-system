// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// ==========================================================================
// Loom IR Textual Format Specification
// ==========================================================================
//
// This header defines the grammar, lexical rules, and semantics of the
// loom IR textual format (.loom files). It is the authoritative
// reference for parser and printer implementations (both Python and C).
//
// ==========================================================================
// Design goals
// ==========================================================================
//
// - Agent-friendly: regular structure, no ambiguity, predictable.
// - Round-trip stable: parse -> print -> parse -> print is identical.
//   SSA value names and comments are preserved exactly.
// - Self-documenting: an agent reading the IR understands the program
//   without external documentation.
// - Compact: minimal syntactic overhead. Every token carries meaning.
// - Declarative: every op's syntax is expressed with reusable format
//   elements. No custom parsers or printers.
// - Greppable: one op per line. `grep "tile.contract" model.loom`
//   gives you the complete op with all operands, types, and results.
//
// ==========================================================================
// Canonical formatting
// ==========================================================================
//
// The canonical format is what the printer produces and what agents
// are trained on. Parsers accept any whitespace arrangement, but
// canonical output follows these rules:
//
// - One op per line. Everything about an op (operands, types,
//   attributes, result types) is on a single line. No line-length
//   limit.
// - Regions: '{' at end of the op line, body ops indented 2 spaces,
//   '}' on its own line at the outer indentation level.
// - Region separators ('else', etc.) at outer indentation: '} else {'.
// - Single space between tokens, except:
//   - No space before: , ) ] }
//   - No space after: ( [ {
//   - IndexList glues to a preceding format element: %src[0, %off].
//     Leading IndexList forms keep a space after the op name: op [0, %off].
//   - BindingList glues to preceding token: iter_args(%acc = %init : f32)
//   - BlockArgs glues to preceding token: do(%arg: f32)
//   - FuncArgs glues to preceding symbol: @name(%a: type)
//   - Explicit Glue suppresses space: @callee(%x) not @callee (%x)
// - Result types: ops with variadic or tied results use parenthesized
//   lists: -> (type) or -> (%op as type, type). Ops with exactly one
//   non-variadic result may use bare syntax: -> type. The choice is
//   per-op (ResultType vs ResultTypeList in the format spec).
// - String literals: parsers accept JSON-compatible escape spellings,
//   but the printer emits one canonical form. The original escape syntax
//   is not semantic state.
//
// Example of canonical output (single line per op, regions indented):
//
//   %c0 = index.constant 0 : index
//   %slice0 = tensor.slice %input[%c0] : tensor<4xf32> -> (tile<4xf32>)
//   %result = scf.for %i = %c0 to %M step %c4 iter_args(%out = %output : tensor<[%M]xf32>) -> (%output as tensor<[%M]xf32>) {
//     %w = tensor.slice %weights[%i, %c0] : tensor<[%M]x4xf32> -> (tile<4x4xf32>)
//     %partial = tile.contract %w, %slice0 : tile<4x4xf32>, tile<4xf32> -> (tile<4xf32>)
//     %update0 = tensor.update %partial, %out[%i] : tile<4xf32> -> (%out as tensor<[%M]xf32>)
//     scf.yield %update0 : tensor<[%M]xf32>
//   }
//
// ==========================================================================
// Value auto-naming
// ==========================================================================
//
// SSA values have either user-assigned names or auto-generated names.
// User-assigned names are identifiers (%x, %result, %input) preserved
// exactly through round-trips. Auto-generated names are digit-only
// (%0, %1, %2) using the value's ID in the module value table.
//
// These occupy separate syntactic namespaces in the grammar:
//
//   SSA value: '%' identifier | '%' digit+
//
// Identifiers start with [a-zA-Z_$]. Digit-only names start with
// [0-9]. No collision between user and auto names is possible.
//
// The value ID is the index into the module's value table, assigned
// in definition order by the builder. The printer emits %N for any
// value without a user-assigned name. The printer requires zero
// allocation and zero state for auto-naming — just the value table.
//
// ==========================================================================
// File structure
// ==========================================================================
//
// A .loom file contains a sequence of top-level items:
//
//   file ::= (attribute-alias | function-def | function-decl)*
//
//   attribute-alias ::= '#' identifier '=' attribute-value
//
// Attribute aliases define shorthand for encoding attributes:
//
//   #enc = #q8_0<block=32>
//   #weights_enc = #q6_k
//
// Alias names are file-local, must be unique, and must not shadow a registered
// encoding family name.
//
// ==========================================================================
// Lexical rules
// ==========================================================================
//
// Whitespace: spaces, tabs, newlines. Insignificant except as token
// separators.
//
// Comments: '//' to end of line. Attached to the next operation or
// block for round-trip preservation. The printer re-emits comments
// at the original indentation level.
//
//   // This comment is attached to the next op.
//   %result = tile.contract %w, %x : tile<4x4xf32>, tile<4xf32> -> (tile<4xf32>)
//
// Tokens:
//
//   SSA value:     '%' identifier | '%' digit+
//                  Examples: %x, %tile, %0, %arg0, %M, %contract0
//
//   Symbol name:   '@' identifier ('.' identifier)*
//                  Examples: @main, @dn_layer, @vnni_q8,
//                            @model36.model.hidden_size
//
//   Hash attr:     '#' identifier
//                  Examples: #q6_k, #enc, #q8_0
//
//   Integer:       [-] digit+ | '0x' hexdigit+
//                  Examples: 42, -1, 0xFF
//
//   Float:         [-] digit+ '.' digit* ([eE] [+-]? digit+)?
//                  | [-] digit+ [eE] [+-]? digit+
//                  Examples: 3.14, 1.0e-5, -0.5
//
//   String:        '"' string-char* '"'
//                  string-char ::= unescaped-char | escape-sequence
//                  unescaped-char ::= any valid UTF-8 codepoint except '"',
//                                     '\\', and U+0000..U+001F
//                  escape-sequence ::=
//                    '\\"' | '\\\\' | '\\/' | '\\b' | '\\f' | '\\n' |
//                    '\\r' | '\\t' | '\\u' HEXDIGIT HEXDIGIT HEXDIGIT
//                    HEXDIGIT
//                  Unicode escapes follow JSON rules. Supplementary-plane
//                  codepoints use a UTF-16 surrogate pair:
//                    '\\uD83D\\uDD25' => 🔥
//                  Lone surrogates and malformed hex digits are parse errors.
//                  In-memory string payloads are decoded UTF-8 without quotes
//                  or escape syntax; source spans preserve the exact quoted
//                  source spelling for diagnostics.
//                  Canonical printing emits decoded UTF-8 directly except
//                  for '"', '\\', '\b', '\f', '\n', '\r', '\t', and other
//                  control bytes U+0000..U+001F, which are escaped as
//                  '\\u00XX'. '/' is printed unescaped.
//                  Examples: "hello", "model.loom", "has \"quotes\"",
//                            "line1\\nline2", "\\u03BB", "\\uD83D\\uDD25"
//
//   Type keyword:  'f16' | 'f32' | 'f64' | 'bf16' | 'f8E4M3' | 'f8E5M2'
//                  | 'i' digit+ | 'index'
//                  | 'tile' | 'tensor' | 'group' | 'encoding'
//                  Note: 'tile', 'tensor', 'group', 'encoding' are type
//                  keywords only when NOT followed by '.'.
//                  'tile.broadcast' is an op name, 'tile<4xf32>' is a type,
//                  'encoding.define' is an op name, bare 'encoding' is a type.
//
//   Op name:       identifier ('.' identifier)+
//                  Examples: tile.contract, scalar.addf, scf.for
//
//   Keyword:       identifier
//                  Examples: func, return, to, step, iter_args, where,
//                            as, right, left, add, max, true, false,
//                            public, host, device, else
//
//   Block label:   '^' identifier
//                  Examples: ^bb0, ^entry, ^then
//
//   Punctuation:   ( ) { } [ ] < > = : , -> ::
//
// Identifiers: [a-zA-Z_$] [a-zA-Z0-9_$.]*
// Op names include dots; bare identifiers do not.
//
// Disambiguation: '#' followed by a letter is a hash attr (encoding
// reference or alias). '#' followed by a digit is a parse error.
//
// ==========================================================================
// Types
// ==========================================================================
//
// type ::= scalar-type | shaped-type | buffer-type | pool-type | group-type
//        | function-type | encoding-type | dialect-type
//
// --- Scalar types ---
//
// scalar-type ::= 'f16' | 'f32' | 'f64' | 'bf16'
//               | 'f8E4M3' | 'f8E5M2'
//               | 'i' DIGIT+
//               | 'index'
//
// --- Shaped types (tile, tensor, vector, view) ---
//
// shaped-type ::= ('tile' | 'tensor') '<' shaped-body
//                   (',' encoding)? '>'
//               | 'view' '<' shaped-body (',' layout)? '>'
//               | 'vector' '<' shape 'x' scalar-type '>'
//
// shaped-body ::= (shape 'x')? scalar-type
//
// shape       ::= dim ('x' dim)*
// dim         ::= INTEGER             // static: 4, 128, 256
//               | '[' dim-ref ']'     // dynamic
//
// dim-ref     ::= SSA-VALUE           // [%M] — references an SSA value
//
// encoding    ::= '#' identifier ('<' static-param (',' static-param)* '>')?
//               | SSA-VALUE
// layout      ::= encoding
// static-param ::= identifier '=' attr-value
//
// Encodings/layouts in the first form are static — known at definition time.
// Static encoding parameters are attributes only; SSA values are rejected here.
// Encodings/layouts in the second form (SSA value) are dynamic — the value
// is an SSA value of type 'encoding' and is resolved at compilation time.
// Dynamic parameters are introduced explicitly by encoding.define:
//
//   %enc = encoding.define #q8_0<block=32> {group_size = %g : index}
//       : encoding<schema>
//
// This enables library functions generic over encoding:
//
//   func.def @generic(%enc: encoding, %t: tile<4xf32, %enc>) -> ()
//
// Examples:
//   tile<4x4xf32>                      All-static tile.
//   tile<[%M]x4xf32>                   First dim dynamic, named %M.
//   tensor<[%M]x[%K]xf32>              Both dims dynamic.
//   vector<[%N]xf32>                   Dynamic 1-D register vector.
//   tile<256x256xf32, #q6_k>           With static encoding (no params).
//   tensor<[%N]xi8, #q8_0<block=32>>   Dynamic dim + parameterized encoding.
//   view<[%N]xf32, #strided<stride=64>> Dynamic view + static layout.
//   view<[%N]xf32, %layout>            Dynamic dim + SSA layout.
//   tile<4xf32, %enc>                  SSA encoding (dynamic).
//   tile<f32>                          0-d (scalar) tile.
//
// Vector types are always rank >= 1 and cannot carry encoding/layout
// attachments. Use explicit splat/broadcast/conversion ops to cross between
// scalar and vector values.
//
// Static encodings are pluggable: each has a name ("q8_0", "q6_k")
// and optional parameters. The encoding name indexes into a vtable
// registered at context creation.
//
// SSA encodings are created by encoding.define and propagated as
// values. The compiler's encoding resolution pass converts SSA
// encodings to static attributes once all call sites are known.
//
// Named dynamic dims: every dynamic dimension references an SSA value.
// In function signatures, named dims in argument types implicitly
// define index-typed SSA values available in the function body.
//
// --- Encoding type ---
//
// encoding-type ::= 'encoding'
//
// The type of an SSA encoding value. encoding.define produces values
// of this type. These values appear in the encoding/layout attachment
// position of types such as tile<4xf32, %enc> and view<[%N]xf32, %layout>.
//
// --- Buffer type ---
//
// buffer-type ::= 'buffer'
//
// Opaque untyped storage identity. Views are typed projections over storage
// roots such as buffers; the buffer type itself carries no shape, element type,
// capacity, or layout.
//
// --- Pool type ---
//
// pool-type ::= 'pool' '<' pool-dim '>'
// pool-dim  ::= INTEGER | '[' SSA-VALUE ']'
//
// Block-managed device memory pool. One parameter: the block size in
// bytes, which is either a static integer or a dynamic SSA reference.
// The pool carries no element type, no encoding, no capacity — it's
// untyped bytes managed at block granularity by the runtime.
//
// Examples:
//   pool<65536>         Static 64KB blocks.
//   pool<[%BS]>         Dynamic block size.
//
// --- Group types ---
//
// group-type ::= 'group' '<' identifier '>'
//
// Example: group<workgroup>
//
// --- Function types ---
//
// function-type ::= '(' type-list ')' '->' '(' type-list ')'
// type-list     ::= (type (',' type)*)?
//
// --- Register types ---
//
// register-type  ::= 'reg' '<' register-class register-units? '>'
// register-class ::= identifier '.' identifier ('.' identifier)*
// register-units ::= 'x' integer | 'x' digit+
//
// Register types are target-low allocation values. The class name is always
// namespace-qualified (for example test.low.i32 or target.vgpr) and the
// optional unit suffix gives the number of class units carried by one SSA
// value.
//
//   reg<amdgpu.vgpr>                   One AMDGPU VGPR.
//   reg<amdgpu.vgpr x4>                Four contiguous VGPR-class units.
//
// --- Dialect types ---
//
// dialect-type ::= dialect-type-name
//                | dialect-type-name '<' type-interior '>'
//
// dialect-type-name ::= identifier ('.' identifier)*
//
// Dialect types are declared with TypeDef in the Python DSL, using
// the same format element vocabulary as ops. Each TypeDef specifies
// its name and the format of its angle-bracket interior (if any).
//
// Opaque types have no parameters and no angle brackets:
//
//   hal.buffer                         Device memory buffer.
//   hal.fence                          Synchronization timepoint.
//   hal.device                         Logical device instance.
//   hal.command_buffer                 Command recording interface.
//
// Parameterized types wrap other types inside angle brackets. The
// interior format is declared in the TypeDef and driven by the same
// format elements (TypeOf, Attr, etc.) used for op syntax:
//
//   vm.ref<hal.buffer>                 Reference-counted handle.
//   vm.ref<vm.list<i32>>              Nested: ref to a list of i32.
//   vm.list<f32>                       Generic container.
//   vm.list<hal.buffer>                Container of device buffers.
//
// The built-in shaped types (tile, tensor, vector, view), buffer, pool,
// and group are also TypeDefs internally. Their format specs use elements
// such as ShapeOf, ScalarOf, and EncodingOf for the dim-x-dim-x-element-type
// syntax that is unique to shaped types.
//
// Type dispatch: the parser checks the token against the type
// registry. Scalar keywords (f32, i32, index) are a fixed set.
// Everything else is looked up by name in the registry. Unknown
// type names are errors. This means every type used in a program
// must have a registered TypeDef (or be a scalar keyword).
//
// Adding a new dialect type: declare a TypeDef in Python, register
// it with the parser/printer. The C code generator produces a type
// format table entry. No hand-written parser or printer code needed.
//
// ==========================================================================
// Dimension references
// ==========================================================================
//
// Dynamic dimensions in types reference SSA values:
//
//   [%name]  SSA value reference. The value must be in scope:
//            - In function arg types: implicitly defines %name as an
//              index-typed SSA value in the function body.
//            - In op result types: %name must be defined on the LHS
//              of the current op or by a prior op in the block.
//            - In function return types: %name must be an arg dim.
//
// At a call site, the caller names results on the LHS and can
// reference them in the result type annotations:
//
//   %tensor, %size = func.call @compute(%input) : (tensor<[%M]xf32>) -> (tensor<[%size]xf32>, index)
//
// ==========================================================================
// Operations
// ==========================================================================
//
// operation ::= (result-list '=')? op-name op-body
//
// result-list ::= ssa-value (',' ssa-value)*
//
// op-name ::= identifier ('.' identifier)+
//
// The op-body is driven by the op's declarative format spec. Each op
// kind registers a format — a sequence of format elements that the
// parser and printer walk to consume/emit tokens. No custom parsers.
//
// --- Format elements ---
//
// The format element vocabulary:
//
//   Ref(field)            Single SSA value reference: %name.
//   Refs(field)           Variadic SSA values: %a, %b, %c.
//   TypedRefs(field)      Variadic typed SSA values:
//                           %a: type, %b: type.
//   Attr(field)           Attribute value: 42, 3.14, slt, "hello".
//   SymbolRef(field)      Symbol reference: @name.
//   TypeOf(field)         Type of a single field: f32, tile<4xf32>.
//   TypesOf(field)        Types of variadic field: f32, tile<4xf32>.
//   ResultType(field)     Single bare result type token.
//                           Use an explicit ARROW keyword before it only when
//                           the operation syntax needs `-> type`. For single
//                           non-variadic results. No tied support.
//   ResultTypeList(field) Result types with tied handling:
//                           -> (type, %op as type). Always uses parens.
//   Keyword(text)         Literal token: , : -> to step else ( ) etc.
//   AttrDict()            Optional {key = value, ...} dictionary.
//   AttrTable(k, v)       Static-keyed value table:
//                           {0 = (%a, %b), 1 = (%c, %d)} default(%x, %y).
//                           Attribute field k stores i64 row keys; operand
//                           field v stores row payloads flattened row-major,
//                           followed by the default row.
//   OperandDict(o, n)     Optional keyed SSA operand dictionary:
//                           {key = %value : type, ...}. Values are ordinary
//                           operands in field o. Attribute field n stores only
//                           key -> operand ordinal metadata.
//   Region(field)         Nested { block+ } region.
//   IndexList(d, s)       Bracket list [0, %x, 4] with static/dynamic
//                           mix. Usually glues to preceding token; format
//                           tables may opt out for named clauses.
//   BindingList(field)    Parenthesized named bindings:
//                           (%elem = %tile : type, ...).
//                           Also accepts grouped types:
//                           (%a = %x, %b = %y : type0, type1).
//                           Always glues to preceding token.
//   BlockArgs(region)     Region entry block args:
//                           (%arg: type, ...).
//                           Always glues to preceding token.
//   FuncArgs(field)       Function argument defs: (%a: type, %b: type).
//                           Always glues to preceding symbol.
//   PredicateList(field)  Where-clause predicates:
//                           [mul(%M, 16), lt(%K, 1024)].
//   OptionalGroup(elems)  Elements that appear only when an anchor
//                           field is present.
//   Glue                  Suppress space before the next element.
//                           Used for bare Keyword("(") after SymbolRef.
//   Flags(field)          Per-op-instance flags: <nuw|nsw>.
//                           Glued to the op name. Stored in instance_flags.
//   OpRef(field)          Op kind reference: <tile.contract>.
//                           Glued to the op name. For template/ukernel.
//   TemplateParam(field)  Required compile-time op parameter: <add>.
//                           Glued to the op name. Parsed as an attribute.
//   TemplateParamFlags(p, f)
//                         Required template parameter plus optional flags:
//                           <add> or <add, reassoc|nnan|nsz>.
//                           Glued to the op name. Parameter p is parsed as an
//                           attribute. Flags f are stored in instance_flags.
//
// --- Spacing rules ---
//
// Default: single space between tokens.
// Backward-glue (no space before): , ) ] }
// Forward-glue (no space after): ( [ {
// Built-in glue: non-leading IndexList, BindingList, BlockArgs, FuncArgs.
// Explicit Glue: suppresses space before the next token.
// ResultTypeList, PredicateList, OperandDict, and AttrTable never glue.
//
// --- Common format patterns ---
//
// Binary scalar:     %result = scalar.addf %a, %b : f32
// Unary scalar:      %result = scalar.negf %x : f32
// Cast:              %result = scalar.sitofp %x : i32 to f32
// Constant:          %c0 = index.constant 0 : index
// Comparison:        %result = scalar.cmpi slt, %a, %b : i32
// Reduce:            %result = tile.reduce %x {axis = 0, combine = "add"} : tile<4xf32> -> (tile<f32>)
// Tile slice:        %subtile = tile.slice %src[%o0, %o1] : tile<64x64xf16> -> (tile<16x16xf16>)
// Tensor update:     %update0 = tensor.update %tile, %tensor[%i] : tile<4xf32> -> (%tensor as tensor<[%M]xf32>)
// Elementwise:       %result = tile.elementwise(%e = %tile : tile<4xf32>) {
//                      %negated = scalar.negf %e : f32
//                      tile.yield %negated : f32
//                    } -> (tile<4xf32>)
// Function def:     func.def @f(%a: f32) -> (f32) { ... func.return %r : f32 }
// Function decl:    func.decl public @f(%a: f32) -> (f32)
// Template:         func.template<tile.contract> device @name(...) -> (...) where [...] { ... }
// Ukernel:          func.ukernel<tile.contract> device @name(...) -> (...)
// Call:             %out, %count = func.call @compute(%x, %y) : (tile<4xf32>, index) -> (%x as tile<4xf32>, index)
// Apply:            %r = func.apply<tile.contract>(%x) : (f32) -> (f32)
// Return:           func.return %r : f32
// Yield:            scf.yield %a, %b : f32, tensor<[%M]xf32>
//
// ==========================================================================
// Tied results (ownership transfer)
// ==========================================================================
//
// A result type is either a fresh allocation or a tied reference to
// an operand (linear ownership transfer). Tied results always carry
// an explicit type via 'as' — every op is self-describing.
//
//   result-type ::= type                        // fresh allocation
//                 | SSA-VALUE 'as' type         // tied (operand consumed)
//
// Result type lists always use parentheses:
//
//   result-type-list ::= '(' ')'
//                       | '(' result-type (',' result-type)* ')'
//
// Examples:
//
//   // Fresh allocation (single result, still has parens):
//   %contract0 = tile.contract %w, %x : tile<4x4xf32>, tile<4xf32> -> (tile<4xf32>)
//
//   // Tied (in-place, %tensor consumed):
//   %update0 = tensor.update %tile, %tensor[%i] : tile<4xf32> -> (%tensor as tensor<[%M]xf32>)
//
//   // Multiple results, some tied:
//   %out, %count = func.call @process(%a, %b) : (tensor<[%M]xf32>, tensor<[%N]xf32>) -> (%a as tensor<[%M]xf32>, tensor<[%K]xf32>)
//
// The consumed operand must have no uses after the consuming op.
// This is verified statically (linear ownership).
//
// Tied results are stored sparsely on the operation: a list of
// (result_index, operand_index) pairs. No sentinel values. An
// absent result in the tied list means it's a fresh allocation.
//
// ==========================================================================
// Functions
// ==========================================================================
//
// The func dialect provides seven ops for program structure:
//
// --- Top-level constructs (module-level symbols) ---
//
// function-def     ::= 'func.def' modifiers? function-sig '{' block+ '}'
// function-decl    ::= 'func.decl' modifiers? function-sig
// function-tmpl    ::= 'func.template' '<' contract-key '>' modifiers?
//                        function-sig '{' block+ '}'
// function-ukernel ::= 'func.ukernel' '<' contract-key '>' modifiers?
//                        function-sig
//
// --- Body ops (inside function/template bodies) ---
//
// func-call   ::= result-list '=' 'func.call' '@' identifier
//                   '(' operand-list ')' ':' '(' type-list ')'
//                   '->' result-type-list
// func-apply  ::= result-list '=' 'func.apply' '<' contract-key '>'
//                   '(' operand-list ')' ':' '(' type-list ')'
//                   '->' result-type-list
// func-return ::= 'func.return' operand-list ':' type-list
//
// --- Shared productions ---
//
// modifiers    ::= modifier+
// modifier     ::= 'public' | 'host' | 'device' | 'initializer'
//
// function-sig ::= '@' identifier '(' arg-list ')' ('->' result-sig)?
//                    where-clause?
//
// arg-list     ::= (arg (',' arg)*)?
// arg          ::= SSA-VALUE ':' type
//
// result-sig   ::= '(' result-type (',' result-type)* ')'
//                 | '(' ')'
//
// where-clause ::= 'where' '[' predicate (',' predicate)* ']'
//
// Modifiers appear AFTER the op name, BEFORE the symbol name. The
// op name is always the first token (greppable, unambiguous at parse
// time). Available modifiers:
//
//   public       — Visible outside the module (default: private).
//   host         — Host calling convention (default, can be omitted).
//   device       — Device calling convention.
//   initializer  — Module initialization function.
//
// --- Op semantics ---
//
// func.def:      Function definition. Must have a body. Callable via
//                func.call by symbol name.
//
// func.decl:     External function declaration. No body. Callable via
//                func.call by symbol name. Linked at compile or load
//                time.
//
// func.template: Constraint-matched visible implementation of an
//                implementation contract key T. Must have a body containing
//                Loom IR. The compiler's selection pass matches templates to
//                live func.apply contract demands. Exact compile-time calls use
//                func.call inline @template and must be inlined before
//                executable lowering.
//
// func.ukernel:  Constraint-matched opaque implementation of an
//                abstract op T. No body. The compiler emits a runtime
//                dispatch call when selecting a ukernel.
//
// func.call:     Function-like symbol call. Calls to func.def/func.decl
//                survive to runtime as call instructions. Required-inline
//                calls to func.template are exact compile-time calls consumed
//                by the inliner before executable lowering.
//
// func.apply:    Compile-time implementation demand. The angle-bracket key
//                names an implementation contract, not a symbol. Selection
//                rewrites resolved applies to func.call inline @selected and
//                unresolved applies are rejected before executable lowering.
//
// func.return:   Return values from a function body. Operand types
//                must match the enclosing function's result types.
//
// Symbol references (@name) are attributes on the operation, not SSA
// value operands. The callee in func.call is stored as a symbol
// attribute, not in the operand list. SymbolRef format elements read
// from the attribute dict and print with @ prefix. Contract keys in
// func.template, func.ukernel, and func.apply are string attributes printed
// in angle brackets.
//
// --- Argument dim semantics ---
//
// Named dims in argument types implicitly define index-typed SSA
// values in the function body:
//
//   func.def @f(%a: tensor<[%M]x[%K]xf32>) -> (tensor<[%M]xf32>)
//   // %M and %K are available as index values in the body.
//
// --- Return dim semantics ---
//
// Return types may reference arg dim names: [%M] where %M was
// defined by an arg type.
//
// --- Tied returns ---
//
// Return types can reference arguments for in-place mutation:
//
//   func.def @append_kv(%cache: tensor<[%S]x[%D]xf32>, %new_kv: tile<1x[%D]xf32>, %pos: index) -> (%cache as tensor<[%S]x[%D]xf32>)
//
// --- Examples ---
//
//   // Definition: function with a body.
//   func.def @negate(%input: tensor<4x4xf32>) -> (tensor<4x4xf32>) {
//     ...
//   }
//
//   // Declaration: external function, no body.
//   func.decl public @extern_matmul(%a: tensor<[%M]x[%K]xf32>, %b: tensor<[%K]x[%N]xf32>) -> (tensor<[%M]x[%N]xf32>)
//
//   // Template: visible implementation of tile.contract, matched by
//   // where-clause constraints. Compiler can inline and optimize.
//   func.template<tile.contract> public device @vnni_q8_matvec(%weights: tensor<[%M]x[%K]xi8, #q8_0<block=32>>, %input: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16), mul(%K, 32)] {
//     ...
//   }
//
//   // Ukernel: opaque implementation, matched by same constraints.
//   func.ukernel<tile.contract> device @vnni_q8_asm(%weights: tensor<[%M]x[%K]xi8, #q8_0<block=32>>, %input: tensor<[%K]xf32>) -> (tensor<[%M]xf32>) where [mul(%M, 16), mul(%K, 32)]
//
//   // Call: runtime function call.
//   %r = func.call @negate(%input) : (tensor<4x4xf32>) -> (tensor<4x4xf32>)
//
//   // Apply: compile-time contract demand.
//   %r = func.apply<tile.contract>(%w, %x) : (tensor<16x32xi8, #q8_0<block=32>>, tensor<32xf32>) -> (tensor<16xf32>)
//
//   // Return: exit function body.
//   func.return %result : tensor<[%M]xf32>
//
// ==========================================================================
// Structured control flow
// ==========================================================================
//
// Control flow ops (scf.for, scf.if, etc.) use the same declarative
// format element vocabulary as all other ops. The BindingList element
// handles the iter_args pattern. Regions are declared on the op and
// parsed/printed automatically.
//
// --- For-like ops ---
//
//   %result = scf.for %i = %c0 to %M step %c4 iter_args(%out = %output : tensor<[%M]xf32>) -> (%output as tensor<[%M]xf32>) {
//     ...
//     scf.yield %out_new : tensor<[%M]xf32>
//   }
//
// The iter_args clause is a BindingList: (%name = %init : type, ...). The
// parser also accepts grouped type-list spelling:
// (%a = %x, %b = %y : type0, type1).
// The body's entry block receives the IV followed by the iter_arg
// values. Tied result names refer to the parent-scope init operands
// (for example %output above), not the region-local iter_arg block args.
// scf.yield returns values matching the iter_args.
//
// --- If-like ops ---
//
//   %result = scf.if %cond -> (tensor<[%M]xf32>) {
//     scf.yield %updated_a : tensor<[%M]xf32>
//   } else {
//     scf.yield %updated_b : tensor<[%M]xf32>
//   }
//
// The else region is separated by the 'else' keyword. Both branches
// must yield the same number and types of values.
//
// --- Yield ops ---
//
//   scf.yield %value : type
//   scf.yield %a, %b : f32, tensor<[%M]xf32>
//   func.return %result : tensor<[%M]xf32>
//
// Values are comma-separated, then ':', then types comma-separated.
//
// ==========================================================================
// Region capture (elementwise, etc.)
// ==========================================================================
//
// Ops with region captures use the BindingList format element to bind
// block arguments to operands with types:
//
//   binding-list ::= '(' binding (',' binding)* ')'
//   binding      ::= SSA-VALUE '=' SSA-VALUE ':' type
//
// Example:
//
//   %result = tile.elementwise(%a = %x : tile<4xf32>, %b = %y : tile<4xf32>) {
//     %sum = scalar.addf %a, %b : f32
//     tile.yield %sum : f32
//   } -> (tile<4xf32>)
//
// ==========================================================================
// Blocks and regions
// ==========================================================================
//
// region ::= '{' block+ '}'
//
// block  ::= (block-label ':')? operation*
//
// block-label ::= '^' identifier ('(' block-arg (',' block-arg)* ')')?
// block-arg   ::= SSA-VALUE ':' type
//
// The first block in a function body is the entry block. Its
// arguments are the function arguments (populated from the function
// signature). Explicit block labels are optional for the entry block.
//
// ==========================================================================
// Attribute dictionary
// ==========================================================================
//
// attr-dict ::= '{' (attr-entry (',' attr-entry)*)? '}'
// attr-entry ::= identifier '=' attr-value   // key-value pair
//              | identifier                    // unit attr (presence = true)
//
// attr-value ::= INTEGER (':' type)?
//              | FLOAT (':' type)?
//              | STRING
//              | 'true' | 'false'
//              | '[' (attr-value (',' attr-value)*)? ']'
//              | '{' (attr-entry (',' attr-entry)*)? '}'
//              | '#' identifier ('<' ... '>')?
//              | type
//
// The attribute dictionary appears in the position specified by the
// op's format spec (typically after operands, before types).
//
// Example:
//   %result = tile.reduce %x {axis = 0, combine = "add"} : tile<4xf32> -> (tile<f32>)
//
// ==========================================================================
// Dynamic index lists
// ==========================================================================
//
// index-list ::= '[' index-entry (',' index-entry)* ']'
// index-entry ::= INTEGER | SSA-VALUE
//
// Mixed static/dynamic indices for slice offsets and sizes. The index
// list always glues to the preceding token (no space before '[').
//
// Examples:
//   %src[0, 0]              All static.
//   %src[%i, %j]            All dynamic.
//   %src[%i, 0]             Mixed.
//
// ==========================================================================
// Source locations
// ==========================================================================
//
// Operations may carry a trailing source location:
//
// location ::= 'loc' '(' location-body ')'
//
// location-body ::= STRING ':' INTEGER ':' INTEGER
//                      ('to' INTEGER ':' INTEGER)?
//                 | 'fused' '<' location-body (',' location-body)* '>'
//                 | 'opaque' '<' STRING ',' STRING '>'
//
// Examples:
//   loc("model.loom":42:3)
//   loc("model.loom":42:3 to 42:58)
//   loc(fused<"model.loom":42:3, "recipe.loom":15:1>)
//   loc(opaque<"torch", "node_id=42">)
//
// Locations are optional in printed output (controlled by
// --print-locations). They always round-trip: parsing captures them
// even when printing omits them.
//
// ==========================================================================
// Predicate vocabulary (where clauses and assume ops)
// ==========================================================================
//
// predicate ::= two-arg-pred '(' pred-arg ',' pred-arg ')'
//             | 'pow2' '(' pred-arg ')'
//             | 'range' '(' pred-arg ',' pred-arg ',' pred-arg ')'
// two-arg-pred ::= 'eq' | 'ne' | 'lt' | 'le' | 'gt' | 'ge'
//                | 'min' | 'max' | 'mul'
// pred-arg  ::= SSA-VALUE | INTEGER
//
// Predicates constrain dim values in where clauses and assume ops.
//
// Examples:
//   mul(%M, 16)          %M is a multiple of 16.
//   lt(%M, 1024)         %M < 1024.
//   range(%K, 32, 512)   32 <= %K <= 512.
//   pow2(%N)             %N is a power of 2.
//   eq(%M, %K)           %M == %K.
//   ne(%M, 0)            %M != 0.
//
// ==========================================================================
// Complete example
// ==========================================================================
//
// With user-assigned names (what a human or agent writes):
//
//   #enc = #q8_0<block=32>
//
//   func.def @tiled_matvec(%weights: tensor<[%M]x4xf32>, %input: tensor<4xf32>, %output: tensor<[%M]xf32>) -> (%output as tensor<[%M]xf32>) {
//     %c0 = index.constant 0 : index
//     %c4 = index.constant 4 : index
//     %tile = tensor.slice %input[%c0] : tensor<4xf32> -> (tile<4xf32>)
//     %result = scf.for %i = %c0 to %M step %c4 iter_args(%out = %output : tensor<[%M]xf32>) -> (%output as tensor<[%M]xf32>) {
//       %w = tensor.slice %weights[%i, %c0] : tensor<[%M]x4xf32> -> (tile<4x4xf32>)
//       %partial = tile.contract %w, %tile : tile<4x4xf32>, tile<4xf32> -> (tile<4xf32>)
//       %updated = tensor.update %partial, %out[%i] : tile<4xf32> -> (%out as tensor<[%M]xf32>)
//       scf.yield %updated : tensor<[%M]xf32>
//     }
//     func.return %result : tensor<[%M]xf32>
//   }
//
// Without user-assigned names (what the printer auto-generates):
//
//   func.def @tiled_matvec(%0: tensor<[%M]x4xf32>, %1: tensor<4xf32>, %2: tensor<[%M]xf32>) -> (%2 as tensor<[%M]xf32>) {
//     %3 = index.constant 0 : index
//     %4 = index.constant 4 : index
//     %5 = tensor.slice %1[%3] : tensor<4xf32> -> (tile<4xf32>)
//     %6 = scf.for %7 = %3 to %M step %4 iter_args(%8 = %2 : tensor<[%M]xf32>) -> (%2 as tensor<[%M]xf32>) {
//       %9 = tensor.slice %0[%7, %3] : tensor<[%M]x4xf32> -> (tile<4x4xf32>)
//       %10 = tile.contract %9, %5 : tile<4x4xf32>, tile<4xf32> -> (tile<4xf32>)
//       %11 = tensor.update %10, %8[%7] : tile<4xf32> -> (%8 as tensor<[%M]xf32>)
//       scf.yield %11 : tensor<[%M]xf32>
//     }
//     func.return %6 : tensor<[%M]xf32>
//   }
// clang-format on

#ifndef LOOM_FORMAT_TEXT_FORMAT_H_
#define LOOM_FORMAT_TEXT_FORMAT_H_

// This header is a specification document. It defines no C types or
// functions. The parser and printer implementations live in parser.h
// and printer.h respectively.
//
// The grammar is LL(1) with two lookahead points:
//   1. After '->', check '%' (tied) vs type keyword (fresh).
//   2. In encoding/layout attachment position (after ',' in shaped type
//      interior), check '%' (SSA value) vs '#' (static attachment).

#endif  // LOOM_FORMAT_TEXT_FORMAT_H_
