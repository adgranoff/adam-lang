/**
 * Adam Language Server — LSP implementation.
 *
 * Provides real-time diagnostics (type errors, parse errors), hover
 * information, auto-completion, and document symbols for .adam files.
 *
 * Communicates with the Rust compiler (adamc) via CLI invocation for
 * type checking and parsing. The server itself handles LSP protocol
 * framing and document synchronization.
 *
 * Supports both stdio and socket transport.
 */

import {
  createConnection,
  TextDocuments,
  ProposedFeatures,
  InitializeParams,
  InitializeResult,
  TextDocumentSyncKind,
  CompletionItem,
  CompletionItemKind,
  Hover,
  MarkupKind,
  SymbolInformation,
  SymbolKind,
  Diagnostic,
  DiagnosticSeverity,
  Position,
  Range,
} from "vscode-languageserver/node";

import { TextDocument } from "vscode-languageserver-textdocument";
import { execFile } from "child_process";
import { promisify } from "util";
import * as path from "path";

const execFileAsync = promisify(execFile);

// ── Connection setup ─────────────────────────────────────────────────

const connection = createConnection(ProposedFeatures.all);
const documents = new TextDocuments(TextDocument);

let compilerPath: string | null = null;

// ── Initialization ───────────────────────────────────────────────────

connection.onInitialize((params: InitializeParams): InitializeResult => {
  // Try to find the compiler in workspace or common locations.
  const workspaceFolders = params.workspaceFolders;
  if (workspaceFolders && workspaceFolders.length > 0) {
    const wsRoot = new URL(workspaceFolders[0].uri).pathname;
    const candidates = [
      path.join(wsRoot, "compiler", "target", "release", "adamc"),
      path.join(wsRoot, "compiler", "target", "release", "adamc.exe"),
      path.join(wsRoot, "compiler", "target", "debug", "adamc"),
      path.join(wsRoot, "compiler", "target", "debug", "adamc.exe"),
    ];
    for (const c of candidates) {
      try {
        require("fs").accessSync(c, require("fs").constants.X_OK);
        compilerPath = c;
        break;
      } catch {
        // not found, try next
      }
    }
  }

  return {
    capabilities: {
      textDocumentSync: TextDocumentSyncKind.Full,
      completionProvider: {
        resolveProvider: false,
        triggerCharacters: [".", "|"],
      },
      hoverProvider: true,
      documentSymbolProvider: true,
    },
  };
});

// ── Diagnostics ──────────────────────────────────────────────────────

/**
 * Run the compiler's type checker on a document and convert errors
 * into LSP diagnostics.
 */
async function validateDocument(doc: TextDocument): Promise<void> {
  const diagnostics: Diagnostic[] = [];
  const text = doc.getText();

  // Parse-level diagnostics from regex patterns
  parseDiagnostics(text, diagnostics);

  // If compiler is available, run type checking
  if (compilerPath) {
    try {
      const typeDiags = await runTypeChecker(text, doc);
      diagnostics.push(...typeDiags);
    } catch {
      // Compiler not available — rely on parse diagnostics only
    }
  }

  connection.sendDiagnostics({ uri: doc.uri, diagnostics });
}

/**
 * Quick parse-level diagnostics using regex patterns.
 * Catches obvious syntax issues without invoking the compiler.
 */
function parseDiagnostics(text: string, diagnostics: Diagnostic[]): void {
  const lines = text.split("\n");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check for unclosed strings
    const stringMatches = line.match(/"/g);
    if (stringMatches && stringMatches.length % 2 !== 0) {
      diagnostics.push({
        severity: DiagnosticSeverity.Error,
        range: {
          start: { line: i, character: 0 },
          end: { line: i, character: line.length },
        },
        message: "Unclosed string literal",
        source: "adam",
      });
    }
  }
}

/**
 * Invoke `adamc check` on the document source and parse error output
 * into diagnostics.
 */
async function runTypeChecker(
  text: string,
  doc: TextDocument
): Promise<Diagnostic[]> {
  if (!compilerPath) return [];

  const tmpFile = path.join(
    require("os").tmpdir(),
    `adam_lsp_${Date.now()}.adam`
  );

  try {
    require("fs").writeFileSync(tmpFile, text);

    const { stderr } = await execFileAsync(compilerPath, ["check", tmpFile], {
      timeout: 5000,
    });

    return parseCompilerErrors(stderr, doc);
  } catch (err: unknown) {
    // The compiler exits non-zero on type errors — parse stderr
    if (
      typeof err === "object" &&
      err !== null &&
      "stderr" in err &&
      typeof (err as { stderr: unknown }).stderr === "string"
    ) {
      return parseCompilerErrors((err as { stderr: string }).stderr, doc);
    }
    return [];
  } finally {
    try {
      require("fs").unlinkSync(tmpFile);
    } catch {
      // ignore cleanup errors
    }
  }
}

/**
 * Parse compiler error output into LSP Diagnostic objects.
 *
 * Expected format: "Type error at <start>..<end>: <message>"
 */
function parseCompilerErrors(
  stderr: string,
  doc: TextDocument
): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];
  const errorPattern = /Type error at (\d+)\.\.(\d+): (.+)/g;
  let match;

  while ((match = errorPattern.exec(stderr)) !== null) {
    const startOffset = parseInt(match[1], 10);
    const endOffset = parseInt(match[2], 10);
    const message = match[3];

    const startPos = doc.positionAt(startOffset);
    const endPos = doc.positionAt(endOffset);

    diagnostics.push({
      severity: DiagnosticSeverity.Error,
      range: { start: startPos, end: endPos },
      message,
      source: "adam-typecheck",
    });
  }

  // Also catch parse errors
  const parsePattern = /(?:Parse error|error): (.+)/g;
  while ((match = parsePattern.exec(stderr)) !== null) {
    if (match[1].startsWith("Type error")) continue; // already handled
    diagnostics.push({
      severity: DiagnosticSeverity.Error,
      range: {
        start: { line: 0, character: 0 },
        end: { line: 0, character: 1 },
      },
      message: match[1],
      source: "adam-parse",
    });
  }

  return diagnostics;
}

// ── Document change handler ──────────────────────────────────────────

documents.onDidChangeContent((change) => {
  validateDocument(change.document);
});

// ── Completion ───────────────────────────────────────────────────────

/** Adam language keywords and builtins for auto-completion. */
const KEYWORDS: CompletionItem[] = [
  { label: "fn", kind: CompletionItemKind.Keyword, detail: "Function declaration" },
  { label: "let", kind: CompletionItemKind.Keyword, detail: "Variable binding" },
  { label: "if", kind: CompletionItemKind.Keyword, detail: "Conditional expression" },
  { label: "else", kind: CompletionItemKind.Keyword, detail: "Else branch" },
  { label: "while", kind: CompletionItemKind.Keyword, detail: "While loop" },
  { label: "for", kind: CompletionItemKind.Keyword, detail: "For-in loop" },
  { label: "in", kind: CompletionItemKind.Keyword, detail: "Iterator keyword" },
  { label: "match", kind: CompletionItemKind.Keyword, detail: "Pattern matching" },
  { label: "return", kind: CompletionItemKind.Keyword, detail: "Return from function" },
  { label: "break", kind: CompletionItemKind.Keyword, detail: "Break from loop" },
  { label: "continue", kind: CompletionItemKind.Keyword, detail: "Continue loop" },
  { label: "true", kind: CompletionItemKind.Constant, detail: "Boolean true" },
  { label: "false", kind: CompletionItemKind.Constant, detail: "Boolean false" },
  { label: "type", kind: CompletionItemKind.Keyword, detail: "Algebraic data type" },
  { label: "struct", kind: CompletionItemKind.Keyword, detail: "Struct declaration" },
  { label: "impl", kind: CompletionItemKind.Keyword, detail: "Implementation block" },
];

const BUILTINS: CompletionItem[] = [
  {
    label: "println",
    kind: CompletionItemKind.Function,
    detail: "fn(value) -> Nil",
    documentation: "Print a value followed by a newline.",
  },
  {
    label: "len",
    kind: CompletionItemKind.Function,
    detail: "fn([T]) -> Int",
    documentation: "Return the length of an array.",
  },
  {
    label: "push",
    kind: CompletionItemKind.Function,
    detail: "fn([T], T) -> Nil",
    documentation: "Append an element to an array.",
  },
  {
    label: "clock",
    kind: CompletionItemKind.Function,
    detail: "fn() -> Float",
    documentation: "Return the current time in seconds.",
  },
];

const TYPE_NAMES: CompletionItem[] = [
  { label: "Int", kind: CompletionItemKind.TypeParameter, detail: "64-bit integer" },
  { label: "Float", kind: CompletionItemKind.TypeParameter, detail: "64-bit float" },
  { label: "String", kind: CompletionItemKind.TypeParameter, detail: "UTF-8 string" },
  { label: "Bool", kind: CompletionItemKind.TypeParameter, detail: "Boolean" },
  { label: "Nil", kind: CompletionItemKind.TypeParameter, detail: "Unit type" },
];

connection.onCompletion((_params) => {
  // Combine keywords, builtins, types, and document-local symbols
  const doc = documents.get(_params.textDocument.uri);
  const items: CompletionItem[] = [
    ...KEYWORDS,
    ...BUILTINS,
    ...TYPE_NAMES,
  ];

  // Extract user-defined names from the document
  if (doc) {
    const text = doc.getText();
    const fnPattern = /\bfn\s+([a-zA-Z_]\w*)/g;
    const letPattern = /\blet\s+([a-zA-Z_]\w*)/g;
    const seen = new Set<string>();

    let m;
    while ((m = fnPattern.exec(text)) !== null) {
      if (!seen.has(m[1])) {
        seen.add(m[1]);
        items.push({
          label: m[1],
          kind: CompletionItemKind.Function,
          detail: "User function",
        });
      }
    }
    while ((m = letPattern.exec(text)) !== null) {
      if (!seen.has(m[1])) {
        seen.add(m[1]);
        items.push({
          label: m[1],
          kind: CompletionItemKind.Variable,
          detail: "User variable",
        });
      }
    }
  }

  return items;
});

// ── Hover ────────────────────────────────────────────────────────────

connection.onHover((params): Hover | null => {
  const doc = documents.get(params.textDocument.uri);
  if (!doc) return null;

  const text = doc.getText();
  const offset = doc.offsetAt(params.position);

  // Find the word under the cursor
  const word = getWordAt(text, offset);
  if (!word) return null;

  // Check builtins
  const builtinInfo: Record<string, string> = {
    println: "```adam\nfn println(value: T) -> Nil\n```\nPrint a value followed by a newline.",
    len: "```adam\nfn len(arr: [T]) -> Int\n```\nReturn the number of elements in an array.",
    push: "```adam\nfn push(arr: [T], elem: T) -> Nil\n```\nAppend an element to an array (mutates in place).",
    clock: "```adam\nfn clock() -> Float\n```\nReturn wall-clock time in seconds.",
    Int: "Built-in type: 64-bit signed integer.",
    Float: "Built-in type: IEEE 754 double-precision float.",
    String: "Built-in type: UTF-8 string.",
    Bool: "Built-in type: boolean (`true` or `false`).",
    Nil: "Built-in type: the unit type (no meaningful value).",
  };

  if (word in builtinInfo) {
    return {
      contents: {
        kind: MarkupKind.Markdown,
        value: builtinInfo[word],
      },
    };
  }

  // Check user-defined functions (extract signature from source)
  const fnPattern = new RegExp(
    `\\bfn\\s+${escapeRegex(word)}\\s*\\(([^)]*)\\)\\s*(?:->\\s*([^{]+))?`,
  );
  const fnMatch = fnPattern.exec(text);
  if (fnMatch) {
    const params = fnMatch[1].trim();
    const ret = fnMatch[2]?.trim() || "?";
    return {
      contents: {
        kind: MarkupKind.Markdown,
        value: `\`\`\`adam\nfn ${word}(${params}) -> ${ret}\n\`\`\``,
      },
    };
  }

  return null;
});

/** Extract the word at a byte offset in the source. */
function getWordAt(text: string, offset: number): string | null {
  if (offset < 0 || offset >= text.length) return null;

  let start = offset;
  let end = offset;

  while (start > 0 && /\w/.test(text[start - 1])) start--;
  while (end < text.length && /\w/.test(text[end])) end++;

  const word = text.slice(start, end);
  return word.length > 0 ? word : null;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ── Document symbols ─────────────────────────────────────────────────

connection.onDocumentSymbol((params): SymbolInformation[] => {
  const doc = documents.get(params.textDocument.uri);
  if (!doc) return [];

  const text = doc.getText();
  const symbols: SymbolInformation[] = [];

  // Functions
  const fnPattern = /\bfn\s+([a-zA-Z_]\w*)\s*\([^)]*\)/g;
  let m;
  while ((m = fnPattern.exec(text)) !== null) {
    const pos = doc.positionAt(m.index);
    symbols.push({
      name: m[1],
      kind: SymbolKind.Function,
      location: {
        uri: params.textDocument.uri,
        range: {
          start: pos,
          end: doc.positionAt(m.index + m[0].length),
        },
      },
    });
  }

  // Let bindings
  const letPattern = /\blet\s+([a-zA-Z_]\w*)/g;
  while ((m = letPattern.exec(text)) !== null) {
    const pos = doc.positionAt(m.index);
    symbols.push({
      name: m[1],
      kind: SymbolKind.Variable,
      location: {
        uri: params.textDocument.uri,
        range: {
          start: pos,
          end: doc.positionAt(m.index + m[0].length),
        },
      },
    });
  }

  // Structs
  const structPattern = /\bstruct\s+([a-zA-Z_]\w*)/g;
  while ((m = structPattern.exec(text)) !== null) {
    const pos = doc.positionAt(m.index);
    symbols.push({
      name: m[1],
      kind: SymbolKind.Struct,
      location: {
        uri: params.textDocument.uri,
        range: {
          start: pos,
          end: doc.positionAt(m.index + m[0].length),
        },
      },
    });
  }

  // Type declarations
  const typePattern = /\btype\s+([a-zA-Z_]\w*)/g;
  while ((m = typePattern.exec(text)) !== null) {
    const pos = doc.positionAt(m.index);
    symbols.push({
      name: m[1],
      kind: SymbolKind.Enum,
      location: {
        uri: params.textDocument.uri,
        range: {
          start: pos,
          end: doc.positionAt(m.index + m[0].length),
        },
      },
    });
  }

  return symbols;
});

// ── Start ────────────────────────────────────────────────────────────

documents.listen(connection);
connection.listen();
