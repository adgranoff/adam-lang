/**
 * Playground application — wires up Monaco editor, example selector,
 * and the tree-walking interpreter for in-browser Adam execution.
 */

import * as monaco from "monaco-editor";
import { Interpreter } from "./interpreter";
import { EXAMPLES } from "./examples";

// ── Monaco editor setup ─────────────────────────────────────────────

// Register Adam as a Monaco language for syntax highlighting.
monaco.languages.register({ id: "adam" });

monaco.languages.setMonarchTokensProvider("adam", {
  keywords: [
    "fn", "let", "if", "else", "while", "for", "in",
    "return", "break", "continue", "true", "false",
    "type", "match", "struct", "impl",
  ],
  builtins: ["println", "len", "clock", "push"],
  operators: [
    "|>", "=>", "->", "&&", "||", "!",
    "==", "!=", "<=", ">=", "<", ">",
    "+", "-", "*", "/", "%", "**", "=", "|",
  ],
  tokenizer: {
    root: [
      [/\/\/.*$/, "comment"],
      [/"[^"]*"/, "string"],
      [/\d+\.\d+/, "number.float"],
      [/\d+/, "number"],
      [/[a-zA-Z_]\w*/, {
        cases: {
          "@keywords": "keyword",
          "@builtins": "support.function",
          "@default": "identifier",
        },
      }],
      [/[{}()\[\]]/, "@brackets"],
      [/\|>|=>|->|&&|\|\||[+\-*/%=<>!]+/, "operator"],
      [/[;,.]/, "delimiter"],
      [/\s+/, "white"],
    ],
  },
});

monaco.editor.defineTheme("adam-dark", {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "keyword", foreground: "ff7b72" },
    { token: "support.function", foreground: "d2a8ff" },
    { token: "string", foreground: "a5d6ff" },
    { token: "number", foreground: "79c0ff" },
    { token: "number.float", foreground: "79c0ff" },
    { token: "comment", foreground: "8b949e" },
    { token: "operator", foreground: "ff7b72" },
    { token: "identifier", foreground: "c9d1d9" },
  ],
  colors: {
    "editor.background": "#0d1117",
    "editor.foreground": "#c9d1d9",
    "editor.lineHighlightBackground": "#161b22",
    "editorCursor.foreground": "#58a6ff",
    "editor.selectionBackground": "#264f78",
    "editorLineNumber.foreground": "#484f58",
    "editorLineNumber.activeForeground": "#c9d1d9",
  },
});

const editor = monaco.editor.create(document.getElementById("editor")!, {
  value: EXAMPLES[0].code,
  language: "adam",
  theme: "adam-dark",
  fontSize: 14,
  fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  minimap: { enabled: false },
  lineNumbers: "on",
  renderLineHighlight: "line",
  scrollBeyondLastLine: false,
  padding: { top: 12 },
  automaticLayout: true,
});

// ── Output panel ────────────────────────────────────────────────────

const outputEl = document.getElementById("output")!;

function setOutput(lines: { text: string; cls: string }[]): void {
  outputEl.innerHTML = lines
    .map((l) => `<span class="${l.cls}">${escapeHtml(l.text)}</span>`)
    .join("\n");
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── Run program ─────────────────────────────────────────────────────

function runProgram(): void {
  const source = editor.getValue();
  const start = performance.now();

  try {
    const interpreter = new Interpreter();
    const result = interpreter.run(source);
    const elapsed = (performance.now() - start).toFixed(1);

    const lines: { text: string; cls: string }[] = [];
    if (result) {
      for (const line of result.split("\n")) {
        lines.push({ text: line, cls: "output-line" });
      }
    }
    lines.push({ text: `\n— executed in ${elapsed}ms`, cls: "output-info" });
    setOutput(lines);
  } catch (err) {
    const elapsed = (performance.now() - start).toFixed(1);
    const message = err instanceof Error ? err.message : String(err);
    setOutput([
      { text: `Error: ${message}`, cls: "output-error" },
      { text: `\n— failed after ${elapsed}ms`, cls: "output-info" },
    ]);
  }
}

// ── Example selector ────────────────────────────────────────────────

const examplesSelect = document.getElementById("examples") as HTMLSelectElement;

for (const example of EXAMPLES) {
  const option = document.createElement("option");
  option.value = example.name;
  option.textContent = example.name;
  examplesSelect.appendChild(option);
}

examplesSelect.addEventListener("change", () => {
  const selected = EXAMPLES.find((e) => e.name === examplesSelect.value);
  if (selected) {
    editor.setValue(selected.code);
    // Clear output when loading a new example
    outputEl.innerHTML =
      '<span class="output-info">Click "Run" or press Ctrl+Enter to execute.</span>';
  }
  // Reset dropdown to placeholder
  examplesSelect.selectedIndex = 0;
});

// ── Keyboard shortcut ───────────────────────────────────────────────

editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, runProgram);

// ── Run button ──────────────────────────────────────────────────────

document.getElementById("run-btn")!.addEventListener("click", runProgram);
