/**
 * Tree-walking interpreter for Adam, running entirely in the browser.
 *
 * This is a simplified interpreter used by the playground for immediate
 * feedback. It directly evaluates the AST without compiling to bytecode.
 * For production use, the Rust compiler + C VM pipeline is used instead.
 *
 * Supports: arithmetic, strings, booleans, variables, functions, closures,
 * arrays, if/else, while, for, pipe operator, recursion.
 */

// ── Token types ──────────────────────────────────────────────────────

type TokenType =
  | "INT" | "FLOAT" | "STRING" | "IDENT" | "TRUE" | "FALSE"
  | "FN" | "LET" | "IF" | "ELSE" | "WHILE" | "FOR" | "IN"
  | "RETURN" | "BREAK" | "CONTINUE"
  | "PLUS" | "MINUS" | "STAR" | "SLASH" | "PERCENT" | "STARSTAR"
  | "EQ" | "EQEQ" | "BANGEQ" | "LT" | "GT" | "LTEQ" | "GTEQ"
  | "AND" | "OR" | "BANG"
  | "PIPEGT" | "ARROW" | "FATARROW" | "PIPE"
  | "LPAREN" | "RPAREN" | "LBRACE" | "RBRACE" | "LBRACKET" | "RBRACKET"
  | "COMMA" | "DOT" | "COLON" | "SEMICOLON" | "NEWLINE"
  | "EOF";

interface Token {
  type: TokenType;
  value: string;
  pos: number;
}

// ── Lexer ────────────────────────────────────────────────────────────

function tokenize(source: string): Token[] {
  const tokens: Token[] = [];
  let pos = 0;

  const KEYWORDS: Record<string, TokenType> = {
    fn: "FN", let: "LET", if: "IF", else: "ELSE",
    while: "WHILE", for: "FOR", in: "IN",
    return: "RETURN", break: "BREAK", continue: "CONTINUE",
    true: "TRUE", false: "FALSE",
  };

  while (pos < source.length) {
    const ch = source[pos];

    // Skip whitespace (not newlines)
    if (ch === " " || ch === "\t" || ch === "\r") {
      pos++;
      continue;
    }

    // Newlines
    if (ch === "\n") {
      tokens.push({ type: "NEWLINE", value: "\n", pos });
      pos++;
      continue;
    }

    // Comments
    if (ch === "/" && source[pos + 1] === "/") {
      while (pos < source.length && source[pos] !== "\n") pos++;
      continue;
    }

    // Numbers
    if (/\d/.test(ch)) {
      let num = "";
      let isFloat = false;
      while (pos < source.length && /[\d.]/.test(source[pos])) {
        if (source[pos] === ".") isFloat = true;
        num += source[pos++];
      }
      tokens.push({ type: isFloat ? "FLOAT" : "INT", value: num, pos });
      continue;
    }

    // Strings
    if (ch === '"') {
      pos++;
      let str = "";
      while (pos < source.length && source[pos] !== '"') {
        if (source[pos] === "\\" && pos + 1 < source.length) {
          pos++;
          switch (source[pos]) {
            case "n": str += "\n"; break;
            case "t": str += "\t"; break;
            case "\\": str += "\\"; break;
            case '"': str += '"'; break;
            default: str += source[pos];
          }
        } else {
          str += source[pos];
        }
        pos++;
      }
      pos++; // closing quote
      tokens.push({ type: "STRING", value: str, pos });
      continue;
    }

    // Identifiers and keywords
    if (/[a-zA-Z_]/.test(ch)) {
      let ident = "";
      while (pos < source.length && /\w/.test(source[pos])) {
        ident += source[pos++];
      }
      const type = KEYWORDS[ident] || "IDENT";
      tokens.push({ type, value: ident, pos });
      continue;
    }

    // Two-character operators
    const two = source.slice(pos, pos + 2);
    const twoMap: Record<string, TokenType> = {
      "|>": "PIPEGT", "=>": "FATARROW", "->": "ARROW",
      "==": "EQEQ", "!=": "BANGEQ", "<=": "LTEQ", ">=": "GTEQ",
      "&&": "AND", "||": "OR", "**": "STARSTAR",
    };
    if (twoMap[two]) {
      tokens.push({ type: twoMap[two], value: two, pos });
      pos += 2;
      continue;
    }

    // Single-character operators
    const oneMap: Record<string, TokenType> = {
      "+": "PLUS", "-": "MINUS", "*": "STAR", "/": "SLASH", "%": "PERCENT",
      "=": "EQ", "<": "LT", ">": "GT", "!": "BANG",
      "(": "LPAREN", ")": "RPAREN", "{": "LBRACE", "}": "RBRACE",
      "[": "LBRACKET", "]": "RBRACKET",
      ",": "COMMA", ".": "DOT", ":": "COLON", ";": "SEMICOLON",
      "|": "PIPE",
    };
    if (oneMap[ch]) {
      tokens.push({ type: oneMap[ch], value: ch, pos });
      pos++;
      continue;
    }

    pos++; // skip unknown
  }

  tokens.push({ type: "EOF", value: "", pos });
  return tokens;
}

// ── AST ──────────────────────────────────────────────────────────────

type Expr =
  | { type: "int"; value: number }
  | { type: "float"; value: number }
  | { type: "string"; value: string }
  | { type: "bool"; value: boolean }
  | { type: "var"; name: string }
  | { type: "binary"; op: string; left: Expr; right: Expr }
  | { type: "unary"; op: string; operand: Expr }
  | { type: "call"; callee: Expr; args: Expr[] }
  | { type: "pipe"; left: Expr; right: Expr }
  | { type: "index"; object: Expr; index: Expr }
  | { type: "field"; object: Expr; field: string }
  | { type: "array"; elements: Expr[] }
  | { type: "if"; condition: Expr; then: Expr; else_?: Expr }
  | { type: "block"; stmts: Stmt[]; expr?: Expr }
  | { type: "lambda"; params: string[]; body: Expr }
  | { type: "assign"; target: Expr; value: Expr }
  | { type: "while"; condition: Expr; body: Expr }
  | { type: "for"; variable: string; iterator: Expr; body: Expr };

type Stmt =
  | { type: "let"; name: string; value: Expr }
  | { type: "fn"; name: string; params: string[]; body: Expr }
  | { type: "expr"; expr: Expr }
  | { type: "return"; value?: Expr };

// ── Parser ───────────────────────────────────────────────────────────

class Parser {
  private tokens: Token[];
  private pos = 0;

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  private peek(): Token {
    return this.tokens[this.pos] || { type: "EOF", value: "", pos: 0 };
  }

  private advance(): Token {
    return this.tokens[this.pos++];
  }

  private expect(type: TokenType): Token {
    const tok = this.advance();
    if (tok.type !== type) {
      throw new Error(`Expected ${type}, got ${tok.type} ('${tok.value}')`);
    }
    return tok;
  }

  private match(type: TokenType): boolean {
    if (this.peek().type === type) {
      this.advance();
      return true;
    }
    return false;
  }

  private skipNewlines(): void {
    while (this.peek().type === "NEWLINE") this.advance();
  }

  parse(): Stmt[] {
    const stmts: Stmt[] = [];
    this.skipNewlines();
    while (this.peek().type !== "EOF") {
      stmts.push(this.parseStmt());
      this.skipNewlines();
    }
    return stmts;
  }

  private parseStmt(): Stmt {
    this.skipNewlines();
    const tok = this.peek();

    if (tok.type === "LET") {
      this.advance();
      const name = this.expect("IDENT").value;
      this.expect("EQ");
      const value = this.parseExpr();
      return { type: "let", name, value };
    }

    if (tok.type === "FN") {
      this.advance();
      const name = this.expect("IDENT").value;
      this.expect("LPAREN");
      const params: string[] = [];
      if (this.peek().type !== "RPAREN") {
        params.push(this.expect("IDENT").value);
        while (this.match("COMMA")) {
          // skip optional type annotations
          params.push(this.expect("IDENT").value);
        }
      }
      this.expect("RPAREN");
      // skip optional return type
      if (this.peek().type === "ARROW") {
        this.advance();
        this.expect("IDENT"); // type name
      }
      const body = this.parseBlock();
      return { type: "fn", name, params, body };
    }

    if (tok.type === "RETURN") {
      this.advance();
      if (
        this.peek().type === "NEWLINE" ||
        this.peek().type === "EOF" ||
        this.peek().type === "RBRACE"
      ) {
        return { type: "return" };
      }
      return { type: "return", value: this.parseExpr() };
    }

    const expr = this.parseExpr();
    return { type: "expr", expr };
  }

  private parseExpr(): Expr {
    return this.parsePipe();
  }

  private parsePipe(): Expr {
    let left = this.parseAssign();
    while (this.peek().type === "PIPEGT") {
      this.advance();
      const right = this.parseAssign();
      left = { type: "pipe", left, right };
    }
    return left;
  }

  private parseAssign(): Expr {
    const left = this.parseOr();
    if (this.peek().type === "EQ") {
      this.advance();
      const value = this.parseExpr();
      return { type: "assign", target: left, value };
    }
    return left;
  }

  private parseOr(): Expr {
    let left = this.parseAnd();
    while (this.peek().type === "OR") {
      this.advance();
      left = { type: "binary", op: "||", left, right: this.parseAnd() };
    }
    return left;
  }

  private parseAnd(): Expr {
    let left = this.parseEquality();
    while (this.peek().type === "AND") {
      this.advance();
      left = { type: "binary", op: "&&", left, right: this.parseEquality() };
    }
    return left;
  }

  private parseEquality(): Expr {
    let left = this.parseComparison();
    while (this.peek().type === "EQEQ" || this.peek().type === "BANGEQ") {
      const op = this.advance().value;
      left = { type: "binary", op, left, right: this.parseComparison() };
    }
    return left;
  }

  private parseComparison(): Expr {
    let left = this.parseAddition();
    while (["LT", "GT", "LTEQ", "GTEQ"].includes(this.peek().type)) {
      const op = this.advance().value;
      left = { type: "binary", op, left, right: this.parseAddition() };
    }
    return left;
  }

  private parseAddition(): Expr {
    let left = this.parseMultiplication();
    while (this.peek().type === "PLUS" || this.peek().type === "MINUS") {
      const op = this.advance().value;
      left = { type: "binary", op, left, right: this.parseMultiplication() };
    }
    return left;
  }

  private parseMultiplication(): Expr {
    let left = this.parsePower();
    while (["STAR", "SLASH", "PERCENT"].includes(this.peek().type)) {
      const op = this.advance().value;
      left = { type: "binary", op, left, right: this.parsePower() };
    }
    return left;
  }

  private parsePower(): Expr {
    const left = this.parseUnary();
    if (this.peek().type === "STARSTAR") {
      this.advance();
      return { type: "binary", op: "**", left, right: this.parsePower() };
    }
    return left;
  }

  private parseUnary(): Expr {
    if (this.peek().type === "MINUS") {
      this.advance();
      return { type: "unary", op: "-", operand: this.parseUnary() };
    }
    if (this.peek().type === "BANG") {
      this.advance();
      return { type: "unary", op: "!", operand: this.parseUnary() };
    }
    return this.parsePostfix();
  }

  private parsePostfix(): Expr {
    let expr = this.parsePrimary();
    while (true) {
      if (this.peek().type === "LPAREN") {
        this.advance();
        const args: Expr[] = [];
        if (this.peek().type !== "RPAREN") {
          args.push(this.parseExpr());
          while (this.match("COMMA")) args.push(this.parseExpr());
        }
        this.expect("RPAREN");
        expr = { type: "call", callee: expr, args };
      } else if (this.peek().type === "LBRACKET") {
        this.advance();
        const index = this.parseExpr();
        this.expect("RBRACKET");
        expr = { type: "index", object: expr, index };
      } else if (this.peek().type === "DOT") {
        this.advance();
        const field = this.expect("IDENT").value;
        expr = { type: "field", object: expr, field };
      } else {
        break;
      }
    }
    return expr;
  }

  private parsePrimary(): Expr {
    const tok = this.peek();

    if (tok.type === "INT") {
      this.advance();
      return { type: "int", value: parseInt(tok.value, 10) };
    }
    if (tok.type === "FLOAT") {
      this.advance();
      return { type: "float", value: parseFloat(tok.value) };
    }
    if (tok.type === "STRING") {
      this.advance();
      return { type: "string", value: tok.value };
    }
    if (tok.type === "TRUE") {
      this.advance();
      return { type: "bool", value: true };
    }
    if (tok.type === "FALSE") {
      this.advance();
      return { type: "bool", value: false };
    }
    if (tok.type === "IDENT") {
      this.advance();
      return { type: "var", name: tok.value };
    }
    if (tok.type === "LPAREN") {
      this.advance();
      const expr = this.parseExpr();
      this.expect("RPAREN");
      return expr;
    }
    if (tok.type === "LBRACKET") {
      this.advance();
      const elements: Expr[] = [];
      if (this.peek().type !== "RBRACKET") {
        elements.push(this.parseExpr());
        while (this.match("COMMA")) elements.push(this.parseExpr());
      }
      this.expect("RBRACKET");
      return { type: "array", elements };
    }
    if (tok.type === "LBRACE") {
      return this.parseBlock();
    }
    if (tok.type === "IF") {
      return this.parseIf();
    }
    if (tok.type === "WHILE") {
      return this.parseWhile();
    }
    if (tok.type === "FOR") {
      return this.parseFor();
    }
    if (tok.type === "PIPE") {
      return this.parseLambda();
    }

    throw new Error(`Unexpected token: ${tok.type} ('${tok.value}')`);
  }

  private parseBlock(): Expr {
    this.expect("LBRACE");
    this.skipNewlines();
    const stmts: Stmt[] = [];
    let tailExpr: Expr | undefined;

    while (this.peek().type !== "RBRACE" && this.peek().type !== "EOF") {
      this.skipNewlines();
      if (this.peek().type === "RBRACE") break;

      const stmt = this.parseStmt();
      this.skipNewlines();

      // If this is an expression statement and the next token is },
      // it's the tail expression.
      if (
        stmt.type === "expr" &&
        this.peek().type === "RBRACE"
      ) {
        tailExpr = stmt.expr;
      } else {
        stmts.push(stmt);
      }
    }
    this.expect("RBRACE");
    return { type: "block", stmts, expr: tailExpr };
  }

  private parseIf(): Expr {
    this.expect("IF");
    const condition = this.parseExpr();
    const then = this.parseBlock();
    let else_: Expr | undefined;
    if (this.peek().type === "ELSE") {
      this.advance();
      if (this.peek().type === "IF") {
        else_ = this.parseIf();
      } else {
        else_ = this.parseBlock();
      }
    }
    return { type: "if", condition, then, else_ };
  }

  private parseWhile(): Expr {
    this.expect("WHILE");
    const condition = this.parseExpr();
    const body = this.parseBlock();
    return { type: "while", condition, body };
  }

  private parseFor(): Expr {
    this.expect("FOR");
    const variable = this.expect("IDENT").value;
    this.expect("IN");
    const iterator = this.parseExpr();
    const body = this.parseBlock();
    return { type: "for", variable, iterator, body };
  }

  private parseLambda(): Expr {
    this.expect("PIPE");
    const params: string[] = [];
    if (this.peek().type !== "PIPE") {
      params.push(this.expect("IDENT").value);
      while (this.match("COMMA")) params.push(this.expect("IDENT").value);
    }
    this.expect("PIPE");
    const body =
      this.peek().type === "LBRACE" ? this.parseBlock() : this.parseExpr();
    return { type: "lambda", params, body };
  }
}

// ── Interpreter ──────────────────────────────────────────────────────

type Value = number | string | boolean | null | Value[] | Closure;

interface Closure {
  __type: "closure";
  params: string[];
  body: Expr;
  env: Env;
}

class BreakSignal {
  constructor() {}
}
class ContinueSignal {
  constructor() {}
}
class ReturnSignal {
  value: Value;
  constructor(value: Value) {
    this.value = value;
  }
}

class Env {
  private bindings: Map<string, Value>;
  private parent: Env | null;

  constructor(parent: Env | null = null) {
    this.bindings = new Map();
    this.parent = parent;
  }

  get(name: string): Value {
    if (this.bindings.has(name)) return this.bindings.get(name)!;
    if (this.parent) return this.parent.get(name);
    throw new Error(`Undefined variable: ${name}`);
  }

  set(name: string, value: Value): void {
    if (this.bindings.has(name)) {
      this.bindings.set(name, value);
      return;
    }
    if (this.parent) {
      try {
        this.parent.set(name, value);
        return;
      } catch {
        // fall through
      }
    }
    this.bindings.set(name, value);
  }

  define(name: string, value: Value): void {
    this.bindings.set(name, value);
  }
}

export class Interpreter {
  private output: string[] = [];
  private env: Env;

  constructor() {
    this.env = new Env();
    this.registerBuiltins();
  }

  private registerBuiltins(): void {
    // println
    this.env.define("println", {
      __type: "closure" as const,
      params: ["x"],
      body: { type: "var" as const, name: "__native_println" },
      env: this.env,
    });

    // len
    this.env.define("len", {
      __type: "closure" as const,
      params: ["x"],
      body: { type: "var" as const, name: "__native_len" },
      env: this.env,
    });

    // clock
    this.env.define("clock", {
      __type: "closure" as const,
      params: [],
      body: { type: "var" as const, name: "__native_clock" },
      env: this.env,
    });

    // push
    this.env.define("push", {
      __type: "closure" as const,
      params: ["arr", "elem"],
      body: { type: "var" as const, name: "__native_push" },
      env: this.env,
    });
  }

  run(source: string): string {
    this.output = [];
    const tokens = tokenize(source);
    const parser = new Parser(tokens);
    const stmts = parser.parse();

    for (const stmt of stmts) {
      this.execStmt(stmt, this.env);
    }

    return this.output.join("\n");
  }

  private execStmt(stmt: Stmt, env: Env): void {
    switch (stmt.type) {
      case "let":
        env.define(stmt.name, this.evalExpr(stmt.value, env));
        break;
      case "fn":
        env.define(stmt.name, {
          __type: "closure",
          params: stmt.params,
          body: stmt.body,
          env,
        });
        break;
      case "expr":
        this.evalExpr(stmt.expr, env);
        break;
      case "return":
        throw new ReturnSignal(
          stmt.value ? this.evalExpr(stmt.value, env) : null
        );
    }
  }

  private evalExpr(expr: Expr, env: Env): Value {
    switch (expr.type) {
      case "int":
      case "float":
        return expr.value;
      case "string":
        return expr.value;
      case "bool":
        return expr.value;

      case "var":
        return env.get(expr.name);

      case "binary":
        return this.evalBinary(expr.op, expr.left, expr.right, env);

      case "unary":
        return this.evalUnary(expr.op, expr.operand, env);

      case "call":
        return this.evalCall(expr.callee, expr.args, env);

      case "pipe": {
        const arg = this.evalExpr(expr.left, env);
        const fn = this.evalExpr(expr.right, env);
        return this.callClosure(fn as Closure, [arg]);
      }

      case "index": {
        const arr = this.evalExpr(expr.object, env) as Value[];
        const idx = this.evalExpr(expr.index, env) as number;
        return arr[idx];
      }

      case "field": {
        const obj = this.evalExpr(expr.object, env) as Record<string, Value>;
        return obj[expr.field];
      }

      case "array":
        return expr.elements.map((e) => this.evalExpr(e, env));

      case "if": {
        const cond = this.evalExpr(expr.condition, env);
        if (cond) {
          return this.evalExpr(expr.then, env);
        } else if (expr.else_) {
          return this.evalExpr(expr.else_, env);
        }
        return null;
      }

      case "block": {
        const blockEnv = new Env(env);
        let result: Value = null;
        for (const s of expr.stmts) {
          this.execStmt(s, blockEnv);
        }
        if (expr.expr) {
          result = this.evalExpr(expr.expr, blockEnv);
        }
        return result;
      }

      case "lambda":
        return {
          __type: "closure",
          params: expr.params,
          body: expr.body,
          env,
        };

      case "assign": {
        const value = this.evalExpr(expr.value, env);
        if (expr.target.type === "var") {
          env.set(expr.target.name, value);
        } else if (expr.target.type === "index") {
          const arr = this.evalExpr(expr.target.object, env) as Value[];
          const idx = this.evalExpr(expr.target.index, env) as number;
          arr[idx] = value;
        }
        return value;
      }

      case "while": {
        while (this.evalExpr(expr.condition, env)) {
          try {
            this.evalExpr(expr.body, env);
          } catch (e) {
            if (e instanceof BreakSignal) break;
            if (e instanceof ContinueSignal) continue;
            throw e;
          }
        }
        return null;
      }

      case "for": {
        const iter = this.evalExpr(expr.iterator, env) as Value[];
        for (const item of iter) {
          const loopEnv = new Env(env);
          loopEnv.define(expr.variable, item);
          try {
            this.evalExpr(expr.body, loopEnv);
          } catch (e) {
            if (e instanceof BreakSignal) break;
            if (e instanceof ContinueSignal) continue;
            throw e;
          }
        }
        return null;
      }
    }
  }

  private evalBinary(op: string, left: Expr, right: Expr, env: Env): Value {
    // Short-circuit for logical operators
    if (op === "&&") {
      const l = this.evalExpr(left, env);
      return l ? this.evalExpr(right, env) : l;
    }
    if (op === "||") {
      const l = this.evalExpr(left, env);
      return l ? l : this.evalExpr(right, env);
    }

    const l = this.evalExpr(left, env);
    const r = this.evalExpr(right, env);

    switch (op) {
      case "+":
        if (typeof l === "string" || typeof r === "string") return `${l}${r}`;
        return (l as number) + (r as number);
      case "-": return (l as number) - (r as number);
      case "*": return (l as number) * (r as number);
      case "/": return (l as number) / (r as number);
      case "%": return (l as number) % (r as number);
      case "**": return Math.pow(l as number, r as number);
      case "==": return l === r;
      case "!=": return l !== r;
      case "<": return (l as number) < (r as number);
      case ">": return (l as number) > (r as number);
      case "<=": return (l as number) <= (r as number);
      case ">=": return (l as number) >= (r as number);
      default: throw new Error(`Unknown operator: ${op}`);
    }
  }

  private evalUnary(op: string, operand: Expr, env: Env): Value {
    const val = this.evalExpr(operand, env);
    switch (op) {
      case "-": return -(val as number);
      case "!": return !val;
      default: throw new Error(`Unknown unary operator: ${op}`);
    }
  }

  private evalCall(callee: Expr, args: Expr[], env: Env): Value {
    const fn = this.evalExpr(callee, env);
    const argValues = args.map((a) => this.evalExpr(a, env));
    return this.callClosure(fn as Closure, argValues);
  }

  private callClosure(closure: Closure, args: Value[]): Value {
    // Handle native functions
    if (
      closure.body.type === "var" &&
      (closure.body as { type: "var"; name: string }).name.startsWith("__native_")
    ) {
      const name = (closure.body as { type: "var"; name: string }).name;
      switch (name) {
        case "__native_println":
          this.output.push(this.stringify(args[0]));
          return null;
        case "__native_len":
          return (args[0] as Value[]).length;
        case "__native_clock":
          return Date.now() / 1000;
        case "__native_push":
          (args[0] as Value[]).push(args[1]);
          return null;
        default:
          throw new Error(`Unknown native: ${name}`);
      }
    }

    const callEnv = new Env(closure.env);
    for (let i = 0; i < closure.params.length; i++) {
      callEnv.define(closure.params[i], args[i] ?? null);
    }

    try {
      return this.evalExpr(closure.body, callEnv);
    } catch (e) {
      if (e instanceof ReturnSignal) return e.value;
      throw e;
    }
  }

  private stringify(value: Value): string {
    if (value === null) return "nil";
    if (value === true) return "true";
    if (value === false) return "false";
    if (Array.isArray(value)) {
      return "[" + value.map((v) => this.stringify(v)).join(", ") + "]";
    }
    if (typeof value === "object" && "__type" in value) {
      return "<closure>";
    }
    return String(value);
  }
}
