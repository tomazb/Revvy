// src/deepReviewer.ts
// Deep Review engine: agentic code review using vscode.lm tool-calling.
//
// The agent explores the workspace with 5 read-only tools before producing a
// structured JSON review identical in shape to Quick Review (ReviewResult).
//
// Cost controls (7 total):
//   1. MAX_AGENT_ROUNDS       — hard cap on conversation turns
//   2. MAX_TOOL_CALLS         — hard cap on total tool executions
//   3. MAX_TOOL_RESULT_CHARS  — truncate individual tool outputs
//   4. MAX_CONVERSATION_CHARS — abort if total message history gets too large
//   5. toolCache              — exact-match dedup (skip redundant tool calls)
//   6. isValidToolInput()     — reject garbage/oversized inputs before execution
//   7. empty-assistant guard  — prevent corrupt conversation history

import * as vscode from 'vscode';
import * as nodePath from 'path';
import { ReviewProfile, ReviewRule } from './ruleLoader';
import { AIKeys, StreamChunkCallback } from './aiBackend';
import { splitDiffByFile } from './diffFilter';
import {
  ReviewResult,
  ReviewComment,
  ReviewTest,
  ReviewSource,
} from './reviewer';

// ─────────────────────────────────────────────────────────────────────────────
// Cost-control constants
// ─────────────────────────────────────────────────────────────────────────────

const MAX_AGENT_ROUNDS               = 30;
const MAX_TOOL_CALLS                 = 50;
const MAX_TOOL_RESULT_CHARS          = 12_000;
const MAX_CONVERSATION_CHARS         = 200_000;
// Tight trim for programmatic grounding searches — we only need the #define
// line, not 12 KB of surrounding code.
const GROUNDING_SYMBOL_RESULT_CHARS  = 3_000;

// ─────────────────────────────────────────────────────────────────────────────
// Output-channel logger (lazy — created on first Deep Review)
// Open in VS Code: View → Output → "Revvy — Deep Review"
// ─────────────────────────────────────────────────────────────────────────────

let _channel: vscode.OutputChannel | undefined;

function getChannel(): vscode.OutputChannel {
  if (!_channel) {
    _channel = vscode.window.createOutputChannel('Revvy — Deep Review');
  }
  return _channel;
}

function log(msg: string): void {
  const t = new Date().toTimeString().slice(0, 8);
  getChannel().appendLine(`[${t}] ${msg}`);
}

function logResult(label: string, result: string): void {
  const preview = result.length > 300 ? result.slice(0, 300) + '…' : result;
  log(`  → ${label}: ${preview.replace(/\n/g, '↵')}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool definitions (JSON Schema inputSchema — no external deps required)
// ─────────────────────────────────────────────────────────────────────────────

const DEEP_REVIEW_TOOLS: vscode.LanguageModelChatTool[] = [
  {
    name: 'readFile',
    description:
      'Read a file from the workspace. Returns up to 500 lines (or the requested range). ' +
      'Use this to understand code that was changed or that references changed symbols.',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'File path relative to the workspace root (e.g. "src/foo.c")',
        },
        startLine: {
          type: 'number',
          description: 'First line to return, 1-indexed (optional)',
        },
        endLine: {
          type: 'number',
          description: 'Last line to return, 1-indexed (optional)',
        },
      },
      required: ['path'],
    },
  },
  {
    name: 'listChangedFiles',
    description:
      'List all files that were changed in this diff. ' +
      'Use this first to understand the scope of the change.',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'getFileDiff',
    description:
      'Get the full diff section for a specific file in this review. ' +
      'Use this to inspect exactly what changed in a file.',
    inputSchema: {
      type: 'object',
      properties: {
        filePath: {
          type: 'string',
          description: 'Path of the file as shown in the diff header',
        },
      },
      required: ['filePath'],
    },
  },
  {
    name: 'searchSymbol',
    description:
      'Search workspace files for occurrences of a symbol or text pattern. ' +
      'Use this to find callers, usages, and cross-file references to changed APIs.',
    inputSchema: {
      type: 'object',
      properties: {
        pattern: {
          type: 'string',
          description: 'Symbol name or text pattern to search for (case-sensitive)',
        },
        fileGlob: {
          type: 'string',
          description: 'Optional glob to restrict search (e.g. "**/*.c", "**/*.ts")',
        },
      },
      required: ['pattern'],
    },
  },
  {
    name: 'listWorkspaceFiles',
    description:
      'List files in the workspace matching a glob pattern. ' +
      'Use this to understand project structure before reading specific files.',
    inputSchema: {
      type: 'object',
      properties: {
        glob: {
          type: 'string',
          description: 'Glob pattern to match (default: "**/*")',
        },
      },
    },
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Cost-control helpers
// ─────────────────────────────────────────────────────────────────────────────

export function trimToolResult(s: string, limit: number = MAX_TOOL_RESULT_CHARS): string {
  if (s.length <= limit) { return s; }
  return s.slice(0, limit) + `\n...[truncated — showing first ${limit} chars]`;
}

export function isValidToolInput(call: vscode.LanguageModelToolCallPart): boolean {
  const inputStr = JSON.stringify(call.input ?? {});

  // Reject oversized inputs
  if (inputStr.length > 4_000) { return false; }

  // Reject empty inputs for tools that require parameters
  if (!call.input || Object.keys(call.input).length === 0) {
    const requiresInput = ['readFile', 'getFileDiff', 'searchSymbol'];
    if (requiresInput.includes(call.name)) { return false; }
  }

  // Per-tool required-field validation
  const inp = call.input as Record<string, unknown>;
  switch (call.name) {
    case 'readFile':
      if (typeof inp.path !== 'string' || inp.path.trim() === '') { return false; }
      break;
    case 'getFileDiff':
      if (typeof inp.filePath !== 'string' || inp.filePath.trim() === '') { return false; }
      break;
    case 'searchSymbol':
      if (typeof inp.pattern !== 'string' || inp.pattern.trim() === '') { return false; }
      break;
  }

  return true;
}

export function getConversationSize(messages: vscode.LanguageModelChatMessage[]): number {
  let total = 0;
  for (const msg of messages) {
    for (const part of (msg.content as any[])) {
      if (part instanceof vscode.LanguageModelTextPart) {
        total += part.value.length;
      } else if (part instanceof vscode.LanguageModelToolResultPart) {
        // content is string | LanguageModelTextPart[]
        const c = (part as any).content;
        if (typeof c === 'string') {
          total += c.length;
        } else if (Array.isArray(c)) {
          for (const item of c) {
            if (item instanceof vscode.LanguageModelTextPart) { total += item.value.length; }
            else if (typeof item?.value === 'string') { total += item.value.length; }
          }
        }
      }
    }
  }
  return total;
}

// ─────────────────────────────────────────────────────────────────────────────
// Symbol extraction (mirrors reviewer.ts — not exported there, duplicated here)
// ─────────────────────────────────────────────────────────────────────────────

// ALL_CAPS identifiers that are language primitives or universal noise — not
// worth searching because they are never meaningfully defined in user code.
const CAPS_NOISE = new Set([
  'NULL', 'TRUE', 'FALSE', 'NONE', 'VOID', 'BOOL', 'BYTE', 'WORD', 'DWORD',
  'QWORD', 'INT', 'CHAR', 'FLOAT', 'DOUBLE', 'LONG', 'SHORT', 'UINT', 'ULONG',
  'EOF', 'NAN', 'INF',
]);

// Language keywords that appear in comparison positions but are not constants.
const COMPARISON_NOISE = new Set([
  'null', 'true', 'false', 'none', 'undefined', 'nil', 'void',
  'this', 'self', 'super',
]);

// C preprocessor directive keywords — never user-defined identifiers.
// Used to filter the output of Pattern A (preprocessor condition scanning).
const PREPROCESSOR_KEYWORD_NOISE = new Set([
  'defined', 'if', 'ifdef', 'ifndef', 'elif', 'else', 'endif',
  'include', 'pragma', 'error', 'warning', 'undef', 'define',
]);

const MAX_SYMBOLS = 50; // soft cap on extracted symbol list

export function extractChangedSymbols(diff: string): string[] {
  const symbols = new Set<string>();
  const capsSymbols = new Set<string>(); // collected separately, appended last

  const addedLines = diff
    .split('\n')
    .filter(l => l.startsWith('+') && !l.startsWith('+++'));

  for (const line of addedLines) {
    const content = line.slice(1).trim();

    // ── Existing patterns ────────────────────────────────────────────────────

    // Function / method calls and definitions
    let m = content.match(/\b([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
    if (m && !['if', 'for', 'while', 'switch', 'return', 'sizeof', 'typeof', 'alignof', 'defined'].includes(m[1])) {
      symbols.add(m[1]);
    }

    // #define NAME
    m = content.match(/^#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)/);
    if (m) { symbols.add(m[1]); }

    // struct / enum / union / typedef NAME
    m = content.match(/\b(?:struct|enum|union|typedef)\s+([A-Za-z_][A-Za-z0-9_]*)/);
    if (m) { symbols.add(m[1]); }

    // JS: function NAME
    m = content.match(/\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)/);
    if (m) { symbols.add(m[1]); }

    // JS/TS: const / let / var NAME =
    m = content.match(/\b(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=/);
    if (m) { symbols.add(m[1]); }

    // Python: def NAME(
    m = content.match(/^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/);
    if (m) { symbols.add(m[1]); }

    // ── Pattern A: preprocessor condition macros ─────────────────────────────
    // Scans ALL identifiers on a preprocessor condition line and filters out
    // directive keywords so that forms like "if defined(X)", "ifdef X", and
    // "if defined(A) && defined(B)" all yield the real macro names.
    if (/^#\s*(?:if|ifdef|ifndef|elif)\b/.test(content)) {
      for (const aMatch of content.matchAll(/\b([A-Za-z_][A-Za-z0-9_]+)\b/g)) {
        if (!PREPROCESSOR_KEYWORD_NOISE.has(aMatch[1])) { symbols.add(aMatch[1]); }
      }
    }

    // ── Pattern B: comparison RHS identifiers (cross-stack) ──────────────────
    // Catches named constants/enums on the right of ==, !=, >=, <=
    for (const rhsMatch of content.matchAll(/(?:==|!=|>=|<=)\s*([A-Za-z_][A-Za-z0-9_]{2,})\b/g)) {
      const sym = rhsMatch[1];
      if (!COMPARISON_NOISE.has(sym) && !CAPS_NOISE.has(sym)) { symbols.add(sym); }
    }

    // ── Pattern C: ALL_CAPS identifiers (universal constant convention) ───────
    // Covers CONFIG_*, K_MAP_*, HTTP_STATUS_OK, MAX_RETRIES, etc.
    // Collected separately so they are appended last (lowest priority).
    for (const capsMatch of content.matchAll(/\b([A-Z][A-Z0-9_]{3,})\b/g)) {
      const sym = capsMatch[1];
      if (!CAPS_NOISE.has(sym)) { capsSymbols.add(sym); }
    }

    // ── Pattern D: comparison LHS identifiers ────────────────────────────────
    // Catches named constants/enums on the LEFT of ==, !=, >=, <=
    // Only uppercase-starting identifiers — named constants start uppercase;
    // local variables in all stacks typically start lowercase.
    for (const lhsMatch of content.matchAll(/\b([A-Z][A-Za-z0-9_]{2,})\s*(?:!=|==|>=|<=)/g)) {
      const sym = lhsMatch[1];
      if (!COMPARISON_NOISE.has(sym) && !CAPS_NOISE.has(sym)) { symbols.add(sym); }
    }

    // ── Pattern E: return value identifiers ──────────────────────────────────
    // Catches named error codes / enum values used in return statements.
    // Only uppercase-starting identifiers (same rationale as Pattern D).
    for (const retMatch of content.matchAll(/\breturn\s+([A-Z][A-Za-z0-9_]{2,})\s*[;,)]/g)) {
      const sym = retMatch[1];
      if (!COMPARISON_NOISE.has(sym) && !CAPS_NOISE.has(sym)) { symbols.add(sym); }
    }

    // ── Pattern F: assignment RHS named constants ─────────────────────────────
    // Catches identifiers on the RHS of assignments, stripping C-style casts.
    // Only uppercase-starting, non-function-call identifiers.
    // Added to the general SYMBOLS list (not tier2) — assignment RHS has more
    // noise than comparisons/returns; left to the model to search selectively.
    for (const assignMatch of content.matchAll(/=\s*(?:\([^)]+\)\s*)?([A-Z][A-Za-z0-9_]{2,})\b(?!\s*[(<])/g)) {
      const sym = assignMatch[1];
      if (!COMPARISON_NOISE.has(sym) && !CAPS_NOISE.has(sym)) { symbols.add(sym); }
    }
  }

  // Merge: prioritised symbols first, ALL_CAPS fill up to MAX_SYMBOLS.
  // Apply the cap on result too — function-call patterns can fill symbols
  // beyond MAX_SYMBOLS when a diff has many call sites.
  const result = Array.from(symbols).slice(0, MAX_SYMBOLS);
  for (const sym of capsSymbols) {
    if (result.length >= MAX_SYMBOLS) { break; }
    if (!symbols.has(sym)) { result.push(sym); }
  }

  return result;
}

// Extracts the highest-priority symbols for programmatic grounding —
// these are resolved in code before the agent loop so the model enters
// with compile-time flag values and named-constant definitions already known.
//
// Tier 1 — Pattern A: preprocessor condition macros (#if / #ifdef / #elif).
//   e.g.  #if (CONFIG_FEATURE_DYNAMIC_EXTFLASH_SIZE == 1)  → CONFIG_FEATURE_DYNAMIC_EXTFLASH_SIZE
//   These determine which code path is compiled; knowing their value converts
//   a conditional finding into a definitive one.
//
// Tier 2 — Pattern B: named constants on the RHS of comparisons.
//   e.g.  result != Ext_Flash_OpSuccess  →  Ext_Flash_OpSuccess
//   These are enum/constant values used in error checks.
//
// No hard cap on either tier — Set deduplication handles repeated symbols;
// the toolCache in runDeepReview handles any duplicate search calls.
// Tier 1 is always resolved first (higher priority).
export function extractHighPrioritySymbols(diff: string): { tier1: string[]; tier2: string[] } {
  const tier1 = new Set<string>();
  const tier2 = new Set<string>();

  for (const rawLine of diff.split('\n')) {
    if (!rawLine.startsWith('+') || rawLine.startsWith('+++')) { continue; }
    const content = rawLine.slice(1).trim();

    // Tier 1: ALL identifiers on preprocessor condition lines, keyword-filtered.
    // Handles "if defined(X)", "ifdef X", "if defined(A) && defined(B)", etc.
    if (/^#\s*(?:if|ifdef|ifndef|elif)\b/.test(content)) {
      for (const aMatch of content.matchAll(/\b([A-Za-z_][A-Za-z0-9_]+)\b/g)) {
        if (!PREPROCESSOR_KEYWORD_NOISE.has(aMatch[1])) { tier1.add(aMatch[1]); }
      }
    }

    // Tier 2: named constants on comparison RHS.
    // Symbols already in tier1 are not duplicated.
    for (const rhsMatch of content.matchAll(/(?:==|!=|>=|<=)\s*([A-Za-z_][A-Za-z0-9_]{2,})\b/g)) {
      const sym = rhsMatch[1];
      if (!tier1.has(sym) && !COMPARISON_NOISE.has(sym) && !CAPS_NOISE.has(sym)) {
        tier2.add(sym);
      }
    }

    // Tier 2: named constants on comparison LHS (uppercase-starting only).
    for (const lhsMatch of content.matchAll(/\b([A-Z][A-Za-z0-9_]{2,})\s*(?:!=|==|>=|<=)/g)) {
      const sym = lhsMatch[1];
      if (!tier1.has(sym) && !COMPARISON_NOISE.has(sym) && !CAPS_NOISE.has(sym)) {
        tier2.add(sym);
      }
    }

    // Tier 2: return value identifiers (uppercase-starting named constants).
    for (const retMatch of content.matchAll(/\breturn\s+([A-Z][A-Za-z0-9_]{2,})\s*[;,)]/g)) {
      const sym = retMatch[1];
      if (!tier1.has(sym) && !COMPARISON_NOISE.has(sym) && !CAPS_NOISE.has(sym)) {
        tier2.add(sym);
      }
    }
  }

  return { tier1: Array.from(tier1), tier2: Array.from(tier2) };
}

/**
 * Extract searchable code identifiers from free-form requirement text.
 *
 * Looks for tokens that are likely real code symbols rather than English prose:
 *   - Contains an underscore  (CONFIG_FEATURE_X, Drv_APP_IN_MTR, etc.)
 *   - Has a mid-word uppercase transition (CamelCase: DrvMTR, GetInstance)
 *   - Is ALL_CAPS of at least 3 characters (HMI, APP, MTR)
 *
 * Exported so it can be unit-tested directly.
 */
export function extractSymbolsFromRequirements(text: string): Set<string> {
  const symbols = new Set<string>();
  // Match any identifier starting with an uppercase letter (3+ total chars).
  for (const m of text.matchAll(/\b([A-Z][A-Za-z0-9_]{2,})\b/g)) {
    const s = m[1];
    const hasUnderscore     = s.includes('_');
    const hasMidUppercase   = /[a-z][A-Z]/.test(s);          // CamelCase
    const isAllCapsWord     = /^[A-Z0-9_]+$/.test(s) && s.length >= 3;
    if (hasUnderscore || hasMidUppercase || isAllCapsWord) {
      symbols.add(s);
    }
  }
  return symbols;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool executor
// ─────────────────────────────────────────────────────────────────────────────

async function executeTool(
  name: string,
  input: Record<string, unknown>,
  diff: string,
): Promise<string> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  const workspaceRoot = workspaceFolders?.[0]?.uri?.fsPath ?? '';

  try {
    switch (name) {
      case 'readFile': {
        const rel = (input.path as string).trim();
        // Block any path traversal attempts
        const resolved = nodePath.resolve(workspaceRoot, rel);
        if (!resolved.startsWith(workspaceRoot)) {
          return '[readFile error: path is outside workspace]';
        }
        const uri = vscode.Uri.file(resolved);
        const bytes = await vscode.workspace.fs.readFile(uri);
        const text = new TextDecoder().decode(bytes);
        const lines = text.split('\n');
        const start = typeof input.startLine === 'number' ? input.startLine - 1 : 0;
        const end = typeof input.endLine === 'number' ? input.endLine : Math.min(lines.length, start + 500);
        return trimToolResult(lines.slice(start, end).join('\n'));
      }

      case 'listChangedFiles': {
        const sections = splitDiffByFile(diff);
        if (sections.length === 0) { return '(no changed files found in diff)'; }
        return [...new Set(sections.map(s => s.filePath))].join('\n');
      }

      case 'getFileDiff': {
        const target = (input.filePath as string).trim();
        const sections = splitDiffByFile(diff);
        const section = sections.find(s =>
          s.filePath === target ||
          s.filePath.endsWith(target) ||
          target.endsWith(s.filePath)
        );
        if (!section) { return `(no diff section found for "${target}")`; }
        return trimToolResult(section.diff);
      }

      case 'searchSymbol': {
        const pattern = (input.pattern as string).trim();
        const fileGlob = typeof input.fileGlob === 'string' ? input.fileGlob : '**/*';
        const files = await vscode.workspace.findFiles(
          fileGlob,
          '{**/node_modules/**,**/out/**,**/dist/**,**/build/**}',
          1000
        );
        const results: string[] = [];
        for (const fileUri of files) {
          if (results.length >= 200) { break; }
          try {
            const bytes = await vscode.workspace.fs.readFile(fileUri);
            const content = new TextDecoder().decode(bytes);
            if (!content.includes(pattern)) { continue; }
            const lines = content.split('\n');
            const relPath = vscode.workspace.asRelativePath(fileUri);
            for (let i = 0; i < lines.length && results.length < 200; i++) {
              if (lines[i].includes(pattern)) {
                results.push(`${relPath}:${i + 1}: ${lines[i].trim()}`);
              }
            }
          } catch { /* skip unreadable files */ }
        }
        if (results.length === 0) { return `(no matches found for "${pattern}")`; }
        return trimToolResult(results.join('\n'));
      }

      case 'listWorkspaceFiles': {
        const glob = typeof input.glob === 'string' ? input.glob : '**/*';
        const files = await vscode.workspace.findFiles(
          glob,
          '{**/node_modules/**,**/out/**,**/dist/**,**/build/**}',
          50
        );
        if (files.length === 0) { return '(no files found)'; }
        return files.map(f => vscode.workspace.asRelativePath(f)).join('\n');
      }

      default:
        return `[unknown tool: ${name}]`;
    }
  } catch (err: any) {
    return `[${name} error: ${err.message ?? String(err)}]`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt builder
// ─────────────────────────────────────────────────────────────────────────────

function buildDeepUserPrompt(
  diff: string,
  symbolList: string[],
  profile: ReviewProfile,
): string {
  const enabledRules = profile.rules.filter(r => r.enabled);
  const ruleLines = enabledRules.map(r => {
    const desc = r.description.length > 80 ? r.description.slice(0, 80) + '…' : r.description;
    const sug = r.suggestion ? ` → ${r.suggestion}` : '';
    return `| ${r.id} | ${r.severity} | ${r.title} | ${desc}${sug}`;
  });

  const symbolBlock = symbolList.length > 0
    ? `Changed symbols in this diff:\n${symbolList.join(', ')}\n\n`
    : '';

  // Requirements section — placed FIRST so the model reads it before rules.
  let ticketSection = '';
  if (profile.ticket_context?.raw_requirements) {
    ticketSection =
      `## Requirements\n` +
      `CRITICAL: Every finding must account for these requirements. ` +
      `Verify the changes implement them correctly and completely.\n` +
      `${profile.ticket_context.raw_requirements.trim()}\n\n`;
  }

  // Step 6 is only emitted when requirements are present.
  const step6 = profile.ticket_context?.raw_requirements
    ? `Step 6 — The ## Requirements section at the top lists what the changed\n` +
      `          code must accomplish. For each requirement:\n` +
      `          (a) Identify the specific function or code path that implements it.\n` +
      `          (b) Use readFile or searchSymbol to confirm the implementation scope\n` +
      `              and behavior.\n` +
      `          (c) Raise an ERROR-severity finding for any requirement that is\n` +
      `              violated, incompletely implemented, or whose scope exceeds what\n` +
      `              the requirement specifies.\n` +
      `          (d) For scope or boundary requirements, check EVERY conditional\n` +
      `              branch — active AND inactive — that affects the function's\n` +
      `              behavior. Other hardware targets or future flag changes may\n` +
      `              enable an inactive branch; flag violations even there.\n` +
      `          (e) For each function called in the relevant code path, search its\n` +
      `              definition to verify the semantics match what the requirement\n` +
      `              expects (for example: does a size function return total flash\n` +
      `              length or only the target partition length?). If the definition\n` +
      `              is not found, flag it as a risk: state the assumption required\n` +
      `              and the consequence if that assumption is wrong.\n` +
      `          Do not skip this step when requirements are present.\n`
    : '';

  const completionSteps = profile.ticket_context?.raw_requirements
    ? 'steps 2, 3, 4, and 6'
    : 'steps 2, 3, and 4';

  return (
    `You are a professional code reviewer specializing in ${profile.label}.\n` +
    `${profile.system_prompt_extra ? profile.system_prompt_extra.trim() + '\n\n' : ''}` +
    ticketSection +
    `## Review Rules (${enabledRules.length} enabled)\n` +
    `ID | Severity | Title | Description\n` +
    `---|----------|-------|------------\n` +
    `${ruleLines.join('\n')}\n\n` +
    `## Response Format (JSON only, no markdown fences)\n` +
    `{"verdict":"APPROVE|REQUEST_CHANGES|NEEDS_DISCUSSION","score":1-10,"summary":"<overview>","comments":[{"file":"<file>","line":<N>,"endLine":<N>,"severity":"error|warning|suggestion","ruleId":"<ID>","ruleTitle":"<title>","message":"<120 chars>","suggestion":"<raw code, \\n joined>","codeFragment":"<verbatim 1-3 lines from diff>"}],"conclusion":"<summary>","tests":[{"title":"<Feature area — failure mode>","category":"functional|security|boundary|performance","steps":["<action>","<action>","<verification>"]}]}\n\n` +
    `## Output Rules\n` +
    `- "message": single sentence <120 chars\n` +
    `- "line": exact line of the violation expression\n` +
    `- "codeFragment": verbatim lines from diff that contain the violation\n` +
    `- "suggestion": raw code only, no fences, <=300 chars\n` +
    `- Score harshly: 7-8=acceptable, 5-6=needs work, 3-4=significant issues, 1-2=major problems\n` +
    `- Reference rule ID in every comment\n` +
    `- No praise, focus on problems only\n\n` +
    `## Instructions\n` +
    `You MUST explore the codebase using tools before writing your review.\n` +
    `Follow this sequence — do not skip steps:\n\n` +
    `Step 1 — listChangedFiles has already been called; results are in your context.\n` +
    `Step 2 — Call readFile on each changed file to read its full current content.\n` +
    `          The diff only shows changed lines; readFile gives you the complete\n` +
    `          function body, surrounding context, full struct/class definitions,\n` +
    `          and the entire header interface. Prioritise header and interface files first.\n` +
    `Step 3 — The priority symbols already resolved during grounding are visible\n` +
    `          above. Do not re-search them.\n\n` +
    `          For any compile-time flag whose value is now known from grounding,\n` +
    `          determine which conditional branch is active and search the named\n` +
    `          constants used within that branch. Do not spend budget on symbols\n` +
    `          that belong to inactive branches — EXCEPTION: when a ## Requirements\n` +
    `          section is present, inactive branches that may affect requirement\n` +
    `          compliance MUST still be checked during Step 6. A requirement applies\n` +
    `          to all build configurations, not just the current flag values.\n\n` +
    `          For remaining symbols in the Changed Symbols list, call searchSymbol\n` +
    `          to find their definitions, callers, and usages. Write definitive\n` +
    `          findings using the values already in context — do not write findings\n` +
    `          that depend on an unknown value. If a symbol search returned no\n` +
    `          result, state that explicitly in your finding.\n` +
    `Step 4 — Call readFile on any related file referenced by the changes (headers,\n` +
    `          base classes, imports) that is not already visible in the diff.\n` +
    `Step 5 — Output the JSON only after you have thoroughly explored the codebase.\n` +
    `          The grounding phase has already used a number of tool calls — use at\n` +
    `          least that many additional calls in your exploration before finalizing.\n` +
    `          Stopping after only a few exploration calls is not sufficient; use the\n` +
    `          full budget available to you.\n` +
    step6 +
    `\nYou must NOT produce the JSON before thoroughly completing ${completionSteps}.\n\n` +
    symbolBlock +
    `When you have gathered sufficient context, output ONLY the JSON review object — no prose, no markdown fences.\n\n` +
    `\`\`\`diff\n${diff}\n\`\`\``
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Response parser (mirrors reviewer.ts — not exported there, duplicated here)
// ─────────────────────────────────────────────────────────────────────────────

function sanitizeControlChars(raw: string): string {
  let inString = false;
  let escape   = false;
  let result   = '';
  for (let i = 0; i < raw.length; i++) {
    const ch   = raw[i];
    const code = raw.charCodeAt(i);
    if (escape) { escape = false; result += ch; continue; }
    if (ch === '\\') { escape = true; result += ch; continue; }
    if (ch === '"') {
      if (inString) {
        // Decide whether this " closes the current string or is an unescaped
        // inner quote the model forgot to escape (e.g. "call foo("arg")").
        // Skip any whitespace after the " and check the first meaningful
        // character: in compact JSON the model emits, a legitimate string-close
        // " is always followed (after optional whitespace) by , : } ] or EOF.
        // Anything else means this is an embedded quote that must be escaped.
        let j = i + 1;
        while (j < raw.length &&
               (raw[j] === ' ' || raw[j] === '\t' || raw[j] === '\r' || raw[j] === '\n')) {
          j++;
        }
        const peek = raw[j];
        const isClose = peek === undefined
          || peek === ',' || peek === ':' || peek === '}' || peek === ']';
        if (isClose) {
          inString = false;
          result += '"';
        } else {
          result += '\\"'; // escape inner quote, stay in string
        }
      } else {
        inString = true;
        result += '"';
      }
      continue;
    }
    if (inString && code < 0x20) {
      switch (code) {
        case 0x08: result += '\\b'; break;
        case 0x09: result += '\\t'; break;
        case 0x0A: result += '\\n'; break;
        case 0x0C: result += '\\f'; break;
        case 0x0D: result += '\\r'; break;
        default:   result += `\\u${code.toString(16).padStart(4, '0')}`; break;
      }
      continue;
    }
    result += ch;
  }
  return result;
}

/**
 * Walk `s` and close any unclosed strings, brackets, and braces.
 * Returns the input unchanged if no closing is needed.
 */
function closeOpenStructures(s: string): string {
  let inString   = false;
  let escape     = false;
  let openBraces = 0;
  let openBrackets = 0;
  for (const ch of s) {
    if (escape)        { escape = false; continue; }
    if (ch === '\\')   { escape = true;  continue; }
    if (ch === '"')    { inString = !inString; continue; }
    if (inString)      { continue; }
    if (ch === '{')      { openBraces++;   }
    else if (ch === '}') { openBraces--;   }
    else if (ch === '[') { openBrackets++; }
    else if (ch === ']') { openBrackets--; }
  }
  let result = s;
  if (inString)          { result += '"'; }
  while (openBrackets > 0) { result += ']'; openBrackets--; }
  while (openBraces   > 0) { result += '}'; openBraces--;   }
  return result;
}

/**
 * Attempt to produce valid JSON from a truncated AI response.
 *
 * Strategies (tried in order):
 *  1. Parse as-is.
 *  2. Position-aware backtrack (uses the byte offset from the parse error):
 *       find the last ',' before the error position, truncate there, close
 *       open structures.  Handles "Expected ':' after property name" —
 *       the key is complete but its ':value' was cut off.
 *  3. Truncate at last '}' / ']' and parse.
 *  4. Close all open structures from the full truncated string.
 *  5. Give up — return the original raw string (caller will re-throw).
 */
function repairTruncatedJson(raw: string, errorPos?: number): string {
  const fixed = raw.trim();
  try { JSON.parse(fixed); return fixed; } catch { /* keep going */ }

  // Strategy 2 — position-aware: remove the incomplete final key-value pair
  if (errorPos !== undefined && errorPos > 0) {
    const pos = Math.min(errorPos, fixed.length);

    // Find the last comma before the error site and truncate there
    const lastComma = fixed.lastIndexOf(',', pos);
    if (lastComma > 0) {
      const candidate = closeOpenStructures(fixed.substring(0, lastComma));
      try { JSON.parse(candidate); return candidate; } catch { /* next */ }
    }

    // Fallback: truncate at the error position itself
    const candidate2 = closeOpenStructures(fixed.substring(0, pos));
    try { JSON.parse(candidate2); return candidate2; } catch { /* next */ }
  }

  // Strategy 2b — position-independent backtrack to last comma.
  // Handles "truncated after a complete key with no :value" — V8 reports
  // "Unexpected end of JSON input" with no position in that case.
  const lastCommaAny = fixed.lastIndexOf(',');
  if (lastCommaAny > 0) {
    const candidate2b = closeOpenStructures(fixed.substring(0, lastCommaAny));
    try { JSON.parse(candidate2b); return candidate2b; } catch { /* next */ }
  }

  // Strategy 3 — last complete brace / bracket
  const lastBrace = Math.max(fixed.lastIndexOf('}'), fixed.lastIndexOf(']'));
  if (lastBrace > 0) {
    const candidate = fixed.substring(0, lastBrace + 1);
    try { JSON.parse(candidate); return candidate; } catch { /* next */ }
  }

  // Strategy 4 — close all open structures from wherever we are
  const candidate = closeOpenStructures(fixed);
  try { JSON.parse(candidate); return candidate; } catch { /* give up */ }

  return raw;
}

const CATEGORY_ORDER: Record<string, number> = {
  functional: 0, boundary: 1, performance: 2, security: 3,
};

export function parseDeepReviewResponse(
  raw: string,
  profile: ReviewProfile,
  durationMs: number,
  modelName = 'deep-review',
): ReviewResult {
  let clean = raw.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
  const jsonMatch = clean.match(/\{[\s\S]*\}/);
  if (jsonMatch) { clean = jsonMatch[0]; }

  clean = sanitizeControlChars(clean);

  let parsed: any;
  try {
    parsed = JSON.parse(clean);
  } catch (err: any) {
    // Extract the byte offset reported by the engine (V8: "at position N")
    // and pass it to repairTruncatedJson so it can backtrack past the
    // incomplete key-value pair that caused "Expected ':' after property name".
    const posMatch = (err.message ?? '').match(/position (\d+)/i);
    const errorPos = posMatch ? parseInt(posMatch[1], 10) : undefined;
    const repaired = repairTruncatedJson(clean, errorPos);
    try {
      parsed = JSON.parse(repaired);
      console.warn('[Revvy] Deep Review AI response was truncated — repaired JSON successfully');
    } catch {
      throw new Error(
        `Deep Review: failed to parse AI response as JSON: ${err.message}\n\n` +
        `Raw (first 500 chars): ${clean.substring(0, 500)}`
      );
    }
  }

  const tests: ReviewTest[] = (parsed.tests || [])
    .map((t: any): ReviewTest => ({
      title: t.title || 'Untitled Test',
      category: (['functional', 'security', 'boundary', 'performance'].includes(t.category)
        ? t.category : 'functional') as ReviewTest['category'],
      steps: Array.isArray(t.steps) ? t.steps.map((s: any) => String(s)) : [],
    }))
    .sort((a: ReviewTest, b: ReviewTest) =>
      (CATEGORY_ORDER[a.category] ?? 99) - (CATEGORY_ORDER[b.category] ?? 99));

  return {
    verdict: parsed.verdict || 'NEEDS_DISCUSSION',
    score: Math.min(10, Math.max(1, parseInt(parsed.score) || 5)),
    summary: parsed.summary || 'No summary provided',
    comments: (parsed.comments || []).map((c: any): ReviewComment => ({
      file: c.file || 'general',
      line: parseInt(c.line) || 0,
      endLine: c.endLine ? parseInt(c.endLine) : undefined,
      severity: c.severity || 'suggestion',
      ruleId: c.ruleId,
      ruleTitle: c.ruleTitle,
      message: c.message || '',
      suggestion: c.suggestion,
      codeFragment: typeof c.codeFragment === 'string' && c.codeFragment.trim()
        ? c.codeFragment.trim() : undefined,
    })),
    conclusion: parsed.conclusion || '',
    tests,
    commitMessages: [],   // Deep Review doesn't generate commit messages
    profileUsed: profile.label,
    modelUsed: modelName,
    backendUsed: 'GitHub Copilot (Deep Review)',
    durationMs,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Force-finalize helper (context cap or budget exhaustion)
// ─────────────────────────────────────────────────────────────────────────────

async function forceFinalize(
  messages: vscode.LanguageModelChatMessage[],
  model: any,
  cts: vscode.CancellationTokenSource,
  profile: ReviewProfile,
  durationMs: number,
  onChunk?: StreamChunkCallback,
): Promise<ReviewResult> {
  messages.push(
    vscode.LanguageModelChatMessage.User(
      'Context limit reached. Based on your exploration so far, output ONLY the final JSON review object now.'
    )
  );

  const finalResp = await model.sendRequest(messages, {}, cts.token);
  let finalText = '';
  for await (const chunk of finalResp.text) {
    finalText += chunk;
    onChunk?.(chunk);
  }

  return parseDeepReviewResponse(finalText, profile, durationMs);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export async function runDeepReview(
  diff: string,
  profile: ReviewProfile,
  keys: AIKeys,  // kept for API symmetry with runReview; Copilot path uses vscode.lm
  onChunk?: StreamChunkCallback,
  sources?: ReviewSource[],
): Promise<ReviewResult> {
  if (!diff.trim()) { throw new Error('No diff to review'); }

  // ── Select model (same logic as callCopilot in aiBackend.ts) ─────────────
  const selectedModelId = vscode.workspace.getConfiguration('revvy').get<string>('selectedModelId', '');
  const allModels: any[] = await (vscode.lm as any).selectChatModels({ vendor: 'copilot' });
  if (!allModels || allModels.length === 0) {
    throw new Error(
      'Deep Review requires GitHub Copilot. No Copilot models are available — ensure GitHub Copilot is enabled in VS Code.'
    );
  }
  const model = selectedModelId
    ? (allModels.find((m: any) => m.id === selectedModelId) ?? allModels[0])
    : allModels[0];

  // Open the output channel so the user sees logs in real time
  getChannel().show(false);
  getChannel().appendLine('');
  getChannel().appendLine('━'.repeat(60));

  const cts = new vscode.CancellationTokenSource();
  const start = Date.now();

  log(`START  model=${model.id ?? model.name ?? 'unknown'}  diff=${diff.length} chars`);

  // ── Build initial context ─────────────────────────────────────────────────
  const symbolList = extractChangedSymbols(diff);
  const { tier1, tier2 } = extractHighPrioritySymbols(diff);
  const initialPrompt = buildDeepUserPrompt(diff, symbolList, profile);
  log(`SYMBOLS  ${symbolList.length > 0 ? symbolList.join(', ') : '(none detected)'}`);
  log(`PRIORITY SYMBOLS  tier1=[${tier1.join(', ')}]  tier2=[${tier2.join(', ')}]`);;

  const messages: vscode.LanguageModelChatMessage[] = [
    vscode.LanguageModelChatMessage.User(initialPrompt),
  ];

  // Cost controls
  const toolCache = new Map<string, string>();
  let toolCallCount = 0;

  // ── Mandatory grounding: always call listChangedFiles before the loop ─────
  // This guarantees toolCallCount >= 1 and primes the conversation so the
  // model sees itself already mid-exploration and continues with steps 2-4.
  {
    const groundingResult = await executeTool('listChangedFiles', {}, diff);
    const groundingCallId = 'grounding-listChangedFiles';
    messages.push(vscode.LanguageModelChatMessage.Assistant([
      new vscode.LanguageModelToolCallPart(groundingCallId, 'listChangedFiles', {}),
    ]));
    messages.push(vscode.LanguageModelChatMessage.User([
      new vscode.LanguageModelToolResultPart(groundingCallId, [new vscode.LanguageModelTextPart(groundingResult)]),
    ]));
    toolCallCount = 1;
    onChunk?.(`\n[Turn 0/${MAX_AGENT_ROUNDS}, tool 1/${MAX_TOOL_CALLS}: listChangedFiles]\n`);
    log(`GROUNDING  listChangedFiles →`);
    log(`  ${groundingResult.replace(/\n/g, '\n  ')}`);
    console.log(`[Revvy] Deep Review: grounding listChangedFiles complete — ${groundingResult.split('\n').length} files`);
  }

  // ── Programmatic symbol grounding: resolve tier1 + tier2 before the loop ──
  // Tier 1 (compile-time flags from #if/#ifdef/#elif) and Tier 2 (named
  // constants on comparison RHS) are searched in code so the model is
  // guaranteed to see their values — no reliance on model compliance.
  // Each result is trimmed to GROUNDING_SYMBOL_RESULT_CHARS (tight: we only
  // need the #define line, not 12 KB of surrounding code).
  // The toolCache deduplicates any re-searches the model may attempt later.
  const groundedSymbols: Array<{ sym: string; preview: string }> = [];
  {
    const autoGroundList = [...tier1, ...tier2]; // tier1 always first
    for (const sym of autoGroundList) {
      if (toolCallCount >= MAX_TOOL_CALLS) { break; } // safety net only
      const raw = await executeTool('searchSymbol', { pattern: sym }, diff);
      const trimmed = trimToolResult(raw, GROUNDING_SYMBOL_RESULT_CHARS);
      const callId = `grounding-searchSymbol-${sym}`;
      messages.push(vscode.LanguageModelChatMessage.Assistant([
        new vscode.LanguageModelToolCallPart(callId, 'searchSymbol', { pattern: sym }),
      ]));
      messages.push(vscode.LanguageModelChatMessage.User([
        new vscode.LanguageModelToolResultPart(callId, [new vscode.LanguageModelTextPart(trimmed)]),
      ]));
      toolCallCount++;
      const preview = trimmed.split('\n')[0].slice(0, 120);
      groundedSymbols.push({ sym, preview });
      onChunk?.(`\n[Turn 0/${MAX_AGENT_ROUNDS}, tool ${toolCallCount}/${MAX_TOOL_CALLS}: searchSymbol(${sym})]\n`);
      log(`GROUNDING  searchSymbol(${sym}) → ${preview}`);
    }
    console.log(`[Revvy] Deep Review: grounding resolved ${groundedSymbols.length} priority symbols`);
  }

  // ── Inject grounding summary so the model knows what is already resolved ──
  // This steers the model away from re-searching grounded symbols and towards
  // the remaining Changed Symbols list and cross-file callers.
  if (groundedSymbols.length > 0) {
    const lines = groundedSymbols.map(({ sym, preview }) =>
      `  • ${sym}: ${preview || '(no matches found)'}`
    );
    const summaryText =
      `[Grounding complete — the following priority symbols have already been resolved:\n` +
      lines.join('\n') +
      `\nDo NOT re-search these. Proceed to Steps 2–4 for remaining exploration.]`;
    messages.push(vscode.LanguageModelChatMessage.User(summaryText));
    log(`GROUNDING SUMMARY  ${groundedSymbols.map(g => g.sym).join(', ')}`);
  }

  // ── Requirement-symbol grounding ─────────────────────────────────────────
  // Extract code identifiers from the requirement text (e.g. function names,
  // config macros, error codes the user explicitly named) and search them
  // before the agent loop so findings are grounded in real code paths.
  // Symbols already resolved in the tier1/tier2 pass are skipped.
  if (profile.ticket_context?.raw_requirements) {
    const alreadyGrounded = new Set(groundedSymbols.map(g => g.sym));
    const reqSymbols = extractSymbolsFromRequirements(
      profile.ticket_context.raw_requirements
    );
    const reqGroundedSymbols: Array<{ sym: string; preview: string }> = [];
    for (const sym of reqSymbols) {
      if (alreadyGrounded.has(sym)) { continue; }
      if (toolCallCount >= MAX_TOOL_CALLS) { break; }
      const raw = await executeTool('searchSymbol', { pattern: sym }, diff);
      const trimmed = trimToolResult(raw, GROUNDING_SYMBOL_RESULT_CHARS);
      const callId = `grounding-req-searchSymbol-${sym}`;
      messages.push(vscode.LanguageModelChatMessage.Assistant([
        new vscode.LanguageModelToolCallPart(callId, 'searchSymbol', { pattern: sym }),
      ]));
      messages.push(vscode.LanguageModelChatMessage.User([
        new vscode.LanguageModelToolResultPart(callId, [new vscode.LanguageModelTextPart(trimmed)]),
      ]));
      toolCallCount++;
      const preview = trimmed.split('\n')[0].slice(0, 120);
      reqGroundedSymbols.push({ sym, preview });
      onChunk?.(`\n[Turn 0/${MAX_AGENT_ROUNDS}, tool ${toolCallCount}/${MAX_TOOL_CALLS}: searchSymbol(${sym}) [requirement]]\n`);
      log(`GROUNDING [req]  searchSymbol(${sym}) → ${preview}`);
    }
    if (reqGroundedSymbols.length > 0) {
      const reqLines = reqGroundedSymbols.map(({ sym, preview }) =>
        `  • ${sym}: ${preview || '(no matches found)'}`
      );
      messages.push(vscode.LanguageModelChatMessage.User(
        `[Requirement-symbol grounding complete — additional symbols from Requirements:\n` +
        reqLines.join('\n') +
        `\nUse these to verify the implementation satisfies the stated requirements (Step 6).]`
      ));
      log(`GROUNDING [req] SUMMARY  ${reqGroundedSymbols.map(g => g.sym).join(', ')}`);
    }
  }

  // ── Agent loop ────────────────────────────────────────────────────────────
  for (let round = 0; round < MAX_AGENT_ROUNDS; round++) {
    const convSize = getConversationSize(messages);
    log(`ROUND ${round + 1}/${MAX_AGENT_ROUNDS}  conversation=${convSize} chars  toolCalls=${toolCallCount}`);

    // Cost control 4: total conversation context cap
    if (convSize > MAX_CONVERSATION_CHARS) {
      log(`CONTEXT CAP hit at round ${round + 1} (${convSize} chars > ${MAX_CONVERSATION_CHARS}) — force-finalizing`);
      console.log(`[Revvy] Deep Review: context cap hit at round ${round} — force-finalizing`);
      onChunk?.('\n[Context limit reached — generating final review]\n');
      const result = await forceFinalize(messages, model, cts, profile, Date.now() - start, onChunk);
      result.toolCallsUsed        = toolCallCount;
      result.estimatedInputTokens = Math.round(convSize / 4);
      log(`DONE  reason=context-cap  toolCalls=${toolCallCount}  duration=${Date.now() - start}ms`);
      return result;
    }

    // Send request with tools available
    const response = await model.sendRequest(
      messages,
      { tools: DEEP_REVIEW_TOOLS },
      cts.token
    );

    // Stream and collect
    const textParts: string[] = [];
    const toolCallParts: vscode.LanguageModelToolCallPart[] = [];

    for await (const chunk of response.stream) {
      if (chunk instanceof vscode.LanguageModelTextPart) {
        textParts.push(chunk.value);
        onChunk?.(chunk.value);
      } else if (chunk instanceof vscode.LanguageModelToolCallPart) {
        toolCallParts.push(chunk);
      }
    }

    const assembledText = textParts.join('').trim();
    log(`RESPONSE  textLen=${assembledText.length}  toolCalls=${toolCallParts.length}  names=[${toolCallParts.map(c => c.name).join(', ')}]`);

    // No tool calls → AI produced its final answer
    if (toolCallParts.length === 0) {
      const text = assembledText;
      if (!text) {
        log(`ERROR  model returned empty response at round ${round + 1}`);
        throw new Error('Deep Review: AI returned an empty response. The diff may be too large or the model is unavailable.');
      }
      log(`FINAL ANSWER  length=${text.length} chars`);
      log(`  preview: ${text.slice(0, 200).replace(/\n/g, '↵')}${text.length > 200 ? '…' : ''}`);
      const result = parseDeepReviewResponse(text, profile, Date.now() - start, model.name || 'copilot');
      result.toolCallsUsed        = toolCallCount;
      result.estimatedInputTokens = Math.round(getConversationSize(messages) / 4);
      log(`DONE  reason=final-answer  toolCalls=${toolCallCount}  verdict=${result.verdict}  score=${result.score}  comments=${result.comments.length}  duration=${Date.now() - start}ms`);
      return result;
    }

    // Cost control 2: tool call budget exhaustion (check before executing)
    if (toolCallCount >= MAX_TOOL_CALLS) {
      log(`BUDGET EXHAUSTED at round ${round + 1} (${toolCallCount}/${MAX_TOOL_CALLS}) — force-finalizing`);
      console.log(`[Revvy] Deep Review: tool call budget (${MAX_TOOL_CALLS}) exhausted at round ${round} — force-finalizing`);
      onChunk?.('\n[Tool budget reached — generating final review]\n');
      const result = await forceFinalize(messages, model, cts, profile, Date.now() - start, onChunk);
      result.toolCallsUsed        = toolCallCount;
      result.estimatedInputTokens = Math.round(getConversationSize(messages) / 4);
      log(`DONE  reason=budget  toolCalls=${toolCallCount}  duration=${Date.now() - start}ms`);
      return result;
    }

    // Cost control 7: empty-assistant guard — include tool call parts always
    const assistantContent: Array<vscode.LanguageModelTextPart | vscode.LanguageModelToolCallPart> = [
      ...toolCallParts,
      ...(assembledText ? [new vscode.LanguageModelTextPart(assembledText)] : []),
    ];
    if (assistantContent.length > 0) {
      messages.push(vscode.LanguageModelChatMessage.Assistant(assistantContent));
    }

    // Execute tool calls
    const toolResultParts: vscode.LanguageModelToolResultPart[] = [];

    for (const call of toolCallParts) {
      // Cost control 6: per-tool input validation
      if (!isValidToolInput(call)) {
        log(`SKIP  ${call.name}  reason=invalid-input  input=${JSON.stringify(call.input ?? {})}`);
        console.log(`[Revvy] Deep Review: skipping invalid tool input for ${call.name}`);
        toolResultParts.push(
          new vscode.LanguageModelToolResultPart(
            call.callId,
            [new vscode.LanguageModelTextPart(`[Tool ${call.name} skipped — invalid or missing input]`)]
          )
        );
        continue;
      }

      // Cost control 2: per-call budget check
      if (toolCallCount >= MAX_TOOL_CALLS) {
        log(`SKIP  ${call.name}  reason=budget-reached  (${toolCallCount}/${MAX_TOOL_CALLS})`);
        toolResultParts.push(
          new vscode.LanguageModelToolResultPart(
            call.callId,
            [new vscode.LanguageModelTextPart(`[Tool ${call.name} skipped — tool call budget (${MAX_TOOL_CALLS}) reached]`)]
          )
        );
        continue;
      }

      // Cost control 5: exact-match dedup cache
      const cacheKey = call.name + JSON.stringify(call.input ?? {});
      let result: string;

      if (toolCache.has(cacheKey)) {
        result = toolCache.get(cacheKey)!;
        log(`CACHE HIT  ${call.name}  input=${JSON.stringify(call.input ?? {})}`);
        console.log(`[Revvy] Deep Review: cache hit for ${call.name}`);
      } else {
        log(`CALL  ${call.name}  input=${JSON.stringify(call.input ?? {})}`);
        result = await executeTool(call.name, (call.input ?? {}) as Record<string, unknown>, diff);
        // Cost control 3: result truncation happens inside executeTool via trimToolResult
        toolCache.set(cacheKey, result);
        toolCallCount++;
        logResult(`result (${result.length} chars)`, result);
        onChunk?.(`\n[Turn ${round + 1}/${MAX_AGENT_ROUNDS}, tool ${toolCallCount}/${MAX_TOOL_CALLS}: ${call.name}]\n`);
        console.log(`[Revvy] Deep Review: round=${round + 1}, tool=${call.name}, calls=${toolCallCount}`);
      }

      toolResultParts.push(new vscode.LanguageModelToolResultPart(call.callId, [new vscode.LanguageModelTextPart(result)]));
    }

    // Feed tool results back
    if (toolResultParts.length > 0) {
      messages.push(vscode.LanguageModelChatMessage.User(toolResultParts));
    }
  }

  // If we exit the loop without returning, all rounds are exhausted
  log(`DONE  reason=rounds-exhausted  toolCalls=${toolCallCount}  duration=${Date.now() - start}ms`);
  throw new Error(
    `Deep Review: impact analysis incomplete — exhausted ${MAX_AGENT_ROUNDS} rounds ` +
    `without a final response. The diff may be too large for one review session.`
  );
}
