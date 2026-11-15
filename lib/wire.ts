// Wire translation: two downstream wires (OpenAI Chat Completions, Anthropic Messages) and three
// upstream wires (the same two plus Gemini), through one canonical form.
//
// The providers answer with an echo of what they parsed: system text, the last user text, the tool
// names, max_tokens, the stop sequences, the last image and the JSON-mode flag. Every check on a
// response compares that echo with what the client sent, so translation is built to carry exactly
// those fields untouched across every wire pair:
//
//   - system text is carried as one exact string (each wire's parser joins blocks with "\n", and a
//     single block holding the joined string parses back to the same text);
//   - each message's text blocks are joined with "\n" once, canonically, and re-emitted as one
//     block, which every wire parses back to the joined string;
//   - images keep their bytes (base64) or URL, in order, so the last image hashes the same;
//   - tools keep name order; max_tokens and stops are set exactly, or omitted exactly;
//   - JSON mode is carried only by wires that can: OpenAI's response_format and Gemini's
//     responseMimeType. Anthropic's wire has no such field, so routing (in main.ts) never sends a
//     JSON-mode request to an Anthropic-wire provider.
//
// Tool-result turns are the one asymmetry: OpenAI's parser ignores `tool` messages when it reads
// the last user text, Anthropic's and Gemini's read the tool-result turn itself (as empty text).
// Each builder therefore simulates its wire's parser over the body it built and, when the last
// user text would come back different from the client's, appends one plain user turn carrying the
// client's exact last user text. With the single-turn workloads the trials send, it never fires.
//
// Streaming is canonicalised too: the upstream stream is read whole, then the downstream event
// sequence is synthesised deterministically from the parsed answer. The checks read the assembled
// text, the stop reason and the final usage, all of which this preserves exactly, and a replay
// produces byte-identical events because nothing here reads a clock or a random source.

import { isRecord, type Json, type Wire } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** SSE text pieces are cut this many characters at a time, as the providers cut theirs. */
const SSE_PIECE = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------
// Canonical forms

export type CImage = { kind: 'b64'; media: string; data: string } | { kind: 'url'; url: string };

export interface CToolCall {
  id: string;
  name: string;
  /** The call's arguments as one JSON string. */
  args: string;
}

export interface CToolResult {
  id: string;
  /** The tool's name, when an earlier assistant turn in the same request named it. */
  name: string;
  text: string;
  isError: boolean;
}

export interface CMsg {
  role: 'user' | 'assistant';
  /** The turn's text blocks joined with "\n" ('' when it holds none). */
  text: string;
  images: CImage[];
  toolCalls: CToolCall[];
  toolResults: CToolResult[];
}

export type CToolChoice = 'auto' | 'required' | 'none' | { name: string } | null;

export interface CReq {
  /** The exact system text, or null when the request carries none. */
  system: string | null;
  messages: CMsg[];
  tools: { name: string; description: string; schema: Json }[];
  toolChoice: CToolChoice;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  /** The client asked for JSON mode (OpenAI's response_format); only some wires can carry it. */
  jsonMode: boolean;
  /** The client's raw response_format, kept so an OpenAI upstream receives it untouched. */
  responseFormat: Json | null;
  stream: boolean;
  /** What the client's own wire parser reads as the last user text: the echo's yardstick. */
  lastUserText: string;
}

export type CStop = 'end' | 'length' | 'tool' | 'filter';

export interface CResp {
  text: string;
  toolCalls: CToolCall[];
  stop: CStop;
  usage: { input: number; output: number };
}

// ---------------------------------------------------------------------------
// Small shared helpers

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text of a string or of a block array's text blocks, joined with "\n"; '' when there is none. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An arguments value as one JSON string. */
function argsString(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === undefined || value === null) return '{}';
  return JSON.stringify(value);
}

/** An arguments JSON string as an object; {} when it does not parse to one. */
function argsObject(value: string): Json {
  try {
    const parsed: unknown = JSON.parse(value);
    if (isRecord(parsed)) return parsed;
  } catch {
    // Not JSON: an empty input.
  }
  return {};
}

/** An OpenAI image_url value as a canonical image: inline base64 where the URL carries it. */
function imageOfUrl(url: string): CImage {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'b64', media: data[1], data: data[2] };
  return { kind: 'url', url };
}

/** A canonical image as an OpenAI image_url value. */
function urlOfImage(image: CImage): string {
  return image.kind === 'b64' ? `data:${image.media};base64,${image.data}` : image.url;
}

function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

// ---------------------------------------------------------------------------
// Downstream requests -> canonical

/** An OpenAI Chat Completions request in canonical form. */
export function parseOpenAIRequest(body: Json): CReq {
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  let sawSystem = false;
  const messages: CMsg[] = [];
  const callNames = new Map<string, string>();
  let lastUserText = '';
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : '';
    if (role === 'system') {
      sawSystem = true;
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'user') {
      const images: CImage[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            images.push(imageOfUrl(block.image_url.url));
          }
        }
      }
      const text = blocksText(raw.content);
      lastUserText = text;
      messages.push({ role: 'user', text, images, toolCalls: [], toolResults: [] });
      continue;
    }
    if (role === 'assistant') {
      const toolCalls: CToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call)) continue;
          const fn = isRecord(call.function) ? call.function : {};
          const id = typeof call.id === 'string' ? call.id : '';
          const name = typeof fn.name === 'string' ? fn.name : '';
          if (id !== '') callNames.set(id, name);
          toolCalls.push({ id, name, args: argsString(fn.arguments) });
        }
      }
      const text = typeof raw.content === 'string' ? raw.content : blocksText(raw.content);
      messages.push({ role: 'assistant', text, images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (role === 'tool' || role === 'function') {
      const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : typeof raw.name === 'string' ? raw.name : '';
      const text = typeof raw.content === 'string' ? raw.content : toolResultText(raw.content);
      messages.push({
        role: 'user',
        text: '',
        images: [],
        toolCalls: [],
        toolResults: [{ id, name: callNames.get(id) ?? '', text, isError: false }],
      });
      continue;
    }
    // Any other role (developer and friends) is read by no wire's echo parser: drop it.
  }
  const tools: CReq['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : '',
        schema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CToolChoice = null;
  if (body.tool_choice === 'auto' || body.tool_choice === 'required' || body.tool_choice === 'none') toolChoice = body.tool_choice;
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  return {
    system: sawSystem ? systemParts.join('\n') : null,
    messages,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens) ?? num(body.max_completion_tokens),
    stops: stopList(body.stop),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    jsonMode: format !== null && (format.type === 'json_object' || format.type === 'json_schema'),
    responseFormat: format,
    stream: body.stream === true,
    lastUserText,
  };
}

/** An Anthropic Messages request in canonical form. */
export function parseAnthropicRequest(body: Json): CReq {
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  const messages: CMsg[] = [];
  const callNames = new Map<string, string>();
  let lastUserText = '';
  for (const raw of rawMessages) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      if (!assistant) lastUserText = content;
      messages.push({ role: assistant ? 'assistant' : 'user', text: content, images: [], toolCalls: [], toolResults: [] });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: CToolCall[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        const id = typeof block.id === 'string' ? block.id : '';
        const name = typeof block.name === 'string' ? block.name : '';
        if (id !== '') callNames.set(id, name);
        toolCalls.push({ id, name, args: argsString(block.input) });
      }
      messages.push({ role: 'assistant', text: blocksText(content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    const images: CImage[] = [];
    const toolResults: CToolResult[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          images.push({ kind: 'b64', media: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          images.push({ kind: 'url', url: source.url });
        }
      } else if (block.type === 'tool_result') {
        const id = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
        toolResults.push({ id, name: callNames.get(id) ?? '', text: toolResultText(block.content), isError: block.is_error === true });
      }
    }
    const text = blocksText(content);
    lastUserText = text;
    messages.push({ role: 'user', text, images, toolCalls: [], toolResults });
  }
  const tools: CReq['tools'] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : '',
        schema: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: CToolChoice = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto') toolChoice = 'auto';
    else if (body.tool_choice.type === 'any') toolChoice = 'required';
    else if (body.tool_choice.type === 'none') toolChoice = 'none';
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { name: body.tool_choice.name };
  }
  return {
    system: body.system === undefined || body.system === null ? null : blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens),
    stops: stopList(body.stop_sequences),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    jsonMode: false,
    responseFormat: null,
    stream: body.stream === true,
    lastUserText,
  };
}

// ---------------------------------------------------------------------------
// The echo guard: what one wire's parser would read as the last user text of a built body.

function upstreamUserText(wire: Wire, body: Json): string {
  let text = '';
  if (wire === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) if (isRecord(m) && m.role === 'user') text = blocksText(m.content);
    return text;
  }
  if (wire === 'anthropic') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) if (isRecord(m) && m.role === 'user') text = blocksText(m.content);
    return text;
  }
  const contents = Array.isArray(body.contents) ? body.contents : [];
  for (const c of contents) {
    if (!isRecord(c) || c.role === 'model') continue;
    const parts = Array.isArray(c.parts) ? c.parts : [];
    text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
  }
  return text;
}

/** Append one plain user turn carrying `expected` when the built body's last user text differs. */
function guardUserText(wire: Wire, body: Json, expected: string): void {
  if (upstreamUserText(wire, body) === expected) return;
  if (wire === 'openai') {
    const messages = Array.isArray(body.messages) ? (body.messages as Json[]) : [];
    messages.push({ role: 'user', content: expected });
    body.messages = messages;
  } else if (wire === 'anthropic') {
    const messages = Array.isArray(body.messages) ? (body.messages as Json[]) : [];
    messages.push({ role: 'user', content: [{ type: 'text', text: expected }] });
    body.messages = messages;
  } else {
    const contents = Array.isArray(body.contents) ? (body.contents as Json[]) : [];
    contents.push({ role: 'user', parts: [{ text: expected }] });
    body.contents = contents;
  }
}

// ---------------------------------------------------------------------------
// Canonical -> upstream request bodies

/** A canonical request as an OpenAI Chat Completions body naming `model`. */
export function buildOpenAIRequest(req: CReq, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== null) messages.push({ role: 'system', content: req.system });
  for (const msg of req.messages) {
    if (msg.role === 'assistant') {
      messages.push({
        role: 'assistant',
        content: msg.text !== '' ? msg.text : msg.toolCalls.length > 0 ? null : '',
        ...(msg.toolCalls.length > 0
          ? { tool_calls: msg.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } })) }
          : {}),
      });
      continue;
    }
    for (const result of msg.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
    }
    const onlyResults = msg.text === '' && msg.images.length === 0 && msg.toolResults.length > 0;
    if (!onlyResults) {
      if (msg.images.length === 0) {
        messages.push({ role: 'user', content: msg.text });
      } else {
        messages.push({
          role: 'user',
          content: [
            { type: 'text', text: msg.text },
            ...msg.images.map((image) => ({ type: 'image_url', image_url: { url: urlOfImage(image) } })),
          ],
        });
      }
    }
  }
  const out: Json = { model, messages };
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
  }
  if (req.toolChoice === 'auto' || req.toolChoice === 'required' || req.toolChoice === 'none') out.tool_choice = req.toolChoice;
  else if (req.toolChoice !== null) out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.jsonMode) out.response_format = req.responseFormat ?? { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  guardUserText('openai', out, req.lastUserText);
  return out;
}

/** A canonical request as an Anthropic Messages body naming `model`. */
export function buildAnthropicRequest(req: CReq, model: string): Json {
  const messages: Json[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'assistant') {
      const blocks: Json[] = [];
      if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
      for (const call of msg.toolCalls) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.args) });
      }
      if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const result of msg.toolResults) {
      blocks.push({
        type: 'tool_result',
        tool_use_id: result.id,
        content: result.text,
        ...(result.isError ? { is_error: true } : {}),
      });
    }
    if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
    for (const image of msg.images) {
      blocks.push(
        image.kind === 'b64'
          ? { type: 'image', source: { type: 'base64', media_type: image.media, data: image.data } }
          : { type: 'image', source: { type: 'url', url: image.url } },
      );
    }
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (req.system !== null) out.system = req.system;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema }));
  }
  if (req.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (req.toolChoice === 'required') out.tool_choice = { type: 'any' };
  else if (req.toolChoice === 'none') out.tool_choice = { type: 'none' };
  else if (req.toolChoice !== null) out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stream) out.stream = true;
  guardUserText('anthropic', out, req.lastUserText);
  return out;
}

/** A canonical request as a Gemini generateContent body (the model travels in the URL path). */
export function buildGeminiRequest(req: CReq): Json {
  const contents: Json[] = [];
  for (const msg of req.messages) {
    if (msg.role === 'assistant') {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ text: msg.text });
      for (const call of msg.toolCalls) {
        parts.push({ functionCall: { name: call.name, args: argsObject(call.args) } });
      }
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const result of msg.toolResults) {
      parts.push({ functionResponse: { name: result.name !== '' ? result.name : 'tool', response: { output: result.text } } });
    }
    if (msg.text !== '') parts.push({ text: msg.text });
    for (const image of msg.images) {
      parts.push(
        image.kind === 'b64'
          ? { inlineData: { mimeType: image.media, data: image.data } }
          : { fileData: { mimeType: 'image/png', fileUri: image.url } },
      );
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (req.system !== null) out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  }
  if (req.toolChoice === 'auto') out.toolConfig = { functionCallingConfig: { mode: 'AUTO' } };
  else if (req.toolChoice === 'required') out.toolConfig = { functionCallingConfig: { mode: 'ANY' } };
  else if (req.toolChoice === 'none') out.toolConfig = { functionCallingConfig: { mode: 'NONE' } };
  else if (req.toolChoice !== null) {
    out.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [req.toolChoice.name] } };
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  guardUserText('gemini', out, req.lastUserText);
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers -> canonical

function openAIStop(finish: unknown): CStop {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool';
  if (finish === 'content_filter') return 'filter';
  return 'end';
}

function anthropicStop(reason: unknown): CStop {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool';
  if (reason === 'refusal') return 'filter';
  return 'end';
}

function geminiStop(reason: unknown): CStop {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST') return 'filter';
  return 'end';
}

/** A Chat Completions JSON answer in canonical form. */
export function parseOpenAIResponse(body: Json): CResp {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  const toolCalls: CToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        args: argsString(call.function.arguments),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: openAIStop(choice.finish_reason),
    usage: { input: numOrZero(usage.prompt_tokens), output: numOrZero(usage.completion_tokens) },
  };
}

/** A Messages JSON answer in canonical form. */
export function parseAnthropicResponse(body: Json): CResp {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: CToolCall[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        args: argsString(block.input),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: anthropicStop(body.stop_reason),
    usage: { input: numOrZero(usage.input_tokens), output: numOrZero(usage.output_tokens) },
  };
}

/** A generateContent JSON answer in canonical form. */
export function parseGeminiResponse(body: Json): CResp {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: CToolCall[] = [];
  let index = 0;
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      const name = typeof part.functionCall.name === 'string' ? part.functionCall.name : '';
      toolCalls.push({ id: `call_${index}`, name, args: argsString(part.functionCall.args) });
      index += 1;
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    stop: geminiStop(candidate.finishReason),
    usage: { input: numOrZero(usage.promptTokenCount), output: numOrZero(usage.candidatesTokenCount) },
  };
}

// ---------------------------------------------------------------------------
// Upstream event streams -> canonical

/** One SSE frame: its event name (or null) and its data payload. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a whole SSE body into frames. */
export function sseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function frameJson(frame: SseFrame): Json | null {
  try {
    const value: unknown = JSON.parse(frame.data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** A Chat Completions SSE body in canonical form. */
export function parseOpenAIStream(text: string): CResp {
  let out = '';
  let stop: CStop = 'end';
  let usage = { input: 0, output: 0 };
  const calls = new Map<number, CToolCall>();
  for (const frame of sseFrames(text)) {
    if (frame.data.trim() === '[DONE]') continue;
    const chunk = frameJson(frame);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      usage = { input: numOrZero(chunk.usage.prompt_tokens), output: numOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const at = typeof call.index === 'number' ? call.index : calls.size;
        const current = calls.get(at) ?? { id: '', name: '', args: '' };
        if (typeof call.id === 'string' && call.id !== '') current.id = call.id;
        if (isRecord(call.function)) {
          if (typeof call.function.name === 'string' && call.function.name !== '') current.name = call.function.name;
          if (typeof call.function.arguments === 'string') current.args += call.function.arguments;
        }
        calls.set(at, current);
      }
    }
    if (typeof choice.finish_reason === 'string') stop = openAIStop(choice.finish_reason);
  }
  return { text: out, toolCalls: [...calls.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c), stop, usage };
}

/** A Messages SSE body in canonical form. */
export function parseAnthropicStream(text: string): CResp {
  let out = '';
  let stop: CStop = 'end';
  let input = 0;
  let output = 0;
  const calls = new Map<number, CToolCall>();
  for (const frame of sseFrames(text)) {
    const data = frameJson(frame);
    if (data === null) continue;
    const type = typeof data.type === 'string' ? data.type : frame.event;
    if (type === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numOrZero(data.message.usage.input_tokens);
    } else if (type === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const at = typeof data.index === 'number' ? data.index : calls.size;
      calls.set(at, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        args: '',
      });
    } else if (type === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const at = typeof data.index === 'number' ? data.index : -1;
        const current = calls.get(at);
        if (current !== undefined) current.args += data.delta.partial_json;
      }
    } else if (type === 'message_delta') {
      if (isRecord(data.delta)) stop = anthropicStop(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, toolCalls: [...calls.entries()].sort((a, b) => a[0] - b[0]).map(([, c]) => c), stop, usage: { input, output } };
}

/** A streamGenerateContent SSE body in canonical form. */
export function parseGeminiStream(text: string): CResp {
  let out = '';
  let stop: CStop = 'end';
  let usage = { input: 0, output: 0 };
  const toolCalls: CToolCall[] = [];
  for (const frame of sseFrames(text)) {
    const body = frameJson(frame);
    if (body === null) continue;
    const partial = parseGeminiResponse(body);
    out += partial.text;
    toolCalls.push(...partial.toolCalls);
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null && typeof candidate.finishReason === 'string') stop = geminiStop(candidate.finishReason);
    if (isRecord(body.usageMetadata)) {
      usage = { input: numOrZero(body.usageMetadata.promptTokenCount), output: numOrZero(body.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: out, toolCalls, stop, usage };
}

// ---------------------------------------------------------------------------
// Canonical -> downstream answers

function openAIStopName(stop: CStop): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool') return 'tool_calls';
  if (stop === 'filter') return 'content_filter';
  return 'stop';
}

function anthropicStopName(stop: CStop): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool') return 'tool_use';
  if (stop === 'filter') return 'refusal';
  return 'end_turn';
}

/** A canonical answer as a Chat Completions JSON body. */
export function openAICompletion(model: string, resp: CResp): Json {
  const message: Json = { role: 'assistant', content: resp.text !== '' ? resp.text : resp.toolCalls.length > 0 ? null : '' };
  if (resp.toolCalls.length > 0) {
    message.tool_calls = resp.toolCalls.map((c) => ({
      id: c.id !== '' ? c.id : 'call_0',
      type: 'function',
      function: { name: c.name, arguments: c.args },
    }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStopName(resp.stop), logprobs: null }],
    usage: {
      prompt_tokens: resp.usage.input,
      completion_tokens: resp.usage.output,
      total_tokens: resp.usage.input + resp.usage.output,
    },
  };
}

/** A canonical answer as a Messages JSON body. */
export function anthropicMessage(model: string, resp: CResp): Json {
  const content: Json[] = [];
  if (resp.text !== '') content.push({ type: 'text', text: resp.text });
  for (const call of resp.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_0', name: call.name, input: argsObject(call.args) });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopName(resp.stop),
    stop_sequence: null,
    usage: { input_tokens: resp.usage.input, output_tokens: resp.usage.output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function pieces(text: string): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += SSE_PIECE) out.push(text.slice(at, at + SSE_PIECE));
  return out;
}

/** A canonical answer as a Chat Completions SSE body, ending with the usage chunk and [DONE]. */
export function openAIStreamBody(model: string, resp: CResp): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of pieces(resp.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  resp.toolCalls.forEach((call, index) => {
    frames.push(
      sseEvent(
        null,
        chunk(
          {
            tool_calls: [
              {
                index,
                id: call.id !== '' ? call.id : 'call_0',
                type: 'function',
                function: { name: call.name, arguments: call.args },
              },
            ],
          },
          null,
        ),
      ),
    );
  });
  frames.push(sseEvent(null, chunk({}, openAIStopName(resp.stop))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: resp.usage.input,
        completion_tokens: resp.usage.output,
        total_tokens: resp.usage.input + resp.usage.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A canonical answer as a Messages SSE body, in the API's event order, ending with message_stop. */
export function anthropicStreamBody(model: string, resp: CResp): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: resp.usage.input, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (resp.text !== '' || resp.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of pieces(resp.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of resp.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_0', name: call.name, input: {} },
      }),
    );
    if (call.args !== '') {
      frames.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.args } }),
      );
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopName(resp.stop), stop_sequence: null },
      usage: { output_tokens: resp.usage.output },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
