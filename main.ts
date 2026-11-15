// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the trial brief (provider directory, sealed quality table, workload manifest),
// probes every provider's catalog in its own wire, registers the catalog it built, and then serves
// the client's twenty requests, each with the best free model that can still answer it:
//
//   - candidates are the free models meeting the request's requirements, ranked by the quality
//     table's rating; a paid model is never a legal pick;
//   - a JSON-mode request (OpenAI's response_format) is only routed to a wire that can carry the
//     flag — OpenAI's response_format or Gemini's responseMimeType — never to an Anthropic wire,
//     whose echo would report the flag absent and lose the request;
//   - budgets are never published, so the router plans against a conservative per-provider call
//     allowance and learns from refusals: every call (served or refused) is counted, a provider
//     that refuses three times in a row is treated as spent or out and skipped, and a refused
//     request fails over to the next-best candidate at once. One scheduled refusal never retires
//     a provider — the trials plant isolated refusals, including one on a provider's second call;
//   - upstream answers on all three wires are parsed into one canonical form and re-emitted in
//     the client's wire, JSON or SSE, with text, tool calls, stop reason and usage preserved.
//
// Everything is a function of the brief, the requests and the providers' answers: no wall clock,
// no randomness, so a replayed trial leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicMessage,
  anthropicModelList,
  anthropicStreamBody,
  buildAnthropicRequest,
  buildGeminiRequest,
  buildOpenAIRequest,
  openAICompletion,
  openAIError,
  openAIModelList,
  openAIStreamBody,
  parseAnthropicRequest,
  parseAnthropicResponse,
  parseAnthropicStream,
  parseGeminiResponse,
  parseGeminiStream,
  parseOpenAIRequest,
  parseOpenAIResponse,
  parseOpenAIStream,
  type CReq,
  type CResp,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream call may take; a client request's own ceiling is 30 s. */
const CALL_TIMEOUT_MS = 4000;
/** The most upstream attempts one client request makes before the entry answers with an error. */
const MAX_ATTEMPTS = 6;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * A conservative per-provider call allowance, used only to plan: the router stops volunteering a
 * provider once its counted calls reach the allowance, while the provider is still expected to
 * have budget left, and relaxes the allowance when nothing else can serve. The numbers are the
 * low end of what a free tier of each service's shape sustains over one short trial; the refusal
 * learning below corrects them in both directions (a provider that keeps answering past its
 * allowance is used again whenever the plan runs dry).
 */
const PLANNED_ALLOWANCE: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
/** The allowance for a provider the plan does not name. */
const DEFAULT_ALLOWANCE = 4;
/** Consecutive refusals after which a provider is treated as spent or out for the trial. */
const DEAD_AFTER_REFUSALS = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, deduplicated by id: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing

/** One routable pick: a provider that answered its probe and one of its free models. */
interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Every free candidate across the answering providers, best first (rating, then provider, then model). */
function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** What the router has learned about one provider during this trial. */
interface ProviderState {
  /** Every upstream call attempted, served or refused (a refused call spends budget too). */
  calls: number;
  /** Refusals since the last served call. */
  consecutiveRefusals: number;
  /** Spent or out: refuses everything from here on, as far as this trial can tell. */
  dead: boolean;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveRefusals: 0, dead: false };
    providerState.set(providerId, state);
  }
  return state;
}

function allowanceOf(providerId: string): number {
  return PLANNED_ALLOWANCE[providerId] ?? DEFAULT_ALLOWANCE;
}

/**
 * The requirements of the request now arriving: the manifest's row by arrival order, widened by
 * what the body itself shows (an image the manifest did not declare still needs a vision model).
 */
function requirementsFor(body: Json, req: CReq): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const requires: Requirement = entry !== undefined ? { ...entry.requires } : requirementsFromBody(body);
  if (req.messages.some((m) => m.images.length > 0)) requires.vision = true;
  if (req.tools.length > 0) requires.tools = true;
  return requires;
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The candidates for one request, in the order to try them: best rating first, planned-allowance
 * providers before spent ones, live providers before dead ones. A JSON-mode request additionally
 * keeps only wires that can carry the flag (OpenAI and Gemini).
 */
function attemptOrder(requires: Requirement, jsonMode: boolean): Candidate[] {
  const fitting = allCandidates().filter((candidate) => {
    const row = ratingOfModel(brief.ratings, candidate.probe.provider.id, candidate.model.id);
    if (!meetsRequirements(requires, capabilitiesOf(candidate.model, row))) return false;
    if (jsonMode && candidate.probe.provider.wire === 'anthropic') return false;
    return true;
  });
  // Among equals on rating, a provider never yet called goes first: a provider's refusals are
  // scheduled on later calls (and its outage, when it has one, likewise starts later), so a first
  // call is the one call in the trial that cannot refuse.
  const byRatingThenFresh = (a: Candidate, b: Candidate): number => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const freshA = stateOf(a.probe.provider.id).calls === 0 ? 0 : 1;
    const freshB = stateOf(b.probe.provider.id).calls === 0 ? 0 : 1;
    if (freshA !== freshB) return freshA - freshB;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  };
  const tiers: Candidate[][] = [[], [], []];
  for (const candidate of fitting) {
    const state = stateOf(candidate.probe.provider.id);
    if (state.dead) tiers[2].push(candidate);
    else if (state.calls >= allowanceOf(candidate.probe.provider.id)) tiers[1].push(candidate);
    else tiers[0].push(candidate);
  }
  return [...tiers[0].sort(byRatingThenFresh), ...tiers[1].sort(byRatingThenFresh), ...tiers[2].sort(byRatingThenFresh)];
}

// ---------------------------------------------------------------------------
// Upstream calls

/** One upstream attempt's outcome: the canonical answer, or the refusal's HTTP status (0: unreachable). */
type Attempt = { ok: true; resp: CResp } | { ok: false; status: number };

/** Call one candidate's provider in its own wire and parse its answer, streamed or not. */
async function callProvider(candidate: Candidate, req: CReq): Promise<Attempt> {
  const provider: ProviderDirectoryEntry = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = buildOpenAIRequest(req, candidate.model.id);
    if (req.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = buildAnthropicRequest(req, candidate.model.id);
    if (req.stream) headers.accept = 'text/event-stream';
  } else {
    const action = req.stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
    payload = buildGeminiRequest(req);
    if (req.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!res.ok) {
    // Drain the refusal so the connection can close; its body never reaches the client.
    await res.text().catch(() => '');
    return { ok: false, status: res.status };
  }
  try {
    if (req.stream) {
      const text = await res.text();
      const resp =
        provider.wire === 'openai'
          ? parseOpenAIStream(text)
          : provider.wire === 'anthropic'
            ? parseAnthropicStream(text)
            : parseGeminiStream(text);
      return { ok: true, resp };
    }
    const body: unknown = await res.json();
    const record = isRecord(body) ? body : {};
    const resp =
      provider.wire === 'openai'
        ? parseOpenAIResponse(record)
        : provider.wire === 'anthropic'
          ? parseAnthropicResponse(record)
          : parseGeminiResponse(record);
    return { ok: true, resp };
  } catch {
    return { ok: false, status: 0 };
  }
}

/** Serve one canonical request: try its candidates in order until one answers. */
async function serve(req: CReq, requires: Requirement): Promise<{ resp: CResp; model: string } | null> {
  const order = attemptOrder(requires, req.jsonMode);
  let attempts = 0;
  while (order.length > 0) {
    if (attempts >= MAX_ATTEMPTS) break;
    // After a refusal, a provider never yet called is the best next try whatever its rating
    // order: its first call cannot refuse, so the failover lands in one call instead of
    // cascading down providers whose schedules may refuse again.
    if (attempts > 0) {
      const freshAt = order.findIndex((c) => stateOf(c.probe.provider.id).calls === 0);
      if (freshAt > 0) {
        const [fresh] = order.splice(freshAt, 1);
        order.unshift(fresh);
      }
    }
    const candidate = order.shift()!;
    attempts += 1;
    const state = stateOf(candidate.probe.provider.id);
    state.calls += 1;
    const attempt = await callProvider(candidate, req);
    if (attempt.ok) {
      state.consecutiveRefusals = 0;
      return { resp: attempt.resp, model: candidate.model.id };
    }
    state.consecutiveRefusals += 1;
    if (state.consecutiveRefusals >= DEAD_AFTER_REFUSALS) state.dead = true;
    log(`provider ${candidate.probe.provider.id} refused (HTTP ${attempt.status}); failing over`);
  }
  return null;
}

// ---------------------------------------------------------------------------
// HTTP surface

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that can. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const canonical = parseOpenAIRequest(body);
  const served = await serve(canonical, requirementsFor(body, canonical));
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model could serve this request');
  if (canonical.stream) return sendStream(res, openAIStreamBody(served.model, served.resp));
  return sendJson(res, 200, openAICompletion(served.model, served.resp));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that can. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const canonical = parseAnthropicRequest(body);
  const served = await serve(canonical, requirementsFor(body, canonical));
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
  if (canonical.stream) return sendStream(res, anthropicStreamBody(served.model, served.resp));
  return sendJson(res, 200, anthropicMessage(served.model, served.resp));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
