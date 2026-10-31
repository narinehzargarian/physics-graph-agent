const assert = require("node:assert");
const { after, before, test } = require("node:test");
const { createApp } = require("./server");

let server;
let base;
const calls = [];
let upstream = async () => new Response(JSON.stringify({ answer: "ok" }), { status: 200 });

before(async () => {
  const fetchImpl = (url, options) => {
    calls.push({ url, options });
    return upstream(url, options);
  };
  const app = createApp({ agentUrl: "http://agent", wakewordUrl: "http://wake", fetchImpl, limit: 1000 });
  await new Promise((resolve) => {
    server = app.listen(0, resolve);
  });
  base = `http://127.0.0.1:${server.address().port}`;
});

after(() => server.close());

const postJson = (path, body) =>
  fetch(base + path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

test("forwards a valid question to the agent", async () => {
  const res = await postJson("/ask", { question: "What is plasma?", session_id: "abc" });
  assert.strictEqual(res.status, 200);
  assert.deepStrictEqual(await res.json(), { answer: "ok" });
  const call = calls.at(-1);
  assert.strictEqual(call.url, "http://agent/ask");
  assert.deepStrictEqual(JSON.parse(call.options.body), { question: "What is plasma?", session_id: "abc" });
});

test("rejects bad questions without calling the agent", async () => {
  const before_ = calls.length;
  for (const question of ["", "hi", 42, undefined, "x".repeat(501)]) {
    const res = await postJson("/ask", { question });
    assert.strictEqual(res.status, 400);
  }
  assert.strictEqual(calls.length, before_);
});

test("rejects malformed JSON", async () => {
  const res = await fetch(base + "/ask", { method: "POST", headers: { "content-type": "application/json" }, body: "{bad" });
  assert.strictEqual(res.status, 400);
});

test("forwards raw audio to the wake word service", async () => {
  const res = await fetch(base + "/wake", {
    method: "POST",
    headers: { "content-type": "application/octet-stream" },
    body: Buffer.from([1, 2, 3, 4]),
  });
  assert.strictEqual(res.status, 200);
  assert.strictEqual(calls.at(-1).url, "http://wake/detect");
  assert.strictEqual(calls.at(-1).options.body.length, 4);
});

test("rejects oversized audio", async () => {
  const res = await fetch(base + "/wake", {
    method: "POST",
    headers: { "content-type": "application/octet-stream" },
    body: Buffer.alloc(100 * 1024),
  });
  assert.strictEqual(res.status, 413);
});

test("returns 502 when the agent is down", async () => {
  upstream = async () => {
    throw new TypeError("connection refused");
  };
  const res = await postJson("/ask", { question: "What is plasma?" });
  assert.strictEqual(res.status, 502);
  upstream = async () => new Response(JSON.stringify({ answer: "ok" }), { status: 200 });
});

test("returns 504 when the agent times out", async () => {
  upstream = async () => {
    throw Object.assign(new Error("timeout"), { name: "TimeoutError" });
  };
  const res = await postJson("/ask", { question: "What is plasma?" });
  assert.strictEqual(res.status, 504);
  upstream = async () => new Response(JSON.stringify({ answer: "ok" }), { status: 200 });
});

test("rate limits per minute and recovers in the next minute", async () => {
  let clock = 0;
  const app = createApp({ agentUrl: "http://agent", wakewordUrl: "http://wake", fetchImpl: upstream, now: () => clock, limit: 2 });
  const limited = await new Promise((resolve) => {
    const s = app.listen(0, () => resolve(s));
  });
  const url = `http://127.0.0.1:${limited.address().port}/health`;
  assert.strictEqual((await fetch(url)).status, 200);
  assert.strictEqual((await fetch(url)).status, 200);
  assert.strictEqual((await fetch(url)).status, 429);
  clock = 60000;
  assert.strictEqual((await fetch(url)).status, 200);
  limited.close();
});
