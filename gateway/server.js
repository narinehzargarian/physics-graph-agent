// Node.js API gateway: one entry point for clients, with validation, rate limiting, and timeouts.
const express = require("express");

const LIMIT_PER_MINUTE = 30;

function rateLimiter(limit, now) {
  const hits = new Map();
  return (req, res, next) => {
    const minute = Math.floor(now() / 60000);
    const key = `${req.ip}:${minute}`;
    const count = (hits.get(key) || 0) + 1;
    hits.set(key, count);
    for (const old of hits.keys()) {
      if (!old.endsWith(`:${minute}`)) hits.delete(old);
    }
    if (count > limit) return res.status(429).json({ error: "too many requests" });
    next();
  };
}

function createApp({ agentUrl, wakewordUrl, fetchImpl = fetch, now = Date.now, limit = LIMIT_PER_MINUTE }) {
  const app = express();
  app.use(rateLimiter(limit, now));

  const forward = async (res, url, options, timeoutMs) => {
    try {
      const upstream = await fetchImpl(url, { ...options, signal: AbortSignal.timeout(timeoutMs) });
      res.status(upstream.status).json(await upstream.json());
    } catch (error) {
      const timedOut = error.name === "TimeoutError";
      res.status(timedOut ? 504 : 502).json({ error: timedOut ? "upstream timed out" : "upstream unavailable" });
    }
  };

  app.get("/health", (_req, res) => res.json({ status: "ok" }));

  app.post("/ask", express.json({ limit: "10kb" }), (req, res) => {
    const { question, session_id: sessionId } = req.body || {};
    if (typeof question !== "string" || question.trim().length < 3 || question.length > 500) {
      return res.status(400).json({ error: "question must be 3 to 500 characters" });
    }
    forward(res, `${agentUrl}/ask`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId || "default" }),
    }, 180000);
  });

  app.post("/wake", express.raw({ type: "application/octet-stream", limit: "64kb" }), (req, res) => {
    if (!Buffer.isBuffer(req.body) || req.body.length === 0) {
      return res.status(400).json({ error: "send raw audio as application/octet-stream" });
    }
    forward(res, `${wakewordUrl}/detect`, { method: "POST", body: req.body }, 5000);
  });

  // Oversized or malformed bodies end up here from the body parsers.
  app.use((error, _req, res, _next) => {
    res.status(error.status || 500).json({ error: error.status ? error.message : "internal error" });
  });
  return app;
}

if (require.main === module) {
  const port = Number(process.env.PORT || 3000);
  createApp({
    agentUrl: process.env.AGENT_URL || "http://localhost:8000",
    wakewordUrl: process.env.WAKEWORD_URL || "http://localhost:8001",
  }).listen(port, () => console.log(`gateway listening on ${port}`));
}

module.exports = { createApp };
