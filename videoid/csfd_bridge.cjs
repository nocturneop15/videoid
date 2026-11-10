// Persistent bridge between VideoID (Python) and node-csfd-api.
// Protocol: one JSON request per stdin line {id, cmd, arg, lang}; one JSON response per stdout line.
// The CSFD browser-verification cookie is persisted next to the package so it is reused across runs.
"use strict";
const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { createRequire } = require("module");

const dir = process.env.CSFD_DIR;
const req = createRequire(path.join(dir, "package.json"));
const { csfd, getAnubisCookie, setAnubisCookie } = req("node-csfd-api");
const cookieFile = path.join(dir, "verification_cookie.txt");

try {
  const c = fs.readFileSync(cookieFile, "utf8").trim();
  if (c) setAnubisCookie(c);
} catch (_) { /* first run */ }

function saveCookie() {
  try {
    const c = getAnubisCookie();
    if (c) fs.writeFileSync(cookieFile, c);
  } catch (_) { /* ignore */ }
}

async function handle(msg) {
  const opts = { language: msg.lang || "cs" };
  switch (msg.cmd) {
    case "movie": return csfd.movie(msg.arg, opts);
    case "search": return csfd.search(msg.arg, opts);
    case "creator": return csfd.creator(msg.arg, opts);
    case "ping": return { pong: true };
    default: throw new Error("unknown command " + msg.cmd);
  }
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  let msg;
  try { msg = JSON.parse(line); } catch (_) { return; }
  let out;
  try {
    const data = await handle(msg);
    out = { id: msg.id, ok: true, data };
    saveCookie();
  } catch (e) {
    out = { id: msg.id, ok: false, error: String(e && e.message || e), reason: e && e.reason || null };
  }
  process.stdout.write(JSON.stringify(out) + "\n");
});
process.stdout.write(JSON.stringify({ id: 0, ok: true, data: "ready" }) + "\n");
