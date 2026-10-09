"""molabi: one OpenAI-compatible endpoint over several molab boxes' marimo AI chat.

    python pool.py [port]        headless; dashboard at http://localhost:<port>/
    python app.py  [port]        same, in a desktop window

Clients use base url http://localhost:<port>/v1 with any api key.

A box needs no auth: its sb-*.molab.run URL is the credential, and the live
marimo session id is discovered from /api/home/running_notebooks. Requests
round-robin over healthy boxes; a 429 cools that (box, model) pair down and the
request moves to the next box before any byte reaches the client.
"""
import itertools, json, os, re, sys, threading, time, uuid, urllib.error, urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = getattr(sys, "frozen", False)  # pyinstaller exe: ui is unpacked to _MEIPASS, config sits beside the exe
RES = getattr(sys, "_MEIPASS", HERE)
CONFIG = os.path.join(os.path.dirname(sys.executable) if FROZEN else HERE, "config.json")
UA = "curl/8"  # cloudflare 403s the urllib default user agent

# probed 2026-10-08 from marimo 0.25.1's wandb model list; these answered.
# Qwen3-Coder-480B, DeepSeek-V4-Pro, DeepSeek-V4-Flash, Nemotron-3-Super-120B
# and Mellum2 are in the list but 404 upstream.
MODELS = [
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "zai-org/GLM-5.3-Flash",
    "zai-org/GLM-5.2",
    "MiniMaxAI/MiniMax-M3",
    "Qwen/Qwen3.8-27B",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "ibm-granite/granite-4.2-8b",
]

# ---------------------------------------------------------------- state

LOCK = threading.Lock()
RR = itertools.count()
BOXES = {}  # url -> {up, sid, err, version, checked, ok, fail, cool: {model: until}}
LOG = deque(maxlen=300)  # recent requests, newest last
TESTS = {}  # model -> {ok, ms, text|err, t}
CFG = {"boxes": [], "disabled": [], "cooldown": 60, "check_every": 60}


def load_cfg():
    if os.path.exists(CONFIG):
        CFG.update(json.load(open(CONFIG, encoding="utf8")))


def save_cfg():
    with LOCK:
        CFG["boxes"] = list(BOXES)
        json.dump(CFG, open(CONFIG, "w", encoding="utf8"), indent=2)


# ---------------------------------------------------------------- openai <-> marimo


def text_of(content):
    if isinstance(content, list):  # ponytail: text parts only, images dropped
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content or ""


def to_ui(messages):
    """OpenAI messages -> marimo/Vercel uiMessages. Tool results fold into the
    assistant's tool part, which is how the AI SDK represents them."""
    outputs = {m["tool_call_id"]: text_of(m["content"]) for m in messages if m["role"] == "tool"}
    # marimo swaps in its own system prompt and drops ours, so fold ours into the first user turn
    system = "\n\n".join(text_of(m["content"]) for m in messages if m["role"] in ("system", "developer"))
    ui = []
    for m in messages:
        role = m["role"]
        if role in ("tool", "system", "developer"):
            continue
        if role == "user" and system:
            m, system = {**m, "content": f"<system>\n{system}\n</system>\n\n{text_of(m['content'])}"}, ""
        parts = [{"type": "text", "text": text_of(m.get("content"))}] if text_of(m.get("content")) else []
        for tc in m.get("tool_calls") or []:
            f, cid = tc["function"], tc["id"]
            part = {"type": f"tool-{f['name']}", "toolCallId": cid, "input": json.loads(f["arguments"] or "{}")}
            if cid in outputs:
                part.update(state="output-available", output=outputs[cid])
            else:
                part["state"] = "input-available"
            parts.append(part)
        ui.append({"role": role, "parts": parts})
    return ui


def to_marimo_tools(tools):
    return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
             "parameters": t["function"].get("parameters", {"type": "object", "properties": {}}),
             "source": "frontend", "mode": ["manual", "ask", "agent"]}
            for t in tools or [] if t.get("type") == "function"]


def upstream_events(box, sid, req_body, model):
    """Yield parsed SSE events from one box."""
    body = {"includeOtherCode": "", "model": "wandb/" + model,
            "uiMessages": to_ui(req_body["messages"]), "tools": to_marimo_tools(req_body.get("tools"))}
    req = urllib.request.Request(f"{box}/api/ai/chat", data=json.dumps(body).encode(), headers={
        "content-type": "application/json", "marimo-session-id": sid, "user-agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data: {"):  # skips the final "data: [DONE]"
                yield json.loads(line[6:])


class UpstreamError(Exception):
    @property
    def status(self):
        m = re.search(r"status_code: (\d{3})", str(self))
        return int(m.group(1)) if m else 502


def chunks(events):
    """marimo events -> (openai delta | None, finish_reason | None). Raises UpstreamError."""
    idx, args_seen = {}, set()
    for ev in events:
        t = ev["type"]
        if t == "error":
            raise UpstreamError(ev["errorText"])
        if t == "text-delta":
            yield {"content": ev["delta"]}, None
        elif t == "reasoning-delta":
            yield {"reasoning_content": ev["delta"]}, None
        elif t == "tool-input-start":
            i = idx.setdefault(ev["toolCallId"], len(idx))
            yield {"tool_calls": [{"index": i, "id": ev["toolCallId"], "type": "function",
                                   "function": {"name": ev["toolName"], "arguments": ""}}]}, None
        elif t == "tool-input-delta":
            args_seen.add(ev["toolCallId"])
            yield {"tool_calls": [{"index": idx[ev["toolCallId"]], "function": {"arguments": ev["inputTextDelta"]}}]}, None
        elif t == "tool-input-available" and ev["toolCallId"] not in args_seen:
            args = json.dumps(ev["input"])
            if ev["toolCallId"] in idx:  # start already sent id + name
                yield {"tool_calls": [{"index": idx[ev["toolCallId"]], "function": {"arguments": args}}]}, None
            else:
                i = idx.setdefault(ev["toolCallId"], len(idx))
                yield {"tool_calls": [{"index": i, "id": ev["toolCallId"], "type": "function",
                                       "function": {"name": ev["toolName"], "arguments": args}}]}, None
        elif t == "finish":
            yield None, "tool_calls" if ev.get("finishReason") == "tool-calls" else "stop"


def collect(stream):
    """Fold a delta stream into one non-streaming assistant message."""
    msg, calls, finish = {"role": "assistant", "content": ""}, {}, "stop"
    for delta, f in stream:
        finish = f or finish
        for k in ("content", "reasoning_content"):
            if delta and k in delta:
                msg[k] = msg.get(k, "") + delta[k]
        for tc in (delta or {}).get("tool_calls", []):
            c = calls.setdefault(tc["index"], {"id": tc.get("id"), "type": "function",
                                               "function": {"name": "", "arguments": ""}})
            c["function"]["name"] += tc.get("function", {}).get("name", "")
            c["function"]["arguments"] += tc.get("function", {}).get("arguments", "")
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg, finish


# ---------------------------------------------------------------- pool


def get_json(url, method="GET"):
    req = urllib.request.Request(url, method=method, data=b"{}" if method == "POST" else None,
                                 headers={"content-type": "application/json", "user-agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def check(url):
    """Health + session discovery for one box; updates BOXES[url]."""
    try:
        st = get_json(f"{url}/api/status")
        files = get_json(f"{url}/api/home/running_notebooks", "POST")["files"]
        sid = next((f["sessionId"] for f in files if f.get("sessionId")), None)
        state = dict(up=bool(sid), sid=sid, version=st.get("version"),
                     err=None if sid else "Box is up but no notebook is open. Open it in molab once.")
    except urllib.error.HTTPError as e:
        state = dict(up=False, sid=None, err={410: "Box is gone (stopped or expired).",
                                              502: "Box is starting or unreachable."}.get(e.code, f"HTTP {e.code}"))
    except Exception as e:
        state = dict(up=False, sid=None, err=f"Unreachable: {e}"[:160])
    with LOCK:
        if url in BOXES:
            BOXES[url].update(state, checked=time.time())
    return state


def norm_url(url):
    url = url.strip().rstrip("/")
    if not re.match(r"^https?://", url):
        url = "https://" + url
    return url


def add_box(url):
    url = norm_url(url)
    with LOCK:
        BOXES.setdefault(url, dict(up=False, sid=None, err="Checking...", version=None,
                                   checked=0, ok=0, limited=0, fail=0, cool={}))
    check(url)
    save_cfg()
    return url


def remove_box(url):
    with LOCK:
        BOXES.pop(url, None)
    save_cfg()


def checker():
    while True:
        time.sleep(CFG["check_every"])
        for url in list(BOXES):
            check(url)


def pick(model, tried):
    """Next healthy box, not yet tried for this request, not cooling down for this model."""
    now = time.time()
    with LOCK:
        ok = [u for u, b in BOXES.items() if b["up"] and u not in tried and b["cool"].get(model, 0) <= now]
        return ok[next(RR) % len(ok)] if ok else None


def bump(box, key):
    with LOCK:
        if box in BOXES:
            BOXES[box][key] += 1


def route(body, model, rec):
    """Try boxes until one starts answering. Returns (delta stream, box) or raises
    PoolError(status, message). rec collects attempt notes for the request log."""
    tried, retried, waited = set(), set(), False
    while True:
        box = pick(model, tried)
        if box is None and not waited:  # cold start: finish first checks before giving up
            waited = True
            pending = [u for u, b in list(BOXES.items()) if not b["checked"]]
            for u in pending:
                check(u)
            if pending:
                continue
        if box is None:
            break
        tried.add(box)
        sid = BOXES[box]["sid"]
        rec["attempts"] += 1
        try:
            gen = chunks(upstream_events(box, sid, body, model))
            first = next(gen, (None, "stop"))  # pull one event so failures surface before we commit
            bump(box, "ok")

            def stream(first=first, gen=gen):
                yield first
                yield from gen
            return stream(), box
        except UpstreamError as e:
            if e.status == 404:  # unknown model: same answer on every box, not this box's fault
                raise PoolError(404, f"Model {model} is not served upstream.")
            bump(box, "limited" if e.status == 429 else "fail")
            if e.status == 429:
                with LOCK:
                    BOXES[box]["cool"][model] = time.time() + CFG["cooldown"]
            rec["notes"].append(f"{short(box)}: {e.status}")
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            bump(box, "fail")
            if e.code == 400 and "session" not in msg.lower():
                raise PoolError(400, msg)  # our request is bad; another box won't fix it
            if check(box)["sid"] not in (None, sid) and box not in retried:  # session rotated: retry once
                retried.add(box)
                tried.discard(box)
            rec["notes"].append(f"{short(box)}: HTTP {e.code}")
        except OSError as e:  # refused, timeout, DNS
            bump(box, "fail")
            check(box)
            rec["notes"].append(f"{short(box)}: {type(e).__name__}")
    now = time.time()
    with LOCK:
        waits = [b["cool"][model] - now for b in BOXES.values() if b["up"] and b["cool"].get(model, 0) > now]
    if waits:
        raise PoolError(429, f"Every box is rate-limited for {model}. Retry in {int(min(waits)) + 1}s.")
    raise PoolError(503, "No healthy box. Add one in the dashboard, or open a box's notebook.")


class PoolError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def short(url):
    return re.sub(r"^https?://(sb-)?|\.sb\.molab\.run$", "", url)[:12]


def log_request(rec):
    with LOCK:
        LOG.append(rec)


# ---------------------------------------------------------------- dashboard actions


def act_state(_):
    now = time.time()
    with LOCK:
        boxes = [dict(url=u, name=short(u), **{k: v for k, v in b.items() if k != "cool"},
                      cooling={m: int(t - now) for m, t in b["cool"].items() if t > now})
                 for u, b in BOXES.items()]
        log = list(LOG)[-150:][::-1]
    return dict(boxes=boxes, log=log, models=[dict(id=m, enabled=m not in CFG["disabled"], test=TESTS.get(m))
                                              for m in MODELS],
                cooldown=CFG["cooldown"], check_every=CFG["check_every"], now=now)


def act_add_box(a):
    return add_box(a["url"])


def act_remove_box(a):
    remove_box(a["url"])


def act_recheck(a):
    for url in [a["url"]] if a.get("url") else list(BOXES):
        check(url)


def act_toggle_model(a):
    with LOCK:
        d = set(CFG["disabled"])
        d.discard(a["id"]) if a["enabled"] else d.add(a["id"])
        CFG["disabled"] = sorted(d)
    save_cfg()


def act_settings(a):
    with LOCK:
        CFG["cooldown"] = max(5, int(a["cooldown"]))
        CFG["check_every"] = max(10, int(a["check_every"]))
    save_cfg()


def act_test_model(a):
    """One tiny real request through the pool, recorded like any other."""
    model, t0 = a["id"], time.time()
    rec = dict(id=uuid.uuid4().hex[:8], t=t0, model=model, box=None, status=200, attempts=0,
               notes=[], stream=False, ms=0, source="test")
    try:
        stream, box = route({"messages": [{"role": "user", "content": "Reply with just: OK"}]}, model, rec)
        rec["box"] = short(box)
        text = collect(stream)[0]["content"].strip()
        TESTS[model] = dict(ok=True, ms=int((time.time() - t0) * 1000), text=text[:40], t=time.time())
    except (PoolError, UpstreamError) as e:
        rec["status"] = getattr(e, "status", 502)
        TESTS[model] = dict(ok=False, ms=int((time.time() - t0) * 1000), err=str(e)[:160], t=time.time())
    rec["ms"] = int((time.time() - t0) * 1000)
    log_request(rec)
    return TESTS[model]


ACTIONS = {n[4:]: f for n, f in globals().items() if n.startswith("act_")}

# ---------------------------------------------------------------- http


STATIC = {"/fonts/geist.woff2": ("fonts/geist.woff2", "font/woff2"),
          "/fonts/geist-mono.woff2": ("fonts/geist-mono.woff2", "font/woff2"),
          "/icon.svg": ("icon.svg", "image/svg+xml")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # the dashboard is the log
        pass

    def send(self, code, data, ctype="application/json"):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def oai_error(self, code, msg):
        self.send(code, {"error": {"message": msg, "type": "pool_error", "code": code}})

    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "" or path in STATIC:
            name, ctype = STATIC.get(path, ("index.html", "text/html; charset=utf-8"))
            return self.send(200, open(os.path.join(RES, "ui", name), "rb").read(), ctype)
        if path.endswith("/v1/models"):
            return self.send(200, {"object": "list", "data": [
                {"id": m, "object": "model", "created": 0, "owned_by": m.split("/")[0]}
                for m in MODELS if m not in CFG["disabled"]]})
        self.oai_error(404, "Not found.")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path.startswith("/ui/") and path[4:] in ACTIONS:
            # ponytail: the dashboard is bound to 127.0.0.1 only, so no auth on these
            return self.send(200, {"result": ACTIONS[path[4:]](self.body())})
        if path.endswith("/chat/completions"):
            return self.chat(self.body())
        self.oai_error(404, "Not found.")

    def chat(self, body):
        model = (body.get("model") or MODELS[0]).removeprefix("wandb/")
        t0, stream_mode = time.time(), bool(body.get("stream"))
        rec = dict(id=uuid.uuid4().hex[:8], t=t0, model=model, box=None, status=200, attempts=0,
                   notes=[], stream=stream_mode, ms=0, source="api")
        try:
            if model in CFG["disabled"]:
                raise PoolError(403, f"Model {model} is disabled in the pool.")
            stream, box = route(body, model, rec)
            rec["box"] = short(box)
            cid, now = "chatcmpl-" + uuid.uuid4().hex, int(t0)

            def guarded():
                try:
                    yield from stream
                except UpstreamError as e:  # mid-stream: headers are out, surface it as text
                    rec["status"] = e.status
                    yield {"content": f"\n[upstream error] {e}"}, None
                    yield None, "stop"

            if stream_mode:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                for delta, finish in guarded():
                    chunk = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model,
                             "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                msg, finish = collect(guarded())
                self.send(200, {"id": cid, "object": "chat.completion", "created": now, "model": model,
                                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
        except PoolError as e:
            rec["status"] = e.status
            rec["notes"].append(str(e))
            self.oai_error(e.status, str(e))
        except OSError:  # client hung up mid-stream
            rec["status"] = 499
        finally:
            rec["ms"] = int((time.time() - t0) * 1000)
            log_request(rec)


def serve(port):
    load_cfg()
    for url in CFG["boxes"]:
        threading.Thread(target=add_box, args=(url,), daemon=True).start()
    threading.Thread(target=checker, daemon=True).start()
    ThreadingHTTPServer.request_queue_size = 128  # default 5 drops connections under agent bursts
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def selftest():
    ui = to_ui([{"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "out"}])
    assert [m["role"] for m in ui] == ["user", "assistant"]
    assert ui[0]["parts"][0]["text"] == "<system>\ns\n</system>\n\nhi"
    assert ui[1]["parts"] == [{"type": "tool-f", "toolCallId": "c1", "input": {"a": 1},
                               "state": "output-available", "output": "out"}]
    evs = [{"type": "tool-input-start", "toolCallId": "x", "toolName": "f"},
           {"type": "tool-input-available", "toolCallId": "x", "toolName": "f", "input": {"a": 1}},
           {"type": "finish", "finishReason": "tool-calls"}]
    msg, finish = collect(chunks(evs))
    assert msg["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a": 1}'} and finish == "tool_calls"
    BOXES.update(a=dict(up=True, cool={"m": time.time() + 99}), b=dict(up=True, cool={}), c=dict(up=False, cool={}))
    assert {pick("m", set()) for _ in range(4)} == {"b"}  # a cooling for m, c down
    assert {pick("x", set()) for _ in range(4)} == {"a", "b"}  # cooldown is per model
    assert pick("m", {"b"}) is None
    BOXES.clear()
    assert norm_url("sb-1.sb.molab.run/ ") == "https://sb-1.sb.molab.run"
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        selftest()
        sys.exit()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    serve(port)
    print(f"dashboard http://localhost:{port}/   openai http://localhost:{port}/v1")
    threading.Event().wait()
