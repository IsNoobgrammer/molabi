# molabi

One OpenAI-compatible endpoint over one or more [molab](https://molab.marimo.io) boxes.

Every molab notebook ships marimo's AI chat, which can talk to a set of hosted open
models. molabi turns that into a plain `/v1/chat/completions` + `/v1/models` endpoint,
so any coding agent or OpenAI SDK can use it. Add several boxes and molabi spreads
requests across them: a box that hits a rate limit (429) cools down for that model and
the request moves to the next box before the client sees anything.

## Use it

1. Open a notebook on molab and copy its address (`https://sb-xxxxxxxxxxxxxxxx.sb.molab.run`).
   Keep the notebook open: molabi uses its live session.
2. Start molabi and paste the address on the **Boxes** page.
3. Point your agent at it:

```
OPENAI_BASE_URL=http://127.0.0.1:8765/v1
OPENAI_API_KEY=molabi        # anything, it is not checked
```

`GET /v1/models` lists the models. Streaming, tool calls and reasoning output are supported.
System prompts are folded into the first user message (marimo drops them otherwise).

The box URL works without a login, so treat it like a password. molabi keeps your boxes in
`config.json` next to the app and only listens on `127.0.0.1`.

## Windows

Download `molabi.exe` from [Releases](../../releases) and run it.

## macOS / Linux (build it yourself)

Needs Python 3.10+.

```sh
git clone https://github.com/IsNoobgrammer/molabi && cd molabi
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python app.py          # desktop window
.venv/bin/python pool.py         # headless, dashboard at http://127.0.0.1:8765/
```

To build a standalone app: `.venv/bin/pip install pyinstaller pillow && ./build.sh`
(gives `dist/molabi.app` on macOS).

Port: pass it as the first argument, e.g. `python pool.py 9000`.
Self-test: `python pool.py test`.

## Files

| File | What |
|---|---|
| `pool.py` | the server: OpenAI <-> marimo translation, box health checks, routing, dashboard API |
| `app.py` | desktop window (pywebview) around the dashboard |
| `ui/` | dashboard page, icon, Geist fonts |
| `build.bat` / `build.sh` | PyInstaller builds |
