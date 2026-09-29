"""Local Chromium + CUDA depth browser. Run with .venv/Scripts/python web_app.py."""
from __future__ import annotations

import argparse
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
import logging
import os
from pathlib import Path
import struct
import time
from urllib.parse import quote_plus, urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from playwright.async_api import async_playwright
from starlette.middleware.trustedhost import TrustedHostMiddleware

from depth_inference import DepthEstimator, colorize

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache/ms-playwright"))
HOME = "https://www.google.com/"
WIDTH, HEIGHT = 1440, 900
log = logging.getLogger("depth-browser")


def navigation_url(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 4096:
        raise ValueError("Enter a URL or search")
    if "://" not in value:
        if value.lower().startswith(("javascript:", "data:", "file:", "about:", "chrome:")):
            raise ValueError("Only HTTP and HTTPS websites are supported")
        value = "https://" + value if "." in value and " " not in value else HOME + "search?q=" + quote_plus(value)
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Enter a valid HTTP or HTTPS URL")
    return value


class DepthWorker:
    """A single GPU session, kept on one worker thread across browser sessions."""

    def __init__(self, device, size):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="depth")
        self.device, self.size = device, size
        self.model = None
        self.lock = asyncio.Lock()

    async def start(self):
        def initialize():
            self.model = DepthEstimator(device=self.device, size=self.size)
            self.model.predict(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        await asyncio.get_running_loop().run_in_executor(self.executor, initialize)

    async def render(self, jpeg):
        def infer():
            tick = time.perf_counter()
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Invalid browser frame")
            depth = self.model.predict(image)
            ok, encoded = cv2.imencode(".jpg", colorize(depth), [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("Could not encode depth frame")
            return encoded.tobytes(), (time.perf_counter() - tick) * 1000
        async with self.lock:
            return await asyncio.get_running_loop().run_in_executor(self.executor, infer)


@asynccontextmanager
async def lifespan(app):
    worker = DepthWorker(app.state.device, app.state.size)
    app.state.worker = worker
    try:
        await worker.start()
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True, args=[
                "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows",
            ])
            app.state.browser = browser
            log.info("Ready: %s", worker.model.describe())
            try:
                yield
            finally:
                await browser.close()
    finally:
        worker.executor.shutdown(wait=True, cancel_futures=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.state.device = "cuda"
app.state.size = 518
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])


@app.get("/")
async def index():
    return FileResponse(ROOT / "web/index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health():
    return {"ready": True, "viewport": [WIDTH, HEIGHT], **app.state.worker.model.describe()}


class BrowserSession:
    def __init__(self, ws):
        self.ws = ws
        self.context = None
        self.page = None
        self.cdp = None
        self.depth = False
        self.revision = 0
        self.latest = None
        self.changed = asyncio.Event()
        self.acknowledged = asyncio.Event()
        self.acknowledged.set()
        self.send_lock = asyncio.Lock()
        self.tasks = set()
        self.navigation = None
        self.navigation_id = 0
        self.closing = False

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return task

    async def send(self, value):
        async with self.send_lock:
            await self.ws.send_json(value)

    async def start(self):
        self.context = await app.state.browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1,
            color_scheme="light", accept_downloads=False)
        self.page = await self.context.new_page()
        self.page.on("framenavigated", lambda frame: self.spawn(self.state()) if frame == self.page.main_frame else None)
        self.page.on("popup", lambda popup: self.spawn(self.popup(popup)))
        self.page.on("dialog", lambda dialog: self.spawn(dialog.dismiss()))
        self.cdp = await self.context.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", self.frame)
        await self.cdp.send("Page.startScreencast", {
            "format": "jpeg", "quality": 92, "maxWidth": WIDTH,
            "maxHeight": HEIGHT, "everyNthFrame": 1})
        await self.send({"type": "ready", "width": WIDTH, "height": HEIGHT})

    async def frame(self, event):
        if self.closing:
            return
        # Acknowledge capture immediately; retain only the newest browser frame.
        self.latest = base64.b64decode(event["data"])
        self.changed.set()
        with suppress(Exception):
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

    async def state(self):
        if self.closing:
            return
        with suppress(Exception):
            history = await self.cdp.send("Page.getNavigationHistory")
            await self.send({"type": "state", "url": self.page.url,
                             "back": history["currentIndex"] > 0,
                             "forward": history["currentIndex"] < len(history["entries"]) - 1})

    async def navigate(self, url=None, action=None, navigation_id=0):
        await self.send({"type": "loading", "value": True})
        try:
            if action == "back":
                await self.page.go_back(wait_until="domcontentloaded", timeout=25000)
            elif action == "forward":
                await self.page.go_forward(wait_until="domcontentloaded", timeout=25000)
            elif action == "reload":
                await self.page.reload(wait_until="domcontentloaded", timeout=25000)
            else:
                await self.page.goto(navigation_url(url), wait_until="domcontentloaded", timeout=25000)
            await self.send({"type": "error", "message": ""})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.info("Navigation failed: %s", exc)
            await self.send({"type": "error", "message": "This page could not load. Try another address or reload."})
        finally:
            if not self.closing and navigation_id == self.navigation_id:
                await self.send({"type": "loading", "value": False})
                await self.state()

    async def popup(self, popup):
        try:
            await popup.wait_for_load_state("domcontentloaded", timeout=15000)
            if popup.url.startswith(("http://", "https://")):
                self.begin_navigation(url=popup.url)
        finally:
            await popup.close()

    def begin_navigation(self, **kwargs):
        self.navigation_id += 1
        if self.navigation and not self.navigation.done():
            self.navigation.cancel()
        self.navigation = self.spawn(self.navigate(**kwargs, navigation_id=self.navigation_id))

    async def stream(self):
        while True:
            await self.changed.wait()
            # Only one frame may be in flight; slow clients cannot accumulate latency.
            await self.acknowledged.wait()
            self.changed.clear()
            jpeg, depth, revision = self.latest, self.depth, self.revision
            if jpeg is None:
                continue
            milliseconds = 0.0
            if depth:
                jpeg, milliseconds = await app.state.worker.render(jpeg)
            if revision != self.revision:
                self.changed.set()
                continue
            # One-byte mode + revision + inference milliseconds, then the JPEG.
            packet = struct.pack("!BIf", int(depth), revision, milliseconds) + jpeg
            self.acknowledged.clear()
            async with self.send_lock:
                await self.ws.send_bytes(packet)

    async def commands(self):
        while True:
            message = await self.ws.receive_json()
            kind = message.get("type")
            try:
                if kind == "ack":
                    self.acknowledged.set()
                elif kind == "navigate":
                    self.begin_navigation(url=navigation_url(str(message.get("url", ""))))
                elif kind in ("back", "forward", "reload"):
                    self.begin_navigation(action=kind)
                elif kind == "mode":
                    self.depth = bool(message["depth"])
                    self.revision = max(0, min(2**32 - 1, int(message["revision"])))
                    self.changed.set()
                elif kind == "pointer":
                    action = message["action"]
                    x = max(0, min(WIDTH - 1, float(message["x"])))
                    y = max(0, min(HEIGHT - 1, float(message["y"])))
                    if action == "move":
                        await self.page.mouse.move(x, y)
                    elif action in ("down", "up"):
                        button = message.get("button", "left")
                        if button not in ("left", "right", "middle"):
                            continue
                        await self.page.mouse.move(x, y)
                        method = self.page.mouse.down if action == "down" else self.page.mouse.up
                        await method(button=button, click_count=min(3, max(1, int(message.get("count", 1)))))
                elif kind == "wheel":
                    await self.page.mouse.move(float(message["x"]), float(message["y"]))
                    await self.page.mouse.wheel(max(-2000, min(2000, float(message["dx"]))),
                                                max(-2000, min(2000, float(message["dy"]))))
                elif kind == "key":
                    key = str(message["key"])
                    if len(key) < 100:
                        await self.page.keyboard.press(key)
                elif kind == "text":
                    await self.page.keyboard.insert_text(str(message["text"])[:10000])
            except (ValueError, KeyError, TypeError) as exc:
                await self.send({"type": "error", "message": str(exc)})
            except Exception as exc:
                log.info("Input failed: %s", exc)
                await self.send({"type": "error", "message": "The page is busy. Try again."})

    async def close(self):
        self.closing = True
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.context:
            await self.context.close()


@app.websocket("/ws")
async def websocket(ws: WebSocket):
    # A remote website must not be able to drive the local browser/GPU.
    origin = urlsplit(ws.headers.get("origin", ""))
    if origin.scheme not in ("http", "https") or origin.netloc != ws.headers.get("host"):
        await ws.close(code=1008)
        return
    await ws.accept()
    session = BrowserSession(ws)
    running = []
    try:
        await session.start()
        running = [asyncio.create_task(session.stream()), asyncio.create_task(session.commands())]
        done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Browser session ended")
        with suppress(Exception):
            await session.send({"type": "error", "message": "Connection interrupted. Reconnecting…"})
            await ws.close(code=1011)
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        await session.close()


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--size", type=int, default=518)
    args = parser.parse_args()
    app.state.device, app.state.size = args.device, args.size
    # Proactor is required for Playwright subprocesses on Windows. Start inside
    # asyncio.run so the server's Windows selector-loop choice cannot replace it.
    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, loop="asyncio",
                            ws_max_size=65536, log_level="info")
    asyncio.run(uvicorn.Server(config).serve())
