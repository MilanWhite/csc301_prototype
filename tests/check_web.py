"""End-to-end checks against a running web_app.py; uses a separate Chromium."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache/ms-playwright"))
OUT = ROOT / "outputs/web-tests"
OUT.mkdir(parents=True, exist_ok=True)
EVENTS = []

FIXTURE = b'''<!doctype html><html><head><style>
*{box-sizing:border-box}body{margin:0;background:#eee;font:24px sans-serif;color:#152324}
.photo{height:650px;overflow:hidden;background:#244d50}.photo img{width:100%;height:100%;object-fit:cover;animation:drift 5s alternate infinite linear}
.controls{position:absolute;top:30px;left:30px;background:white;padding:20px;display:flex;gap:20px}
button,input,a{font:22px sans-serif;padding:10px}section{height:1400px;background:linear-gradient(#abc4b0,#344c6c);padding:50px}
@keyframes drift{from{transform:scale(1.05) translateX(-2%)}to{transform:scale(1.2) translateX(2%)}}
</style></head><body><div class="photo"><img src="/scene.jpg"></div>
<div class="controls"><button onclick="fetch('/event?click');this.textContent='Clicked'">Click me</button>
<input aria-label="Test input" oninput="fetch('/event?text='+encodeURIComponent(this.value))" style="width:240px">
<a href="/second">Next</a><a href="/second" target="_blank">Popup</a></div>
<section>Scroll target</section><script>let last=0;addEventListener('scroll',()=>{if(Date.now()-last>100){fetch('/event?scroll='+scrollY);last=Date.now()}})</script></body></html>'''


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.startswith('/event?'):
            EVENTS.append(self.path)
            content, kind = b'ok', 'text/plain'
        elif self.path == '/scene.jpg':
            sample = ROOT / 'outputs/sample_cats.jpg'
            if sample.exists():
                content = sample.read_bytes()
            else:
                rng = np.random.default_rng(44)
                photo = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
                content = cv2.imencode('.jpg', photo)[1].tobytes()
            kind = 'image/jpeg'
        elif self.path == '/second':
            content, kind = b'<html><body style="background:#5c7b83"><h1>Second page</h1></body></html>', 'text/html'
        else:
            content, kind = FIXTURE, 'text/html'
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(content)))
        # Prove this browser does not depend on iframe embedding permission.
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy', "frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)


async def eventually(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(.05)
    raise AssertionError('Expected input event was not received')


async def main():
    fixture = ThreadingHTTPServer(('127.0.0.1', 8766), Fixture)
    threading.Thread(target=fixture.serve_forever, daemon=True).start()
    errors = []
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1)
            page.on('pageerror', lambda error: errors.append(str(error)))
            await page.goto('http://127.0.0.1:8765/')
            await page.wait_for_function("!document.querySelector('#depth-toggle').disabled")
            address = page.get_by_role('textbox', name='Website address or search')
            await address.fill('http://127.0.0.1:8766/')
            await address.press('Enter')
            await page.wait_for_function("document.querySelector('#viewport').dataset.url==='http://127.0.0.1:8766/' && Number(document.querySelector('#viewport').dataset.frame)>2")
            await page.screenshot(path=str(OUT / 'normal.png'))
            screen = page.get_by_role('application')
            bounds = await screen.bounding_box()
            async def click(x, y):
                await page.mouse.click(bounds['x'] + x * bounds['width'] / 1440,
                                       bounds['y'] + y * bounds['height'] / 900)
            await click(110, 75)
            await eventually(lambda: '/event?click' in EVENTS)
            await click(310, 75)
            await page.keyboard.type('Depth test 123')
            await eventually(lambda: any('Depth%20test%20123' in event for event in EVENTS))

            toggle = page.get_by_role('switch', name='Depth view')
            tick = time.perf_counter()
            await toggle.click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.mode==='depth' && !document.body.classList.contains('pending')")
            toggle_ms = (time.perf_counter() - tick) * 1000
            await page.screenshot(path=str(OUT / 'depth.png'))
            first_frame = int(await page.locator('#viewport').get_attribute('data-frame'))
            tick = time.perf_counter()
            await page.wait_for_function('(first) => Number(document.querySelector("#viewport").dataset.frame) >= first + 20', arg=first_frame, timeout=15000)
            depth_fps = 20 / (time.perf_counter() - tick)
            await page.screenshot(path=str(OUT / 'depth-later.png'))
            shape = await screen.evaluate('(canvas) => [canvas.width, canvas.height]')
            assert shape == [1440, 900], shape
            inference_ms = await page.locator('#viewport').get_attribute('data-inference-ms')

            await click(310, 75)
            await page.keyboard.press('End')
            await page.keyboard.type(' live')
            await eventually(lambda: any('123%20live' in event for event in EVENTS))
            await page.mouse.move(bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2)
            await page.mouse.wheel(0, 620)
            await eventually(lambda: any('scroll=' in event and event != '/event?scroll=0' for event in EVENTS))
            await toggle.click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.mode==='normal'")
            # Repeated switches must settle in the latest requested mode.
            for _ in range(5):
                await toggle.click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.mode==='depth' && !document.body.classList.contains('pending')")
            await toggle.click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.mode==='normal'")
            await address.fill('http://127.0.0.1:8766/second')
            await address.press('Enter')
            await page.wait_for_function("document.querySelector('#viewport').dataset.url?.endsWith('/second')")
            await page.get_by_role('button', name='Go back', exact=True).click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.url==='http://127.0.0.1:8766/'")
            await page.get_by_role('button', name='Go forward', exact=True).click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.url?.endsWith('/second')")
            await page.get_by_role('button', name='Reload page').click()
            await page.wait_for_function("!document.body.classList.contains('loading')")

            # Real in-page links, including target=_blank, stay inside this browser.
            await page.get_by_role('button', name='Go back', exact=True).click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.url==='http://127.0.0.1:8766/'")
            await page.mouse.move(bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2)
            await page.mouse.wheel(0, -2000)
            await eventually(lambda: '/event?scroll=0' in EVENTS)
            await click(560, 75)
            await page.wait_for_function("document.querySelector('#viewport').dataset.url?.endsWith('/second')")
            await page.get_by_role('button', name='Go back', exact=True).click()
            await page.wait_for_function("document.querySelector('#viewport').dataset.url==='http://127.0.0.1:8766/'")
            await click(470, 75)
            await page.wait_for_function("document.querySelector('#viewport').dataset.url?.endsWith('/second')")

            await page.get_by_role('button', name='Toggle fullscreen').click()
            await page.wait_for_function('Boolean(document.fullscreenElement)')
            await page.get_by_role('button', name='Toggle fullscreen').click()
            await page.wait_for_function('!document.fullscreenElement')
            before = int(await page.locator('#viewport').get_attribute('data-frame'))
            await page.evaluate('socket.close()')
            await page.wait_for_function("(before) => document.body.classList.contains('connected') && Number(document.querySelector('#viewport').dataset.frame)>before", arg=before)

            await address.fill('file:///C:/Windows')
            await address.press('Enter')
            await page.get_by_role('alert').wait_for(state='visible')
            await page.get_by_role('button', name='Reload page').click()

            await page.set_viewport_size({'width': 390, 'height': 844})
            await page.screenshot(path=str(OUT / 'mobile.png'))
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            await page.set_viewport_size({'width': 1600, 'height': 1100})
            before = int(await page.locator('#viewport').get_attribute('data-frame'))
            await address.fill('https://www.wikipedia.org/')
            await address.press('Enter')
            await page.wait_for_function("(before) => document.querySelector('#viewport').dataset.url?.includes('wikipedia.org') && !document.body.classList.contains('loading') && Number(document.querySelector('#viewport').dataset.frame)>before", arg=before, timeout=30000)
            await page.screenshot(path=str(OUT / 'public-website.png'))
            assert await page.locator('#error').is_hidden(), await page.locator('#error').inner_text()
            before = int(await page.locator('#viewport').get_attribute('data-frame'))
            await address.fill('https://www.google.com/')
            await address.press('Enter')
            await page.wait_for_function("(before) => document.querySelector('#viewport').dataset.url?.includes('google.com') && !document.body.classList.contains('loading') && Number(document.querySelector('#viewport').dataset.frame)>before", arg=before, timeout=30000)
            await page.screenshot(path=str(OUT / 'google.png'))
            assert await page.locator('#error').is_hidden(), await page.locator('#error').inner_text()
            assert not errors, errors
            result = {'canvas': shape, 'depth_stream_fps': round(depth_fps, 1),
                      'toggle_latency_ms': round(toggle_ms, 1), 'last_inference_ms': inference_ms,
                      'javascript_errors': errors, 'checks': ['iframe-blocked site', 'mouse click',
                      'typing', 'depth toggle', 'live depth frames', 'typing in depth mode',
                      'scrolling in depth mode', 'rapid toggles', 'back', 'forward', 'reload',
                      'popup link', 'same-tab link', 'fullscreen', 'reconnect', 'invalid URL',
                      'mobile layout', 'Wikipedia', 'Google']}
            (OUT / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps(result, indent=2))
            await browser.close()
    finally:
        fixture.shutdown()


if __name__ == '__main__':
    asyncio.run(main())
