const screen = document.querySelector('#screen');
const viewport = document.querySelector('#viewport');
const context = screen.getContext('2d', { alpha: false });
const address = document.querySelector('#address');
const toggle = document.querySelector('#depth-toggle');
const error = document.querySelector('#error');
let socket, depth = false, revision = 0, frames = 0, connected = false;
let currentURL = address.value, retryTimer, moveTimer;
let lastPointer, pointerDown = false;

function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}
function showError(message) { error.textContent = message; error.hidden = !message; }
function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onmessage = async ({ data }) => {
    if (typeof data === 'string') {
      const message = JSON.parse(data);
      if (message.type === 'ready') {
        connected = true;
        document.body.classList.add('connected');
        toggle.disabled = false;
        showError('');
        send({ type: 'mode', depth, revision });
        send({ type: 'navigate', url: currentURL });
      } else if (message.type === 'state') {
        if (message.url !== 'about:blank') {
          currentURL = message.url;
          viewport.dataset.url = currentURL;
          if (document.activeElement !== address) address.value = currentURL;
        }
        document.querySelector('#back').disabled = !message.back;
        document.querySelector('#forward').disabled = !message.forward;
      } else if (message.type === 'loading') {
        document.body.classList.toggle('loading', message.value);
      } else if (message.type === 'error') showError(message.message);
      return;
    }
    try {
      const header = new DataView(data);
      const frameDepth = Boolean(header.getUint8(0));
      const frameRevision = header.getUint32(1);
      if (frameDepth !== depth || frameRevision !== revision) return;
      const bitmap = await createImageBitmap(new Blob([data.slice(9)], { type: 'image/jpeg' }));
      // Toggling while decoding must not paint an obsolete mode.
      if (frameDepth === depth && frameRevision === revision) {
        if (screen.width !== bitmap.width || screen.height !== bitmap.height) {
          screen.width = bitmap.width; screen.height = bitmap.height;
        }
        context.drawImage(bitmap, 0, 0);
        viewport.dataset.frame = String(++frames);
        viewport.dataset.mode = depth ? 'depth' : 'normal';
        viewport.dataset.inferenceMs = header.getFloat32(5).toFixed(1);
        viewport.setAttribute('aria-busy', 'false');
        document.body.classList.remove('pending');
      }
      bitmap.close();
    } catch (exception) { showError('Unable to display this frame. Try reloading.'); }
    finally { send({ type: 'ack' }); }
  };
  socket.onclose = () => {
    connected = false;
    document.body.classList.remove('connected', 'loading');
    toggle.disabled = true;
    showError('Reconnecting…');
    clearTimeout(retryTimer);
    retryTimer = setTimeout(connect, 1500);
  };
}

document.querySelector('#navigation').addEventListener('submit', event => {
  event.preventDefault();
  if (!connected) return;
  showError('');
  send({ type: 'navigate', url: address.value });
  screen.focus({ preventScroll: true });
});
address.addEventListener('focus', () => address.select());
address.addEventListener('keydown', event => {
  if (event.key === 'Escape') { address.value = currentURL; screen.focus(); }
});
for (const action of ['back', 'forward', 'reload']) {
  document.querySelector(`#${action}`).addEventListener('click', () => {
    showError(''); send({ type: action }); screen.focus({ preventScroll: true });
  });
}
toggle.addEventListener('click', () => {
  depth = !depth; revision++;
  toggle.setAttribute('aria-checked', String(depth));
  document.body.classList.toggle('depth', depth);
  document.body.classList.add('pending');
  viewport.setAttribute('aria-busy', 'true');
  send({ type: 'mode', depth, revision });
});
document.querySelector('#expand').addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector('.workspace').requestFullscreen();
  } catch { showError('Fullscreen is unavailable in this browser.'); }
});

function point(event) {
  const bounds = screen.getBoundingClientRect();
  return { x: Math.max(0, Math.min(1439, (event.clientX - bounds.left) * 1440 / bounds.width)),
           y: Math.max(0, Math.min(899, (event.clientY - bounds.top) * 900 / bounds.height)) };
}
const buttons = ['left', 'middle', 'right'];
screen.addEventListener('pointerdown', event => {
  event.preventDefault();
  screen.focus({ preventScroll: true });
  screen.setPointerCapture(event.pointerId); pointerDown = true;
  send({ type: 'pointer', action: 'down', ...point(event), button: buttons[event.button] || 'left', count: event.detail || 1 });
});
screen.addEventListener('pointerup', event => {
  pointerDown = false;
  send({ type: 'pointer', action: 'up', ...point(event), button: buttons[event.button] || 'left', count: event.detail || 1 });
});
screen.addEventListener('pointercancel', event => {
  if (pointerDown) send({ type: 'pointer', action: 'up', ...point(event), button: 'left' });
  pointerDown = false;
});
screen.addEventListener('pointermove', event => {
  lastPointer = point(event);
  if (!moveTimer) moveTimer = setTimeout(() => {
    send({ type: 'pointer', action: 'move', ...lastPointer }); moveTimer = null;
  }, 25);
});
screen.addEventListener('wheel', event => {
  event.preventDefault();
  const scale = event.deltaMode === 1 ? 20 : event.deltaMode === 2 ? 900 : 1;
  send({ type: 'wheel', ...point(event), dx: event.deltaX * scale, dy: event.deltaY * scale });
}, { passive: false });
screen.addEventListener('contextmenu', event => event.preventDefault());
screen.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'l') {
    event.preventDefault(); address.focus(); return;
  }
  // Let the normal paste event read only the text the user explicitly pastes.
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'v') return;
  if (['Shift', 'Control', 'Alt', 'Meta', 'Dead', 'Unidentified'].includes(event.key) || event.isComposing) return;
  if (event.key === 'Escape' && document.fullscreenElement) return;
  event.preventDefault();
  if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
    send({ type: 'text', text: event.key });
  } else {
    let modifiers = [];
    if (event.ctrlKey) modifiers.push('Control');
    if (event.metaKey) modifiers.push('Meta');
    if (event.altKey) modifiers.push('Alt');
    if (event.shiftKey) modifiers.push('Shift');
    send({ type: 'key', key: [...modifiers, event.key === ' ' ? 'Space' : event.key].join('+') });
  }
});
screen.addEventListener('paste', event => {
  event.preventDefault(); send({ type: 'text', text: event.clipboardData.getData('text/plain') });
});
screen.addEventListener('compositionend', event => send({ type: 'text', text: event.data }));
window.addEventListener('beforeunload', () => { socket.onclose = null; socket.close(); });
connect();
