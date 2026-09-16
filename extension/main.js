// Pinned to specific versions rather than @latest: an extension loaded
// straight from a CDN on every page load has no build step to catch a
// breaking (or malicious) change in an upstream package before it reaches
// users — @latest means anyone able to publish to either package can affect
// this extension the moment they do. Bump these deliberately when needed.
import OBR, { buildShape } from 'https://cdn.jsdelivr.net/npm/@owlbear-rodeo/sdk@3.1.0/+esm'
import { io } from 'https://cdn.jsdelivr.net/npm/socket.io-client@4.8.3/dist/socket.io.esm.min.js'

document.querySelector('#app').innerHTML = `
  <div class="container">
    <h2>Token Sync</h2>
    <div id="build-info" class="build-info">Not connected</div>
    <div class="connection-box">
      <input type="text" id="ws-url" value="http://localhost:5000/" placeholder="Tracker URL (e.g. http://localhost:5000/)" />
      <button id="connect-btn">Connect</button>
      <div id="status" class="status disconnected">Disconnected</div>
    </div>

    <div class="mapping-section">
      <h3>Token Assignment</h3>
      <p class="subtitle">Assign physical tokens to virtual ones.</p>
      <div id="mapping-list" class="mapping-list">
        <p class="empty-msg">No physical tokens detected yet.</p>
      </div>
    </div>

    <div class="mapping-section">
      <h3>Sync Performance</h3>
      <div class="control-row">
        <label>Max Update Rate (FPS): <span id="fps-val">30</span></label>
        <input type="range" id="sync-fps" min="5" max="60" value="30">
      </div>
      <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">How often positions are pushed to Owlbear. Motion is smoothly interpolated between updates from the tracker, so higher just means less network/host load, not choppier motion.</p>
    </div>

    <div class="blackout-settings">
      <h3>Blackout Settings</h3>
      <div class="control-row">
        <label>Color:</label>
        <select id="blackout-color">
          <option value="#000000">Black</option>
          <option value="#ffffff">White</option>
        </select>
      </div>
      <button id="test-blackout-btn" class="secondary">Test Blackout (1s)</button>
    </div>
  </div>
`

let socket = null;
let isReady = false;
let tokenMapping = {}; // physicalId -> virtualItemId
let assignedNames = {}; // physicalId -> virtualName (Used to re-sync across scenes)
let currentPhysicalTokens = [];
let virtualTokens = [];
let sceneItemsCache = [];
let viewportSize = { width: 0, height: 0 };

// --- Motion smoothing ---------------------------------------------------
// The tracker only emits a fresh position ~10-15x/sec (whatever the camera
// captures at), and applying each one directly to Owlbear the instant it
// arrives — which is what this used to do — makes tokens visibly hop from
// point to point rather than glide. Instead, `latestTokens` holds the most
// recent raw sample from the server (updated instantly, no throttling), and
// a setInterval-driven loop continuously eases `smoothState` toward it
// using exponential smoothing (a fixed fraction of the remaining distance
// per unit time, independent of the server's actual sample rate/jitter).
// That loop is what actually pushes to Owlbear, at up to TARGET_FPS times a
// second — decoupling "how often we learn something new" from "how often we
// tell Owlbear about it" is what turns the jumps into a smooth glide.
let latestTokens = {}; // physicalId -> {id, alias, x, y} raw, normalized [0,1]
let smoothState = {};  // physicalId -> {x, y} currently-applied, normalized [0,1]
let latestBlank = false;
let lastServerBlankScreen = false; // most recent value from the server (for test-blackout restore)
let lastAppliedBlank = false;      // what's actually been pushed to Owlbear
let TARGET_FPS = 30;
const SMOOTH_RATE = 20; // 1/s — higher = snappier/less lag, lower = smoother but more visible delay
const POS_EPSILON = 0.0005; // normalized units; below this, treat as "arrived" and stop nudging

// The periodic render loop below calls applyBlackout() using the tracker's
// real blank_screen state. Without a guard, that would immediately stomp on
// a manual "Test Blackout" click — the overlay would get added, then
// removed again on the very next tick, which is why the test only ever
// flashed for a few frames instead of holding for its full duration.
let manualBlackoutUntil = 0; // Date.now()-based deadline; 0 = no active override

// UI-only listeners are bound immediately, synchronously, with no
// dependency on any OBR.* call succeeding first. Previously these lived
// inside the OBR.onReady(async () => {...}) callback, after a couple of
// awaited OBR calls — if any of those calls rejected (e.g. scene not fully
// available yet), the rest of the callback silently never ran, which meant
// nothing after that point (including these listeners) was ever wired up.
// That's the most likely explanation for "the sliders don't do anything":
// not that the slider logic was wrong, but that it never got attached at
// all. Binding these first, unconditionally, means a later OBR failure
// can't take them down with it.
document.getElementById('sync-fps').addEventListener('input', (e) => {
  TARGET_FPS = parseInt(e.target.value, 10);
  document.getElementById('fps-val').innerText = TARGET_FPS;
  if (_intervalHandle) startRenderLoop(); // apply the new rate immediately instead of waiting for a reconnect
});

document.getElementById('test-blackout-btn').addEventListener('click', async () => {
  const TEST_DURATION_MS = 1000;
  manualBlackoutUntil = Date.now() + TEST_DURATION_MS;
  await applyBlackout(true);
  setTimeout(async () => {
    manualBlackoutUntil = 0;
    // Hand control back to whatever the tracker actually wants right now,
    // rather than unconditionally turning the overlay off — if a real
    // blackout started during the test window, this keeps it on.
    await applyBlackout(lastServerBlankScreen);
  }, TEST_DURATION_MS);
});

// Re-apply color immediately when changed, so switching the dropdown has a
// visible effect right away instead of waiting for the next
// activate/deactivate transition (which might not happen for a while).
document.getElementById('blackout-color').addEventListener('change', async () => {
  if (Date.now() < manualBlackoutUntil || lastServerBlankScreen) {
    await applyBlackout(true);
  }
});

document.getElementById('connect-btn').addEventListener('click', () => {
  const url = document.getElementById('ws-url').value;
  connectSocketIO(url);
});

OBR.onReady(async () => {
  isReady = true;
  document.getElementById('status').innerText = "Ready. Connect to Tracker.";
  document.getElementById('status').className = "status ready";

  // Name-Based Re-sync: Update mapping whenever the scene changes. Also
  // keeps sceneItemsCache current so the render loop never has to await a
  // fresh OBR.scene.items.getItems() on every push — that round trip used
  // to happen up to 15x/sec purely to re-fetch a list that only actually
  // changes when this callback fires anyway.
  OBR.scene.items.onChange((items) => {
    sceneItemsCache = items;
    virtualTokens = items.filter(item => item.layer === "CHARACTER" || item.layer === "MOUNT");

    // Merge name-based matches into existing mapping (don't overwrite manual assignments)
    for (const [physicalId, name] of Object.entries(assignedNames)) {
      const match = virtualTokens.find(vt => (vt.text && vt.text.plainText === name) || vt.name === name);
      if (match) {
        tokenMapping[physicalId] = match.id;
      }
    }
    renderMappingUI();
  });

  try {
    // Initial fetch — everything after this used to live in this same
    // callback and would never run if this particular call failed. It's
    // now the *only* thing gated on it; onChange above will keep things
    // current from here on regardless.
    const items = await OBR.scene.items.getItems();
    sceneItemsCache = items;
    virtualTokens = items.filter(item => item.layer === "CHARACTER" || item.layer === "MOUNT");
  } catch (e) {
    console.error("Initial scene items fetch failed:", e);
  }

  await refreshViewportSize();
  // Viewport pixel size only changes on window resize, not per-frame — a
  // slow background poll is enough and avoids a getWidth()/getHeight()
  // round trip on every single position push.
  setInterval(refreshViewportSize, 2000);

  startRenderLoop();
});

async function refreshViewportSize() {
  try {
    const [width, height] = await Promise.all([OBR.viewport.getWidth(), OBR.viewport.getHeight()]);
    viewportSize = { width, height };
  } catch (e) {
    console.error("Viewport size fetch failed:", e);
  }
}

// Shows which build of the tracker (and by extension, which build of this
// extension's own served files, since both are baked into the same Docker
// image) is actually running — so a mismatch against what you just pushed
// tells you the browser is holding onto stale cached JS, not the server.
function fetchBuildInfo(url) {
  const buildEl = document.getElementById('build-info');
  fetch(new URL('api/build_info', url).toString())
    .then(res => res.json())
    .then(info => {
      const commit = (info.build_commit || 'unknown').substring(0, 7);
      buildEl.innerText = `tracker: ${info.build_branch || 'unknown'} · v${info.build_version || 'dev'} · ${commit}`;
    })
    .catch(() => {
      buildEl.innerText = 'Build info unavailable';
    });
}

// Measures how long an actual push to Owlbear takes (transform round trips
// + updateItems), separate from how often we attempt one. Every OBR SDK
// call is a postMessage round-trip to the host, so this is a meaningful
// slice of end-to-end delay. Averaged over a window rather than logged per
// call so it doesn't flood the console at push rate.
let _syncSamples = [];
let _syncLastReport = performance.now();
function recordSyncLatency(ms) {
  _syncSamples.push(ms);
  const elapsed = performance.now() - _syncLastReport;
  if (elapsed >= 5000 && _syncSamples.length) {
    const avg = _syncSamples.reduce((a, b) => a + b, 0) / _syncSamples.length;
    const worst = Math.max(..._syncSamples);
    console.log(`[SyncPerf] OBR push: avg ${avg.toFixed(0)}ms, worst ${worst.toFixed(0)}ms over ${_syncSamples.length} pushes`);
    _syncSamples = [];
    _syncLastReport = performance.now();
  }
}

function connectSocketIO(url) {
  if (socket) socket.disconnect();

  document.getElementById('status').innerText = "Connecting...";
  document.getElementById('status').className = "status ready";
  fetchBuildInfo(url);

  try {
    // Engine.IO defaults to starting every connection on HTTP polling and
    // only upgrading to WebSocket after an extra round-trip. Listing
    // websocket first tries it immediately, falling back to polling only if
    // it's genuinely unavailable — skips that handshake delay on every
    // connect/reconnect on a LAN setup like this one.
    socket = io(url, { transports: ['websocket', 'polling'] });

    socket.on('connect', () => {
      document.getElementById('status').innerText = "Connected to Tracker";
      document.getElementById('status').className = "status connected";
    });

    socket.on('disconnect', () => {
      document.getElementById('status').innerText = "Disconnected";
      document.getElementById('status').className = "status disconnected";
    });

    // Deliberately cheap: no OBR calls, no awaits. This just records the
    // latest known state so the render loop (below) can pick it up on its
    // own schedule. Previously this handler did the entire OBR round trip
    // itself and dropped any message that arrived while a previous one was
    // still in flight — since each OBR round trip can plausibly take
    // longer than the ~66ms gap between camera frames, that dropped the
    // effective update rate to whatever the OBR round trip allowed, often
    // just 1-2/sec, regardless of how fast the tracker was actually
    // emitting.
    socket.on('tokens_update', (data) => {
      if (!isReady) return;

      const tokens = data.tokens || [];
      latestBlank = data.blank_screen || false;
      lastServerBlankScreen = latestBlank;

      const seenIds = new Set();
      for (const t of tokens) {
        seenIds.add(t.id);
        latestTokens[t.id] = t;
        if (!smoothState[t.id]) {
          // First time seeing this token: snap immediately rather than
          // easing in from (0,0) or some other undefined starting point.
          smoothState[t.id] = { x: t.x, y: t.y };
        }
      }
      for (const id of Object.keys(latestTokens)) {
        if (!seenIds.has(id)) {
          delete latestTokens[id];
          delete smoothState[id];
        }
      }

      // Mapping UI / auto-assignment bookkeeping — all local, no OBR calls,
      // safe to do synchronously on every message.
      let mappingChanged = false;
      const oldIds = currentPhysicalTokens.map(t => t.id).sort().join(',');
      currentPhysicalTokens = tokens;

      currentPhysicalTokens.forEach(pt => {
        if (pt.alias && !tokenMapping[pt.id]) {
          const match = virtualTokens.find(vt => (vt.text && vt.text.plainText === pt.alias) || vt.name === pt.alias);
          if (match) {
            tokenMapping[pt.id] = match.id;
            assignedNames[pt.id] = pt.alias;
            mappingChanged = true;
          }
        }
      });

      const newIds = currentPhysicalTokens.map(t => t.id).sort().join(',');
      if (newIds !== oldIds || mappingChanged) {
        renderMappingUI();
      }
    });
  } catch (e) {
    document.getElementById('status').innerText = "Connection Error";
    document.getElementById('status').className = "status disconnected";
  }
}

// Drives smoothing + pushes to Owlbear on a fixed-rate setInterval rather
// than requestAnimationFrame. rAF is deliberately suspended by the browser
// the instant this tab isn't the visible/foreground one — fine for a
// typical animation, but exactly wrong here: a projector/second-screen fog
// of war setup is meant to keep updating while the DM works in a different
// browser tab or window, which is precisely when rAF stops firing at all.
// setInterval keeps running in background tabs (browsers only clamp its
// rate after several minutes of continuous backgrounding, rather than
// freezing it outright the way rAF does), so updates keep flowing to the
// visible-but-unfocused Owlbear tab.
let _intervalHandle = null;
let _lastTickTime = performance.now();
let _isPushing = false;

function tick() {
  const now = performance.now();
  const dt = Math.min((now - _lastTickTime) / 1000, 0.25); // clamp so a long gap (e.g. throttled interval) doesn't lurch
  _lastTickTime = now;

  if (!isReady || !socket || !socket.connected) return;

  const alpha = 1 - Math.exp(-SMOOTH_RATE * dt);
  for (const id in smoothState) {
    const target = latestTokens[id];
    if (!target) continue;
    const s = smoothState[id];
    s.x += (target.x - s.x) * alpha;
    s.y += (target.y - s.y) * alpha;
    if (Math.abs(target.x - s.x) < POS_EPSILON) s.x = target.x;
    if (Math.abs(target.y - s.y) < POS_EPSILON) s.y = target.y;
  }

  pushToOwlbear();
}

function startRenderLoop() {
  if (_intervalHandle) clearInterval(_intervalHandle);
  _lastTickTime = performance.now();
  _intervalHandle = setInterval(tick, 1000 / TARGET_FPS);
}

async function pushToOwlbear() {
  if (_isPushing) return; // previous push still in flight — skip this tick rather than queue up stale ones
  _isPushing = true;
  const _t0 = performance.now();

  try {
    if (Date.now() >= manualBlackoutUntil && latestBlank !== lastAppliedBlank) {
      await applyBlackout(latestBlank);
    }

    const tokensToSync = Object.keys(smoothState).map(id => ({
      id,
      alias: (latestTokens[id] || {}).alias,
      x: smoothState[id].x,
      y: smoothState[id].y,
    }));

    if (tokensToSync.length > 0) {
      await syncTokensWithOwlbear(tokensToSync, sceneItemsCache, viewportSize.width, viewportSize.height);
    }
  } catch (err) {
    console.error("Sync Error:", err);
  } finally {
    _isPushing = false;
    recordSyncLatency(performance.now() - _t0);
  }
}

function renderMappingUI() {
  const listEl = document.getElementById('mapping-list');

  if (currentPhysicalTokens.length === 0) {
    listEl.innerHTML = '<p class="empty-msg">No physical tokens detected yet.</p>';
    return;
  }

  // Remove the empty message if it exists
  const emptyMsg = listEl.querySelector('.empty-msg');
  if (emptyMsg) {
    emptyMsg.remove();
  }

  // Map existing elements by physical ID
  const existingElements = {};
  listEl.querySelectorAll('.mapping-item').forEach(el => {
    existingElements[el.dataset.id] = el;
  });

  const currentIds = new Set(currentPhysicalTokens.map(pt => pt.id));

  // Remove elements for tokens that disappeared
  for (const id in existingElements) {
    if (!currentIds.has(id)) {
      existingElements[id].remove();
    }
  }

  currentPhysicalTokens.forEach(pt => {
    const displayName = pt.alias || pt.id.split('_')[0];

    if (existingElements[pt.id]) {
      // Update label if alias changed, leave select untouched
      const labelStrong = existingElements[pt.id].querySelector('strong');
      if (labelStrong && labelStrong.innerText !== displayName) {
        labelStrong.innerText = displayName;
      }
      // Update select if auto-mapped behind the scenes
      const selectEl = existingElements[pt.id].querySelector('select');
      if (selectEl && tokenMapping[pt.id] && selectEl.value !== tokenMapping[pt.id]) {
         selectEl.value = tokenMapping[pt.id];
      }
    } else {
      // Create new mapping item
      const itemEl = document.createElement('div');
      itemEl.className = 'mapping-item';
      itemEl.dataset.id = pt.id; // Store ID for incremental updates

      const label = document.createElement('div');
      label.className = 'mapping-label';
      label.innerHTML = `<strong>${displayName}</strong> <span class="id-tag">${pt.id}</span>`;

      const select = document.createElement('select');
      select.className = 'mapping-select';

      const defaultOpt = document.createElement('option');
      defaultOpt.value = "";
      defaultOpt.text = "-- Select Virtual Token --";
      select.appendChild(defaultOpt);

      virtualTokens.forEach(vt => {
        const opt = document.createElement('option');
        opt.value = vt.id;
        opt.text = vt.text && vt.text.plainText ? vt.text.plainText : (vt.name || 'Unnamed Token');
        select.appendChild(opt);
      });

      if (tokenMapping[pt.id]) {
        select.value = tokenMapping[pt.id];
      }

      select.addEventListener('change', (e) => {
        const virtualId = e.target.value;
        const selectedToken = virtualTokens.find(vt => vt.id === virtualId);

        if (virtualId === "") {
          delete tokenMapping[pt.id];
          delete assignedNames[pt.id];
        } else {
          tokenMapping[pt.id] = virtualId;
          assignedNames[pt.id] = selectedToken.text && selectedToken.text.plainText ? selectedToken.text.plainText : (selectedToken.name || 'Unnamed Token');
        }
      });

      itemEl.appendChild(label);
      itemEl.appendChild(select);
      listEl.appendChild(itemEl);
    }
  });
}

// Minimum scene-unit movement before bothering to push a position update.
// Was previously a user-facing "Sensitivity (px)" slider, but the value was
// actually compared against scene-space distance, not screen pixels —
// mislabeled and, at the range the slider offered, either had no
// perceptible effect or (if turned up) fought against the smoothing above.
// Now that motion is continuously eased toward the target every push
// anyway, this only needs to be just large enough to stop float noise from
// generating pushes once a token has settled.
const MIN_MOVE_DIST = 0.5;

async function syncTokensWithOwlbear(physicalTokens, items, screenWidth, screenHeight) {
  const itemsToUpdate = [];

  // Run all inverseTransformPoint queries concurrently
  const transformPromises = physicalTokens.map(async (pt) => {
    const virtualId = tokenMapping[pt.id];
    if (!virtualId) return null;

    const targetItem = items.find(item => item.id === virtualId);
    if (!targetItem) return null;

    // Map normalized [0,1] token coordinates directly to the Owlbear Viewport pixel size
    const screenPoint = {
      x: pt.x * screenWidth,
      y: pt.y * screenHeight
    };

    // Convert physical screen pixels exactly to the underlying map grid coordinates!
    const scenePoint = await OBR.viewport.inverseTransformPoint(screenPoint);

    const dist = Math.sqrt(Math.pow(targetItem.position.x - scenePoint.x, 2) + Math.pow(targetItem.position.y - scenePoint.y, 2));

    // Check if name needs updating
    let needsNameUpdate = false;
    if (pt.alias) {
      if (targetItem.text && targetItem.text.plainText !== pt.alias) needsNameUpdate = true;
      else if (!targetItem.text && targetItem.name !== pt.alias) needsNameUpdate = true;
    }

    // Update if moved more than threshold or needs name sync
    if (dist > MIN_MOVE_DIST || needsNameUpdate) {
      return {
        id: targetItem.id,
        position: { x: scenePoint.x, y: scenePoint.y },
        alias: pt.alias
      };
    }
    return null;
  });

  const results = await Promise.all(transformPromises);
  results.forEach(res => {
    if (res) itemsToUpdate.push(res);
  });

  if (itemsToUpdate.length > 0) {
    await OBR.scene.items.updateItems(
      itemsToUpdate.map(i => i.id),
      (items) => {
        for (let i = 0; i < items.length; i++) {
          const update = itemsToUpdate.find(u => u.id === items[i].id);
          if (update) {
            items[i].position = update.position;
            if (update.alias) {
              items[i].name = update.alias;
              if (items[i].text) {
                items[i].text.plainText = update.alias;
              }
            }
          }
        }
      }
    );
  }
}

// Remembers the scene's real fog color from just before we first touch it,
// so turning blackout off can restore it — see the note below on why we
// have to touch scene.fog at all.
let savedFogColor = null;

async function applyBlackout(active, items) {
  try {
    // Hex, not the CSS keyword "black"/"white": nothing in the SDK actually
    // validates this string (checked the bundled source — it's a raw
    // assignment), but the Owlbear app's own color handling almost
    // certainly expects hex like a native color picker would produce, and
    // silently falls back to a default (black) on anything else.
    const color = document.getElementById('blackout-color').value || "#000000";
    if (!items) {
      items = sceneItemsCache.length ? sceneItemsCache : await OBR.scene.items.getItems();
    }
    const hasItem = items.some(i => i.id === "blackout-overlay");

    if (active && !hasItem) {
      // Confirmed by testing: a shape on the "FOG" layer is rendered using
      // the scene's shared fog color (OBR.scene.fog), not its own
      // style.fillColor — the SDK type technically has fillColor on Shape,
      // but the app ignores it for this layer. FOG is still the right layer
      // to use (it's the one confirmed to stack above tokens/map to
      // actually hide them), so we drive the real fog color instead of
      // fighting it. Saved/restored around the blackout's lifetime so this
      // doesn't permanently change fog color for anyone using OBR's actual
      // dynamic fog-of-war for unrelated reasons.
      if (savedFogColor === null) {
        savedFogColor = await OBR.scene.fog.getColor();
      }
      await OBR.scene.fog.setColor(color);

      const item = buildShape()
        .shapeType("RECTANGLE")
        .width(500000)
        .height(500000)
        .position({ x: -250000, y: -250000 })
        .fillColor(color)
        .fillOpacity(1.0)
        .strokeWidth(0)
        .layer("FOG")
        .locked(true)
        .id("blackout-overlay")
        .build();
      await OBR.scene.items.addItems([item]);
    } else if (!active && hasItem) {
      await OBR.scene.items.deleteItems(["blackout-overlay"]);
      if (savedFogColor !== null) {
        await OBR.scene.fog.setColor(savedFogColor);
        savedFogColor = null;
      }
    } else if (active && hasItem) {
      // Already active — just the color changed. Update both the real fog
      // color (what's actually visible) and the shape's own style (kept in
      // sync for correctness even though the app doesn't render it here).
      await OBR.scene.fog.setColor(color);
      await OBR.scene.items.updateItems(["blackout-overlay"], (items) => {
        for (let item of items) {
          item.style.fillColor = color;
        }
      });
    }
    lastAppliedBlank = active;
  } catch (e) {
    console.error("Error updating blackout:", e);
  }
}
