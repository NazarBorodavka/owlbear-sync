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
        <label>Sync Rate (FPS): <span id="fps-val">15</span></label>
        <input type="range" id="sync-fps" min="1" max="30" value="15">
      </div>
      <div class="control-row">
        <label>Sensitivity (px): <span id="sens-val">3</span></label>
        <input type="range" id="sync-sens" min="0" max="20" value="3" step="0.5">
      </div>
      <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 5px;">Higher FPS or lower sensitivity means smoother movement but more network traffic.</p>
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
let isUpdating = false;
let lastUpdateTime = 0;
// Was 100ms (10fps) — slower than the camera itself (15fps ~= 66ms/frame),
// meaning the extension was artificially adding latency below what the
// backend can already deliver. Matches the camera's real rate now instead
// of a leftover conservative default from before the CPU limit was fixed.
let THROTTLE_MS = 66;
let SYNC_THRESHOLD = 3;
// The periodic tokens_update handler below calls updateBlackout() on every
// tick using the tracker's real blank_screen state. Without a guard, that
// would immediately stomp on a manual "Test Blackout" click — the overlay
// would get added, then removed again on the very next tick (as little as
// ~100ms later), which is why the test only ever flashed for a few frames
// instead of holding for its full duration.
let manualBlackoutUntil = 0; // Date.now()-based deadline; 0 = no active override
let lastServerBlankScreen = false;

OBR.onReady(async () => {
  isReady = true;
  document.getElementById('status').innerText = "Ready. Connect to Tracker.";
  document.getElementById('status').className = "status ready";
  
  // Name-Based Re-sync: Update mapping whenever the scene changes
  OBR.scene.items.onChange((items) => {
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
  
  // Initial fetch of virtual tokens
  const items = await OBR.scene.items.getItems();
  virtualTokens = items.filter(item => item.layer === "CHARACTER" || item.layer === "MOUNT");

  // Add Test Blackout listener
  document.getElementById('test-blackout-btn').addEventListener('click', async () => {
    const TEST_DURATION_MS = 1000;
    manualBlackoutUntil = Date.now() + TEST_DURATION_MS;
    await updateBlackout(true);
    setTimeout(async () => {
      manualBlackoutUntil = 0;
      // Hand control back to whatever the tracker actually wants right now,
      // rather than unconditionally turning the overlay off — if a real
      // blackout started during the test window, this keeps it on.
      await updateBlackout(lastServerBlankScreen);
    }, TEST_DURATION_MS);
  });

  // Re-apply color immediately when changed, so switching the dropdown has
  // a visible effect right away instead of waiting for the next
  // activate/deactivate transition (which might not happen for a while).
  document.getElementById('blackout-color').addEventListener('change', async () => {
    if (Date.now() < manualBlackoutUntil || lastServerBlankScreen) {
      await updateBlackout(true);
    }
  });

  // UI Listeners for Sync Performance
  document.getElementById('sync-fps').addEventListener('input', (e) => {
    const fps = parseInt(e.target.value, 10);
    document.getElementById('fps-val').innerText = fps;
    THROTTLE_MS = Math.floor(1000 / fps);
  });

  document.getElementById('sync-sens').addEventListener('input', (e) => {
    SYNC_THRESHOLD = parseFloat(e.target.value);
    document.getElementById('sens-val').innerText = SYNC_THRESHOLD;
  });
});

document.getElementById('connect-btn').addEventListener('click', () => {
  const url = document.getElementById('ws-url').value;
  connectSocketIO(url);
});


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

// Measures the one leg of the latency chain the server-side [Perf] logs
// can't see: how long it takes from a tokens_update arriving here to
// Owlbear actually having applied the move. Every OBR SDK call is a
// postMessage round-trip to the host, so this can be a meaningful slice of
// end-to-end delay. Averaged over a window rather than logged per frame so
// it doesn't flood the console at sync rate.
let _syncSamples = [];
let _syncLastReport = performance.now();
function recordSyncLatency(ms) {
  _syncSamples.push(ms);
  const elapsed = performance.now() - _syncLastReport;
  if (elapsed >= 5000 && _syncSamples.length) {
    const avg = _syncSamples.reduce((a, b) => a + b, 0) / _syncSamples.length;
    const worst = Math.max(..._syncSamples);
    console.log(`[SyncPerf] OBR update: avg ${avg.toFixed(0)}ms, worst ${worst.toFixed(0)}ms over ${_syncSamples.length} updates`);
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
    
    socket.on('tokens_update', async (data) => {
      if (!isReady) return;
      
      const now = Date.now();
      if (now - lastUpdateTime < THROTTLE_MS) return; // Throttle to the configured sync rate
      
      if (isUpdating) return; // Skip if we're still processing the previous frame
      isUpdating = true;
      const _syncStart = performance.now();

      try {
        const tokens = data.tokens || [];
        const blankScreen = data.blank_screen || false;
        lastServerBlankScreen = blankScreen;

        // Fetch items and viewport details concurrently to reduce latency
        const [items, screenWidth, screenHeight] = await Promise.all([
          OBR.scene.items.getItems(),
          OBR.viewport.getWidth(),
          OBR.viewport.getHeight()
        ]);

        // 1. Prioritize blackout (Critical for projector setup) — unless a
        // manual test is currently overriding it (see test-blackout-btn).
        if (Date.now() >= manualBlackoutUntil) {
          await updateBlackout(blankScreen, items);
        }

        // 2. Sync positions
        await syncTokensWithOwlbear(tokens, items, screenWidth, screenHeight);
        
        // 3. UI Update & Auto-Mapping
        let mappingChanged = false;
        
        const oldIds = currentPhysicalTokens.map(t => t.id).sort().join(',');
        currentPhysicalTokens = tokens;
        
        // Auto-assign virtual tokens based on physical token alias
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
        
        lastUpdateTime = Date.now();
        recordSyncLatency(performance.now() - _syncStart);
      } catch (err) {
        console.error("Sync Error:", err);
      } finally {
        isUpdating = false;
      }
    });
  } catch (e) {
    document.getElementById('status').innerText = "Connection Error";
    document.getElementById('status').className = "status disconnected";
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
    if (dist > SYNC_THRESHOLD || needsNameUpdate) {
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

async function updateBlackout(active, items) {
  try {
    // Hex, not the CSS keyword "black"/"white": nothing in the SDK actually
    // validates this string (checked the bundled source — it's a raw
    // assignment), but the Owlbear app's own color handling almost
    // certainly expects hex like a native color picker would produce, and
    // silently falls back to a default (black) on anything else.
    const color = document.getElementById('blackout-color').value || "#000000";
    if (!items) {
      items = await OBR.scene.items.getItems();
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
  } catch (e) {
    console.error("Error updating blackout:", e);
  }
}
