import OBR, { buildShape } from 'https://cdn.jsdelivr.net/npm/@owlbear-rodeo/sdk@latest/+esm'
import { io } from 'https://cdn.jsdelivr.net/npm/socket.io-client@latest/dist/socket.io.esm.min.js'

document.querySelector('#app').innerHTML = `
  <div class="container">
    <h2>Token Sync</h2>
    <div class="connection-box">
      <input type="text" id="ws-url" value="https://owlbear-tracker.n4z4r.com/" placeholder="Tracker URL" />
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

    <div class="blackout-settings">
      <h3>Blackout Settings</h3>
      <div class="control-row">
        <label>Color:</label>
        <select id="blackout-color">
          <option value="black">Black</option>
          <option value="white">White</option>
        </select>
      </div>
      <button id="test-blackout-btn" class="secondary">Test Blackout (3s)</button>
    </div>
  </div>
`

let socket = null;
let isReady = false;
let tokenMapping = {}; // physicalId -> virtualItemId
let currentPhysicalTokens = [];
let virtualTokens = [];
let blackoutItemId = null;
let filterItemId = null;

OBR.onReady(async () => {
  isReady = true;
  document.getElementById('status').innerText = "Ready. Connect to Tracker.";
  document.getElementById('status').className = "status ready";
  
  // Listen for changes in the scene items to refresh our virtual token list
  OBR.scene.items.onChange((items) => {
    virtualTokens = items.filter(item => item.layer === "CHARACTER" || item.layer === "MOUNT");
    renderMappingUI();
  });
  
  // Initial fetch of virtual tokens
  const items = await OBR.scene.items.getItems();
  virtualTokens = items.filter(item => item.layer === "CHARACTER" || item.layer === "MOUNT");

  // Add Test Blackout listener
  document.getElementById('test-blackout-btn').addEventListener('click', async () => {
    console.log("Testing blackout...");
    await updateBlackout(true);
    setTimeout(() => updateBlackout(false), 3000);
  });
});

document.getElementById('connect-btn').addEventListener('click', () => {
  const url = document.getElementById('ws-url').value;
  connectSocketIO(url);
});

function connectSocketIO(url) {
  if (socket) socket.disconnect();
  
  document.getElementById('status').innerText = "Connecting...";
  document.getElementById('status').className = "status ready";
  
  try {
    socket = io(url);
    
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
      
      const tokens = data.tokens || [];
      const blankScreen = data.blank_screen || false;
      
      console.log(`Received tokens_update: ${tokens.length} tokens, blackout: ${blankScreen}`);
      
      // Prioritize blackout for immediate anti-reflection
      await updateBlackout(blankScreen);
      await syncTokensWithOwlbear(tokens);
      
      // Check if physical tokens changed to avoid unnecessary re-renders
      const newIds = tokens.map(t => t.id).sort().join(',');
      const oldIds = currentPhysicalTokens.map(t => t.id).sort().join(',');
      
      currentPhysicalTokens = tokens;
      if (newIds !== oldIds) {
        renderMappingUI();
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
  
  listEl.innerHTML = '';
  
  currentPhysicalTokens.forEach(pt => {
    const itemEl = document.createElement('div');
    itemEl.className = 'mapping-item';
    
    // Use alias if available, otherwise fallback to the color name from ID
    const displayName = pt.alias || pt.id.split('_')[0];
    
    const label = document.createElement('div');
    label.className = 'mapping-label';
    label.innerHTML = `<strong>${displayName}</strong> <span class="id-tag">${pt.id}</span>`;
    
    const select = document.createElement('select');
    select.className = 'mapping-select';
    
    // Default empty option
    const defaultOpt = document.createElement('option');
    defaultOpt.value = "";
    defaultOpt.text = "-- Select Virtual Token --";
    select.appendChild(defaultOpt);
    
    // Populate virtual tokens
    virtualTokens.forEach(vt => {
      const opt = document.createElement('option');
      opt.value = vt.id;
      // Use text.plainText if it's a text item, otherwise use name
      opt.text = vt.text && vt.text.plainText ? vt.text.plainText : (vt.name || 'Unnamed Token');
      select.appendChild(opt);
    });
    
    // Set current value if mapped
    if (tokenMapping[pt.id]) {
      select.value = tokenMapping[pt.id];
    }
    
    // Handle changes
    select.addEventListener('change', (e) => {
      if (e.target.value === "") {
        delete tokenMapping[pt.id];
      } else {
        tokenMapping[pt.id] = e.target.value;
      }
    });
    
    itemEl.appendChild(label);
    itemEl.appendChild(select);
    listEl.appendChild(itemEl);
  });
}

async function syncTokensWithOwlbear(physicalTokens) {
  const items = await OBR.scene.items.getItems();
  const itemsToUpdate = [];
  const dpi = await OBR.scene.grid.getDpi();
  
  const screenWidth = await OBR.viewport.getWidth();
  const screenHeight = await OBR.viewport.getHeight();
  
  for (const pt of physicalTokens) {
    // Lookup the assigned virtual token ID
    const virtualId = tokenMapping[pt.id];
    if (!virtualId) continue; // Skip if not assigned
    
    const targetItem = items.find(item => item.id === virtualId);
    
    if (targetItem) {
      // Map normalized [0,1] token coordinates directly to the Owlbear Viewport pixel size
      const screenPoint = {
        x: pt.x * screenWidth,
        y: pt.y * screenHeight
      };
      
      // Convert physical screen pixels exactly to the underlying map grid coordinates!
      const scenePoint = await OBR.viewport.inverseTransformPoint(screenPoint);
      
      const targetX = scenePoint.x;
      const targetY = scenePoint.y;
      
      const dist = Math.sqrt(Math.pow(targetItem.position.x - targetX, 2) + Math.pow(targetItem.position.y - targetY, 2));
      
      // Smooth movement, only apply if moved more than 1/4th of a square
      if (dist > (dpi / 4)) {
        itemsToUpdate.push({
          id: targetItem.id,
          position: { x: targetX, y: targetY }
        });
      }
    }
  }
  
  if (itemsToUpdate.length > 0) {
    await OBR.scene.items.updateItems(
      itemsToUpdate.map(i => i.id),
      (items) => {
        for (let i = 0; i < items.length; i++) {
          const update = itemsToUpdate.find(u => u.id === items[i].id);
          if (update) {
            items[i].position = update.position;
          }
        }
      }
    );
  }
}

async function updateBlackout(active) {
  try {
    const color = document.getElementById('blackout-color').value || "black";
    const items = await OBR.scene.items.getItems();
    const hasItem = items.some(i => i.id === "blackout-overlay");

    if (active && !hasItem) {
      console.log(`Adding ${color} blackout overlay...`);
      const item = buildShape()
        .shapeType("RECTANGLE")
        .width(500000) 
        .height(500000)
        .position({ x: -250000, y: -250000 })
        .fillColor(color)
        .fillOpacity(1.0)
        .strokeWidth(0)
        .layer("DRAWING") 
        .locked(true)
        .id("blackout-overlay")
        .build();
      await OBR.scene.items.addItems([item]);
    } else if (!active && hasItem) {
      console.log("Removing blackout overlay...");
      await OBR.scene.items.deleteItems(["blackout-overlay"]);
    } else if (active && hasItem) {
      // If active and exists, ensure color matches (in case it was changed)
      await OBR.scene.items.updateItems(["blackout-overlay"], (items) => {
        for (let item of items) {
          item.fillColor = color;
        }
      });
    }
  } catch (e) {
    console.error("Error updating blackout:", e);
  }
}

// updateSceneFilters removed per user request
