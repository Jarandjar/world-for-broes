import { WebSocketServer, WebSocket } from 'ws';

const PORT = Number(process.env.PORT) || 8088;
const wss = new WebSocketServer({ port: PORT });

console.log(`====================================================`);
console.log(`🌲 WORLD 4 BRO'S — MULTIPLAYER SERVER (ROOMS SUPPORT)`);
console.log(`   WebSocket Server listening on ws://localhost:${PORT}`);
console.log(`====================================================`);

// Multi-Room State Management
const rooms = new Map();

function getOrCreateRoom(roomCode = 'BROS') {
  const code = (String(roomCode || 'BROS')).toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8) || 'BROS';
  if (!rooms.has(code)) {
    rooms.set(code, {
      code,
      timeOfDay: 80,
      dayCount: 1,
      cycleDuration: 1200,
      players: new Map(), // playerId -> { data: player, ws }
      campfires: new Map(),
      trees: new Map(),
      itemDrops: new Map(),
    });
    console.log(`[ROOM CREATED] Room ${code} initialized.`);
  }
  return rooms.get(code);
}

function broadcastToRoom(room, type, payload, exceptWs = null) {
  const message = JSON.stringify({ type, payload });
  for (const client of room.players.values()) {
    if (client.ws !== exceptWs && client.ws.readyState === WebSocket.OPEN) {
      client.ws.send(message);
    }
  }
}

// Server Tick Loop for each active room (10 Hz)
setInterval(() => {
  const delta = 0.1;
  for (const [code, room] of rooms.entries()) {
    if (room.players.size === 0) continue; // Sleep idle rooms

    room.timeOfDay += delta;
    if (room.timeOfDay >= room.cycleDuration) {
      room.timeOfDay = 0;
      room.dayCount += 1;
      broadcastToRoom(room, 'TIME_SYNC', { timeOfDay: room.timeOfDay, dayCount: room.dayCount });
    }

    // Burn down campfires
    for (const [id, fire] of room.campfires.entries()) {
      if (fire.fuelSeconds > 0) {
        fire.fuelSeconds -= delta;
        if (fire.fuelSeconds <= 0) {
          broadcastToRoom(room, 'CAMPFIRE_EXTINGUISH', { id });
        }
      }
    }
  }
}, 100);

let nextId = 1;

wss.on('connection', (ws) => {
  const playerId = `survivor_${nextId++}`;
  const playerName = `Survivor ${nextId - 1}`;
  let currentRoom = getOrCreateRoom('BROS');

  // Helper to join a specific room
  const joinRoom = (roomCode) => {
    // Leave previous room if joined
    if (currentRoom && currentRoom.players.has(playerId)) {
      currentRoom.players.delete(playerId);
      broadcastToRoom(currentRoom, 'PLAYER_LEAVE', { id: playerId, name: playerName });
    }

    currentRoom = getOrCreateRoom(roomCode);

    // Smart Co-Op Spawning: spawn near fellow survivors in room
    let spawnX = 0;
    let spawnZ = 0;
    const living = Array.from(currentRoom.players.values())
      .map(p => p.data)
      .filter(p => !p.isDead && p.id !== playerId);

    if (living.length > 0) {
      const existing = living[0];
      const angle = Math.random() * Math.PI * 2;
      const distance = 2.0 + Math.random() * 1.2;
      spawnX = Math.round((existing.x + Math.cos(angle) * distance) * 10) / 10;
      spawnZ = Math.round((existing.z + Math.sin(angle) * distance) * 10) / 10;
    }

    const newPlayer = {
      id: playerId,
      name: playerName,
      x: spawnX,
      z: spawnZ,
      facingLeft: false,
      isMoving: false,
      isChopping: false,
      isDancing: false,
      tool: null,
      health: 100,
      isDead: false,
    };
    currentRoom.players.set(playerId, { data: newPlayer, ws });

    console.log(`[JOIN] ${playerName} (${playerId}) joined Room [${currentRoom.code}]. Players in room: ${currentRoom.players.size}`);

    // Send INIT to player
    const initPayload = {
      selfId: playerId,
      roomCode: currentRoom.code,
      spawnPos: { x: spawnX, z: spawnZ },
      timeOfDay: currentRoom.timeOfDay,
      dayCount: currentRoom.dayCount,
      players: Array.from(currentRoom.players.values()).map(p => p.data),
      trees: Array.from(currentRoom.trees.entries()),
      campfires: Array.from(currentRoom.campfires.entries()),
      itemDrops: Array.from(currentRoom.itemDrops.entries()),
    };
    ws.send(JSON.stringify({ type: 'INIT', payload: initPayload }));

    // Announce to others in room
    broadcastToRoom(currentRoom, 'PLAYER_JOIN', newPlayer, ws);
  };

  // Join default room on connect
  joinRoom('BROS');

  // Handle messages
  ws.on('message', (raw) => {
    try {
      const data = JSON.parse(raw.toString());
      switch (data.type) {
        case 'JOIN_ROOM': {
          const targetRoom = data.payload?.room || 'BROS';
          joinRoom(targetRoom);
          break;
        }

        case 'PLAYER_UPDATE': {
          const clientEntry = currentRoom.players.get(playerId);
          if (clientEntry) {
            Object.assign(clientEntry.data, data.payload);
            clientEntry.data.id = playerId;
            broadcastToRoom(currentRoom, 'PLAYER_UPDATE', clientEntry.data, ws);
          }
          break;
        }

        case 'TREE_CHOP': {
          const { treeIndex, damage, x, z } = data.payload;
          let tree = currentRoom.trees.get(treeIndex) || { hits: 0, isStump: false };
          if (!tree.isStump) {
            tree.hits += damage;
            if (tree.hits >= 4) {
              tree.isStump = true;
              const drop1 = { id: `drop_${Date.now()}_1`, itemId: 'log', x: x + 0.8, z: z };
              const drop2 = { id: `drop_${Date.now()}_2`, itemId: 'log', x: x - 0.8, z: z + 0.5 };
              const drop3 = { id: `drop_${Date.now()}_3`, itemId: 'twigs', x: x, z: z - 0.8 };
              currentRoom.itemDrops.set(drop1.id, drop1);
              currentRoom.itemDrops.set(drop2.id, drop2);
              currentRoom.itemDrops.set(drop3.id, drop3);
              broadcastToRoom(currentRoom, 'ITEM_SPAWN', drop1);
              broadcastToRoom(currentRoom, 'ITEM_SPAWN', drop2);
              broadcastToRoom(currentRoom, 'ITEM_SPAWN', drop3);
            }
            currentRoom.trees.set(treeIndex, tree);
            broadcastToRoom(currentRoom, 'TREE_STATE', { treeIndex, ...tree });
          }
          break;
        }

        case 'ITEM_PICKUP': {
          const { dropId } = data.payload;
          if (currentRoom.itemDrops.has(dropId)) {
            currentRoom.itemDrops.delete(dropId);
            broadcastToRoom(currentRoom, 'ITEM_REMOVE', { dropId, pickedBy: playerId });
          }
          break;
        }

        case 'CAMPFIRE_PLACE': {
          const fireId = `fire_${Date.now()}`;
          const fire = {
            id: fireId,
            x: data.payload.x,
            z: data.payload.z,
            fuelSeconds: 240,
            placedBy: playerName,
          };
          currentRoom.campfires.set(fireId, fire);
          broadcastToRoom(currentRoom, 'CAMPFIRE_SPAWN', fire);
          break;
        }

        case 'CAMPFIRE_REFUEL': {
          const { fireId, seconds } = data.payload;
          const fire = currentRoom.campfires.get(fireId);
          if (fire) {
            fire.fuelSeconds = Math.min(360, fire.fuelSeconds + seconds);
            broadcastToRoom(currentRoom, 'CAMPFIRE_SYNC', fire);
          }
          break;
        }

        case 'CHAT_MESSAGE': {
          broadcastToRoom(currentRoom, 'CHAT_BROADCAST', {
            senderId: playerId,
            senderName: playerName,
            text: String(data.payload.text).slice(0, 80),
          });
          break;
        }
      }
    } catch (err) {
      console.error('Error handling WS packet:', err);
    }
  });

  ws.on('close', () => {
    if (currentRoom) {
      currentRoom.players.delete(playerId);
      console.log(`[LEAVE] ${playerName} left Room [${currentRoom.code}]. Remaining: ${currentRoom.players.size}`);
      broadcastToRoom(currentRoom, 'PLAYER_LEAVE', { id: playerId, name: playerName });
    }
  });
});

