import * as THREE from 'three';
import { RemotePlayer } from '../entities/RemotePlayer';

export class NetworkManager {
  private ws: WebSocket | null = null;
  public isConnected: boolean = false;
  public selfId: string | null = null;
  public roomCode: string = 'BROS';
  public remotePlayers: Map<string, RemotePlayer> = new Map();

  private scene: THREE.Scene;
  private onTimeSyncCallback?: (time: number, day: number) => void;
  private onCampfireSpawnCallback?: (fire: { id: string; x: number; z: number; fuelSeconds: number }) => void;
  private onInitCallback?: (data: { selfId: string; spawnPos?: { x: number; z: number }; othersOnline?: number }) => void;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    const urlParams = new URLSearchParams(window.location.search);
    const roomParam = urlParams.get('room');
    if (roomParam) {
      this.roomCode = roomParam.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8) || 'BROS';
    }
    this.updateInviteLinks();
    this.connect();
  }

  public updateInviteLinks() {
    const currentUrl = new URL(window.location.href);
    currentUrl.searchParams.set('room', this.roomCode);
    currentUrl.searchParams.set('autostart', '1');
    const inviteUrl = currentUrl.toString();

    const phoneQr = document.getElementById('phone-qr-code') as HTMLImageElement;
    const phoneUrlLink = document.getElementById('phone-game-url') as HTMLAnchorElement;
    if (phoneQr) {
      phoneQr.src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(inviteUrl)}`;
    }
    if (phoneUrlLink) {
      phoneUrlLink.href = inviteUrl;
      phoneUrlLink.innerText = inviteUrl;
    }
  }

  public switchRoom(newCode: string) {
    const sanitized = newCode.toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 8) || 'BROS';
    this.roomCode = sanitized;

    // Update browser URL query
    const url = new URL(window.location.href);
    url.searchParams.set('room', this.roomCode);
    window.history.replaceState({}, '', url.toString());
    this.updateInviteLinks();

    // Clear existing remote players
    this.remotePlayers.forEach(rp => rp.destroy(this.scene));
    this.remotePlayers.clear();

    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'JOIN_ROOM', payload: { room: this.roomCode } }));
    }
    this.updateHUD();
  }

  public onInit(fn: (data: { selfId: string; spawnPos?: { x: number; z: number }; othersOnline?: number }) => void) {
    this.onInitCallback = fn;
  }

  public connect() {
    const urlParams = new URLSearchParams(window.location.search);
    const queryWs = urlParams.get('ws');
    const envUrl = (import.meta as any).env?.VITE_WS_URL;
    const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname || 'localhost';
    
    // In local dev use local port 8088; on Vercel / public use live tunnel or envUrl
    const defaultUrl = isLocal ? `${protocol}//${host}:8088` : 'wss://wildsurvivors.loca.lt';
    const wsUrl = queryWs || envUrl || defaultUrl;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.isConnected = true;
        console.log(`[MULTIPLAYER] Connected to server at ${wsUrl} in Room [${this.roomCode}]`);
        // Immediately notify server of room join
        this.ws?.send(JSON.stringify({ type: 'JOIN_ROOM', payload: { room: this.roomCode } }));
        this.updateHUD();
      };

      this.ws.onmessage = (event) => {
        try {
          const { type, payload } = JSON.parse(event.data);
          this.handlePacket(type, payload);
        } catch (e) {
          console.error('[MULTIPLAYER] Packet parse error', e);
        }
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        console.log('[MULTIPLAYER] Disconnected from server. Running in offline/local mode.');
        this.updateHUD();
        // Retry connection in 4s
        setTimeout(() => this.connect(), 4000);
      };

      this.ws.onerror = () => {
        // Will close and trigger onclose fallback
      };
    } catch {
      console.log('[MULTIPLAYER] Running offline');
    }
  }

  public onTimeSync(fn: (time: number, day: number) => void) {
    this.onTimeSyncCallback = fn;
  }

  public onCampfireSpawn(fn: (fire: { id: string; x: number; z: number; fuelSeconds: number }) => void) {
    this.onCampfireSpawnCallback = fn;
  }

  private handlePacket(type: string, payload: any) {
    switch (type) {
      case 'INIT': {
        this.selfId = payload.selfId;
        if (payload.roomCode) {
          this.roomCode = payload.roomCode;
          this.updateInviteLinks();
        }
        if (this.onInitCallback) {
          const othersOnline = (payload.players || []).filter((p: any) => p.id !== payload.selfId && !p.isDead).length;
          this.onInitCallback({ selfId: payload.selfId, spawnPos: payload.spawnPos, othersOnline });
        }
        if (this.onTimeSyncCallback) {
          this.onTimeSyncCallback(payload.timeOfDay, payload.dayCount);
        }
        // Spawn existing players in this room
        payload.players.forEach((p: any) => {
          if (p.id !== this.selfId) {
            this.spawnRemotePlayer(p);
          }
        });
        this.updateHUD();
        break;
      }

      case 'PLAYER_JOIN': {
        if (payload.id !== this.selfId && !this.remotePlayers.has(payload.id)) {
          this.spawnRemotePlayer(payload);
          this.updateHUD();
        }
        break;
      }

      case 'PLAYER_UPDATE': {
        const rp = this.remotePlayers.get(payload.id);
        if (rp) {
          rp.updateData(payload);
        }
        break;
      }

      case 'PLAYER_LEAVE': {
        const rp = this.remotePlayers.get(payload.id);
        if (rp) {
          rp.destroy(this.scene);
          this.remotePlayers.delete(payload.id);
          this.updateHUD();
        }
        break;
      }

      case 'TIME_SYNC': {
        if (this.onTimeSyncCallback) {
          this.onTimeSyncCallback(payload.timeOfDay, payload.dayCount);
        }
        break;
      }

      case 'CAMPFIRE_SPAWN': {
        if (this.onCampfireSpawnCallback) {
          this.onCampfireSpawnCallback(payload);
        }
        break;
      }

      case 'CHAT_BROADCAST': {
        const rp = this.remotePlayers.get(payload.senderId);
        if (rp) {
          rp.showSpeechBubble(payload.text);
        }
        break;
      }
    }
  }

  public sendChatMessage(text: string) {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'CHAT_MESSAGE', payload: { text } }));
    }
  }

  private spawnRemotePlayer(p: any) {
    const rp = new RemotePlayer(this.scene, p.id, p.name || 'Survivor', { x: p.x || 0, z: p.z || 0 });
    this.remotePlayers.set(p.id, rp);
  }

  public sendPlayerUpdate(payload: {
    x: number;
    z: number;
    facingLeft: boolean;
    isMoving: boolean;
    isChopping: boolean;
    isDancing?: boolean;
    tool: string | null;
    health: number;
    isDead: boolean;
  }) {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'PLAYER_UPDATE', payload }));
    }
  }

  public sendCampfirePlace(x: number, z: number) {
    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'CAMPFIRE_PLACE', payload: { x, z } }));
    }
  }

  public update(delta: number, camera: THREE.Camera) {
    this.remotePlayers.forEach(rp => rp.update(delta, camera));
  }

  public copyInviteLink() {
    const url = new URL(window.location.href);
    url.searchParams.set('room', this.roomCode);
    navigator.clipboard?.writeText(url.toString()).then(() => {
      alert(`Invite link for room [${this.roomCode}] copied to clipboard!`);
    });
  }

  private updateHUD() {
    let badge = document.getElementById('mp-badge');
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'mp-badge';
      badge.style.position = 'absolute';
      badge.style.top = '20px';
      badge.style.right = '145px';
      badge.style.background = 'rgba(18, 14, 10, 0.9)';
      badge.style.border = '1px solid #6b5338';
      badge.style.borderRadius = '6px';
      badge.style.padding = '6px 12px';
      badge.style.fontSize = '12px';
      badge.style.color = '#ebd197';
      badge.style.display = 'flex';
      badge.style.alignItems = 'center';
      badge.style.gap = '8px';
      badge.style.boxShadow = '0 2px 10px rgba(0,0,0,0.6)';
      badge.style.zIndex = '30';
      badge.style.cursor = 'pointer';
      badge.title = 'Click to copy room invite link';
      badge.onclick = () => this.copyInviteLink();
      document.getElementById('hud-container')?.appendChild(badge);
    }

    const totalSurvivors = this.remotePlayers.size + 1;
    if (this.isConnected) {
      badge.innerHTML = `<span style="color: #6edb65;">●</span> <span>${totalSurvivors} Online</span> <span style="background: #38281a; padding: 2px 6px; border-radius: 4px; font-weight: bold; border: 1px solid #735332;">${this.roomCode}</span>`;
    } else {
      badge.innerHTML = `<span style="color: #e09836;">○</span> <span>Solo / Offline</span> <span style="background: #38281a; padding: 2px 6px; border-radius: 4px; border: 1px solid #735332;">${this.roomCode}</span>`;
    }
  }
}

