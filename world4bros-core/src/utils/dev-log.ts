/**
 * DevLog & Bug Tracker Subsystem for World4Bros
 * Captures runtime events, errors, performance stats, and combat telemetries.
 * Provides live log streams, filtering, and one-click Markdown/JSON bug report generation.
 */

export type LogLevel = 'info' | 'warn' | 'error' | 'bug' | 'combat' | 'sim' | 'perf';
export type LogCategory = 'combat' | 'zombie' | 'bot' | 'world' | 'render' | 'audio' | 'ui' | 'network' | 'system';

export interface LogEntry {
  id: number;
  timestamp: string;
  simTime: number;
  level: LogLevel;
  category: LogCategory;
  message: string;
  data?: unknown;
  stack?: string;
}

export interface BugReportSnapshot {
  exportedAt: string;
  url: string;
  userAgent: string;
  simTime: number;
  fps: number;
  gameState: {
    seed: number;
    mode: string;
    playerPos: { x: number; y: number };
    playerHealth: number;
    activeArena: string | null;
    entityCounts: Record<string, number>;
  };
  totalLogs: number;
  errorsCount: number;
  recentLogs: LogEntry[];
}

export class DevLog {
  private static instance: DevLog;
  private logs: LogEntry[] = [];
  private nextId = 1;
  private maxLogs = 500;
  private listeners: ((entry: LogEntry) => void)[] = [];
  private isPaused = false;

  private constructor() {
    this.hookGlobalErrors();
  }

  static get(): DevLog {
    if (!DevLog.instance) {
      DevLog.instance = new DevLog();
      // Expose globally for browser developer console access
      if (typeof window !== 'undefined') {
        (window as unknown as { devLog: DevLog }).devLog = DevLog.instance;
      }
    }
    return DevLog.instance;
  }

  private hookGlobalErrors(): void {
    if (typeof window === 'undefined') return;

    window.addEventListener('error', (event) => {
      this.bug(
        'system',
        `Uncaught error: ${event.message} (${event.filename}:${event.lineno}:${event.colno})`,
        undefined,
        event.error?.stack
      );
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.bug('system', `Unhandled Promise Rejection: ${String(event.reason)}`, undefined, event.reason?.stack);
    });
  }

  onLog(listener: (entry: LogEntry) => void): () => void {
    this.listeners.push(listener);
    return () => {
      const idx = this.listeners.indexOf(listener);
      if (idx >= 0) this.listeners.splice(idx, 1);
    };
  }

  log(category: LogCategory, message: string, data?: unknown): void {
    this.addEntry('info', category, message, data);
  }

  warn(category: LogCategory, message: string, data?: unknown): void {
    this.addEntry('warn', category, message, data);
  }

  error(category: LogCategory, message: string, data?: unknown, stack?: string): void {
    this.addEntry('error', category, message, data, stack);
  }

  bug(category: LogCategory, message: string, data?: unknown, stack?: string): void {
    this.addEntry('bug', category, message, data, stack ?? new Error().stack);
  }

  combat(message: string, data?: unknown): void {
    this.addEntry('combat', 'combat', message, data);
  }

  sim(category: LogCategory, message: string, data?: unknown): void {
    this.addEntry('sim', category, message, data);
  }

  perf(message: string, data?: unknown): void {
    this.addEntry('perf', 'system', message, data);
  }

  private addEntry(level: LogLevel, category: LogCategory, message: string, data?: unknown, stack?: string): void {
    if (this.isPaused) return;

    const now = new Date();
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;

    const entry: LogEntry = {
      id: this.nextId++,
      timestamp: timeStr,
      simTime: 0,
      level,
      category,
      message,
      data,
      stack,
    };

    this.logs.push(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    const colors: Record<LogLevel, string> = {
      info: '#38bdf8',
      warn: '#facc15',
      error: '#f87171',
      bug: '#ef4444',
      combat: '#f97316',
      sim: '#4ade80',
      perf: '#c084fc',
    };

    if (level === 'bug' || level === 'error') {
      console.error(`%c[${level.toUpperCase()}][${category}] ${message}`, `color:${colors[level]}; font-weight:bold;`, data ?? '');
    } else if (level === 'warn') {
      console.warn(`%c[WARN][${category}] ${message}`, `color:${colors[level]};`, data ?? '');
    }

    for (const listener of this.listeners) {
      try {
        listener(entry);
      } catch (err) {
        console.error('Error in DevLog listener:', err);
      }
    }
  }

  getLogs(filter?: { level?: LogLevel; category?: LogCategory; search?: string }): LogEntry[] {
    return this.logs.filter((entry) => {
      if (filter?.level && entry.level !== filter.level) return false;
      if (filter?.category && entry.category !== filter.category) return false;
      if (filter?.search) {
        const q = filter.search.toLowerCase();
        const textMatch = entry.message.toLowerCase().includes(q) || entry.category.toLowerCase().includes(q);
        if (!textMatch) return false;
      }
      return true;
    });
  }

  getBugCount(): number {
    return this.logs.filter((l) => l.level === 'bug' || l.level === 'error').length;
  }

  clear(): void {
    this.logs = [];
  }

  togglePause(): boolean {
    this.isPaused = !this.isPaused;
    return this.isPaused;
  }

  getPaused(): boolean {
    return this.isPaused;
  }

  /**
   * Generates a complete diagnostic bug snapshot including game context and recent logs.
   */
  generateBugReport(gameContext?: {
    seed: number;
    mode: string;
    playerPos: { x: number; y: number };
    playerHealth: number;
    activeArena: string | null;
    entityCounts: Record<string, number>;
    fps: number;
    simTime: number;
  }): BugReportSnapshot {
    return {
      exportedAt: new Date().toISOString(),
      url: typeof window !== 'undefined' ? window.location.href : '',
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
      simTime: gameContext?.simTime ?? 0,
      fps: gameContext?.fps ?? 0,
      gameState: {
        seed: gameContext?.seed ?? 0,
        mode: gameContext?.mode ?? 'unknown',
        playerPos: gameContext?.playerPos ?? { x: 0, y: 0 },
        playerHealth: gameContext?.playerHealth ?? 100,
        activeArena: gameContext?.activeArena ?? null,
        entityCounts: gameContext?.entityCounts ?? {},
      },
      totalLogs: this.logs.length,
      errorsCount: this.getBugCount(),
      recentLogs: this.logs.slice(-60),
    };
  }

  /**
   * Formats the bug snapshot as clean GitHub-flavored Markdown ready to paste into GitHub issues or chat.
   */
  formatMarkdownReport(snapshot: BugReportSnapshot): string {
    const lines: string[] = [
      `# 🐛 World4Bros Diagnostic Bug Report`,
      `**Exported:** ${snapshot.exportedAt}  `,
      `**URL:** \`${snapshot.url}\`  `,
      `**User-Agent:** \`${snapshot.userAgent}\`  `,
      `**FPS:** ${snapshot.fps.toFixed(1)} | **Sim Time:** ${snapshot.simTime.toFixed(2)}s  `,
      ``,
      `## 🕹️ Game State`,
      `- **Seed:** \`${snapshot.gameState.seed}\``,
      `- **Mode:** \`${snapshot.gameState.mode}\``,
      `- **Player Position:** \`(${snapshot.gameState.playerPos.x.toFixed(2)}, ${snapshot.gameState.playerPos.y.toFixed(2)})\``,
      `- **Player Health:** \`${snapshot.gameState.playerHealth.toFixed(0)} / 100\``,
      `- **Active Arena:** \`${snapshot.gameState.activeArena ?? 'None'}\``,
      `- **Active Entities:**`,
    ];

    for (const [kind, count] of Object.entries(snapshot.gameState.entityCounts)) {
      lines.push(`  - **${kind}:** ${count}`);
    }

    lines.push(``, `## 📜 Log Stream (Last ${snapshot.recentLogs.length} Events)`);
    lines.push(`| Time | Level | Category | Message |`);
    lines.push(`| :--- | :--- | :--- | :--- |`);

    for (const l of snapshot.recentLogs) {
      const badge = l.level === 'bug' ? '🔴 BUG'
        : l.level === 'error' ? '❌ ERROR'
        : l.level === 'warn' ? '⚠️ WARN'
        : l.level === 'combat' ? '⚔️ COMBAT'
        : l.level === 'sim' ? '🌱 SIM'
        : l.level === 'perf' ? '⚡ PERF'
        : 'ℹ️ INFO';
      lines.push(`| \`${l.timestamp}\` | ${badge} | \`${l.category}\` | ${l.message.replace(/\|/g, '\\|')} |`);
    }

    const bugsWithStack = snapshot.recentLogs.filter((l) => l.stack);
    if (bugsWithStack.length > 0) {
      lines.push(``, `## 💥 Stack Traces`);
      for (const b of bugsWithStack) {
        lines.push(`### [${b.level.toUpperCase()}] ${b.message}`);
        lines.push('```');
        lines.push(b.stack ?? 'No stack available');
        lines.push('```');
      }
    }

    return lines.join('\n');
  }
}

export const devLog = DevLog.get();
