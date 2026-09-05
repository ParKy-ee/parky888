import { promises as fs } from "fs";
import path from "path";
import { exec } from "child_process";
import { promisify } from "util";
import { NextResponse } from "next/server";
import { readRunHistory } from "../../_lib/run_history";

export const dynamic = "force-dynamic";

const execAsync = promisify(exec);
const prototypeDir = path.join(process.cwd(), "..", "prototype");
const dataDir = path.join(prototypeDir, "data");
const paperDir = path.join(dataDir, "paper_trades");
const auditDir = path.join(dataDir, "audit");
const decisionPath = path.join(paperDir, "paper_trade_decisions.csv");
const systemErrorsPath = path.join(auditDir, "system_errors.jsonl");
const historyPath = path.join(auditDir, "prototype_run_history.jsonl");
const liveServiceLogPath = path.join(prototypeDir, "live_service.log");
const STALE_LOCK_SECONDS = 300;
const STALE_TMP_SECONDS = 3600;
const LOCK_PROBE_TIMEOUT_MS = 2000;
const LOCK_PROBE_POLL_MS = 50;

async function checkLiveService() {
  try {
    // Windows: Use powershell to find the python process by command line
    const cmd = `powershell -Command "Get-CimInstance Win32_Process | Where-Object Name -eq 'python.exe' | Where-Object CommandLine -like '*run_live_service.py*' | Select-Object -ExpandProperty ProcessId"`;
    const { stdout } = await execAsync(cmd, { windowsHide: true });
    const pids = stdout.split(/\r?\n/).map(line => line.trim()).filter(line => /^\d+$/.test(line));
    return pids.length > 0;
  } catch {
    return false;
  }
}

function parseCsvLine(line: string) {
  const cells: string[] = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === '"' && quoted && next === '"') {
      current += '"';
      index += 1;
      continue;
    }

    if (char === '"') {
      quoted = !quoted;
      continue;
    }

    if (char === "," && !quoted) {
      cells.push(current);
      current = "";
      continue;
    }

    current += char;
  }

  cells.push(current);
  return cells;
}

function parseCsv(csv: string) {
  const lines = csv.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const headers = parseCsvLine(lines[0]);

  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    return headers.reduce<Record<string, string>>((row, header, index) => {
      row[header] = cells[index] ?? "";
      return row;
    }, {});
  });
}

async function readCsv(relativePath: string) {
  try {
    const content = await fs.readFile(path.join(dataDir, relativePath), "utf8");
    return parseCsv(content);
  } catch {
    return [];
  }
}

async function readJsonlTail(relativePath: string, limit = 20) {
  try {
    const content = await fs.readFile(path.join(dataDir, relativePath), "utf8");
    return content
      .trim()
      .split(/\r?\n/)
      .filter(Boolean)
      .slice(-limit)
      .map((line) => {
        try {
          return JSON.parse(line) as Record<string, unknown>;
        } catch {
          return { raw: line };
        }
      })
      .reverse();
  } catch {
    return [];
  }
}

type LiveCycleInfo = {
  status: "OK" | "ERROR" | "UNKNOWN";
  at: string | null;
  message: string;
};

function toIsoLike(localStamp: string) {
  return localStamp.replace(" ", "T");
}

async function readLastLiveCycleFromLog(): Promise<LiveCycleInfo | null> {
  try {
    const content = await fs.readFile(liveServiceLogPath, "utf8");
    const lines = content.split(/\r?\n/).filter(Boolean);
    const cycleLine = [...lines].reverse().find((line) => {
      return line.includes("Cycle completed.") || line.includes("Error in cycle:");
    });
    if (!cycleLine) return null;

    const match = cycleLine.match(/^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+\[(\w+)\]\s+(.*)$/);
    if (!match) {
      return { status: "UNKNOWN", at: null, message: cycleLine };
    }

    const [, stamp, level, message] = match;
    return {
      status: level === "ERROR" ? "ERROR" : "OK",
      at: toIsoLike(stamp),
      message
    };
  } catch {
    return null;
  }
}

function countBy<T extends Record<string, string>>(rows: T[], key: keyof T) {
  return rows.reduce<Record<string, number>>((acc, row) => {
    const value = row[key] || "UNKNOWN";
    acc[value] = (acc[value] ?? 0) + 1;
    return acc;
  }, {});
}

function lockPathFor(targetPath: string) {
  const ext = path.extname(targetPath);
  return targetPath.slice(0, -ext.length) + ext + ".lock";
}

async function probeTempWrite(dir: string) {
  const probeFile = path.join(dir, `.writable-probe-${process.pid}-${Date.now()}.tmp`);
  try {
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(probeFile, "probe", "utf8");
    await fs.unlink(probeFile);
    return true;
  } catch {
    try {
      await fs.unlink(probeFile);
    } catch {
      // ignore cleanup errors
    }
    return false;
  }
}

async function probeLock(targetPath: string) {
  const lockFile = lockPathFor(targetPath);
  const started = Date.now();
  let handle: fs.FileHandle | null = null;

  while (!handle) {
    try {
      handle = await fs.open(lockFile, "wx");
      break;
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code !== "EEXIST") {
        return false;
      }

      try {
        const stat = await fs.stat(lockFile);
        const lockAgeMs = Date.now() - stat.mtimeMs;
        if (lockAgeMs > STALE_LOCK_SECONDS * 1000) {
          await fs.unlink(lockFile);
          continue;
        }
      } catch {
        // race: lock removed between stat and unlink
        continue;
      }

      if (Date.now() - started > LOCK_PROBE_TIMEOUT_MS) {
        return false;
      }
      await new Promise((resolve) => setTimeout(resolve, LOCK_PROBE_POLL_MS));
    }
  }

  try {
    await handle.close();
    await fs.unlink(lockFile);
    return true;
  } catch {
    try {
      await fs.unlink(lockFile);
    } catch {
      // ignore
    }
    return false;
  }
}

/** Mirrors write_csv_atomic / append_jsonl: real temp write + lock acquire/release. */
async function probePathWritable(targetPath: string) {
  const dir = path.dirname(targetPath);
  const [tempOk, lockOk] = await Promise.all([probeTempWrite(dir), probeLock(targetPath)]);
  return tempOk && lockOk;
}

async function collectStaleLocks() {
  const locks: Array<{ path: string; ageSeconds: number }> = [];
  try {
    const entries = await fs.readdir(paperDir);
    const now = Date.now();
    for (const name of entries) {
      if (!name.endsWith(".lock")) continue;
      const fullPath = path.join(paperDir, name);
      const stat = await fs.stat(fullPath);
      const ageSeconds = Math.round((now - stat.mtimeMs) / 1000);
      if (ageSeconds > STALE_LOCK_SECONDS) {
        locks.push({ path: name, ageSeconds });
      }
    }
  } catch {
    return locks;
  }
  return locks;
}

async function countStaleTempFiles() {
  let count = 0;
  const now = Date.now();

  async function walk(dir: string) {
    let entries: string[];
    try {
      entries = await fs.readdir(dir);
    } catch {
      return;
    }
    for (const name of entries) {
      const fullPath = path.join(dir, name);
      const stat = await fs.stat(fullPath).catch(() => null);
      if (!stat) continue;
      if (stat.isDirectory()) {
        await walk(fullPath);
        continue;
      }
      if (!name.endsWith(".tmp")) continue;
      if ((now - stat.mtimeMs) / 1000 > STALE_TMP_SECONDS) {
        count += 1;
      }
    }
  }

  await walk(dataDir);
  return count;
}

const EXPECTED_OPEN_FIELDS = [
  "trade_id", "signal_id", "mode", "signal_cluster_id", "cluster_status", "market",
  "symbol", "status", "direction", "decision", "signal_type", "source_run_at",
  "model_score", "confidence_pct", "timeframe", "model_profile", "watch_threshold",
  "trade_ready_threshold", "entry_time", "entry_price", "stop_loss", "initial_stop_loss",
  "initial_risk_distance", "take_profit_1", "quantity", "lot_size", "notional_usd",
  "risk_pct", "risk_amount_usd", "risk_distance", "entry_source", "last_checked_at",
  "last_market_source", "reason", "trade_thesis", "correlation_group",
  "active_same_thesis_count", "recent_sl_same_thesis_4h", "entry_delay_minutes",
  "entry_price_drift_pct", "trading_session", "market_regime", "volatility_bucket",
  "replay_verified", "replay_timeframe", "ambiguous_intrabar", "evaluation_included",
  "highest_favorable_price", "lowest_adverse_price", "max_favorable_r", "max_adverse_r",
  "breakeven_activated", "partial_close_done", "remaining_quantity_pct",
  "realized_partial_pct", "partial_close_count", "trailing_stop_active", "schema_version"
];

const EXPECTED_CLOSED_FIELDS = [
  ...EXPECTED_OPEN_FIELDS,
  "exit_time",
  "exit_price",
  "exit_reason",
  "exit_source",
  "gross_return_pct",
  "cost_pct",
  "net_return_pct",
  "win",
  "hold_minutes",
  "label_quality",
  "is_trainable"
];

const EXPECTED_DECISION_FIELDS = [
  "decision_time", "outcome", "mode", "signal_cluster_id", "cluster_status",
  "signal_id", "symbol", "direction", "decision", "model_score", "confidence_pct",
  "final_action", "skip_reason", "block_rule", "price_source", "entry_allowed",
  "risk_allowed", "correlation_allowed", "stale_signal", "price_drift_pct",
  "current_price", "entry_price", "stop_loss", "take_profit_1", "schema_version"
];

const EXPECTED_BLOCKED_FIELDS = [
  "blocked_at", "outcome", "mode", "signal_id", "signal_cluster_id", "cluster_status",
  "market", "symbol", "direction", "decision", "signal_type", "source_run_at",
  "model_score", "confidence_pct", "block_rule", "block_reason", "price_source",
  "current_price", "entry_price", "stop_loss", "take_profit_1", "timeframe",
  "trade_thesis", "reason", "price_drift_pct", "schema_version"
];

async function checkCsvHeaderMatches(filePath: string, expectedFields: string[]) {
  try {
    const content = await fs.readFile(filePath, "utf8");
    const firstLine = content.split(/\r?\n/)[0];
    if (!firstLine) return false;
    const actualFields = parseCsvLine(firstLine);
    if (actualFields.length !== expectedFields.length) return false;
    for (let i = 0; i < expectedFields.length; i++) {
      if (actualFields[i].trim() !== expectedFields[i]) return false;
    }
    return true;
  } catch {
    return true; // If missing, we don't treat as mismatch since it's initialized on run
  }
}

async function checkLegacyFilesDetected() {
  const legacyDir = path.join(paperDir, "legacy_snapshots");
  try {
    const entries = await fs.readdir(legacyDir);
    const csvFiles = entries.filter(name => name.endsWith(".csv"));
    return csvFiles.length > 0;
  } catch {
    return false;
  }
}

async function checkBlockedArtefactsPresent() {
  const blockedPath = path.join(paperDir, "paper_trade_blocked.csv");
  try {
    await fs.access(blockedPath);
    return true;
  } catch {
    return false;
  }
}

async function checkConfigRegimePolicyActive() {
  const settingsPath = path.join(prototypeDir, "config", "settings.json");
  try {
    const content = await fs.readFile(settingsPath, "utf8");
    const settings = JSON.parse(content);
    const allowed = settings.allowed_market_regimes || [];
    const blocked = settings.blocked_market_regimes || [];
    return allowed.length > 0 || blocked.length > 0;
  } catch {
    return false;
  }
}

async function buildHealth() {
  const [
    decisionWritable,
    auditWritable,
    staleLocks,
    staleTempFiles,
    runHistory,
    openHeaderOk,
    closedHeaderOk,
    decisionHeaderOk,
    blockedHeaderOk,
    legacyDetected,
    blockedPresent,
    regimePolicyActive
  ] = await Promise.all([
    probePathWritable(decisionPath),
    probePathWritable(systemErrorsPath),
    collectStaleLocks(),
    countStaleTempFiles(),
    readRunHistory(historyPath, 64),
    checkCsvHeaderMatches(path.join(paperDir, "open_trades.csv"), EXPECTED_OPEN_FIELDS),
    checkCsvHeaderMatches(path.join(paperDir, "closed_trades.csv"), EXPECTED_CLOSED_FIELDS),
    checkCsvHeaderMatches(path.join(paperDir, "paper_trade_decisions.csv"), EXPECTED_DECISION_FIELDS),
    checkCsvHeaderMatches(path.join(paperDir, "paper_trade_blocked.csv"), EXPECTED_BLOCKED_FIELDS),
    checkLegacyFilesDetected(),
    checkBlockedArtefactsPresent(),
    checkConfigRegimePolicyActive()
  ]);

  const lastManualRun = runHistory.find((run) => (run as { runSource?: string }).runSource !== "live_cycle") ?? null;
  const headerSchemaMatch = openHeaderOk && closedHeaderOk && decisionHeaderOk && blockedHeaderOk;

  return {
    decisionCsvWritable: decisionWritable,
    auditWritable,
    hasStaleLocks: staleLocks.length > 0,
    staleLocks,
    staleTempFileCount: staleTempFiles,
    headerSchemaMatch,
    legacyFilesDetected: legacyDetected,
    blockedArtefactsPresent: blockedPresent,
    configRegimePolicyActive: regimePolicyActive,
    lastManualRun: lastManualRun
      ? {
          status: lastManualRun.status,
          startedAt: lastManualRun.startedAt,
          finishedAt: lastManualRun.finishedAt,
          failedStep: lastManualRun.failedStep ?? null
        }
      : null
  };
}

export async function GET() {
  const [
    forexSignals,
    cryptoSignals,
    openTrades,
    closedTrades,
    tradeEvents,
    summary,
    aiDataset,
    observationDataset,
    decisionAudit,
    executionAudit,
    liveServiceRunning,
    health,
    runHistory,
    lastLiveCycle
  ] = await Promise.all([
    readCsv("signals/forex_signals.csv"),
    readCsv("signals/crypto_signals.csv"),
    readCsv("paper_trades/open_trades.csv"),
    readCsv("paper_trades/closed_trades.csv"),
    readCsv("paper_trades/trade_events.csv"),
    readCsv("paper_trades/paper_trade_summary.csv"),
    readCsv("ai/ai_training_dataset.csv"),
    readCsv("ai/signal_observation_dataset.csv"),
    readJsonlTail("audit/decision_audit_log.jsonl"),
    readJsonlTail("audit/execution_audit_log.jsonl"),
    checkLiveService(),
    buildHealth(),
    readRunHistory(historyPath, 16),
    readLastLiveCycleFromLog()
  ]);

  const manualRunHistory = runHistory.filter((entry) => (entry as { runSource?: string }).runSource !== "live_cycle").slice(0, 8);

  const allSignals = [...forexSignals, ...cryptoSignals];
  const closedReturns = closedTrades
    .map((row) => Number(row.net_return_pct))
    .filter((value) => Number.isFinite(value));
  const wins = closedTrades.filter((row) => row.win === "True").length;

  return NextResponse.json({
    generatedAt: new Date().toISOString(),
    liveServiceRunning,
    summary: summary[0] ?? {
      open_trades: String(openTrades.length),
      closed_trades: String(closedTrades.length),
      wins: String(wins),
      losses: String(closedTrades.length - wins),
      winrate_pct: closedTrades.length ? ((wins / closedTrades.length) * 100).toFixed(2) : "0",
      total_net_return_pct: closedReturns.reduce((total, value) => total + value, 0).toFixed(4),
      avg_net_return_pct: closedReturns.length
        ? (closedReturns.reduce((total, value) => total + value, 0) / closedReturns.length).toFixed(4)
        : "0"
    },
    counts: {
      totalSignals: allSignals.length,
      forexSignals: forexSignals.length,
      cryptoSignals: cryptoSignals.length,
      openTrades: openTrades.length,
      closedTrades: closedTrades.length,
      aiRows: aiDataset.length,
      observationRows: observationDataset.length
    },
    breakdown: {
      signalsByMarket: countBy(allSignals, "market"),
      signalsByDecision: countBy(allSignals, "decision"),
      openByMarket: countBy(openTrades, "market"),
      openByDirection: countBy(openTrades, "direction")
    },
    forexSignals,
    cryptoSignals,
    openTrades,
    closedTrades,
    tradeEvents,
    aiDataset,
    observationDataset,
    audit: {
      decision: decisionAudit,
      execution: executionAudit
    },
    health,
    runHistory: manualRunHistory,
    lastLiveCycle
  }, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    }
  });
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    }
  });
}
