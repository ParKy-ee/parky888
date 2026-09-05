"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type Row = Record<string, string>;

type RunHistoryEntry = {
  market: string;
  startedAt: string;
  finishedAt: string;
  status: "OK" | "ERROR";
  stdout?: string;
  stderr?: string;
  error?: string;
  failedStep?: string | null;
  runModels?: boolean;
};

type PrototypeHealth = {
  decisionCsvWritable: boolean;
  auditWritable: boolean;
  hasStaleLocks: boolean;
  staleLocks: Array<{ path: string; ageSeconds: number }>;
  staleTempFileCount: number;
  headerSchemaMatch: boolean;
  legacyFilesDetected: boolean;
  blockedArtefactsPresent: boolean;
  configRegimePolicyActive: boolean;
  lastManualRun: {
    status: string;
    startedAt: string;
    finishedAt: string;
    failedStep: string | null;
  } | null;
};

type LastLiveCycle = {
  status: "OK" | "ERROR" | "UNKNOWN";
  at: string | null;
  message: string;
} | null;

type PrototypeStatus = {
  generatedAt: string;
  liveServiceRunning: boolean;
  summary: Row;
  counts: {
    totalSignals: number;
    forexSignals: number;
    cryptoSignals: number;
    openTrades: number;
    closedTrades: number;
    aiRows: number;
    observationRows: number;
  };
  breakdown: {
    signalsByMarket: Record<string, number>;
    signalsByDecision: Record<string, number>;
    openByMarket: Record<string, number>;
    openByDirection: Record<string, number>;
  };
  forexSignals: Row[];
  cryptoSignals: Row[];
  openTrades: Row[];
  closedTrades: Row[];
  tradeEvents: Row[];
  aiDataset: Row[];
  observationDataset: Row[];
  audit: {
    decision: Array<Record<string, unknown>>;
    execution: Array<Record<string, unknown>>;
  };
  health?: PrototypeHealth;
  runHistory?: RunHistoryEntry[];
  lastLiveCycle?: LastLiveCycle;
};

type View = "signals" | "open" | "closed" | "dataset" | "audit";

async function readJsonResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  return text ? (JSON.parse(text) as T) : ({} as T);
}

function value(value: string | number | undefined) {
  if (value === undefined || value === "") return "-";
  return value;
}

function compactReason(reason: string | undefined) {
  if (!reason) return "-";
  return reason.length > 96 ? `${reason.slice(0, 96)}...` : reason;
}

function decisionClass(decision: string | undefined) {
  if (!decision) return "neutral";
  if (decision.includes("BULLISH")) return "bullish";
  if (decision.includes("BEARISH")) return "bearish";
  return "neutral";
}

function formatDateTime(value: string | undefined) {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function historyOutput(entry: RunHistoryEntry) {
  return [entry.stdout, entry.stderr, entry.error].filter(Boolean).join("\n\n").trim();
}

export default function PrototypePage() {
  const [data, setData] = useState<PrototypeStatus | null>(null);
  const [view, setView] = useState<View>("signals");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [serviceActionLoading, setServiceActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runOutput, setRunOutput] = useState<string | null>(null);
  const [replaying, setReplaying] = useState(false);
  const [runModelsBefore, setRunModelsBefore] = useState(false);
  const [auditing, setAuditing] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 10;

  async function loadStatus() {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/prototype/status", { cache: "no-store" });
      const payload = (await response.json()) as PrototypeStatus;
      if (!response.ok) throw new Error("Cannot load prototype status");
      setData(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }

  async function runPrototype() {
    setRunning(true);
    setError(null);
    setRunOutput("Running CSV-first prototype (One-shot)...");
    try {
      const response = await fetch("/api/prototype/run", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ runModels: runModelsBefore })
      });
      const payload = await readJsonResponse<{ stdout?: string; stderr?: string; error?: string }>(response);
      const output = [payload.stdout, payload.stderr].filter(Boolean).join("\n\n").trim();
      setRunOutput(output || "Prototype run completed.");
      if (!response.ok) throw new Error(payload.error || "Prototype run failed");
      await loadStatus();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  }

  async function toggleLiveService() {
    setServiceActionLoading(true);
    setError(null);
    const action = data?.liveServiceRunning ? "stop" : "start";
    setRunOutput(`${action === "start" ? "Starting" : "Stopping"} 24/7 Live Service Mode...`);
    
    try {
      const response = await fetch("/api/prototype/live", {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action })
      });
      const payload = await readJsonResponse<{ message?: string; error?: string }>(response);
      if (!response.ok) throw new Error(payload.error || `Failed to ${action} service`);
      
      setRunOutput(payload.message || `Service ${action}ed successfully.`);
      await loadStatus();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setServiceActionLoading(false);
    }
  }

  async function runReplay() {
    setReplaying(true);
    setError(null);
    setRunOutput("Syncing trades via MT5 replay...");
    try {
      const response = await fetch("/api/prototype/replay", {
        method: "POST",
        cache: "no-store",
      });
      const payload = await readJsonResponse<{ stdout?: string; stderr?: string; error?: string }>(response);
      const output = [payload.stdout, payload.stderr].filter(Boolean).join("\n\n").trim();
      setRunOutput(output || "Replay sync completed.");
      if (!response.ok) throw new Error(payload.error || "Replay sync failed");
      await loadStatus();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setReplaying(false);
    }
  }

  async function runAudit() {
    setAuditing(true);
    setError(null);
    setRunOutput("Auditing paper trade quality...");
    try {
      const response = await fetch("/api/prototype/audit", {
        method: "POST",
        cache: "no-store",
      });
      const payload = await readJsonResponse<{ stdout?: string; stderr?: string; error?: string }>(response);
      const output = [payload.stdout, payload.stderr].filter(Boolean).join("\n\n").trim();
      setRunOutput(output || "Audit completed.");
      if (!response.ok) throw new Error(payload.error || "Audit failed");
      await loadStatus();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setAuditing(false);
    }
  }

  useEffect(() => {
    void loadStatus();
  }, []);

  useEffect(() => {
    const refreshMs = data?.liveServiceRunning ? 5000 : 30000;
    const interval = setInterval(() => void loadStatus(), refreshMs);
    return () => clearInterval(interval);
  }, [data?.liveServiceRunning]);

  useEffect(() => {
    setCurrentPage(1);
  }, [view]);

  const allSignals = useMemo(() => {
    const list = [...(data?.forexSignals ?? []), ...(data?.cryptoSignals ?? [])];
    return list.sort((a, b) => (b.source_run_at ?? "").localeCompare(a.source_run_at ?? ""));
  }, [data]);

  const closedTradesSorted = useMemo(() => {
    const list = [...(data?.closedTrades ?? [])];
    return list.sort((a, b) => (b.exit_time ?? "").localeCompare(a.exit_time ?? ""));
  }, [data]);

  const observationDatasetSorted = useMemo(() => {
    const list = [...(data?.observationDataset ?? [])];
    return list.sort((a, b) => (b.source_run_at ?? "").localeCompare(a.source_run_at ?? ""));
  }, [data]);

  const visibleAudit = useMemo(() => {
    const decision = data?.audit.decision ?? [];
    const execution = data?.audit.execution ?? [];
    return [...execution, ...decision].slice(0, 20);
  }, [data]);

  function Paginate({ total }: { total: number }) {
    const totalPages = Math.ceil(total / pageSize);
    if (totalPages <= 1) return null;

    return (
      <div className="pagination">
        <button disabled={currentPage === 1} onClick={() => setCurrentPage((p) => p - 1)}>
          Previous
        </button>
        <span>
          Page {currentPage} of {totalPages}
        </span>
        <button disabled={currentPage === totalPages} onClick={() => setCurrentPage((p) => p + 1)}>
          Next
        </button>
      </div>
    );
  }

  return (
    <main className="prototypeShell">
      <section className="prototypeHero">
        <div>
          <p className="eyebrow">CSV-First Auto Trade Prototype v2 (Real-time)</p>
          <h1>Command & Monitor</h1>
          <p>
            Control panel for running the prototype, inspecting signals, paper trades, audit logs, and datasets
            for AI training without routing actual orders to any broker/exchange.
          </p>
        </div>
        <div className="prototypeHeroActions">
          <Link className="ghostLink" href="/">
            Model Dashboard
          </Link>
          <button className="refreshButton" disabled={loading || running || replaying} onClick={() => void loadStatus()}>
            {loading ? "Loading" : "Refresh"}
          </button>
          <button className="runButton" disabled={loading || running || replaying} style={{ backgroundColor: "#2457c5", borderColor: "#2457c5", color: "white" }} onClick={() => void runPrototype()}>
            {running ? "Running" : "Run Once"}
          </button>
          <button className="runButton" disabled={loading || running || replaying || auditing} style={{ backgroundColor: "#087f5b", borderColor: "#087f5b", color: "white" }} onClick={() => void runReplay()}>
            {replaying ? "Syncing..." : "Replay Trades"}
          </button>
          <button className="runButton" disabled={loading || running || replaying || auditing} style={{ backgroundColor: "#f08c00", borderColor: "#f08c00", color: "white" }} onClick={() => void runAudit()}>
            {auditing ? "Auditing..." : "Run Audit"}
          </button>
        </div>
      </section>

      <section className="runOptionsBar">
        <label className="checkboxLabel">
          <input
            checked={runModelsBefore}
            onChange={(e) => setRunModelsBefore(e.target.checked)}
            type="checkbox"
          />
          <span>Run AI Models before prototype (Full Pipeline - takes ~6-10 min)</span>
        </label>
      </section>

      {error ? <div className="alert">Prototype error: {error}</div> : null}

      <section className="schedulerBar" aria-label="24/7 Live Service Mode">
        <div>
          <p className="eyebrow">24/7 Live Service Mode (New)</p>
          <strong style={{ color: data?.liveServiceRunning ? "#087f5b" : "#e03131" }}>
            {data?.liveServiceRunning
              ? `RUNNING (Background Process ACTIVE)`
              : "STOPPED (Offline)"}
          </strong>
        </div>
        <div className="scheduleActions">
          <button 
            className={data?.liveServiceRunning ? "stopButton" : "runButton"} 
            disabled={serviceActionLoading || loading}
            onClick={toggleLiveService}
            style={{ 
              backgroundColor: data?.liveServiceRunning ? "#e03131" : "#087f5b",
              color: "white",
              padding: "0.5rem 2rem"
            }}
          >
            {serviceActionLoading ? "Processing..." : data?.liveServiceRunning ? "STOP SERVICE" : "START LIVE SERVICE"}
          </button>
        </div>
      </section>

      <section className="prototypeStatusBand" aria-label="Pipeline health">
        <div>
          <span>Decision CSV</span>
          <strong style={{ color: data?.health?.decisionCsvWritable ? "#087f5b" : "#e03131" }}>
            {data?.health?.decisionCsvWritable ? "Writable" : "Not writable"}
          </strong>
        </div>
        <div>
          <span>Stale locks</span>
          <strong style={{ color: data?.health?.hasStaleLocks ? "#e03131" : "#087f5b" }}>
            {data?.health?.hasStaleLocks
              ? `${data.health.staleLocks.length} orphaned`
              : "None"}
          </strong>
        </div>
        <div>
          <span>Audit log</span>
          <strong style={{ color: data?.health?.auditWritable ? "#087f5b" : "#e03131" }}>
            {data?.health?.auditWritable ? "Writable" : "Not writable"}
          </strong>
        </div>
        <div>
          <span>Last Manual Prototype Run</span>
          <strong>
            {data?.health?.lastManualRun
              ? `${data.health.lastManualRun.status} at ${formatDateTime(data.health.lastManualRun.finishedAt)}`
              : "Never"}
          </strong>
        </div>
        <div>
          <span>Orphan temp files</span>
          <strong>{data?.health?.staleTempFileCount ?? 0}</strong>
        </div>
      </section>

      <section className="prototypeStatusBand">
        <div>
          <span>Mode</span>
          <strong>Research / Paper Only</strong>
        </div>
        <div>
          <span>Last refresh</span>
          <strong>{data?.generatedAt ? new Date(data.generatedAt).toLocaleString() : "-"}</strong>
        </div>
        <div>
          <span>Last Live Cycle</span>
          <strong>
            {data?.lastLiveCycle?.at
              ? `${data.lastLiveCycle.status} at ${formatDateTime(data.lastLiveCycle.at)}`
              : "No live cycle yet"}
          </strong>
        </div>
        <div>
          <span>Safety</span>
          <strong>No live orders</strong>
        </div>
      </section>

      <section className="prototypeStatusBand" aria-label="State consistency and alignment">
        <div>
          <span>Schema Integrity</span>
          <strong style={{ color: data?.health?.headerSchemaMatch ? "#087f5b" : "#e03131" }}>
            {data?.health?.headerSchemaMatch ? "Match" : "Mismatch"}
          </strong>
        </div>
        <div>
          <span>Legacy Snapshots</span>
          <strong style={{ color: data?.health?.legacyFilesDetected ? "#f08c00" : "#087f5b" }}>
            {data?.health?.legacyFilesDetected ? "Detected" : "None"}
          </strong>
        </div>
        <div>
          <span>Blocked CSV</span>
          <strong style={{ color: data?.health?.blockedArtefactsPresent ? "#087f5b" : "#e03131" }}>
            {data?.health?.blockedArtefactsPresent ? "Present" : "Missing"}
          </strong>
        </div>
        <div>
          <span>Regime Filter</span>
          <strong style={{ color: data?.health?.configRegimePolicyActive ? "#087f5b" : "#f08c00" }}>
            {data?.health?.configRegimePolicyActive ? "Active" : "Inactive"}
          </strong>
        </div>
      </section>

      {runOutput ? (
        <section className="runConsole">
          <div className="panelHeader">
            <div>
              <p className="eyebrow">Command Output</p>
              <h2>Latest Activity</h2>
            </div>
          </div>
          <pre>{runOutput}</pre>
        </section>
      ) : null}

      <section className="panel historyPanel" aria-label="Prototype run history">
        <div className="panelHeader">
          <div>
            <p className="eyebrow">Run History</p>
            <h2>Latest prototype runs</h2>
          </div>
        </div>
        {(data?.runHistory ?? []).length > 0 ? (
          <div className="historyList">
            {(data?.runHistory ?? []).map((entry, index) => (
              <article className="historyCard" key={`${entry.startedAt}-${index}`}>
                <div className="historyTop">
                  <span className={`pill ${entry.status === "OK" ? "bullish" : "bearish"}`}>{entry.status}</span>
                  {entry.failedStep ? <span className="pill neutral">failed: {entry.failedStep}</span> : null}
                  <button
                    className="refreshButton"
                    onClick={() => setRunOutput(historyOutput(entry) || entry.status)}
                    type="button"
                  >
                    View output
                  </button>
                </div>
                <dl className="historyMeta">
                  <div>
                    <dt>Started</dt>
                    <dd>{formatDateTime(entry.startedAt)}</dd>
                  </div>
                  <div>
                    <dt>Finished</dt>
                    <dd>{formatDateTime(entry.finishedAt)}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>
        ) : (
          <p className="historyEmpty">No prototype run history yet.</p>
        )}
      </section>

      <section className="prototypeMetrics">
        <article>
          <span>Total Signals</span>
          <strong>{data?.counts.totalSignals ?? "-"}</strong>
          <p>Forex {data?.counts.forexSignals ?? 0} / Crypto {data?.counts.cryptoSignals ?? 0}</p>
        </article>
        <article>
          <span>Open Paper Trades</span>
          <strong>{data?.counts.openTrades ?? "-"}</strong>
          <p>Simulated positions that are not closed yet</p>
        </article>
        <article>
          <span>Closed Trades</span>
          <strong>{data?.counts.closedTrades ?? "-"}</strong>
          <p>Winrate {value(data?.summary.winrate_pct)}%</p>
        </article>
        <article>
          <span>AI Dataset Rows</span>
          <strong>{data?.counts.aiRows ?? "-"}</strong>
          <p>Paper {data?.counts.aiRows ?? 0} / Observations {data?.counts.observationRows ?? 0}</p>
        </article>
      </section>

      <section className="prototypeGrid">
        <div className="prototypePanel">
          <div className="panelHeader">
            <div>
              <p className="eyebrow">Capabilities (v2)</p>
              <h2>What the prototype can do currently</h2>
            </div>
          </div>
          <div className="capabilityList">
            <span style={{ fontWeight: "bold", color: "#2457c5" }}>[New] 24/7 Background Live Service Mode</span>
            <span style={{ fontWeight: "bold", color: "#2457c5" }}>[New] Rate Limit Protection (Surgical Fetch & Throttling)</span>
            <span>Read Forex/Crypto logs and normalize them to a signal CSV</span>
            <span>Prevent duplicates using signal_id and trade_id</span>
            <span>Separate research observations from realistic paper entries with explicit mode labels</span>
            <span>Calculate position size from balance, risk %, entry, and initial SL</span>
            <span>Anchor R metrics (MFE/MAE) to initial risk distance throughout trade lifecycle</span>
            <span>Use candle high/low path to check TP/SL when market feed is ready</span>
            <span>Block realistic entries when no executable live price is available</span>
            <span>Block signals where source_run_at is unsafe for data leakage</span>
            <span>Forex position sizing uses pip value / contract size standard formula</span>
            <span>Paper trade fees configured per symbol via config</span>
            <span>Write decision/execution audit logs to JSONL</span>
            <span>Export AI training dataset from closed trades</span>
            <span>Export signal observation dataset separating WATCH/WAIT for Forex and Crypto</span>
          </div>
        </div>

        <div className="prototypePanel">
          <div className="panelHeader">
            <div>
              <p className="eyebrow">Risk Snapshot</p>
              <h2>Paper Trade Summary</h2>
            </div>
          </div>
          <dl className="prototypeDl">
            <div>
              <dt>Open</dt>
              <dd>{value(data?.summary.open_trades)}</dd>
            </div>
            <div>
              <dt>Closed</dt>
              <dd>{value(data?.summary.closed_trades)}</dd>
            </div>
            <div>
              <dt>Wins</dt>
              <dd>{value(data?.summary.wins)}</dd>
            </div>
            <div>
              <dt>Losses</dt>
              <dd>{value(data?.summary.losses)}</dd>
            </div>
            <div>
              <dt>Total Net Return</dt>
              <dd>{value(data?.summary.total_net_return_pct)}%</dd>
            </div>
            <div>
              <dt>Avg Net Return</dt>
              <dd>{value(data?.summary.avg_net_return_pct)}%</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="prototypePanel">
        <div className="monitorTabs">
          <button className={view === "signals" ? "selected" : ""} onClick={() => setView("signals")}>
            Signals
          </button>
          <button className={view === "open" ? "selected" : ""} onClick={() => setView("open")}>
            Open Trades
          </button>
          <button className={view === "closed" ? "selected" : ""} onClick={() => setView("closed")}>
            Closed Trades
          </button>
          <button className={view === "dataset" ? "selected" : ""} onClick={() => setView("dataset")}>
            Dataset
          </button>
          <button className={view === "audit" ? "selected" : ""} onClick={() => setView("audit")}>
            Audit
          </button>
        </div>

        {view === "signals" ? (
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Market</th>
                  <th>Symbol</th>
                  <th>Time</th>
                  <th>Decision</th>
                  <th>Score</th>
                  <th>Confidence</th>
                  <th>Price</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {allSignals.slice((currentPage - 1) * pageSize, currentPage * pageSize).map((row) => (
                  <tr key={row.signal_id}>
                    <td>{row.market}</td>
                    <td>{row.symbol}</td>
                    <td>{row.source_run_at}</td>
                    <td>
                      <span className={`pill ${decisionClass(row.decision)}`}>{row.decision}</span>
                    </td>
                    <td>{row.final_score}</td>
                    <td>{row.confidence_pct}%</td>
                    <td>{row.price}</td>
                    <td>{compactReason(row.reason)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Paginate total={allSignals.length} />
          </div>
        ) : null}

        {view === "open" ? (
          <div className="tradeCards">
            {(data?.openTrades ?? []).map((trade) => (
              <article className="tradeCard" key={trade.trade_id}>
                <div className="cardTop">
                  <strong>{trade.symbol}</strong>
                  <span className={`pill ${trade.direction === "LONG" ? "bullish" : "bearish"}`}>
                    {trade.direction}
                  </span>
                </div>
                <dl>
                  <div>
                    <dt>Entry Time</dt>
                    <dd>{trade.entry_time}</dd>
                  </div>
                  <div>
                    <dt>Entry</dt>
                    <dd>{trade.entry_price}</dd>
                  </div>
                  <div>
                    <dt>Initial SL</dt>
                    <dd>{value(trade.initial_stop_loss)}</dd>
                  </div>
                  <div>
                    <dt>Current SL</dt>
                    <dd>{trade.stop_loss}</dd>
                  </div>
                  <div>
                    <dt>Initial Risk Dist</dt>
                    <dd>{value(trade.initial_risk_distance)}</dd>
                  </div>
                  <div>
                    <dt>Risk</dt>
                    <dd>{trade.risk_amount_usd} USD</dd>
                  </div>
                  <div>
                    <dt>Max MFE R</dt>
                    <dd>{value(trade.max_favorable_r)}</dd>
                  </div>
                  <div>
                    <dt>Max MAE R</dt>
                    <dd>{value(trade.max_adverse_r)}</dd>
                  </div>
                  <div>
                    <dt>Feed</dt>
                    <dd>{value(trade.last_market_source)}</dd>
                  </div>
                </dl>
                <p>{compactReason(trade.reason)}</p>
              </article>
            ))}
          </div>
        ) : null}

        {view === "closed" ? (
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Exit Time</th>
                  <th>Direction</th>
                  <th>Exit</th>
                  <th>Reason</th>
                  <th>Source</th>
                  <th>Net Return</th>
                  <th>Win</th>
                  <th>Hold</th>
                </tr>
              </thead>
              <tbody>
                {closedTradesSorted.slice((currentPage - 1) * pageSize, currentPage * pageSize).map((trade) => (
                  <tr key={trade.trade_id}>
                    <td>{trade.symbol}</td>
                    <td>{trade.exit_time}</td>
                    <td>{trade.direction}</td>
                    <td>{trade.exit_price}</td>
                    <td>{trade.exit_reason}</td>
                    <td>{trade.exit_source}</td>
                    <td>{trade.net_return_pct}%</td>
                    <td>{trade.win}</td>
                    <td>{trade.hold_minutes}m</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Paginate total={closedTradesSorted.length} />
          </div>
        ) : null}

        {view === "dataset" ? (
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Market</th>
                  <th>Symbol</th>
                  <th>Time</th>
                  <th>Role</th>
                  <th>Decision</th>
                  <th>Score</th>
                  <th>RR</th>
                  <th>1h</th>
                  <th>4h</th>
                  <th>24h</th>
                </tr>
              </thead>
              <tbody>
                {observationDatasetSorted.slice((currentPage - 1) * pageSize, currentPage * pageSize).map((row) => (
                  <tr key={row.signal_id}>
                    <td>{row.dataset_label}</td>
                    <td>{row.market}</td>
                    <td>{row.symbol}</td>
                    <td>{row.source_run_at}</td>
                    <td>{row.observation_role}</td>
                    <td>{row.decision}</td>
                    <td>{row.model_score}</td>
                    <td>{value(row.risk_reward_ratio)}</td>
                    <td>{value(row.forward_return_1h_pct)}</td>
                    <td>{value(row.forward_return_4h_pct)}</td>
                    <td>{value(row.forward_return_24h_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <Paginate total={observationDatasetSorted.length} />
          </div>
        ) : null}

        {view === "audit" ? (
          <div className="auditList">
            {visibleAudit.map((item, index) => (
              <pre key={index}>{JSON.stringify(item, null, 2)}</pre>
            ))}
          </div>
        ) : null}
      </section>
    </main>
  );
}
