"use client";

import { useEffect, useMemo, useState } from "react";

type SummaryRow = {
  symbol: string;
  status: string;
  final_bias: "BULLISH" | "BEARISH" | "NEUTRAL" | string;
  decision: string;
  final_score: string;
  confidence: string;
  alignment: string;
  price: string;
  best_timeframe: string;
  entry_price: string;
  entry_zone: string;
  stop_loss: string;
  take_profit_1: string;
  reason: string;
  returncode: string;
};

type Timeframe = {
  interval?: string;
  price?: number;
  bias?: string;
  action?: string;
  confidence?: number;
  score?: number;
  adx?: number;
  rsi?: number;
  atr_ratio?: number;
  support?: number;
  resistance?: number;
  stop_loss?: number;
  take_profit_1?: number;
  reasons?: string;
};

type LatestForecast = {
  summary?: {
    run_id?: string;
    model_version?: string;
    symbol?: string;
  };
  timeframes?: Timeframe[];
} | null;

type RunHistoryEntry = {
  market: "forex" | "crypto";
  startedAt: string;
  finishedAt: string;
  status: "OK" | "ERROR";
  stdout: string;
  stderr: string;
  error?: string;
  code?: number | string | null;
  signal?: string | null;
};

type ForexLogResponse = {
  generatedAt: string | null;
  modelsRun: number;
  summary: SummaryRow[];
  latest: Record<string, LatestForecast>;
  humanLog: string;
  runHistory: RunHistoryEntry[];
  error?: string;
  detail?: string;
};

const marketSymbols = {
  forex: [
    "AUDCAD",
    "AUDCHF",
    "AUDJPY",
    "AUDNZD",
    "AUDUSD",
    "CADCHF",
    "CADJPY",
    "CHFJPY",
    "EURAUD",
    "EURCAD",
    "EURCHF",
    "EURGBP",
    "EURJPY",
    "EURNZD",
    "EURUSD",
    "GBPAUD",
    "GBPCAD",
    "GBPCHF",
    "GBPJPY",
    "GBPNZD",
    "GBPUSD",
    "NZDCAD",
    "NZDCHF",
    "NZDJPY",
    "NZDUSD",
    "USDCAD",
    "USDCHF",
    "USDJPY"
  ],
  crypto: ["BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD"]
} as const;
const labels = {
  en: {
    appName: "AI Chatbot Forex Forecast",
    title: "Model Run Logs",
    runModel: "Run model",
    runningModel: "Running model",
    runPaper: "Run paper",
    runningPaper: "Running paper",
    prototypeMonitor: "Paper monitor",
    refreshLogs: "Refresh logs",
    loading: "Loading",
    errorPrefix: "Unable to read logs",
    runnerOutput: "Runner Output",
    runHistory: "Run History",
    latestRuns: "Latest model runs",
    viewOutput: "View output",
    startedAt: "Started",
    finishedAt: "Finished",
    noRunHistory: "No run history yet",
    running: "Running...",
    latestRunResult: "Latest run result",
    generated: "Generated",
    models: "Models",
    watchSignals: "Watch Signals",
    avgConfidence: "Avg Confidence",
    summaryCsv: "Summary CSV",
    signalTable: "Signal Table",
    symbol: "Symbol",
    bias: "Bias",
    decision: "Decision",
    score: "Score",
    confidence: "Confidence",
    entryZone: "Entry Zone",
    price: "Price",
    bestTf: "Best TF",
    entry: "Entry",
    latestJson: "Latest JSON",
    symbols: "Symbols",
    timeframes: "Timeframes",
    forecastDetail: "forecast detail",
    humanMarkdown: "Human Markdown",
    rawRunLogPreview: "Raw Run Log Preview",
    loadingLog: "Loading log...",
    noSignal: "No signal selected",
    modelRunStart: "Running all forex forecast models...",
    modelRunDone: "Model run completed.",
    paperRunStart: "Running paper trade prototype...",
    paperRunDone: "Paper trade prototype completed.",
    language: "Language"
  },
  th: {
    appName: "ระบบพยากรณ์ Forex ด้วย AI Chatbot",
    title: "บันทึกการรันโมเดล",
    runModel: "รันโมเดล",
    runningModel: "กำลังรันโมเดล",
    runPaper: "Run paper",
    runningPaper: "Running paper",
    prototypeMonitor: "Paper monitor",
    refreshLogs: "โหลด log ใหม่",
    loading: "กำลังโหลด",
    errorPrefix: "อ่าน log ไม่สำเร็จ",
    runnerOutput: "ผลลัพธ์จากการรัน",
    running: "กำลังรัน...",
    latestRunResult: "ผลรันล่าสุด",
    generated: "สร้างเมื่อ",
    models: "จำนวนโมเดล",
    watchSignals: "สัญญาณเฝ้าดู",
    avgConfidence: "ความมั่นใจเฉลี่ย",
    summaryCsv: "ไฟล์สรุป CSV",
    signalTable: "ตารางสัญญาณ",
    symbol: "คู่เงิน",
    bias: "ทิศทาง",
    decision: "ผลตัดสินใจ",
    score: "คะแนน",
    confidence: "ความมั่นใจ",
    entryZone: "โซนเข้า",
    price: "ราคา",
    bestTf: "TF เด่น",
    entry: "จุดเข้า",
    latestJson: "JSON ล่าสุด",
    symbols: "คู่เงิน",
    timeframes: "รายละเอียด Timeframe",
    forecastDetail: "รายละเอียด forecast",
    humanMarkdown: "Markdown สำหรับอ่าน",
    rawRunLogPreview: "ตัวอย่าง log ล่าสุด",
    loadingLog: "กำลังโหลด log...",
    noSignal: "ยังไม่ได้เลือกสัญญาณ",
    modelRunStart: "กำลังรันโมเดล forecast ทุกคู่เงิน...",
    modelRunDone: "รันโมเดลเสร็จแล้ว",
    paperRunStart: "Running paper trade prototype...",
    paperRunDone: "Paper trade prototype completed.",
    language: "ภาษา"
  }
} as const;

type Language = keyof typeof labels;
type AutoRunMinutes = 5 | 10 | 15;
type Market = keyof typeof marketSymbols;

function numberValue(value: string | number | undefined) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatNumber(value: string | number | undefined, digits = 5) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "-";
  return parsed.toFixed(digits);
}

function biasClass(value: string) {
  if (value.includes("BULLISH")) return "bullish";
  if (value.includes("BEARISH")) return "bearish";
  return "neutral";
}

function markdownToBlocks(markdown: string) {
  return markdown
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .slice(0, 80);
}

function splitSignalReason(reason?: string) {
  if (!reason) {
    return { summary: "", tags: [] as string[] };
  }

  const [summary, detail = ""] = reason.split("|");
  const tags = detail
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  return {
    summary: summary.trim(),
    tags
  };
}

function formatDateTime(value?: string | null) {
  if (!value) return "-";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function historyOutput(entry: RunHistoryEntry) {
  return [entry.stdout, entry.stderr, entry.error].filter(Boolean).join("\n\n").trim();
}

export default function Home() {
  const [data, setData] = useState<ForexLogResponse | null>(null);
  const [market, setMarket] = useState<Market>("forex");
  const [selectedSymbol, setSelectedSymbol] = useState("EURUSD");
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [paperRunning, setPaperRunning] = useState(false);
  const [runOutput, setRunOutput] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState<Language>("en");
  const [autoRunMinutes, setAutoRunMinutes] = useState<AutoRunMinutes | null>(null);
  const [nextRunAt, setNextRunAt] = useState<number | null>(null);
  const [now, setNow] = useState(Date.now());
  const text = { ...labels.en, ...labels[language] };
  const symbols = marketSymbols[market];

  async function loadLogs() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/${market}/logs`, { cache: "no-store" });
      const payload = (await response.json()) as ForexLogResponse;

      if (!response.ok) {
        throw new Error(payload.detail || payload.error || "Cannot load logs");
      }

      setData(payload);
      if (!payload.summary.some((row) => row.symbol === selectedSymbol)) {
        setSelectedSymbol(payload.summary[0]?.symbol ?? "EURUSD");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadLogs();
  }, [market]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (!running && !paperRunning) {
        void loadLogs();
      }
    }, 15000);
    return () => window.clearInterval(interval);
  }, [market, running, paperRunning]);

  async function runModels(options?: { keepSchedule?: boolean }) {
    setRunning(true);
    setError(null);
    setRunOutput(`${market.toUpperCase()}: ${text.modelRunStart}`);

    try {
      const response = await fetch(`/api/${market}/run`, {
        method: "POST",
        cache: "no-store"
      });
      const payload = (await response.json()) as {
        status?: string;
        error?: string;
        stdout?: string;
        stderr?: string;
      };

      const output = [payload.stdout, payload.stderr].filter(Boolean).join("\n\n").trim();
      setRunOutput(output || payload.status || text.modelRunDone);

      if (!response.ok) {
        throw new Error(payload.error || "Model run failed");
      }

      await loadLogs();

      if (options?.keepSchedule && autoRunMinutes) {
        setNextRunAt(Date.now() + autoRunMinutes * 60 * 1000);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  }

  async function runPaperPrototype() {
    setPaperRunning(true);
    setError(null);
    setRunOutput(text.paperRunStart);

    try {
      const response = await fetch("/api/prototype/run", {
        method: "POST",
        cache: "no-store"
      });
      const payload = (await response.json()) as {
        stdout?: string;
        stderr?: string;
        error?: string;
      };

      const output = [payload.stdout, payload.stderr].filter(Boolean).join("\n\n").trim();
      setRunOutput(output || text.paperRunDone);

      if (!response.ok) {
        throw new Error(payload.error || "Paper prototype run failed");
      }

      await loadLogs();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setPaperRunning(false);
    }
  }

  function startAutoRun(minutes: AutoRunMinutes) {
    setAutoRunMinutes(minutes);
    setNextRunAt(Date.now() + minutes * 60 * 1000);
  }

  function stopAutoRun() {
    setAutoRunMinutes(null);
    setNextRunAt(null);
  }

  function changeMarket(nextMarket: Market) {
    setMarket(nextMarket);
    setData(null);
    setRunOutput(null);
    setSelectedSymbol(marketSymbols[nextMarket][0]);
    stopAutoRun();
  }

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!autoRunMinutes || !nextRunAt || running || loading || now < nextRunAt) {
      return;
    }

    void runModels({ keepSchedule: true });
  }, [autoRunMinutes, nextRunAt, running, loading, now]);

  const selectedSummary = useMemo(
    () => data?.summary.find((row) => row.symbol === selectedSymbol),
    [data?.summary, selectedSymbol]
  );
  const selectedLatest = data?.latest[selectedSymbol];
  const selectedReason = splitSignalReason(selectedSummary?.reason);
  const watchCount = data?.summary.filter((row) => row.decision.includes("WATCH")).length ?? 0;
  const avgConfidence =
    data && data.summary.length > 0
      ? data.summary.reduce((total, row) => total + numberValue(row.confidence.replace("%", "")), 0) /
        data.summary.length
      : 0;
  const secondsUntilNextRun = nextRunAt ? Math.max(0, Math.ceil((nextRunAt - now) / 1000)) : null;
  const nextRunText =
    secondsUntilNextRun === null
      ? "-"
      : `${Math.floor(secondsUntilNextRun / 60)
          .toString()
          .padStart(2, "0")}:${(secondsUntilNextRun % 60).toString().padStart(2, "0")}`;
  const runHistory = data?.runHistory ?? [];

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">{market === "crypto" ? "AI Chatbot Crypto Forecast" : text.appName}</p>
          <h1>{market === "crypto" ? `Crypto ${text.title}` : text.title}</h1>
        </div>
        <div className="actions">
          <div className="marketSwitch" aria-label="Market">
            <button className={market === "forex" ? "selected" : ""} onClick={() => changeMarket("forex")}>
              Forex
            </button>
            <button className={market === "crypto" ? "selected" : ""} onClick={() => changeMarket("crypto")}>
              Crypto
            </button>
          </div>
          <div className="languageSwitch" aria-label={text.language}>
            <button className={language === "en" ? "selected" : ""} onClick={() => setLanguage("en")}>
              EN
            </button>
            <button className={language === "th" ? "selected" : ""} onClick={() => setLanguage("th")}>
              TH
            </button>
          </div>
          <button className="runButton" onClick={() => void runModels()} disabled={running || loading}>
            {running ? text.runningModel : text.runModel}
          </button>
          <button className="refreshButton" onClick={() => void runPaperPrototype()} disabled={paperRunning || running}>
            {paperRunning ? text.runningPaper : text.runPaper}
          </button>
          <a className="refreshButton" href="/prototype">
            {text.prototypeMonitor}
          </a>
          <button className="refreshButton" onClick={() => void loadLogs()} disabled={loading || running}>
            {loading ? text.loading : text.refreshLogs}
          </button>
        </div>
      </section>

      <section className="schedulerBar" aria-label="Auto run model scheduler">
        <div>
          <p className="eyebrow">{language === "th" ? "รันอัตโนมัติ" : "Auto Run"}</p>
          <strong>
            {autoRunMinutes
              ? language === "th"
                ? `รันทุก ${autoRunMinutes} นาที | รอบถัดไป ${nextRunText}`
                : `Every ${autoRunMinutes} minutes | Next run ${nextRunText}`
              : language === "th"
                ? "ยังไม่ได้เปิดรันอัตโนมัติ"
                : "Auto run is off"}
          </strong>
        </div>
        <div className="scheduleActions">
          {[5, 10, 15].map((minutes) => (
            <button
              className={autoRunMinutes === minutes ? "selected" : ""}
              disabled={running}
              key={minutes}
              onClick={() => startAutoRun(minutes as AutoRunMinutes)}
            >
              {minutes}m
            </button>
          ))}
          <button className="stopButton" disabled={!autoRunMinutes} onClick={stopAutoRun}>
            {language === "th" ? "หยุด" : "Stop"}
          </button>
        </div>
      </section>

      {error ? <div className="alert">{text.errorPrefix}: {error}</div> : null}
      {runOutput ? (
        <section className="runConsole" aria-label="Model run output">
          <div className="panelHeader">
            <div>
              <p className="eyebrow">{text.runnerOutput}</p>
              <h2>{running ? text.running : text.latestRunResult}</h2>
            </div>
          </div>
          <pre>{runOutput}</pre>
        </section>
      ) : null}

      <section className="panel historyPanel" aria-label="Model run history">
        <div className="panelHeader">
          <div>
            <p className="eyebrow">{text.runHistory}</p>
            <h2>{text.latestRuns}</h2>
          </div>
        </div>
        {runHistory.length > 0 ? (
          <div className="historyList">
            {runHistory.map((entry, index) => (
              <article className="historyCard" key={`${entry.startedAt}-${index}`}>
                <div className="historyTop">
                  <span className={`pill ${entry.status === "OK" ? "bullish" : "bearish"}`}>{entry.status}</span>
                  <button
                    className="refreshButton"
                    onClick={() => setRunOutput(historyOutput(entry) || entry.status)}
                    type="button"
                  >
                    {text.viewOutput}
                  </button>
                </div>
                <dl className="historyMeta">
                  <div>
                    <dt>{text.startedAt}</dt>
                    <dd>{formatDateTime(entry.startedAt)}</dd>
                  </div>
                  <div>
                    <dt>{text.finishedAt}</dt>
                    <dd>{formatDateTime(entry.finishedAt)}</dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>
        ) : (
          <p className="historyEmpty">{text.noRunHistory}</p>
        )}
      </section>

      <section className="metrics" aria-label="Run overview">
        <div className="metric">
          <span>{text.generated}</span>
          <strong>{data?.generatedAt ?? "-"}</strong>
        </div>
        <div className="metric">
          <span>{text.models}</span>
          <strong>{data?.modelsRun ?? "-"}</strong>
        </div>
        <div className="metric">
          <span>{text.watchSignals}</span>
          <strong>{watchCount}</strong>
        </div>
        <div className="metric">
          <span>{text.avgConfidence}</span>
          <strong>{avgConfidence ? `${avgConfidence.toFixed(2)}%` : "-"}</strong>
        </div>
      </section>

      <section className="signalGrid" aria-label="Signal cards">
        {(data?.summary ?? []).map((row) => (
          <button
            className={`signalCard ${selectedSymbol === row.symbol ? "active" : ""}`}
            key={row.symbol}
            onClick={() => setSelectedSymbol(row.symbol)}
          >
            <span className="cardTop">
              <strong>{row.symbol}</strong>
              <span className={`pill ${biasClass(row.final_bias)}`}>{row.final_bias}</span>
            </span>
            <span className="decision">{row.decision}</span>
            <span className="scoreLine">
              <span>Score {row.final_score}</span>
              <span>{row.confidence}</span>
            </span>
          </button>
        ))}
      </section>

      <section className="workspace">
        <div className="panel tablePanel">
          <div className="panelHeader">
            <div>
              <p className="eyebrow">{text.summaryCsv}</p>
              <h2>{text.signalTable}</h2>
            </div>
          </div>
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>{text.symbol}</th>
                  <th>{text.bias}</th>
                  <th>{text.decision}</th>
                  <th>{text.score}</th>
                  <th>{text.confidence}</th>
                  <th>{text.entryZone}</th>
                  <th>SL</th>
                  <th>TP1</th>
                </tr>
              </thead>
              <tbody>
                {(data?.summary ?? []).map((row) => (
                  <tr key={row.symbol}>
                    <td>{row.symbol}</td>
                    <td>
                      <span className={`pill ${biasClass(row.final_bias)}`}>{row.final_bias}</span>
                    </td>
                    <td>{row.decision}</td>
                    <td>{row.final_score}</td>
                    <td>{row.confidence}</td>
                    <td>{row.entry_zone || "-"}</td>
                    <td>{row.stop_loss}</td>
                    <td>{row.take_profit_1}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <aside className="panel detailPanel">
          <p className="eyebrow">{text.latestJson}</p>
          <h2>{selectedSymbol}</h2>
          <div className="detailStats">
            <span>{text.price} {formatNumber(selectedSummary?.price)}</span>
            <span>{text.bestTf} {selectedSummary?.best_timeframe ?? "-"}</span>
            <span>{text.entry} {selectedSummary?.entry_price || "-"}</span>
          </div>
          <div className="reasonBox">
            {selectedReason.summary ? <p>{selectedReason.summary}</p> : <p>{text.noSignal}</p>}
            {selectedReason.tags.length > 0 ? (
              <div className="reasonTags">
                {selectedReason.tags.map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
            ) : null}
          </div>
          <div className="tabs" aria-label={text.symbols}>
            {symbols.map((symbol) => (
              <button
                className={selectedSymbol === symbol ? "selected" : ""}
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
              >
                {symbol}
              </button>
            ))}
          </div>
        </aside>
      </section>

      <section className="panel">
        <div className="panelHeader">
          <div>
            <p className="eyebrow">{text.timeframes}</p>
            <h2>{selectedLatest?.summary?.model_version ?? `${selectedSymbol} ${text.forecastDetail}`}</h2>
          </div>
          <span className="runId">{selectedLatest?.summary?.run_id ?? "-"}</span>
        </div>
        <div className="timeframeGrid">
          {(selectedLatest?.timeframes ?? []).map((item) => (
            <article className="timeframeCard" key={item.interval}>
              <div className="cardTop">
                <strong>{item.interval}</strong>
                <span className={`pill ${biasClass(item.bias ?? "")}`}>{item.bias}</span>
              </div>
              <dl>
                <div>
                  <dt>{text.price}</dt>
                  <dd>{formatNumber(item.price)}</dd>
                </div>
                <div>
                  <dt>{text.score}</dt>
                  <dd>{item.score ?? "-"}</dd>
                </div>
                <div>
                  <dt>{text.confidence}</dt>
                  <dd>{item.confidence ? `${(item.confidence * 100).toFixed(2)}%` : "-"}</dd>
                </div>
                <div>
                  <dt>RSI</dt>
                  <dd>{formatNumber(item.rsi, 2)}</dd>
                </div>
                <div>
                  <dt>ADX</dt>
                  <dd>{formatNumber(item.adx, 2)}</dd>
                </div>
                <div>
                  <dt>ATR Ratio</dt>
                  <dd>{formatNumber(item.atr_ratio, 2)}</dd>
                </div>
              </dl>
              <p>{item.reasons}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel logPanel">
        <div className="panelHeader">
          <div>
            <p className="eyebrow">{text.humanMarkdown}</p>
            <h2>{text.rawRunLogPreview}</h2>
          </div>
        </div>
        <div className="logText">
          {data ? (
            markdownToBlocks(data.humanLog).map((line, index) => <pre key={`${line}-${index}`}>{line}</pre>)
          ) : (
            <pre>{text.loadingLog}</pre>
          )}
        </div>
      </section>
    </main>
  );
}
