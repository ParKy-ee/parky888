import { promises as fs } from "fs";
import path from "path";

export type RunHistoryEntry = {
  market: "forex" | "crypto" | "prototype";
  startedAt: string;
  finishedAt: string;
  status: "OK" | "ERROR";
  stdout: string;
  stderr: string;
  error?: string;
  code?: number | string | null;
  signal?: string | null;
  runModels?: boolean;
  failedStep?: string | null;
  steps?: Array<{
    step: string;
    status: "OK" | "ERROR";
    result?: unknown;
    error?: string;
  }>;
};

export async function appendRunHistory(historyPath: string, entry: RunHistoryEntry) {
  await fs.mkdir(path.dirname(historyPath), { recursive: true });
  await fs.appendFile(historyPath, `${JSON.stringify(entry)}\n`, "utf8");
}

export async function readRunHistory(historyPath: string, limit = 12): Promise<RunHistoryEntry[]> {
  try {
    const content = await fs.readFile(historyPath, "utf8");
    return content
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => JSON.parse(line) as RunHistoryEntry)
      .slice(-limit)
      .reverse();
  } catch {
    return [];
  }
}
