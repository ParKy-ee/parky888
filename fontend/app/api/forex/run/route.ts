import { execFile } from "child_process";
import path from "path";
import { promisify } from "util";
import { NextResponse } from "next/server";
import { appendRunHistory } from "../../_lib/run_history";

export const dynamic = "force-dynamic";

const execFileAsync = promisify(execFile);
const projectRoot = path.join(process.cwd(), "..", "ai_chatbot");
const runnerPath = path.join(projectRoot, "forex", "models_", "run_all_models.py");
const historyPath = path.join(projectRoot, "logs", "forex_run_history.jsonl");

let running = false;

export async function POST() {
  if (running) {
    return NextResponse.json(
      {
        error: "Models are already running"
      },
      { status: 409 }
    );
  }

  running = true;
  const startedAt = new Date().toISOString();

  try {
    const candidates: Array<{ cmd: string; args: string[]; label: string }> = [
      { cmd: "py", args: ["-3.10", runnerPath], label: "py -3.10" },
      { cmd: "python", args: [runnerPath], label: "python" }
    ];

    let result:
      | {
          stdout: string;
          stderr: string;
        }
      | undefined;
    let lastError:
      | (Error & {
          stdout?: string;
          stderr?: string;
          code?: number | string;
          signal?: string;
        })
      | undefined;
    let runnerUsed = "";

    for (const candidate of candidates) {
      try {
        result = await execFileAsync(candidate.cmd, candidate.args, {
          cwd: projectRoot,
          timeout: 10 * 60 * 1000,
          maxBuffer: 1024 * 1024 * 10,
          windowsHide: true
        });
        runnerUsed = candidate.label;
        break;
      } catch (error) {
        lastError = error as Error & {
          stdout?: string;
          stderr?: string;
          code?: number | string;
          signal?: string;
        };
      }
    }

    if (!result) {
      throw lastError ?? new Error("Failed to execute forex model runner.");
    }

    const finishedAt = new Date().toISOString();
    await appendRunHistory(historyPath, {
      market: "forex",
      startedAt,
      finishedAt,
      status: "OK",
      stdout: `[runner=${runnerUsed}]\n${result.stdout}`,
      stderr: result.stderr,
      code: 0,
      signal: null
    });

    return NextResponse.json({
      status: "OK",
      startedAt,
      finishedAt,
      runner: runnerUsed,
      stdout: result.stdout,
      stderr: result.stderr
    });
  } catch (error) {
    const detail = error as Error & {
      stdout?: string;
      stderr?: string;
      code?: number | string;
      signal?: string;
    };
    const finishedAt = new Date().toISOString();
    await appendRunHistory(historyPath, {
      market: "forex",
      startedAt,
      finishedAt,
      status: "ERROR",
      stdout: detail.stdout ?? "",
      stderr: detail.stderr ?? "",
      error: detail.message,
      code: detail.code ?? null,
      signal: detail.signal ?? null
    });

    return NextResponse.json(
      {
        status: "ERROR",
        startedAt,
        finishedAt,
        error: detail.message,
        code: detail.code ?? null,
        signal: detail.signal ?? null,
        stdout: detail.stdout ?? "",
        stderr: detail.stderr ?? ""
      },
      { status: 500 }
    );
  } finally {
    running = false;
  }
}
