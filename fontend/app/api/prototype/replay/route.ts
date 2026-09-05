import { execFile } from "child_process";
import path from "path";
import { promisify } from "util";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const execFileAsync = promisify(execFile);
const projectRoot = path.join(process.cwd(), "..");
const runnerPath = path.join(projectRoot, "prototype", "run_replay.py");

let running = false;

export async function POST() {
  if (running) {
    return NextResponse.json({ error: "Replay is already running" }, { status: 409 });
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
          timeout: 5 * 60 * 1000,
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
      throw lastError ?? new Error("Failed to execute replay runner.");
    }

    return NextResponse.json({
      status: "OK",
      startedAt,
      runner: runnerUsed,
      finishedAt: new Date().toISOString(),
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

    const stderr = detail.stderr ?? "";
    const stdout = detail.stdout ?? "";
    const combined = `${detail.message}\n${stderr}\n${stdout}`;
    const missingMt5 = combined.includes("No module named 'MetaTrader5'");

    return NextResponse.json(
      {
        status: "ERROR",
        startedAt,
        finishedAt: new Date().toISOString(),
        error: missingMt5
          ? "MetaTrader5 is not installed in the Python environment used by Replay. Install with: py -3.10 -m pip install MetaTrader5"
          : detail.message,
        code: detail.code ?? null,
        signal: detail.signal ?? null,
        stdout,
        stderr
      },
      { status: 500 }
    );
  } finally {
    running = false;
  }
}
