import { execFile } from "child_process";
import path from "path";
import { promisify } from "util";
import { NextResponse } from "next/server";
import { appendRunHistory } from "../../_lib/run_history";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const execFileAsync = promisify(execFile);
const projectRoot = path.join(process.cwd(), "..");
const runnerPath = path.join(projectRoot, "prototype", "run_prototype.py");
const historyPath = path.join(projectRoot, "prototype", "data", "audit", "prototype_run_history.jsonl");

let running = false;

export async function POST(request: Request) {
  if (running) {
    return NextResponse.json({ error: "Prototype is already running" }, { 
      status: 409,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
      }
    });
  }

  const body = await request.json().catch(() => ({}));
  const runModels = body.runModels === true;

  running = true;
  const startedAt = new Date().toISOString();

  try {
    const args = [runnerPath];
    if (runModels) {
      args.push("--run-models");
    }

    const result = await execFileAsync("python", args, {
      cwd: projectRoot,
      timeout: 15 * 60 * 1000,
      maxBuffer: 1024 * 1024 * 10,
      windowsHide: true
    });

    const finishedAt = new Date().toISOString();
    await appendRunHistory(historyPath, {
      market: "prototype",
      startedAt,
      finishedAt,
      status: "OK",
      stdout: result.stdout,
      stderr: result.stderr,
      code: 0,
      signal: null,
      runModels
    });

    return NextResponse.json({
      status: "OK",
      startedAt,
      finishedAt,
      stdout: result.stdout,
      stderr: result.stderr
    }, {
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
      }
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
      market: "prototype",
      startedAt,
      finishedAt,
      status: "ERROR",
      stdout: detail.stdout ?? "",
      stderr: detail.stderr ?? "",
      error: detail.message,
      code: detail.code ?? null,
      signal: detail.signal ?? null,
      runModels
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
      { 
        status: 500,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type"
        }
      }
    );
  } finally {
    running = false;
  }
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
