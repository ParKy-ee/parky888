import { execFile } from "child_process";
import path from "path";
import { promisify } from "util";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const execFileAsync = promisify(execFile);
const projectRoot = path.join(process.cwd(), "..");
const auditScriptPath = path.join(projectRoot, "prototype", "audit_paper_trade_quality.py");

export async function POST() {
  const startedAt = new Date().toISOString();

  try {
    const result = await execFileAsync("python", [auditScriptPath], {
      cwd: projectRoot,
      timeout: 60 * 1000, // 1 minute
      maxBuffer: 1024 * 1024 * 5,
      windowsHide: true
    });

    return NextResponse.json({
      status: "OK",
      startedAt,
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

    return NextResponse.json(
      {
        status: "ERROR",
        startedAt,
        finishedAt: new Date().toISOString(),
        error: detail.message,
        code: detail.code ?? null,
        signal: detail.signal ?? null,
        stdout: detail.stdout ?? "",
        stderr: detail.stderr ?? ""
      },
      { status: 500 }
    );
  }
}
