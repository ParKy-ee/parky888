import { exec, spawn } from "child_process";
import { promises as fs } from "fs";
import path from "path";
import { promisify } from "util";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const execAsync = promisify(exec);
const projectRoot = path.join(process.cwd(), "..");
const liveServicePath = path.join(projectRoot, "prototype", "run_live_service.py");
const livePidPath = path.join(projectRoot, "prototype", "data", "audit", "live_service.pid");

async function readPidFile(): Promise<number | null> {
  try {
    const raw = (await fs.readFile(livePidPath, "utf8")).trim();
    const pid = Number(raw);
    if (!Number.isInteger(pid) || pid <= 0) return null;
    return pid;
  } catch {
    return null;
  }
}

async function writePidFile(pid: number) {
  await fs.mkdir(path.dirname(livePidPath), { recursive: true });
  await fs.writeFile(livePidPath, String(pid), "utf8");
}

async function clearPidFile() {
  try {
    await fs.unlink(livePidPath);
  } catch {
    // ignore
  }
}

function isPidAlive(pid: number) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export async function getServicePid() {
  const filePid = await readPidFile();
  if (filePid && isPidAlive(filePid)) {
    return String(filePid);
  }
  if (filePid && !isPidAlive(filePid)) {
    await clearPidFile();
  }

  try {
    // Windows: Use powershell to find the process by command line
    const cmd = `powershell -Command "Get-CimInstance Win32_Process | Where-Object Name -eq 'python.exe' | Where-Object CommandLine -like '*run_live_service.py*' | Select-Object -ExpandProperty ProcessId"`;
    const { stdout } = await execAsync(cmd, { windowsHide: true });
    const pids = stdout.split(/\r?\n/).map(line => line.trim()).filter(line => /^\d+$/.test(line));
    if (pids.length > 0) {
      await writePidFile(Number(pids[0]));
      return pids[0];
    }
    return null;
  } catch {
    return null;
  }
}

export async function verifyServiceStarted(getServicePidFn = getServicePid): Promise<string | null> {
  // Wait a bit and check if it started
  await new Promise(resolve => setTimeout(resolve, 2000));
  return await getServicePidFn();
}

export async function verifyServiceStopped(getServicePidFn = getServicePid): Promise<boolean> {
  let isGone = false;
  for (let i = 0; i < 3; i++) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    const checkPid = await getServicePidFn();
    if (!checkPid) {
      isGone = true;
      break;
    }
  }
  return isGone;
}

export async function handleLiveAction(
  action: string,
  options: {
    currentPid: string | null;
    spawnFn: typeof spawn;
    execAsyncFn: typeof execAsync;
    getServicePidFn: typeof getServicePid;
    verifyStartedFn: typeof verifyServiceStarted;
    verifyStoppedFn: typeof verifyServiceStopped;
    liveServicePath: string;
    projectRoot: string;
  }
) {
  if (action === "start") {
    if (options.currentPid) {
      return { error: "Service is already running", pid: options.currentPid, status: 409 };
    }

    try {
      // Prefer pinned interpreter to avoid launching with a Python env that lacks MetaTrader5.
      const launchCandidates: Array<{ cmd: string; args: string[] }> = [
        { cmd: "py", args: ["-3.10", options.liveServicePath] },
        { cmd: "python", args: [options.liveServicePath] }
      ];

      let startedPid: string | null = null;
      let lastError: unknown = null;

      for (const candidate of launchCandidates) {
        try {
          const child = options.spawnFn(candidate.cmd, candidate.args, {
            cwd: options.projectRoot,
            detached: true,
            stdio: "ignore",
            windowsHide: true
          });
          child.unref();
          if (child.pid) {
            await writePidFile(child.pid);
          }

          startedPid = await options.verifyStartedFn(options.getServicePidFn);
          if (startedPid) break;
        } catch (error) {
          lastError = error;
        }
      }

      if (!startedPid) {
        return {
          error: "Failed to start service",
          message: "Process start was not successful (interpreter/env mismatch or immediate crash)",
          details: lastError ? String(lastError) : undefined,
          status: 500
        };
      }

      return {
        status: "OK",
        message: "Service started",
        pid: startedPid
      };
    } catch (error) {
      return {
        error: "Failed to start service",
        details: String(error),
        status: 500
      };
    }
  }

  if (action === "stop") {
    try {
      // Prefer PID file based stop, fallback to CIM query.
      if (options.currentPid) {
        try {
          process.kill(Number(options.currentPid));
        } catch {
          // fallback below
        }
      }

      const killCmd = `powershell -Command "Get-CimInstance Win32_Process | Where-Object Name -eq 'python.exe' | Where-Object CommandLine -like '*run_live_service.py*' | ForEach-Object { Stop-Process -Id \\$_.ProcessId -Force }"`;
      await options.execAsyncFn(killCmd, { windowsHide: true }).catch(() => undefined);
      
      const isGone = await options.verifyStoppedFn(options.getServicePidFn);

      if (!isGone) {
        return {
          error: "Failed to stop service",
          message: "Service process still running after stop request",
          status: 500
        };
      }

      await clearPidFile();
      return { status: "OK", message: "All service processes stopped" };
    } catch (error) {
      const pid = await options.getServicePidFn();
      if (!pid) return { status: "OK", message: "Service already stopped" };
      return { error: "Failed to stop service", details: String(error), status: 500 };
    }
  }

  return { error: "Invalid action", status: 400 };
}

export async function GET() {
  const pid = await getServicePid();
  return NextResponse.json({
    running: pid !== null,
    pid: pid
  }, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type"
    }
  });
}

export async function POST(request: Request) {
  const { action } = await request.json();
  const currentPid = await getServicePid();

  const result = await handleLiveAction(action, {
    currentPid,
    spawnFn: spawn,
    execAsyncFn: execAsync,
    getServicePidFn: getServicePid,
    verifyStartedFn: verifyServiceStarted,
    verifyStoppedFn: verifyServiceStopped,
    liveServicePath,
    projectRoot
  });

  if ("error" in result) {
    const { status, ...body } = result;
    return NextResponse.json(body, { 
      status: status as number,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
      }
    });
  }

  return NextResponse.json(result, {
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
