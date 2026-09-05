import assert from "assert";
import { handleLiveAction } from "./app/api/prototype/live/route";

async function runTests() {
  console.log("Running Next.js API Route unit tests on real exported handleLiveAction...");

  // Mock common options defaults
  const baseOptions = {
    currentPid: null,
    spawnFn: (() => ({ unref: () => {} })) as any,
    execAsyncFn: (async () => ({ stdout: "", stderr: "" })) as any,
    getServicePidFn: async () => null,
    verifyStartedFn: async () => "12345",
    verifyStoppedFn: async () => true,
    liveServicePath: "/path/to/service.py",
    projectRoot: "/path/to/root"
  };

  // --- START ACTION TESTS ---

  // Test Case 1: Start success (spawn success, PID found)
  {
    const res = await handleLiveAction("start", {
      ...baseOptions,
      currentPid: null,
      verifyStartedFn: async () => "12345"
    });
    assert.strictEqual(res.status, "OK");
    assert.strictEqual((res as any).pid, "12345");
    console.log("PASS: Test Case 1: Start success returns success with PID");
  }

  // Test Case 2: Start fails (spawn succeeds but verifyStarted returns null PID)
  {
    const res = await handleLiveAction("start", {
      ...baseOptions,
      currentPid: null,
      verifyStartedFn: async () => null
    });
    assert.strictEqual(res.status, 500);
    assert.strictEqual(res.error, "Failed to start service");
    assert.strictEqual((res as any).message, "Process start was not successful");
    console.log("PASS: Test Case 2: Start failure returns 500 error code");
  }

  // Test Case 3: Start fails because already running
  {
    const res = await handleLiveAction("start", {
      ...baseOptions,
      currentPid: "5555"
    });
    assert.strictEqual(res.status, 409);
    assert.strictEqual(res.error, "Service is already running");
    assert.strictEqual((res as any).pid, "5555");
    console.log("PASS: Test Case 3: Already running start fails with 409");
  }

  // Test Case 7: Start fails due to spawn exception (e.g. executable not found)
  {
    const res = await handleLiveAction("start", {
      ...baseOptions,
      currentPid: null,
      spawnFn: (() => { throw new Error("spawn ENOENT"); }) as any
    });
    assert.strictEqual(res.status, 500);
    assert.strictEqual(res.error, "Failed to start service");
    assert.ok(res.details && res.details.includes("spawn ENOENT"));
    console.log("PASS: Test Case 7: Start fails with 500 on spawn exception");
  }

  // --- STOP ACTION TESTS ---

  // Test Case 4: Stop success (process is killed and disappears within retry period)
  {
    let execCalled = false;
    const res = await handleLiveAction("stop", {
      ...baseOptions,
      currentPid: "12345",
      execAsyncFn: (async (cmd: string) => {
        execCalled = true;
        return { stdout: "", stderr: "" };
      }) as any,
      verifyStoppedFn: async () => true // Process disappeared!
    });
    assert.strictEqual(res.status, "OK");
    assert.strictEqual(res.message, "All service processes stopped");
    assert.strictEqual(execCalled, true);
    console.log("PASS: Test Case 4: Stop success returns success when process is killed");
  }

  // Test Case 5: Stop fails (process is still running after all retries)
  {
    const res = await handleLiveAction("stop", {
      ...baseOptions,
      currentPid: "12345",
      verifyStoppedFn: async () => false // Process still active!
    });
    assert.strictEqual(res.status, 500);
    assert.strictEqual(res.error, "Failed to stop service");
    assert.strictEqual((res as any).message, "Service process still running after stop request");
    console.log("PASS: Test Case 5: Stop failure returns 500 when process persists");
  }

  // Test Case 6: Stop returns OK immediately when process is already stopped/gone
  {
    const res = await handleLiveAction("stop", {
      ...baseOptions,
      execAsyncFn: (async () => { throw new Error("Some kill error"); }) as any,
      getServicePidFn: async () => null // Process already gone!
    });
    assert.strictEqual(res.status, "OK");
    assert.strictEqual(res.message, "Service already stopped");
    console.log("PASS: Test Case 6: Stop returns OK when service is already stopped");
  }

  console.log("All Next.js API Route tests passed successfully!");
}

runTests().catch(err => {
  console.error("Test execution failed:", err);
  process.exit(1);
});
