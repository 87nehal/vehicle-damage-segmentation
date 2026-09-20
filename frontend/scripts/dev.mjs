import { spawn } from "node:child_process"
import path from "node:path"
import process from "node:process"
import { fileURLToPath } from "node:url"

const scriptsDir = path.dirname(fileURLToPath(import.meta.url))
const frontendDir = path.resolve(scriptsDir, "..")
const repositoryRoot = path.resolve(frontendDir, "..")
const apiHealthUrl = "http://127.0.0.1:8001/api/health"
const python = process.env.PYTHON_EXECUTABLE || "python"
const nextBin = path.join(
  frontendDir,
  "node_modules",
  "next",
  "dist",
  "bin",
  "next"
)

let apiProcess = null
let nextProcess = null
let stopping = false

async function apiIsReady() {
  try {
    const response = await fetch(apiHealthUrl, {
      signal: AbortSignal.timeout(1500),
      cache: "no-store",
    })
    if (!response.ok) return false
    const payload = await response.json()
    return Boolean(payload?.damage?.ok && payload?.damage?.model_loaded)
  } catch {
    return false
  }
}

async function waitForApi() {
  for (let attempt = 0; attempt < 180; attempt += 1) {
    if (await apiIsReady()) return
    if (apiProcess && apiProcess.exitCode !== null) {
      throw new Error(`model API exited with code ${apiProcess.exitCode}`)
    }
    await new Promise((resolve) => setTimeout(resolve, 500))
  }
  throw new Error("model API did not become ready within 90 seconds")
}

function terminate(child) {
  if (child && child.exitCode === null) child.kill()
}

function shutdown(code = 0) {
  if (stopping) return
  stopping = true
  terminate(nextProcess)
  terminate(apiProcess)
  setTimeout(() => process.exit(code), 1000).unref()
}

async function main() {
  if (await apiIsReady()) {
    console.log("[dev] Reusing model API on http://127.0.0.1:8001")
  } else {
    console.log("[dev] Starting hash-verified model API...")
    apiProcess = spawn(python, ["-m", "vehicle_damage.api"], {
      cwd: repositoryRoot,
      env: {
        ...process.env,
        PYTHONPATH: [path.join(repositoryRoot, "src"), process.env.PYTHONPATH]
          .filter(Boolean)
          .join(path.delimiter),
      },
      stdio: "inherit",
      windowsHide: true,
    })
    apiProcess.on("error", (error) => {
      console.error(`[dev] Unable to start Python: ${error.message}`)
      shutdown(1)
    })
    await waitForApi()
    console.log("[dev] Model API ready")
  }

  nextProcess = spawn(process.execPath, [nextBin, "dev", "--webpack"], {
    cwd: frontendDir,
    env: process.env,
    stdio: "inherit",
    windowsHide: true,
  })
  nextProcess.on("error", (error) => {
    console.error(`[dev] Unable to start Next.js: ${error.message}`)
    shutdown(1)
  })
  nextProcess.on("exit", (code) => shutdown(code ?? 0))
  apiProcess?.on("exit", (code) => {
    if (!stopping) {
      console.error(`[dev] Model API stopped unexpectedly with code ${code}`)
      shutdown(code ?? 1)
    }
  })
}

process.on("SIGINT", () => shutdown(0))
process.on("SIGTERM", () => shutdown(0))

main().catch((error) => {
  console.error(
    `[dev] ${error instanceof Error ? error.message : String(error)}`
  )
  shutdown(1)
})
