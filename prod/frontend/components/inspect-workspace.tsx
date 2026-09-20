"use client"

import * as React from "react"
import { Camera01Icon, Upload01Icon } from "@hugeicons/core-free-icons"
import { HugeiconsIcon } from "@hugeicons/react"
import { toast } from "sonner"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  getHealth,
  predictDamageFile,
  type DamageDetection,
  type DamageReport,
  type HealthResponse,
} from "@/lib/api"

const CLASS_COLORS: Record<string, string> = {
  dent: "#ea580c",
  scratch: "#0284c7",
  crack_or_breakage: "#dc2626",
  paint_damage: "#ca8a04",
  deformation_or_detachment: "#9333ea",
}

const DECISION_LABELS: Record<DamageReport["decision"], string> = {
  damage_detected: "Damage detected",
  no_damage_detected: "No damage detected",
  manual_review_required: "Manual review required",
  recapture_required: "Recapture required",
}

function humanize(value: string) {
  return value.replaceAll("_", " ").replaceAll(":", ": ")
}

function downloadBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = name
  anchor.click()
  URL.revokeObjectURL(url)
}

export function InspectWorkspace() {
  const [health, setHealth] = React.useState<HealthResponse | null>(null)
  const [result, setResult] = React.useState<DamageReport | null>(null)
  const [busy, setBusy] = React.useState(false)
  const [drag, setDrag] = React.useState(false)
  const [error, setError] = React.useState("")
  const [healthError, setHealthError] = React.useState("")
  const [filename, setFilename] = React.useState("")
  const fileRef = React.useRef<HTMLInputElement>(null)

  React.useEffect(() => {
    let active = true
    async function refreshHealth() {
      try {
        const nextHealth = await getHealth()
        if (!active) return
        setHealth(nextHealth)
        setHealthError("")
      } catch (err) {
        if (!active) return
        setHealth(null)
        const detail = err instanceof Error ? err.message : "API unreachable"
        setHealthError(
          `Model API unavailable. Restart the app with npm run dev. ${detail}`
        )
      }
    }
    void refreshHealth()
    const interval = window.setInterval(refreshHealth, 5000)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [])

  async function runFile(file?: File) {
    if (!file) return
    if (!file.type.startsWith("image/")) {
      setError("Choose a JPEG, PNG, or WebP image.")
      return
    }
    setBusy(true)
    setError("")
    setResult(null)
    setFilename(file.name)
    try {
      const data = await predictDamageFile(file)
      setResult(data)
      if (data.recapture_required) toast.error("Recapture required")
      else if (data.manual_review_required)
        toast.warning("Manual review required")
      else toast.success("Inspection complete")
    } catch (err) {
      const message = err instanceof Error ? err.message : "Inspection failed"
      setError(message)
      toast.error(message)
    } finally {
      setBusy(false)
    }
  }

  React.useEffect(() => {
    function onPaste(event: ClipboardEvent) {
      const item = [...(event.clipboardData?.items ?? [])].find((entry) =>
        entry.type.startsWith("image/")
      )
      const file = item?.getAsFile()
      if (file) void runFile(file)
    }
    document.addEventListener("paste", onPaste)
    return () => document.removeEventListener("paste", onPaste)
  }, [])

  const modelReady = Boolean(health?.damage?.ok && health.damage.model_loaded)
  const findings = result
    ? [...result.detections, ...result.review_candidates]
    : []
  const outputStem = filename.replace(/\.[^.]+$/, "") || "inspection"

  function downloadReport() {
    if (!result) return
    const report = { ...result }
    delete report.annotated_jpeg_b64
    downloadBlob(
      new Blob([JSON.stringify(report, null, 2)], {
        type: "application/json",
      }),
      `${outputStem}-damage-report.json`
    )
  }

  function downloadOverlay() {
    if (!result?.annotated_jpeg_b64) return
    const bytes = atob(result.annotated_jpeg_b64)
    const payload = Uint8Array.from(bytes, (character) =>
      character.charCodeAt(0)
    )
    downloadBlob(
      new Blob([payload], { type: "image/jpeg" }),
      `${outputStem}-overlay.jpg`
    )
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[minmax(0,23rem)_minmax(0,1fr)]">
      <div className="space-y-6">
        <Alert>
          <AlertTitle>Development model</AlertTitle>
          <AlertDescription>
            This checkpoint is for supervised testing only. It has not passed
            the clean-vehicle release gates and must not make insurance or
            repair decisions.
          </AlertDescription>
        </Alert>

        <Card>
          <CardHeader>
            <CardTitle>Source image</CardTitle>
            <CardDescription>
              Upload one exterior vehicle photo. Frozen thresholds and safety
              routing are applied automatically.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <button
              type="button"
              disabled={busy || !modelReady}
              onClick={() => fileRef.current?.click()}
              onDragOver={(event) => {
                event.preventDefault()
                setDrag(true)
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDrag(false)
                void runFile(event.dataTransfer.files[0])
              }}
              className={`flex min-h-40 w-full flex-col items-center justify-center rounded-2xl border border-dashed px-4 py-8 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                drag
                  ? "border-foreground bg-muted/60"
                  : "border-border bg-muted/30"
              }`}
            >
              <HugeiconsIcon
                icon={Upload01Icon}
                strokeWidth={2}
                className="mb-2 size-5 text-muted-foreground"
              />
              <p className="text-sm font-medium">
                Drop an image or click to browse
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                JPEG, PNG, or WebP - maximum 20 MB - paste supported
              </p>
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(event) => void runFile(event.target.files?.[0])}
            />

            <div className="rounded-xl border bg-muted/20 p-3 text-xs text-muted-foreground">
              <div className="flex items-center justify-between gap-3">
                <span>Model</span>
                <Badge variant={modelReady ? "secondary" : "destructive"}>
                  {modelReady ? "Loaded on GPU" : "Unavailable"}
                </Badge>
              </div>
              <div className="mt-2 flex items-center justify-between gap-3">
                <span>Frozen thresholds</span>
                <span className="font-mono">
                  {health?.damage?.triage_threshold?.toFixed(2) ?? "-"} /{" "}
                  {health?.damage?.segmentation_threshold?.toFixed(2) ?? "-"}
                </span>
              </div>
              {filename ? (
                <p className="mt-2 truncate">File: {filename}</p>
              ) : null}
            </div>

            {error || healthError ? (
              <Alert variant="destructive">
                <AlertTitle>
                  {error ? "Inspection failed" : "Model API unavailable"}
                </AlertTitle>
                <AlertDescription>{error || healthError}</AlertDescription>
              </Alert>
            ) : null}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Model findings</CardTitle>
          <CardDescription>
            Solid class colors are precise-mask regions. Amber regions are
            recall-only evidence that requires review.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-3">
            <Metric
              label="Decision"
              value={result ? DECISION_LABELS[result.decision] : "-"}
              tone={result ? decisionTone(result.decision) : "idle"}
            />
            <Metric
              label="Recall-first alert"
              value={
                result ? (result.triage_alert ? "Positive" : "Clear") : "-"
              }
              tone={result ? (result.triage_alert ? "bad" : "ok") : "idle"}
            />
            <Metric
              label="Precise regions"
              value={result ? String(result.n_detections) : "-"}
              tone={result ? (result.n_detections ? "bad" : "ok") : "idle"}
            />
          </div>

          {result ? (
            <Alert
              variant={result.recapture_required ? "destructive" : "default"}
            >
              <AlertTitle>{DECISION_LABELS[result.decision]}</AlertTitle>
              <AlertDescription>
                <p>{result.reason}</p>
                {result.decision_reasons.length ? (
                  <p className="mt-2">
                    Reasons: {result.decision_reasons.map(humanize).join(", ")}
                  </p>
                ) : null}
              </AlertDescription>
            </Alert>
          ) : null}

          {result ? (
            <div className="flex flex-wrap items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={downloadOverlay}
              >
                Download overlay
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={downloadReport}
              >
                Download JSON report
              </Button>
              <span className="ml-auto text-xs text-muted-foreground">
                {result.runtime.elapsed_ms.toFixed(1)} ms on{" "}
                {result.runtime.device.toUpperCase()}
              </span>
            </div>
          ) : null}

          <div className="overflow-hidden rounded-2xl border bg-muted/20">
            {result?.annotated_jpeg_b64 ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                alt="Vehicle with damage segmentation overlay"
                src={`data:image/jpeg;base64,${result.annotated_jpeg_b64}`}
                className="mx-auto max-h-[34rem] w-full object-contain"
              />
            ) : busy ? (
              <Skeleton className="h-80 w-full rounded-none" />
            ) : (
              <div className="flex h-72 flex-col items-center justify-center gap-2 text-muted-foreground">
                <HugeiconsIcon
                  icon={Camera01Icon}
                  strokeWidth={2}
                  className="size-6"
                />
                <p className="text-sm">No image inspected</p>
              </div>
            )}
          </div>

          {result?.quality.review_reasons.length ? (
            <div className="flex flex-wrap gap-2">
              {result.quality.review_reasons.map((reason) => (
                <Badge key={reason} variant="outline">
                  {humanize(reason)}
                </Badge>
              ))}
            </div>
          ) : null}

          <Separator />
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Class</TableHead>
                <TableHead>Evidence</TableHead>
                <TableHead>Score</TableHead>
                <TableHead>Area</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {!findings.length ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-muted-foreground">
                    {result ? "No regions passed the frozen thresholds" : "-"}
                  </TableCell>
                </TableRow>
              ) : (
                findings.map((detection, index) => (
                  <FindingRow
                    key={`${detection.kind}-${detection.class_name}-${index}`}
                    detection={detection}
                  />
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}

function decisionTone(
  decision: DamageReport["decision"]
): "ok" | "bad" | "warn" {
  if (decision === "no_damage_detected") return "ok"
  if (decision === "manual_review_required") return "warn"
  return "bad"
}

function FindingRow({ detection }: { detection: DamageDetection }) {
  return (
    <TableRow>
      <TableCell>
        <span className="inline-flex items-center gap-2 capitalize">
          <span
            className="size-2 rounded-full"
            style={{
              background:
                detection.kind === "triage_only"
                  ? "#f59e0b"
                  : CLASS_COLORS[detection.class_name] || "#64748b",
            }}
          />
          {humanize(detection.class_name)}
        </span>
      </TableCell>
      <TableCell>
        <Badge variant="outline">
          {detection.kind === "triage_only" ? "Review only" : "Precise mask"}
        </Badge>
      </TableCell>
      <TableCell className="font-mono tabular-nums">
        {detection.score.toFixed(3)}
      </TableCell>
      <TableCell className="font-mono tabular-nums">
        {detection.area_frac == null
          ? "-"
          : `${(detection.area_frac * 100).toFixed(2)}%`}
      </TableCell>
    </TableRow>
  )
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone: "ok" | "bad" | "warn" | "idle"
}) {
  return (
    <div className="rounded-2xl border bg-muted/20 p-4">
      <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </p>
      <p className="mt-1 text-lg font-semibold tracking-tight">{value}</p>
      {tone !== "idle" ? (
        <Badge
          variant={tone === "bad" ? "destructive" : "secondary"}
          className="mt-2"
        >
          {tone === "ok"
            ? "Clear"
            : tone === "warn"
              ? "Human check"
              : "Action required"}
        </Badge>
      ) : null}
    </div>
  )
}
