export type HealthResponse = {
  ok: boolean
  telemetry?: {
    ok?: boolean
    model_exists?: boolean
    error?: string | null
  }
  damage?: {
    ok?: boolean
    weights_exist?: boolean
    model_loaded?: boolean
    cuda?: boolean
    device?: string
    model_status?: string
    production_approved?: boolean
    triage_threshold?: number
    segmentation_threshold?: number
    error?: string | null
  }
  error?: string
}

export type DamageSample = {
  id: string
  file: string
  label: string
  expected_damage: boolean
  url: string
}

export type DamageDetection = {
  class_name: string
  score: number
  bbox?: number[] | null
  area_frac?: number | null
  area_pixels?: number
  kind: "segmentation" | "triage_only"
}

export type DamageReport = {
  model_status: string
  production_approved: boolean
  decision:
    | "damage_detected"
    | "no_damage_detected"
    | "manual_review_required"
    | "recapture_required"
  automated_decision_allowed: boolean
  manual_review_required: boolean
  recapture_required: boolean
  decision_reasons: string[]
  recapture_reasons: string[]
  damage_present: boolean
  triage_alert: boolean
  high_confidence_alert: boolean
  reason: string
  n_detections: number
  detections: DamageDetection[]
  review_candidates: DamageDetection[]
  quality: {
    width: number
    height: number
    dark_fraction: number
    clipped_highlight_fraction: number
    specular_highlight_fraction: number
    sharpness: number
    review_reasons: string[]
  }
  runtime: { device: string; elapsed_ms: number }
  thresholds: { triage: number; segmentation: number }
  annotated_jpeg_b64?: string
}

export type TelemetrySample = {
  id: string
  label: string
  available: boolean
}

export type TelemetryReport = {
  has_issue: boolean
  needs_mechanic: boolean
  issue: string
  mechanic_needed: string
  issue_probability: number
  reason: string
  n_rows: number
  used_sensors: string[]
  used_sensor_labels?: string[]
  rule_hits?: string[]
  source: string
}

async function readJson<T>(res: Response): Promise<T> {
  const data = (await res.json().catch(() => ({}))) as Record<string, unknown>
  if (!res.ok) {
    const detail = data.detail ?? data.error ?? data.message ?? res.statusText
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail)
    )
  }
  return data as T
}

export async function getHealth() {
  const res = await fetch("/api/health")
  return readJson<HealthResponse>(res)
}

export async function getDamageSamples() {
  const res = await fetch("/api/damage/samples")
  return readJson<DamageSample[]>(res)
}

export async function predictDamageFile(file: File) {
  const fd = new FormData()
  fd.append("file", file)
  const res = await fetch("/api/damage/predict", {
    method: "POST",
    body: fd,
  })
  return readJson<DamageReport>(res)
}

export async function getTelemetrySamples() {
  const res = await fetch("/api/telemetry/samples")
  return readJson<TelemetrySample[]>(res)
}

export async function diagnoseTelemetry(input: {
  sample?: string
  csvText?: string
  file?: File
}) {
  if (input.file) {
    const fd = new FormData()
    fd.append("file", input.file)
    const res = await fetch("/api/telemetry/diagnose", {
      method: "POST",
      body: fd,
    })
    return readJson<TelemetryReport>(res)
  }
  const res = await fetch("/api/telemetry/diagnose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sample: input.sample,
      csv_text: input.csvText,
    }),
  })
  return readJson<TelemetryReport>(res)
}
