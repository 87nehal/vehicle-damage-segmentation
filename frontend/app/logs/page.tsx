import type { Metadata } from "next"

import { TelemetryWorkspace } from "@/components/telemetry-workspace"

export const metadata: Metadata = {
  title: "Telemetry",
}

export default function LogsPage() {
  return <TelemetryWorkspace />
}
