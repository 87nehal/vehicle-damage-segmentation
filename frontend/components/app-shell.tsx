"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Analytics01Icon,
  Camera01Icon,
  Car01Icon,
  CpuIcon,
  Menu01Icon,
  Moon02Icon,
  Sun01Icon,
} from "@hugeicons/core-free-icons"
import { useTheme } from "next-themes"

import { getHealth, type HealthResponse } from "@/lib/api"
import { cn } from "@/lib/utils"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"

const NAV = [
  {
    href: "/",
    label: "Inspection",
    description: "Body damage",
    icon: Camera01Icon,
  },
  {
    href: "/logs",
    label: "Telemetry",
    description: "ECU health logs",
    icon: Analytics01Icon,
  },
]

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname()
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map((item) => {
        const active = pathname === item.href
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            className={cn(
              buttonVariants({ variant: active ? "secondary" : "ghost" }),
              "h-auto w-full justify-start gap-3 rounded-2xl px-3 py-2.5"
            )}
          >
            <HugeiconsIcon
              icon={item.icon}
              strokeWidth={2}
              className="size-4 shrink-0"
            />
            <span className="flex min-w-0 flex-col items-start text-left">
              <span className="text-sm font-medium">{item.label}</span>
              <span className="text-xs font-normal text-muted-foreground">
                {item.description}
              </span>
            </span>
          </Link>
        )
      })}
    </nav>
  )
}

function StatusBadges({ health }: { health: HealthResponse | null }) {
  if (!health) {
    return <Badge variant="outline">Connecting</Badge>
  }
  if (health.error) {
    return <Badge variant="destructive">API unreachable</Badge>
  }
  const inspectReady = Boolean(health.damage?.ok && health.damage?.model_loaded)
  const logsReady = Boolean(health.telemetry?.ok)
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge variant={inspectReady ? "secondary" : "destructive"}>
        <HugeiconsIcon
          icon={Camera01Icon}
          strokeWidth={2}
          data-icon="inline-start"
        />
        Inspection{" "}
        {inspectReady ? (health.damage?.cuda ? "GPU" : "CPU") : "offline"}
      </Badge>
      <Badge variant={logsReady ? "secondary" : "destructive"}>
        <HugeiconsIcon
          icon={CpuIcon}
          strokeWidth={2}
          data-icon="inline-start"
        />
        Telemetry {logsReady ? "ready" : "offline"}
      </Badge>
    </div>
  )
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const mounted = React.useSyncExternalStore(
    () => () => undefined,
    () => true,
    () => false
  )
  if (!mounted) {
    return <Button variant="outline" size="icon" aria-label="Toggle theme" />
  }
  const dark = resolvedTheme === "dark"
  return (
    <Button
      variant="outline"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(dark ? "light" : "dark")}
    >
      <HugeiconsIcon icon={dark ? Sun01Icon : Moon02Icon} strokeWidth={2} />
    </Button>
  )
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-3 px-1">
      <span className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground">
        <HugeiconsIcon icon={Car01Icon} strokeWidth={2} className="size-4" />
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold tracking-tight">
          Car Health
        </span>
        <span className="block text-xs text-muted-foreground">
          Vehicle intelligence
        </span>
      </span>
    </Link>
  )
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [health, setHealth] = React.useState<HealthResponse | null>(null)
  const [open, setOpen] = React.useState(false)
  const pathname = usePathname()
  const current = NAV.find((item) => item.href === pathname) ?? NAV[0]

  React.useEffect(() => {
    let active = true
    async function refreshHealth() {
      try {
        const nextHealth = await getHealth()
        if (active) setHealth(nextHealth)
      } catch (err) {
        if (!active) return
        const message = err instanceof Error ? err.message : "API unreachable"
        setHealth({ ok: false, error: message })
      }
    }
    void refreshHealth()
    const interval = window.setInterval(refreshHealth, 5000)
    return () => {
      active = false
      window.clearInterval(interval)
    }
  }, [])

  return (
    <div className="min-h-svh bg-background">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 border-r bg-sidebar md:flex md:flex-col">
        <div className="px-4 py-5">
          <Brand />
        </div>
        <Separator />
        <div className="flex-1 overflow-y-auto p-3">
          <p className="mb-2 px-3 text-[11px] font-medium tracking-[0.14em] text-muted-foreground uppercase">
            Workspace
          </p>
          <NavLinks />
        </div>
        <Separator />
        <div className="flex items-center justify-between gap-2 p-4">
          <span className="text-xs text-muted-foreground">Appearance</span>
          <ThemeToggle />
        </div>
      </aside>

      <div className="md:pl-64">
        <header className="sticky top-0 z-20 border-b bg-background">
          <div className="flex items-center gap-3 px-4 py-3 md:px-6">
            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger
                render={
                  <Button variant="outline" size="icon" className="md:hidden" />
                }
              >
                <HugeiconsIcon icon={Menu01Icon} strokeWidth={2} />
                <span className="sr-only">Open menu</span>
              </SheetTrigger>
              <SheetContent side="left" className="w-72 p-0">
                <SheetHeader className="border-b">
                  <SheetTitle className="sr-only">Navigation</SheetTitle>
                  <Brand />
                </SheetHeader>
                <div className="p-3">
                  <NavLinks onNavigate={() => setOpen(false)} />
                </div>
              </SheetContent>
            </Sheet>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-medium tracking-[0.14em] text-muted-foreground uppercase">
                {current.label}
              </p>
              <h1 className="truncate text-base font-semibold tracking-tight md:text-lg">
                {current.href === "/"
                  ? "Damage inspection"
                  : "Telemetry analysis"}
              </h1>
            </div>
            <div className="hidden sm:block">
              <StatusBadges health={health} />
            </div>
            <div className="md:hidden">
              <ThemeToggle />
            </div>
          </div>
        </header>
        <main className="px-4 py-6 md:px-6 md:py-8">{children}</main>
      </div>
    </div>
  )
}
