// The parts every screen shares: the sidebar (what am I comparing or learning), the page
// header with breadcrumbs, the jump palette and the run sheet.
import { useQuery } from "@tanstack/react-query"
import { Link, Outlet, useNavigate, useParams } from "@tanstack/react-router"
import { Monitor, Moon, Search, Sun } from "lucide-react"
import { cloneElement, Fragment, useEffect, useState, type ReactElement, type ReactNode } from "react"

import { RunSheet } from "@/components/run-sheet"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { Kbd } from "@/components/ui/kbd"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { kindsQuery, runsQuery, suitesQuery, tasksQuery, worldsQuery } from "@/lib/api"
import { useTheme } from "@/lib/theme"
import { useHidden, useScopeRuns } from "@/lib/view"

export function Frame() {
  const [jump, setJump] = useState(false)
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        setJump((j) => !j)
      }
    }
    window.addEventListener("keydown", on)
    return () => window.removeEventListener("keydown", on)
  }, [])
  return (
    <SidebarProvider>
      <AppSidebar onJump={() => setJump(true)} />
      <SidebarInset className="min-w-0">
        <Outlet />
      </SidebarInset>
      <Jump open={jump} onOpenChange={setJump} />
      <RunSheet />
    </SidebarProvider>
  )
}

/** The top of a screen: the sidebar toggle, breadcrumbs, and the screen's own controls. */
/** `to` returns the crumb's link without children; the label is put inside it. */
export function PageHeader({ crumbs, children }: { crumbs: { label: ReactNode; to?: () => ReactElement<{ children?: ReactNode }> }[]; children?: ReactNode }) {
  return (
    <header className="sticky top-0 z-10 flex h-14 shrink-0 items-center gap-3 border-b bg-background/95 px-4 backdrop-blur supports-backdrop-filter:bg-background/80">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="h-4!" />
      <Breadcrumb className="min-w-0">
        <BreadcrumbList className="flex-nowrap">
          {crumbs.map((c, i) => (
            <Fragment key={i}>
              {i > 0 && <BreadcrumbSeparator />}
              <BreadcrumbItem className="min-w-0">
                {i === crumbs.length - 1 || !c.to ? (
                  <BreadcrumbPage className="truncate">{c.label}</BreadcrumbPage>
                ) : (
                  <BreadcrumbLink asChild>{cloneElement(c.to(), { children: c.label })}</BreadcrumbLink>
                )}
              </BreadcrumbItem>
            </Fragment>
          ))}
        </BreadcrumbList>
      </Breadcrumb>
      <div className="ml-auto flex items-center gap-2">{children}</div>
    </header>
  )
}

function useScope() {
  const p = useParams({ strict: false }) as { suite?: string; split?: string; kind?: string }
  return { suite: p.suite ?? "1a", split: p.split ?? "dev", kind: p.kind }
}

function AppSidebar({ onJump }: { onJump: () => void }) {
  const { suite, split, kind } = useScope()
  const navigate = useNavigate()
  const { data: runs = [] } = useQuery(runsQuery)
  const { data: suites = [] } = useQuery(suitesQuery)
  const { data: worlds = [] } = useQuery(worldsQuery)
  const { data: kinds } = useQuery(kindsQuery(suite))
  const { all, color } = useScopeRuns(suite, split)
  const hidden = useHidden()
  const path = useParams({ strict: false }) as { task?: string; world?: string }
  const onOverview = !kind && !path.task && !path.world

  // Scopes with runs, then each suite's dev split for reading its tasks with no run.
  const scopes = [...new Set([...runs.filter((r) => r.suite).map((r) => `${r.suite}/${r.split}`), ...suites.map((s) => `${s}/dev`)])]

  const setHidden = (name: string, hide: boolean) => {
    const next = new Set(hidden)
    if (hide) next.add(name)
    else next.delete(name)
    navigate({ to: ".", search: (s: Record<string, unknown>) => ({ ...s, hide: next.size ? [...next].join(",") : undefined }) })
  }

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="gap-3 p-3">
        <div className="px-1 pt-1 text-sm font-semibold tracking-tight group-data-[collapsible=icon]:hidden">wa-bench view</div>
        <Select
          value={`${suite}/${split}`}
          onValueChange={(v) => {
            const [s, sp] = v.split("/")
            navigate({ to: "/$suite/$split", params: { suite: s, split: sp }, search: { hide: undefined } })
          }}
        >
          <SelectTrigger className="w-full group-data-[collapsible=icon]:hidden" aria-label="Suite and split">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {scopes.map((s) => (
              <SelectItem key={s} value={s}>
                Suite {s.replace("/", " · ")}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarMenu>
            <SidebarMenuItem>
              <SidebarMenuButton asChild isActive={onOverview} tooltip="Overview">
                <Link to="/$suite/$split" params={{ suite, split }}>
                  <span className="grid size-4 place-items-center text-xs">≡</span>
                  <span>Overview</span>
                </Link>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </SidebarMenu>
        </SidebarGroup>
        {worlds.length > 0 && (
          <SidebarGroup>
            <SidebarGroupLabel>Worlds</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {worlds.map((w) => (
                  <SidebarMenuItem key={w.name}>
                    <SidebarMenuButton asChild isActive={path.world === w.name} tooltip={w.name}>
                      <Link to="/world/$world" params={{ world: w.name }}>
                        <span className="grid size-4 place-items-center text-xs">◎</span>
                        <span className="truncate">{w.name}</span>
                        <span className="ml-auto text-xs text-muted-foreground tabular-nums">{w.rollouts}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
        <SidebarGroup>
          <SidebarGroupLabel>Task kinds</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {kinds?.kinds.map((k) => {
                const played = Object.values(k.rows).some((n) => n > 0)
                return (
                  <SidebarMenuItem key={k.kind}>
                    <SidebarMenuButton asChild isActive={kind === k.kind} tooltip={k.title} className="h-auto py-1.5">
                      <Link to="/$suite/$split/kind/$kind/tasks" params={{ suite, split, kind: k.kind }}>
                        <span className="hidden size-4 shrink-0 place-items-center text-[10px] font-semibold uppercase group-data-[collapsible=icon]:grid">
                          {k.title.slice(0, 2)}
                        </span>
                        <span className="flex min-w-0 flex-col leading-tight">
                          <span className={played ? "" : "text-muted-foreground"}>{k.title}</span>
                          <span className="truncate font-mono text-[11px] text-muted-foreground">{k.kind}</span>
                        </span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarGroup>
          <SidebarGroupLabel>Runs</SidebarGroupLabel>
          <SidebarGroupContent>
            {all.length === 0 ? (
              <p className="px-2 text-xs leading-relaxed text-muted-foreground group-data-[collapsible=icon]:hidden">
                No run loaded for this suite and split. Start the page with <code className="font-mono">--run runs/&lt;id&gt;</code> to
                see what models wrote.
              </p>
            ) : (
              <SidebarMenu>
                {all.map((r) => (
                  <SidebarMenuItem key={r.name} className="flex items-center gap-1">
                    <Checkbox
                      className="ml-2 group-data-[collapsible=icon]:hidden"
                      checked={!hidden.has(r.name)}
                      onCheckedChange={(c) => setHidden(r.name, c !== true)}
                      aria-label={`Show ${r.endpoint}`}
                    />
                    <SidebarMenuButton asChild tooltip={r.endpoint}>
                      <Link to="." search={(s: Record<string, unknown>) => ({ ...s, about: r.name })}>
                        <span className="size-3 shrink-0 rounded-sm" style={{ background: color(r.name) }} />
                        <span className={`truncate ${hidden.has(r.name) ? "text-muted-foreground line-through" : ""}`}>{r.endpoint}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="gap-2 p-3 group-data-[collapsible=icon]:hidden">
        <Button variant="outline" size="sm" className="justify-between text-muted-foreground" onClick={onJump}>
          <span className="flex items-center gap-2">
            <Search className="size-3.5" /> Jump to…
          </span>
          <Kbd>Ctrl K</Kbd>
        </Button>
        <ThemeSwitch />
      </SidebarFooter>
    </Sidebar>
  )
}

function ThemeSwitch() {
  const { choice, set } = useTheme()
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size="sm"
      value={choice}
      onValueChange={(v) => v && set(v as "light" | "dark" | "system")}
      className="w-full"
    >
      <ToggleGroupItem value="light" aria-label="Light" className="flex-1">
        <Sun />
      </ToggleGroupItem>
      <ToggleGroupItem value="dark" aria-label="Dark" className="flex-1">
        <Moon />
      </ToggleGroupItem>
      <ToggleGroupItem value="system" aria-label="As the system" className="flex-1">
        <Monitor />
      </ToggleGroupItem>
    </ToggleGroup>
  )
}

function Jump({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const { suite, split } = useScope()
  const navigate = useNavigate()
  const { data: kinds } = useQuery(kindsQuery(suite))
  const { data: tasks } = useQuery({ ...tasksQuery(suite, false), enabled: open })
  const { all } = useScopeRuns(suite, split)
  const go = (f: () => void) => {
    onOpenChange(false)
    f()
  }
  const title = (k: string) => kinds?.kinds.find((x) => x.kind === k)?.title ?? k
  return (
    <CommandDialog open={open} onOpenChange={onOpenChange} title="Jump to" description="A task kind, a task or a run">
      <CommandInput placeholder="Task id, template, POV, author, kind or run…" />
      <CommandList>
        <CommandEmpty>Nothing matches.</CommandEmpty>
        <CommandGroup heading="Task kinds">
          {kinds?.kinds.map((k) => (
            <CommandItem
              key={k.kind}
              value={`${k.title} ${k.kind}`}
              onSelect={() => go(() => navigate({ to: "/$suite/$split/kind/$kind/tasks", params: { suite, split, kind: k.kind } }))}
            >
              {k.title}
              <span className="ml-auto font-mono text-xs text-muted-foreground">{k.kind}</span>
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandGroup heading="Runs">
          {all.map((r) => (
            <CommandItem key={r.name} value={`run ${r.name}`} onSelect={() => go(() => navigate({ to: ".", search: (s: Record<string, unknown>) => ({ ...s, about: r.name }) }))}>
              {r.endpoint}
              <span className="ml-auto font-mono text-xs text-muted-foreground">{r.name}</span>
            </CommandItem>
          ))}
        </CommandGroup>
        <CommandGroup heading={`Tasks · ${suite} ${split}`}>
          {tasks?.tasks
            .filter((t) => t.split === split)
            .map((t) => (
              <CommandItem
                key={t.task_id}
                value={`${t.task_id} ${Object.values(t.row).join(" ")}`}
                onSelect={() => go(() => navigate({ to: "/$suite/$split/task/$task", params: { suite, split, task: t.task_id } }))}
              >
                <span className="truncate font-mono text-xs">{t.task_id}</span>
                <span className="ml-auto shrink-0 text-xs text-muted-foreground">{title(t.kind)}</span>
              </CommandItem>
            ))}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  )
}
