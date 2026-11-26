// Overview: per task kind, its outcome metrics per arm, one value per visible run. The numbers
// are `report`'s own (`build_report`); a metric no item was judged on shows as not judged.
import { useQueries, useQuery } from "@tanstack/react-query"
import { Link, useParams } from "@tanstack/react-router"
import { useState } from "react"

import { PageHeader } from "@/components/frame"
import { Button } from "@/components/ui/button"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { kindsQuery, reportQuery, type KindInfo, type ReportGroup } from "@/lib/api"
import { digits, fmt, isProfile, useScopeRuns, type Run } from "@/lib/view"

type Cell = { mean: number | null; n: number; total: number; absent?: string }

export function Overview() {
  const { suite, split } = useParams({ from: "/$suite/$split/" })
  const { visible, all, color } = useScopeRuns(suite, split)
  const { data: kinds } = useQuery(kindsQuery(suite))
  const reports = useQueries({ queries: visible.map((r) => reportQuery(r.name)) })
  const loading = reports.some((q) => q.isPending) || !kinds

  const header = <PageHeader crumbs={[{ label: `Suite ${suite} · ${split}` }, { label: "Overview" }]} />
  if (all.length === 0)
    return (
      <>
        {header}
        <Empty className="mt-16">
          <EmptyHeader>
            <EmptyTitle>No run to compare</EmptyTitle>
            <EmptyDescription>
              No run of suite {suite} on the {split} split is loaded. The task kinds in the sidebar list its tasks; start the page with{" "}
              <code className="font-mono">wa-bench view --run runs/&lt;id&gt;</code> to compare models on them.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </>
    )

  // kind -> arm -> metric -> run -> cell
  const byKind = new Map<string, Map<string, Map<string, Map<string, Cell>>>>()
  const notes = new Map<string, string[]>()
  reports.forEach((q, i) => {
    const run = visible[i]
    for (const g of q.data?.groups ?? []) collect(g, run, byKind, notes)
  })
  const order = (kinds?.kinds ?? []).map((k) => k.kind)
  const kindNames = [...byKind.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b))

  return (
    <>
      {header}
      <div className="mx-auto w-full max-w-6xl space-y-10 px-6 py-8">
        {visible
          .filter((r) => r.open)
          .map((r) => (
            <p key={r.name} className="text-sm text-muted-foreground">
              <span className="font-medium text-foreground">{r.endpoint}</span>:{" "}
              {r.open === 1 ? "1 episode is" : `${r.open} episodes are`} open and not shown here. A run is still playing, or was
              killed; running the same <code className="font-mono">wa-bench run</code> command again continues each from its last
              saved turn.
            </p>
          ))}
        {loading && <Skeleton className="h-64 w-full" />}
        {!loading && visible.length === 0 && (
          <p className="text-sm text-muted-foreground">Every run is hidden: tick one in the sidebar.</p>
        )}
        {!loading &&
          kindNames.map((k) => (
            <KindTable
              key={k}
              suite={suite}
              split={split}
              info={kinds?.kinds.find((x) => x.kind === k)}
              kind={k}
              arms={byKind.get(k)!}
              runs={visible}
              color={color}
              notes={notes.get(k) ?? []}
              about={kinds?.metrics ?? {}}
            />
          ))}
        {!loading && kinds && (
          <Unplayed kinds={kinds.kinds.filter((k) => !byKind.has(k.kind))} split={split} />
        )}
      </div>
    </>
  )
}

function collect(
  g: ReportGroup,
  run: Run,
  byKind: Map<string, Map<string, Map<string, Map<string, Cell>>>>,
  notes: Map<string, string[]>,
) {
  const { kind, condition } = g.key
  const arms = byKind.get(kind) ?? new Map()
  byKind.set(kind, arms)
  const metrics = arms.get(condition) ?? new Map()
  arms.set(condition, metrics)
  const total = Object.entries(g.counts)
    .filter(([k]) => k.startsWith("status:"))
    .reduce((a, [, v]) => a + v, 0)
  const put = (m: string, c: Cell) => {
    const runs = metrics.get(m) ?? new Map()
    metrics.set(m, runs)
    runs.set(run.name, c)
  }
  for (const r of g.rows) put(r.metric, { mean: r.mean, n: r.n, total })
  for (const a of g.absent) put(a.metric, { mean: null, n: 0, total, absent: a.reason })
  const out = notes.get(kind) ?? []
  notes.set(kind, out)
  const uses = new Set(g.rows.flatMap((r) => r.notes.filter((n) => n.startsWith("ungated")).map((n) => n.slice(9, -1))))
  for (const use of uses) {
    const line = `The ${use} judge is not yet checked against human labels.`
    if (!out.includes(line)) out.push(line)
  }
  for (const [what, reasons] of Object.entries(g.missing)) {
    const said = Object.entries(reasons).map(([why, n]) => `${n} ${why.replace("_", " ")}`).join(", ")
    out.push(`${run.endpoint}, ${condition}: ${what}: ${said}.`)
  }
}

function KindTable({
  suite,
  split,
  info,
  kind,
  arms,
  runs,
  color,
  notes,
  about,
}: {
  suite: string
  split: string
  info?: KindInfo
  kind: string
  arms: Map<string, Map<string, Map<string, Cell>>>
  runs: Run[]
  color: (name: string) => string
  notes: string[]
  about: Record<string, string>
}) {
  const [everything, setEverything] = useState(false)
  const armOrder = Object.keys(info?.arms ?? {})
  const armNames = [...arms.keys()].sort((a, b) => armOrder.indexOf(a) - armOrder.indexOf(b))
  const metricSet = new Set<string>()
  for (const ms of arms.values()) for (const m of ms.keys()) metricSet.add(m)
  const all = [...metricSet].sort()
  const metrics = everything ? all : all.filter((m) => !isProfile(m))
  const hiddenCount = all.length - all.filter((m) => !isProfile(m)).length
  const tasks = info?.rows[split]

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-lg font-semibold tracking-tight">
          <Link to="/$suite/$split/kind/$kind/tasks" params={{ suite, split, kind }} className="hover:underline">
            {info?.title ?? kind}
          </Link>
        </h2>
        <span className="font-mono text-xs text-muted-foreground">{kind}</span>
        {tasks !== undefined && <span className="text-sm text-muted-foreground">{tasks} tasks</span>}
        {hiddenCount > 0 && (
          <Button variant="ghost" size="sm" className="ml-auto text-muted-foreground" onClick={() => setEverything((e) => !e)}>
            {everything ? "Outcome metrics only" : `All metrics (+${hiddenCount})`}
          </Button>
        )}
      </div>
      {info && <p className="max-w-prose text-sm text-muted-foreground">{info.tests}</p>}
      <div className="overflow-x-auto rounded-lg border">
        <Table className="tabular-nums">
          <TableHeader>
            {armNames.length > 1 && (
              <TableRow className="hover:bg-transparent">
                <TableHead />
                {armNames.map((a) => (
                  <TableHead key={a} colSpan={runs.length} className="border-l text-center font-mono text-xs">
                    {a}
                  </TableHead>
                ))}
              </TableRow>
            )}
            <TableRow className="hover:bg-transparent">
              <TableHead className="w-[28%]">{armNames.length === 1 ? <span className="font-mono text-xs">{armNames[0]}</span> : "Metric"}</TableHead>
              {armNames.flatMap((a) =>
                runs.map((r, i) => (
                  <TableHead key={`${a}-${r.name}`} className={`text-right ${i === 0 ? "border-l" : ""}`}>
                    <span className="inline-flex items-center gap-1.5">
                      <span className="size-2 rounded-full" style={{ background: color(r.name) }} />
                      {r.endpoint}
                    </span>
                  </TableHead>
                )),
              )}
            </TableRow>
          </TableHeader>
          <TableBody>
            {metrics.map((m) => {
              const cells = armNames.flatMap((a) => runs.map((r) => arms.get(a)?.get(m)?.get(r.name)))
              const max = Math.max(...cells.map((c) => Math.abs(c?.mean ?? 0)), 1e-9)
              const d = digits(cells.map((c) => c?.mean))
              return (
                <TableRow key={m}>
                  <TableCell className="font-mono text-xs">
                    {about[m] ? (
                      <Tooltip>
                        <TooltipTrigger className="cursor-help underline decoration-muted-foreground/40 decoration-dotted underline-offset-4">{m}</TooltipTrigger>
                        <TooltipContent className="max-w-xs">{about[m]}</TooltipContent>
                      </Tooltip>
                    ) : (
                      m
                    )}
                  </TableCell>
                  {armNames.flatMap((a) =>
                    runs.map((r, i) => {
                      const c = arms.get(a)?.get(m)?.get(r.name)
                      return (
                        <TableCell key={`${a}-${r.name}`} className={`p-0 text-right ${i === 0 ? "border-l" : ""}`}>
                          <Link
                            to="/$suite/$split/kind/$kind/tasks"
                            params={{ suite, split, kind }}
                            search={{ arm: a, metric: m, sort: r.name }}
                            className="block px-3 py-2 hover:bg-muted"
                          >
                            <Value cell={c} max={max} digits={d} color={color(r.name)} />
                          </Link>
                        </TableCell>
                      )
                    }),
                  )}
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
      </div>
      {notes.length > 0 && (
        <ul className="space-y-0.5 text-xs text-muted-foreground">
          {notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </section>
  )
}

function Value({ cell, max, digits, color }: { cell?: Cell; max: number; digits: number; color: string }) {
  if (!cell) return <span className="text-muted-foreground">not played</span>
  if (cell.absent) return <span className="text-muted-foreground">not judged ({cell.absent})</span>
  return (
    <span className="inline-flex flex-col items-end gap-1">
      <span>
        {fmt(cell.mean, digits)}
        {cell.n < cell.total && <span className="ml-1 text-xs text-muted-foreground">n {cell.n}</span>}
      </span>
      <span className="h-1 w-16 overflow-hidden rounded-full bg-muted">
        <span className="block h-full rounded-full" style={{ width: `${(Math.abs(cell.mean ?? 0) / max) * 100}%`, background: color }} />
      </span>
    </span>
  )
}

function Unplayed({ kinds, split }: { kinds: KindInfo[]; split: string }) {
  if (!kinds.length) return null
  return (
    <p className="text-sm text-muted-foreground">
      No loaded run played {kinds.map((k) => k.title).join(", ")}
      {kinds.some((k) => !k.rows[split]) ? ` (some have no ${split} tasks)` : ""}. Their tasks are in the sidebar.
    </p>
  )
}
