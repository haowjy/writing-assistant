// A kind's tasks: one row per task, one column per visible run, on one metric of one arm.
// With no run loaded, the kind's rows alone (held-out test rows behind an explicit switch).
import { useQueries, useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router"
import { ArrowDown, ArrowUp } from "lucide-react"

import { PageHeader } from "@/components/frame"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { kindsQuery, reportQuery, tasksQuery, type EpisodeItem } from "@/lib/api"
import { choices, taskRows, useKindEpisodes } from "@/lib/tasks"
import { digits, fmt, shortTask, useScopeRuns } from "@/lib/view"
import type { TasksSearch } from "@/router"

export function KindTasks() {
  const { suite, split, kind } = useParams({ from: "/$suite/$split/kind/$kind/tasks" })
  const search = useSearch({ from: "/$suite/$split/kind/$kind/tasks" })
  const navigate = useNavigate({ from: "/$suite/$split/kind/$kind/tasks" })
  const { data: kinds } = useQuery(kindsQuery(suite))
  const info = kinds?.kinds.find((k) => k.kind === kind)
  const { all, visible, color } = useScopeRuns(suite, split)
  const { pending, byRun } = useKindEpisodes(visible, search.judges as string | undefined)
  const reports = useQueries({ queries: visible.map((r) => reportQuery(r.name)) })
  const set = (patch: Partial<TasksSearch>) => navigate({ search: (s) => ({ ...s, ...patch }), replace: true })

  const header = (
    <PageHeader
      crumbs={[
        { label: `Suite ${suite} · ${split}`, to: () => <Link to="/$suite/$split" params={{ suite, split }} /> },
        { label: info?.title ?? kind },
        { label: "Tasks" },
      ]}
    />
  )
  const intro = (
    <div className="space-y-1">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <h1 className="text-xl font-semibold tracking-tight">{info?.title ?? kind}</h1>
        <span className="font-mono text-xs text-muted-foreground">{kind}</span>
        {info && (
          <span className="text-sm text-muted-foreground">
            {Object.entries(info.rows)
              .map(([sp, n]) => `${sp} ${n}`)
              .join(" · ")}
          </span>
        )}
      </div>
      {info && <p className="max-w-prose text-sm text-muted-foreground">{info.tests}</p>}
    </div>
  )

  if (all.length === 0) return <RowsOnly suite={suite} split={split} kind={kind} header={header} intro={intro} test={!!search.test} onTest={(t) => set({ test: t || undefined })} />

  const { arms, arm, metrics } = choices(byRun, kind, Object.keys(info?.arms ?? {}), search.arm)
  const metric = search.metric && metrics.includes(search.metric) ? search.metric : metrics[0]
  const rows = taskRows(byRun, visible, kind, arm, metric, search)
  const fields = rowFields(rows)
  const total = taskRows(byRun, visible, kind, arm, metric, {}).length
  // An outcome metric no item of this arm was judged on has no column to pick; say so, or the
  // profile metric left as the default reads as the score.
  const unjudged = new Map<string, string>()
  for (const q of reports)
    for (const g of q.data?.groups ?? [])
      if (g.key.kind === kind && g.key.condition === arm) for (const a of g.absent) unjudged.set(a.metric, a.reason)
  const d = digits(rows.flatMap((t) => visible.map((r) => (metric ? t.cells.get(r.name)?.metrics[metric] : undefined))))

  return (
    <>
      {header}
      <div className="mx-auto w-full max-w-6xl space-y-6 px-6 py-8">
        {intro}
        <div className="flex flex-wrap items-center gap-3">
          {arms.length > 1 && (
            <ToggleGroup type="single" variant="outline" size="sm" spacing={0} value={arm} onValueChange={(v) => v && set({ arm: v })}>
              {arms.map((a) => (
                <ToggleGroupItem key={a} value={a} className="px-3 font-mono text-xs">
                  {a}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          )}
          <Select value={metric} onValueChange={(v) => set({ metric: v })}>
            <SelectTrigger size="sm" className="w-64 font-mono text-xs" aria-label="Metric">
              <SelectValue placeholder="No metric scored" />
            </SelectTrigger>
            <SelectContent>
              {metrics.map((m) => (
                <SelectItem key={m} value={m} className="font-mono text-xs">
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <ToggleGroup type="single" variant="outline" size="sm" spacing={0} value={search.only ?? "all"} onValueChange={(v) => v && set({ only: v === "all" ? undefined : (v as TasksSearch["only"]) })}>
            <ToggleGroupItem value="all" className="px-3">All</ToggleGroupItem>
            <ToggleGroupItem value="any" className="px-3">Above 0 for any run</ToggleGroupItem>
            <ToggleGroupItem value="every" className="px-3">Above 0 for every run</ToggleGroupItem>
            <ToggleGroupItem value="unscored" className="px-3">Not scored</ToggleGroupItem>
          </ToggleGroup>
          <Input
            className="ml-auto h-8 w-56"
            placeholder="Filter by id or field"
            defaultValue={search.q}
            onChange={(e) => set({ q: e.target.value || undefined })}
            aria-label="Filter tasks"
          />
        </div>
        {kinds && info && metric && kinds.metrics[metric] && <p className="text-sm text-muted-foreground"><span className="font-mono text-xs">{metric}</span>: {kinds.metrics[metric]}</p>}
        {unjudged.size > 0 && (
          <p className="text-sm text-muted-foreground">
            Not judged in these runs:{" "}
            {[...unjudged].map(([m, why], i) => (
              <span key={m}>
                {i > 0 && ", "}
                <span className="font-mono text-xs">{m}</span> ({why})
              </span>
            ))}
            .
          </p>
        )}
        {pending ? (
          <Skeleton className="h-96 w-full" />
        ) : rows.length === 0 ? (
          <Empty className="border">
            <EmptyHeader>
              <EmptyTitle>{total ? "No task matches these filters" : "No visible run played this kind"}</EmptyTitle>
              <EmptyDescription>{total ? `${total} tasks without them.` : "Show another run in the sidebar, or load one that played it."}</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground tabular-nums">
              {rows.length} of {total} tasks
            </p>
            <div className="overflow-x-auto rounded-lg border">
              <Table className="tabular-nums">
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead>Task</TableHead>
                    {fields.map((f) => (
                      <TableHead key={f}>{f}</TableHead>
                    ))}
                    {visible.map((r) => {
                      const sorted = search.sort === r.name
                      return (
                        <TableHead key={r.name} className="text-right">
                          <button
                            className="inline-flex items-center gap-1.5 hover:text-foreground"
                            onClick={() => set({ sort: r.name, desc: sorted && search.desc !== false ? false : undefined })}
                          >
                            <span className="size-2 rounded-full" style={{ background: color(r.name) }} />
                            {r.endpoint}
                            {sorted && (search.desc === false ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" />)}
                          </button>
                        </TableHead>
                      )
                    })}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((t) => (
                    <TableRow key={t.task_id}>
                      <TableCell className="font-mono text-xs">
                        <Link
                          to="/$suite/$split/task/$task/$arm"
                          params={{ suite, split, task: t.task_id, arm: arm! }}
                          search={(s: Record<string, unknown>) => ({ ...s, run: visible.find((r) => t.cells.get(r.name))?.name })}
                          className="hover:underline"
                        >
                          {shortTask(t.task_id, suite, kind)}
                        </Link>
                      </TableCell>
                      {fields.map((f) => (
                        <TableCell key={f} className="text-muted-foreground">
                          {String(t.row[f] ?? "")}
                        </TableCell>
                      ))}
                      {visible.map((r) => (
                        <TableCell key={r.name} className="p-0 text-right">
                          <Link
                            to="/$suite/$split/task/$task/$arm"
                            params={{ suite, split, task: t.task_id, arm: arm! }}
                            search={(s: Record<string, unknown>) => ({ ...s, run: r.name })}
                            className="block px-3 py-2 hover:bg-muted"
                          >
                            <CellValue e={t.cells.get(r.name)} metric={metric} digits={d} />
                          </Link>
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        )}
      </div>
    </>
  )
}

function CellValue({ e, metric, digits }: { e?: EpisodeItem; metric?: string; digits: number }) {
  if (!e) return <span className="text-muted-foreground">not played</span>
  if (e.no_output.length) return <span className="text-muted-foreground">no output ({e.no_output.join(", ").replace("_", " ")})</span>
  if (e.not_scored) return <span className="text-muted-foreground">not scored</span>
  const v = metric ? e.metrics[metric] : undefined
  return v === undefined ? <span className="text-muted-foreground">not scored</span> : <span>{fmt(v, digits)}</span>
}

/** The row fields worth a column: short, differing between tasks, not already in the task id,
 * and not a long number (a seed). */
function rowFields(rows: { task_id: string; row: Record<string, string | number> }[]): string[] {
  if (!rows.length) return []
  return Object.keys(rows[0].row)
    .filter((k) => {
      const vals = rows.map((r) => String(r.row[k] ?? ""))
      return (
        new Set(vals).size > 1 &&
        vals.every((v) => v.length <= 24 && !/^\d{6,}$/.test(v)) &&
        !rows.every((r) => r.task_id.includes(String(r.row[k])))
      )
    })
    .slice(0, 3)
}

function RowsOnly({
  suite,
  split,
  kind,
  header,
  intro,
  test,
  onTest,
}: {
  suite: string
  split: string
  kind: string
  header: React.ReactNode
  intro: React.ReactNode
  test: boolean
  onTest: (t: boolean) => void
}) {
  const { data, isPending } = useQuery(tasksQuery(suite, test))
  const rows = (data?.tasks ?? []).filter((t) => t.kind === kind)
  const fields = rowFields(rows)
  return (
    <>
      {header}
      <div className="mx-auto w-full max-w-6xl space-y-6 px-6 py-8">
        {intro}
        <div className="flex items-center gap-2">
          <Switch id="held-out" checked={test} onCheckedChange={onTest} />
          <Label htmlFor="held-out">Held-out test rows</Label>
          <span className="text-xs text-muted-foreground">Reading them counts as a use of the test split.</span>
        </div>
        {isPending ? (
          <Skeleton className="h-96 w-full" />
        ) : (
          <div className="overflow-x-auto rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Task</TableHead>
                  <TableHead>Split</TableHead>
                  {fields.map((f) => (
                    <TableHead key={f}>{f}</TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((t) => (
                  <TableRow key={t.task_id}>
                    <TableCell className="font-mono text-xs">{shortTask(t.task_id, suite, kind)}</TableCell>
                    <TableCell className={t.split === split ? "" : "text-muted-foreground"}>{t.split}</TableCell>
                    {fields.map((f) => (
                      <TableCell key={f} className="text-muted-foreground">
                        {String(t.row[f] ?? "")}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    </>
  )
}
