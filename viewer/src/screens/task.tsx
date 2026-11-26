// What the models wrote on one task: one run's prose as a reading column with its verdicts in
// one line, everything else closed below it; side by side across the visible runs
// (`?compare=1`); a verdict opens in a sheet (`?verdict=<id>`).
import { useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router"
import { ChevronDown, ChevronLeft, ChevronRight, Columns2 } from "lucide-react"
import { useEffect, type ReactNode } from "react"

import { PageHeader } from "@/components/frame"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Kbd } from "@/components/ui/kbd"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import {
  compareQuery,
  familyQuery,
  kindsQuery,
  type Call,
  type DiffLine,
  type Episode,
  type Family,
  type Judgment,
} from "@/lib/api"
import { taskRows, useKindEpisodes } from "@/lib/tasks"
import { answerText, fmt, isProfile, shortTask, useScopeRuns } from "@/lib/view"
import type { TaskSearch } from "@/router"

const FROM = "/$suite/$split/task/$task/$arm" as const

export function TaskView() {
  const { suite, split, task, arm } = useParams({ from: FROM })
  const search = useSearch({ from: FROM })
  const navigate = useNavigate({ from: FROM })
  const { data: kinds } = useQuery(kindsQuery(suite))
  const { visible, color } = useScopeRuns(suite, split)
  const { pending, byRun } = useKindEpisodes(visible, search.judges as string | undefined)

  const mine = visible.flatMap((r) => (byRun.get(r.name) ?? []).filter((e) => e.task_id === task).map((e) => ({ run: r, ep: e })))
  const kind = mine[0]?.ep.kind ?? ""
  const info = kinds?.kinds.find((k) => k.kind === kind)
  const played = visible.filter((r) => mine.some((m) => m.run.name === r.name))
  const run = played.find((r) => r.name === search.run) ?? played[0]
  const armOrder = Object.keys(info?.arms ?? {})
  const arms = [...new Set(mine.map((m) => m.ep.condition))].sort((a, b) => armOrder.indexOf(a) - armOrder.indexOf(b))
  const ep = mine.find((m) => m.run.name === run?.name && m.ep.condition === arm)?.ep
  const { data: family, isPending: familyPending } = useQuery({ ...familyQuery(run?.name ?? "", ep?.ep_id ?? "", search.judges as string | undefined), enabled: !!ep })

  // the list this task was reached from, as filtered and sorted in its URL
  const list = kind ? taskRows(byRun, visible, kind, arm, search.metric, search).map((t) => t.task_id) : []
  const at = list.indexOf(task)
  const go = (patch: Partial<TaskSearch>, to?: { task?: string; arm?: string }) =>
    navigate({ to: FROM, params: { suite, split, task: to?.task ?? task, arm: to?.arm ?? arm }, search: (s) => ({ ...s, ...patch }) })

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.closest("input, textarea, [role=dialog]") || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === "j" && at >= 0 && at < list.length - 1) go({ verdict: undefined }, { task: list[at + 1] })
      else if (e.key === "k" && at > 0) go({ verdict: undefined }, { task: list[at - 1] })
      else if (e.key === "s") go({ compare: search.compare ? undefined : true })
      else if (e.key === "]" || e.key === "[") {
        const i = arms.indexOf(arm) + (e.key === "]" ? 1 : -1)
        if (arms[i]) go({ verdict: undefined }, { arm: arms[i] })
      } else return
      e.preventDefault()
    }
    window.addEventListener("keydown", on)
    return () => window.removeEventListener("keydown", on)
  })

  const label = shortTask(task, suite, kind)
  const header = (
    <PageHeader
      crumbs={[
        { label: `Suite ${suite} · ${split}`, to: () => <Link to="/$suite/$split" params={{ suite, split }} /> },
        {
          label: info?.title ?? kind,
          to: () => (
            <Link
              to="/$suite/$split/kind/$kind/tasks"
              params={{ suite, split, kind }}
              search={{ arm, metric: search.metric, only: search.only, sort: search.sort, desc: search.desc, q: search.q }}
            />
          ),
        },
        { label },
        { label: arm },
      ]}
    >
      {list.length > 0 && (
        <>
          <span className="text-xs text-muted-foreground tabular-nums">{at >= 0 ? `${at + 1} of ${list.length}` : `not in the list`}</span>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" disabled={at <= 0} onClick={() => go({ verdict: undefined }, { task: list[at - 1] })} aria-label="Previous task">
                <ChevronLeft />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Previous task <Kbd>k</Kbd>
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button variant="ghost" size="icon-sm" disabled={at < 0 || at >= list.length - 1} onClick={() => go({ verdict: undefined }, { task: list[at + 1] })} aria-label="Next task">
                <ChevronRight />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              Next task <Kbd>j</Kbd>
            </TooltipContent>
          </Tooltip>
        </>
      )}
    </PageHeader>
  )

  if (pending) return <>{header}<div className="mx-auto max-w-3xl px-6 py-10"><Skeleton className="h-96" /></div></>
  if (!mine.length)
    return (
      <>
        {header}
        <p className="mx-auto max-w-3xl px-6 py-10 text-sm text-muted-foreground">No visible run played this task. Show another run in the sidebar.</p>
      </>
    )

  return (
    <>
      {header}
      <div className="mx-auto w-full max-w-6xl px-6 py-8">
        <div className="mb-6 space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">{label}</h1>
          <p className="text-sm text-muted-foreground">
            {info?.title}
            {Object.entries(mine[0].ep.row)
              .filter(([, v]) => String(v).length < 30 && !/^\d{6,}$/.test(String(v)))
              .map(([k, v]) => (
                <span key={k} className="ml-3">
                  <span className="text-muted-foreground/70">{k}</span> {String(v)}
                </span>
              ))}
          </p>
        </div>
        <div className="mb-8 flex flex-wrap items-center gap-3 border-b pb-3">
          <Tabs value={arm} onValueChange={(a) => go({ verdict: undefined }, { arm: a })}>
            <TabsList variant="line">
              {arms.map((a) => (
                <TabsTrigger key={a} value={a} className="px-3 font-mono text-xs">
                  {a}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          <div className="ml-auto flex items-center gap-2">
            {!search.compare && played.length > 1 && (
              <ToggleGroup type="single" variant="outline" size="sm" spacing={0} value={run?.name} onValueChange={(v) => v && go({ run: v, verdict: undefined })}>
                {played.map((r) => (
                  <ToggleGroupItem key={r.name} value={r.name} className="gap-1.5 px-3">
                    <span className="size-2 rounded-full" style={{ background: color(r.name) }} />
                    {r.endpoint}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            )}
            {played.length > 1 && (
              <Button aria-pressed={!!search.compare} variant={search.compare ? "secondary" : "outline"} size="sm" onClick={() => go({ compare: search.compare ? undefined : true })}>
                <Columns2 /> Side by side <Kbd>s</Kbd>
              </Button>
            )}
          </div>
        </div>
        {search.compare ? (
          <SideBySide task={task} arm={arm} runs={played} color={color} onVerdict={(r, id) => go({ compare: undefined, run: r, verdict: id })} />
        ) : !ep ? (
          <p className="text-sm text-muted-foreground">{run?.endpoint} did not play the {arm} arm of this task.</p>
        ) : familyPending || !family ? (
          <Skeleton className="mx-auto h-96 max-w-[68ch]" />
        ) : (
          <Chapter family={family} epId={ep.ep_id} metricsAbout={kinds?.metrics ?? {}} onVerdict={(id) => go({ verdict: id })} />
        )}
      </div>
      {family && <VerdictSheet family={family} id={search.verdict} onClose={() => go({ verdict: undefined })} onGo={(id) => go({ verdict: id })} />}
    </>
  )
}

// -- one run's chapter ---------------------------------------------------------------------------

function verdictLine(ep: Episode, judgments: Judgment[]) {
  return Object.entries(ep.score.metrics ?? {})
    .filter(([m]) => !isProfile(m))
    .map(([m, v]) => {
      const items = judgments.filter((j) => j.metric === m)
      const asked = items.filter((j) => typeof j.model === "boolean")
      const yes = asked.filter((j) => j.model === true).length
      return { metric: m, value: v, counts: asked.length ? `${yes} of ${asked.length} yes` : "", first: items[0]?.id }
    })
}

function Chapter({
  family,
  epId,
  metricsAbout,
  onVerdict,
}: {
  family: Family
  epId: string
  metricsAbout: Record<string, string>
  onVerdict: (id: string) => void
}) {
  const ep = family.episodes.find((e) => e.ep_id === epId)!
  const twin = family.episodes.find((e) => e.twin_of === epId)
  const judgments = ep.judgments ?? []
  const line = verdictLine(ep, judgments)
  const words = ep.written.reduce((a, w) => a + w.text.split(/\s+/).filter(Boolean).length, 0)
  const lastCall = ep.turns.flatMap((t) => t.calls).at(-1)
  const against = family.episodes.find((e) => e.ep_id === ep.shown.against)

  return (
    <article className="mx-auto max-w-[68ch]">
      <div className="mb-6 flex flex-wrap items-center gap-2">
        {ep.score.not_scored && <Badge variant="outline">{ep.score.not_scored}</Badge>}
        {line.map((v) => (
          <Tooltip key={v.metric}>
            <TooltipTrigger asChild>
              <Badge asChild variant="secondary" className="cursor-pointer font-mono text-xs font-normal">
                <button onClick={() => v.first && onVerdict(v.first)}>
                  {v.metric} <span className="font-semibold">{fmt(v.value)}</span>
                  {v.counts && <span className="text-muted-foreground">· {v.counts}</span>}
                </button>
              </Badge>
            </TooltipTrigger>
            {metricsAbout[v.metric] && <TooltipContent className="max-w-xs">{metricsAbout[v.metric]}</TooltipContent>}
          </Tooltip>
        ))}
        <span className="text-xs text-muted-foreground tabular-nums">
          {words.toLocaleString("en")} words{lastCall?.finish_reason ? ` · ${stopped(lastCall.finish_reason)}` : ""}
        </span>
      </div>

      {ep.status === "failed" && <p className="mb-4 text-sm text-destructive">The episode failed: {ep.failure ?? "no reason recorded"}</p>}
      {ep.written.length === 0 ? (
        <p className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">The model wrote nothing{lastCall?.finish_reason ? ` (stopped: ${lastCall.finish_reason})` : ""}.</p>
      ) : (
        ep.written.map((w) => (
          <section key={w.label} className="mb-8">
            {ep.written.length > 1 || w.label !== "reply" ? <h3 className="mb-3 font-mono text-xs text-muted-foreground">{w.label}</h3> : null}
            <Prose text={w.text} />
          </section>
        ))
      )}

      <div className="mt-10 divide-y border-y">
        {judgments.length > 0 && (
          <Row title="Verdicts" meta={`${judgments.length} questions`}>
            <Verdicts judgments={judgments} epId={epId} onVerdict={onVerdict} />
          </Row>
        )}
        {ep.judgments === null && <div className="py-3 text-sm text-muted-foreground">This run stored no judge items.</div>}
        {twin && (
          <Row title={`Twin: ${twin.condition}`} meta={difference(twin)}>
            {twin.status === "failed" && <p className="text-sm text-destructive">The twin failed: {twin.failure}</p>}
            {twin.written.length ? twin.written.map((w) => <Prose key={w.label} text={w.text} small />) : <p className="text-sm text-muted-foreground">The twin wrote nothing.</p>}
          </Row>
        )}
        {ep.scene?.y_star && (
          <Row title="The author's real scene" meta={`${ep.scene.y_star.text.split(/\s+/).length} words, kept from the model`}>
            <Prose text={ep.scene.y_star.text} small />
          </Row>
        )}
        {ep.scene?.exemplars && ep.scene.exemplars.length > 0 && (
          <Row title="The samples it was shown" meta={`${ep.scene.exemplars.length} passages`}>
            {ep.scene.exemplars.map((x, i) => (
              <div key={i} className="mb-6">
                <h4 className="mb-2 text-xs text-muted-foreground">Passage {i + 1}</h4>
                <Prose text={x.text} small />
              </div>
            ))}
          </Row>
        )}
        {ep.edit && (
          <Row title={`The edit to ${ep.edit.path}`} meta={ep.edit.objection ?? ""}>
            <Diff lines={ep.edit.diff} />
          </Row>
        )}
        <Row title="What it was shown" meta={against ? `${ep.shown.removed.length + ep.shown.added.length} lines differ from ${against.condition}` : "the system prompt and every message"}>
          <Diff lines={ep.shown.lines} />
          {ep.shown.files.map((f) => (
            <div key={f.path} className="mt-4">
              <h4 className="mb-1 font-mono text-xs text-muted-foreground">{f.path} differs</h4>
              <Diff lines={f.lines} />
            </div>
          ))}
        </Row>
        <Row title="How it was written" meta={callsMeta(ep)}>
          <Calls ep={ep} />
        </Row>
        <Row title="All metrics" meta="words, readability, overlap and counts">
          <MetricTable ep={ep} />
        </Row>
      </div>
    </article>
  )
}

const stopped = (reason: string) =>
  reason === "stop" ? "ended on its own" : reason === "length" ? "cut off at the token limit" : `stopped: ${reason}`

function difference(twin: Episode) {
  const { removed, added, files } = twin.shown
  if (removed.length === 1 && added.length === 1) return `“${removed[0]}” → “${added[0]}”`
  const parts = [removed.length ? `drops ${removed.length} lines` : "", added.length ? `adds ${added.length}` : "", ...files.map((f) => `${f.path} differs`)].filter(Boolean)
  return parts.join(", ") || "no line of its prompt or files differs"
}

function callsMeta(ep: Episode) {
  const calls = ep.turns.flatMap((t) => t.calls)
  const out = calls.reduce((a, c) => a + (c.usage?.output_tokens ?? 0), 0)
  return `${calls.length} call${calls.length === 1 ? "" : "s"} · ${out.toLocaleString("en")} output tokens`
}

function Row({ title, meta, children }: { title: string; meta?: string; children: ReactNode }) {
  return (
    <Collapsible className="group/row">
      <CollapsibleTrigger className="flex w-full items-center gap-3 py-3 text-left text-sm hover:text-foreground">
        <ChevronDown className="size-4 shrink-0 -rotate-90 text-muted-foreground transition-transform group-data-[state=open]/row:rotate-0" />
        <span className="shrink-0 font-medium">{title}</span>
        {meta && <span className="ml-auto truncate pl-4 text-xs text-muted-foreground">{meta}</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="pb-6 pl-7">{children}</CollapsibleContent>
    </Collapsible>
  )
}

export function Prose({ text, small }: { text: string; small?: boolean }) {
  return (
    <div className={`space-y-4 font-serif ${small ? "text-[0.95rem] leading-7" : "text-[1.0625rem] leading-[1.75]"} text-foreground/90`}>
      {text
        .split(/\n\s*\n/)
        .filter((p) => p.trim())
        .map((p, i) => (
          <p key={i} className="whitespace-pre-line">
            {p.trim()}
          </p>
        ))}
    </div>
  )
}

function Verdicts({ judgments, epId, onVerdict }: { judgments: Judgment[]; epId: string; onVerdict: (id: string) => void }) {
  const groups = new Map<string, Judgment[]>()
  for (const j of judgments) groups.set(j.metric, [...(groups.get(j.metric) ?? []), j])
  return (
    <div className="space-y-5">
      {[...groups].map(([m, items]) => (
        <div key={m}>
          <h4 className="mb-1 font-mono text-xs text-muted-foreground">{m}</h4>
          <ul>
            {items.map((j) => (
              <li key={j.id}>
                <button onClick={() => onVerdict(j.id)} className="flex w-full items-baseline gap-3 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted">
                  <span className="flex-1">
                    {j.question}
                    {j.about !== epId && <span className="ml-2 text-xs text-muted-foreground">asked of the twin</span>}
                  </span>
                  <Answer value={j.model} unscored={j.unscored} />
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}

function Answer({ value, unscored }: { value: boolean | number | null; unscored?: string | null }) {
  return (
    <Badge variant={value === true ? "default" : "outline"} className="shrink-0 tabular-nums">
      {value === null && unscored ? `unscored: ${unscored}` : answerText(value)}
    </Badge>
  )
}

function Diff({ lines }: { lines: DiffLine[] }) {
  return (
    <pre className="max-h-[32rem] overflow-auto rounded-md border bg-muted/40 py-2 font-mono text-xs leading-5">
      {lines.map(([op, t], i) =>
        op === "@" ? (
          <div key={i} className="px-3 text-muted-foreground">⋯</div>
        ) : (
          <div
            key={i}
            className={`px-3 whitespace-pre-wrap ${op === "-" ? "bg-red-500/10 text-red-700 line-through decoration-red-500/40 dark:text-red-300" : op === "+" ? "bg-emerald-500/10 text-emerald-800 dark:text-emerald-300" : ""}`}
          >
            <span className="mr-2 inline-block w-3 select-none text-muted-foreground">{op === " " ? "" : op}</span>
            {t || " "}
          </div>
        ),
      )}
    </pre>
  )
}

function Calls({ ep }: { ep: Episode }) {
  return (
    <ol className="space-y-6">
      {ep.turns.map((t, i) => (
        <li key={i} className="space-y-3">
          {t.message !== null && (
            <div className="text-sm">
              <span className="text-xs text-muted-foreground">to {t.actor}</span>
              <pre className="mt-1 max-h-40 overflow-auto rounded-md bg-muted/40 p-3 font-mono text-xs whitespace-pre-wrap">{t.message}</pre>
            </div>
          )}
          {t.calls.map((c, k) => (
            <CallView key={k} call={c} n={k + 1} />
          ))}
        </li>
      ))}
    </ol>
  )
}

function CallView({ call, n }: { call: Call; n: number }) {
  return (
    <div className="space-y-2 border-l-2 pl-4">
      <div className="text-xs text-muted-foreground tabular-nums">
        Call {n} · {call.actor} · stopped: {call.finish_reason ?? "not recorded"}
        {call.usage && ` · ${call.usage.input_tokens ?? "?"} in, ${call.usage.output_tokens ?? "?"} out`}
      </div>
      {call.reasoning && (
        <Collapsible className="group/r">
          <CollapsibleTrigger className="flex items-center gap-1 text-xs font-medium hover:underline">
            <ChevronDown className="size-3 -rotate-90 transition-transform group-data-[state=open]/r:rotate-0" />
            Reasoning · {call.reasoning.split(/\s+/).length.toLocaleString("en")} words
          </CollapsibleTrigger>
          <CollapsibleContent>
            <pre className="mt-2 max-h-96 overflow-auto rounded-md bg-muted/40 p-3 font-mono text-xs whitespace-pre-wrap">{call.reasoning}</pre>
          </CollapsibleContent>
        </Collapsible>
      )}
      {call.tool_calls.map((tc, i) => (
        <div key={i} className="text-xs">
          <code className="font-mono">
            {tc.name}({typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments)})
          </code>
          {tc.result ? (
            <pre className="mt-1 max-h-48 overflow-auto rounded-md bg-muted/40 p-2 font-mono whitespace-pre-wrap">
              {tc.result.summary}
              {tc.result.text && `\n\n${tc.result.text}`}
            </pre>
          ) : (
            <span className="ml-2 text-muted-foreground">result not recorded</span>
          )}
        </div>
      ))}
    </div>
  )
}

function MetricTable({ ep }: { ep: Episode }) {
  const all = { ...(ep.score.metrics ?? {}), ...(ep.score.prose ?? {}), ...(ep.score.counts ?? {}) }
  const names = Object.keys(all).sort()
  if (!names.length) return <p className="text-sm text-muted-foreground">No metrics: {ep.score.not_scored ?? "not scored"}.</p>
  return (
    <dl className="grid grid-cols-[1fr_auto] gap-x-6 gap-y-1 font-mono text-xs tabular-nums">
      {names.map((m) => (
        <div key={m} className="contents">
          <dt className="text-muted-foreground">{m}</dt>
          <dd className="text-right">{fmt(all[m])}</dd>
        </div>
      ))}
      {ep.score.code && <p className="col-span-2 mt-2 font-sans text-muted-foreground">Scored by code {ep.score.code}.</p>}
    </dl>
  )
}

// -- side by side --------------------------------------------------------------------------------

function SideBySide({
  task,
  arm,
  runs,
  color,
  onVerdict,
}: {
  task: string
  arm: string
  runs: { name: string; endpoint: string }[]
  color: (n: string) => string
  onVerdict: (run: string, id: string) => void
}) {
  const { data, isPending } = useQuery(compareQuery(task))
  if (isPending || !data) return <Skeleton className="h-96" />
  const cols = runs.map((r) => ({ r, c: data.find((d) => d.run === r.name) }))
  return (
    <div className="grid gap-8" style={{ gridTemplateColumns: `repeat(${cols.length}, minmax(0, 1fr))` }}>
      {cols.map(({ r, c }) => {
        const ep = c?.episodes.find((e) => e.condition === arm && !e.twin_of)
        return (
          <section key={r.name} className="min-w-0 space-y-4">
            <h3 className="flex items-center gap-2 border-b pb-2 text-sm font-semibold">
              <span className="size-2.5 rounded-full" style={{ background: color(r.name) }} />
              {r.endpoint}
            </h3>
            {!ep ? (
              <p className="text-sm text-muted-foreground">Did not play this arm.</p>
            ) : (
              <>
                <div className="flex flex-wrap gap-1.5">
                  {ep.not_scored && <Badge variant="outline">{ep.not_scored}</Badge>}
                  {Object.entries(ep.metrics)
                    .filter(([m]) => !isProfile(m))
                    .map(([m, v]) => {
                      const items = ep.judged.filter((j) => j.metric === m)
                      return (
                        <Badge key={m} asChild variant="secondary" className="cursor-pointer font-mono text-xs font-normal">
                          <button onClick={() => items[0] && onVerdict(r.name, items[0].id)}>
                            {m} <span className="font-semibold">{fmt(v)}</span>
                          </button>
                        </Badge>
                      )
                    })}
                </div>
                {ep.written.length ? ep.written.map((w) => <Prose key={w.label} text={w.text} small />) : <p className="text-sm text-muted-foreground">Wrote nothing.</p>}
              </>
            )}
          </section>
        )
      })}
    </div>
  )
}

// -- the verdict sheet ---------------------------------------------------------------------------

function VerdictSheet({ family, id, onClose, onGo }: { family: Family; id?: string; onClose: () => void; onGo: (id: string) => void }) {
  // the verdicts of the episode this one was scored with, in the order they were asked
  const scored = family.episodes.find((e) => e.judgments?.some((j) => j.id === id))
  const all = (scored?.judgments ?? []).map((j) => ({ j, scored: scored! }))
  const i = all.findIndex((x) => x.j.id === id)
  const hit = all[i]
  const about = hit && family.episodes.find((e) => e.ep_id === hit.j.about)
  const isTwin = !!about?.twin_of
  return (
    <Sheet open={!!hit} onOpenChange={(o) => !o && onClose()} modal={false}>
      {/* not modal: the prose it judged stays readable beside it */}
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl" showOverlay={false} onInteractOutside={(e) => e.preventDefault()}>
        {hit && (
          <>
            <SheetHeader>
              <SheetTitle className="flex items-center gap-2">
                <span className="font-mono text-sm">{hit.j.metric}</span>
                <Answer value={hit.j.model} unscored={hit.j.unscored} />
              </SheetTitle>
              <SheetDescription className="text-base text-foreground">{hit.j.question}</SheetDescription>
            </SheetHeader>
            <div className="space-y-6 px-4 pb-6 text-sm">
              <p className="text-muted-foreground">
                Asked of {isTwin ? "the twin's output" : `the ${about?.condition ?? hit.scored.condition} output`}
                {hit.j.judge && ` · ${hit.j.judge.use} judge, ${hit.j.judge.endpoint || "no endpoint"}`}
              </p>
              <section className="space-y-2">
                <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">The judge's reply</h3>
                {hit.j.replies?.length ? (
                  hit.j.replies.map((r, k) => (
                    <pre key={k} className="rounded-md border bg-muted/40 p-3 font-sans text-sm whitespace-pre-wrap">
                      {r}
                    </pre>
                  ))
                ) : (
                  <p className="text-muted-foreground">No reply stored{hit.j.unscored ? ` (${hit.j.unscored})` : ""}.</p>
                )}
              </section>
              <section className="space-y-2">
                <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">The text judged</h3>
                {hit.j.evidence ? (
                  <pre className="max-h-80 overflow-auto rounded-md border p-3 font-mono text-xs whitespace-pre-wrap">{hit.j.evidence}</pre>
                ) : hit.j.a !== undefined ? (
                  <div className="grid gap-3">
                    {(["a", "b"] as const).map((side) => (
                      <div key={side}>
                        <h4 className="mb-1 text-xs text-muted-foreground">
                          {side.toUpperCase()}: {hit.j[`${side}_is`] ?? "a passage"}
                        </h4>
                        <pre className="max-h-60 overflow-auto rounded-md border p-3 font-serif text-sm whitespace-pre-wrap">{hit.j[side]}</pre>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground">{hit.j.evidence_is ? `${hit.j.evidence_is[0].toUpperCase()}${hit.j.evidence_is.slice(1)}.` : "Not stored."}</p>
                )}
              </section>
              <div className="flex gap-2 border-t pt-4">
                <Button variant="outline" size="sm" disabled={i <= 0} onClick={() => onGo(all[i - 1].j.id)}>
                  <ChevronLeft /> Previous verdict
                </Button>
                <Button variant="outline" size="sm" disabled={i >= all.length - 1} onClick={() => onGo(all[i + 1].j.id)}>
                  Next verdict <ChevronRight />
                </Button>
                <span className="ml-auto self-center text-xs text-muted-foreground tabular-nums">
                  {i + 1} of {all.length}
                </span>
              </div>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
