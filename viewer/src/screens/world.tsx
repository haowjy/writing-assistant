// A library world (`wa-bench view --world NAME`): its situations, each one's groups of
// rollouts with what every rubric scored, and one rollout (`?ep=<name>`) read on the episode
// screen's parts: the conversation in order, what the model wrote, each seat's calls with
// their thinking, the files it changed and every judge's findings.
import { useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router"
import type { ReactElement, ReactNode } from "react"

import { PageHeader } from "@/components/frame"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { rolloutQuery, worldQuery, type Episode, type World, type WorldRollout, type WorldSituation } from "@/lib/api"
import { fmt } from "@/lib/view"
import { Chapter, VerdictSheet } from "@/screens/task"

const FROM = "/world/$world" as const

export function WorldScreen() {
  const { world } = useParams({ from: FROM })
  const search = useSearch({ from: FROM })
  const judges = search.judges as string | undefined
  const { data, isPending, error } = useQuery(worldQuery(world, judges))
  const home = () => <Link to={FROM} params={{ world }} search={{ ep: undefined, verdict: undefined }} />
  if (search.ep) return <Rollout world={world} data={data} ep={search.ep} verdict={search.verdict} judges={judges} home={home} />
  return (
    <>
      <PageHeader crumbs={[{ label: `World ${world}` }]} />
      <div className="mx-auto w-full max-w-5xl px-6 py-8">
        {error ? (
          <p className="text-sm text-destructive">{String(error.message)}</p>
        ) : isPending || !data ? (
          <Skeleton className="h-96" />
        ) : (
          <>
            <h1 className="text-xl font-semibold tracking-tight">{data.name}</h1>
            <p className="mt-1 mb-8 text-sm text-muted-foreground">
              {data.situations.length} situations, {data.rollouts} closed rollouts, split {data.split ?? "?"}.{" "}
              {data.profile
                ? `Scores are the judge profile ${data.profile}'s${data.profiles.length > 1 ? ` (the world also has ${data.profiles.filter((p) => p !== data.profile).join(", ")}: add ?judges=<profile>)` : ""}.`
                : "Nothing in it is scored yet: wa-bench score --world judges its rollouts."}
            </p>
            {data.situations.length === 0 && <p className="text-sm text-muted-foreground">No situation yet: wa-bench situate --world sets a task up at a position.</p>}
            {data.situations.map((s) => (
              <Situation key={s.name} world={world} s={s} />
            ))}
          </>
        )}
      </div>
    </>
  )
}

function Situation({ world, s }: { world: string; s: WorldSituation }) {
  return (
    <section className="mb-10">
      <h2 className="font-mono text-sm font-semibold">{s.name}</h2>
      <p className="mt-1 text-xs text-muted-foreground">
        {s.kind} · seats {Object.entries(s.seats).map(([seat, cast]) => `${seat} (${cast})`).join(", ")} · rubrics {s.rubrics.join(", ") || "none listed"}
      </p>
      {s.request && <p className="mt-2 line-clamp-3 font-serif text-sm text-foreground/80">{s.request}</p>}
      <p className="mt-1 text-xs text-muted-foreground">Starts with {s.files.join(", ") || "no files"}</p>
      {s.groups.length === 0 ? (
        <p className="mt-3 text-sm text-muted-foreground">Not played yet: wa-bench run --world {world} plays a group of it.</p>
      ) : (
        s.groups.map((g) => (
          <div key={g.group} className="mt-4">
            <h3 className="text-xs text-muted-foreground">
              Group <span className="font-mono">{g.group}</span> · {g.rollouts.length} of {g.size ?? "?"} rollouts closed · played by {g.endpoint ?? "?"}
              {g.model ? ` (${g.model})` : ""} · round {g.round ?? "?"}
            </h3>
            <ul className="mt-1 divide-y rounded-md border">
              {g.rollouts.map((r) => (
                <RolloutLine key={r.ep_id} world={world} r={r} />
              ))}
            </ul>
          </div>
        ))
      )}
    </section>
  )
}

function RolloutLine({ world, r }: { world: string; r: WorldRollout }) {
  const scored = Object.entries(r.scores)
  return (
    <li>
      <Link to={FROM} params={{ world }} search={{ ep: r.ep_id }} className="flex flex-wrap items-center gap-2 px-3 py-2 text-sm hover:bg-muted">
        <span className="font-mono text-xs">{r.ep_id.split(".").at(-1)}</span>
        {r.status !== "completed" && <Badge variant="destructive">{r.status}</Badge>}
        <span className="text-xs text-muted-foreground tabular-nums">{r.words.toLocaleString("en")} words</span>
        {scored.length === 0 && <Badge variant="outline">not scored</Badge>}
        {scored.map(([rubric, sc]) =>
          sc.not_scored ? (
            <Badge key={rubric} variant="outline">
              {rubric}: {sc.not_scored}
            </Badge>
          ) : (
            Object.entries(sc.metrics)
              .filter(([m]) => rubric !== "counts" || m === "turns")
              .map(([m, v]) => (
                <Badge key={`${rubric}.${m}`} variant="secondary" className="font-mono text-xs font-normal">
                  {m} <span className="font-semibold">{fmt(v)}</span>
                </Badge>
              ))
          ),
        )}
      </Link>
    </li>
  )
}

function Rollout({ world, data, ep, verdict, judges, home }: { world: string; data?: World; ep: string; verdict?: string; judges?: string; home: () => ReactElement<{ children?: ReactNode }> }) {
  const navigate = useNavigate({ from: FROM })
  const { data: family, isPending, error } = useQuery(rolloutQuery(world, ep, judges))
  const go = (id?: string) => navigate({ to: FROM, params: { world }, search: (s) => ({ ...s, verdict: id }) })
  // the rollouts of the same situation, in the order the world screen lists them
  const siblings = data?.situations.find((s) => s.groups.some((g) => g.rollouts.some((r) => r.ep_id === ep)))
  const list = siblings?.groups.flatMap((g) => g.rollouts.map((r) => r.ep_id)) ?? []
  const at = list.indexOf(ep)
  const episode = family?.episodes[0]
  return (
    <>
      <PageHeader crumbs={[{ label: `World ${world}`, to: home }, { label: siblings?.name ?? family?.task_id ?? "rollout" }, { label: ep.split("@").at(-1) ?? ep }]}>
        {list.length > 1 && (
          <span className="flex items-center gap-3 text-xs text-muted-foreground tabular-nums">
            {at + 1} of {list.length}
            {at > 0 && (
              <Link to={FROM} params={{ world }} search={{ ep: list[at - 1] }} className="underline">
                previous
              </Link>
            )}
            {at < list.length - 1 && (
              <Link to={FROM} params={{ world }} search={{ ep: list[at + 1] }} className="underline">
                next
              </Link>
            )}
          </span>
        )}
      </PageHeader>
      <div className="mx-auto w-full max-w-6xl px-6 py-8">
        {error ? (
          <p className="text-sm text-destructive">{String(error.message)}</p>
        ) : isPending || !family || !episode ? (
          <Skeleton className="mx-auto h-96 max-w-[68ch]" />
        ) : (
          <>
            <div className="mx-auto mb-6 max-w-[68ch] space-y-1">
              <h1 className="font-mono text-sm font-semibold break-all">{ep}</h1>
              <p className="text-sm text-muted-foreground">
                {family.kind} · seats {Object.entries(episode.actors).map(([seat, cast]) => `${seat} (${cast})`).join(", ")}
                {family.profile ? ` · judged under ${family.profile}` : ""}
              </p>
            </div>
            <Conversation ep={episode} />
            <Chapter family={family} epId={ep} metricsAbout={{}} onVerdict={(id) => go(id)} />
          </>
        )}
      </div>
      {family && <VerdictSheet family={family} id={verdict} onClose={() => go(undefined)} onGo={(id) => go(id)} />}
    </>
  )
}

/** What was said and written, in order, by every seat: the scripted ones too. */
function Conversation({ ep }: { ep: Episode }) {
  const seats = Object.keys(ep.actors)
  const lines = ep.events.filter((e) => e.kind !== "deliver" && e.kind !== "tool")
  if (!lines.length) return null
  return (
    <section className="mx-auto mb-10 max-w-[68ch]">
      <h2 className="mb-3 text-xs font-medium tracking-wide text-muted-foreground uppercase">The conversation</h2>
      <ol className="space-y-3">
        {lines.map((e) => {
          const model = ep.actors[e.actor] !== undefined && ep.actors[e.actor] !== "scripted"
          const heard = seats.filter((s) => s !== e.actor && e.seen_by.includes(s))
          const toAll = heard.length === seats.filter((s) => s !== e.actor).length
          return (
            <li key={e.idx} className={`border-l-2 pl-3 ${model ? "border-foreground/60" : "border-muted"}`}>
              <div className="text-xs text-muted-foreground">
                <span className={model ? "font-semibold text-foreground" : ""}>{e.actor}</span>
                {model ? ` · ${ep.actors[e.actor]}` : " · scripted"}
                {e.kind !== "say" && ` · ${e.kind}`}
                {e.kind === "say" && !toAll && ` · heard only by ${heard.join(", ") || "nobody"}`}
              </div>
              <p className="font-serif text-[0.95rem] leading-7 whitespace-pre-line">{e.text}</p>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
