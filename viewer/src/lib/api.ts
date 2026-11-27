// The JSON the server (`src/writing_assistant/cli/view.py`) answers with, and one query per
// endpoint. The page asks for one thing at a time: a run is never fetched whole.
import { queryOptions } from "@tanstack/react-query"

export type RunSummary = {
  name: string
  suite?: string
  endpoint?: string
  model?: string
  split?: string
  started?: string
  profiles?: string[]
  episodes?: number
  twins?: number
  /** episodes a run is still playing, or a killed run left: shown nowhere, counted here */
  open?: number
  judged?: string[]
  guardrails?: string
}

export type ReportRow = {
  metric: string
  mean: number | null
  n: number
  unflagged: number | null
  refused: number | null
  ungated: boolean
  notes: string[]
}

export type ReportGroup = {
  key: { kind: string; condition: string }
  counts: Record<string, number>
  missing: Record<string, Record<string, number>>
  rows: ReportRow[]
  left_out: string[]
  absent: { metric: string; reason: string; n: number }[]
}

export type Report = {
  info: Record<string, unknown> & { served?: Record<string, string[]>; spent_usd?: Record<string, number> }
  header: string[]
  groups: ReportGroup[]
  profile: string
  profiles: string[]
}

export type EpisodeItem = {
  ep_id: string
  task_id: string
  kind: string
  condition: string
  status: string
  failure: string | null
  row: Record<string, string | number>
  metrics: Record<string, number>
  not_scored: string | null
  no_output: string[]
  words: number
}

export type DiffLine = [op: " " | "-" | "+" | "@", text: string]

export type Shown = {
  against: string | null
  lines: DiffLine[]
  files: { path: string; lines: DiffLine[] }[]
  removed: string[]
  added: string[]
}

export type Judgment = {
  id: string
  metric: string
  question: string
  /** the judge's answer: yes/no, or a probability for A/B items; null when unscored */
  model: boolean | number | null
  unscored: string | null
  judge: { use: string; endpoint: string; model: string } | null
  replies: string[] | null
  about: string
  evidence?: string | null
  evidence_is?: string | null
  context?: string[]
  a?: string
  b?: string
  a_is?: string | null
  b_is?: string | null
  /** a finding of a judge that read the whole rollout: the turns and files it cites */
  cites?: string[]
}

export type ToolCall = {
  name: string
  arguments: unknown
  result: { summary: string; text: string } | null
}

export type Call = {
  actor: string
  endpoint: string
  finish_reason: string | null
  usage: Record<string, number> | null
  reasoning: string | null
  response: string
  tool_calls: ToolCall[]
  served: string | null
}

export type Turn = { actor: string; turn_idx: number | null; message: string | null; calls: Call[] }

export type EventView = { idx: number; actor: string; kind: string; text: string; seen_by: string[] }

export type Score = {
  not_scored?: string
  metrics?: Record<string, number>
  prose?: Record<string, number>
  counts?: Record<string, number>
  code?: string
}

export type Scene = {
  mode?: string
  k?: number
  exemplars?: { text: string; [k: string]: unknown }[]
  y_star?: { text: string; [k: string]: unknown }
  beat_sheet?: unknown
}

export type Episode = {
  ep_id: string
  condition: string
  twin_of: string | null
  status: string
  failure: string | null
  actors: Record<string, string>
  meta: Record<string, unknown>
  system: Record<string, string>
  written: { label: string; text: string }[]
  turns: Turn[]
  files: { path: string; chars: number; final?: string; initial?: string | null; writers?: string[]; diff?: DiffLine[] | null }[]
  events: EventView[]
  score: Score
  judgments: Judgment[] | null
  shown: Shown
  scene?: Scene
  edit?: { path: string; objection: string | null; diff: DiffLine[] }
}

export type Family = {
  run: RunSummary
  profile: string
  profiles: string[]
  task_id: string
  kind: string
  row: Record<string, string | number>
  split: string
  episodes: Episode[]
  elsewhere: string[]
}

export type TaskItem = { task_id: string; kind: string; split: string; row: Record<string, string | number> }

export type TaskEpisode = {
  ep_id: string
  condition: string
  twin_of: string | null
  parent: string | null
  actors: Record<string, string>
  tools: Record<string, string[]>
  system: Record<string, string>
  shown: Shown
  files: { path: string; chars: number; text: string; readable: string[] }[]
  events: EventView[]
}

export type TaskRow = {
  task_id: string
  suite: string
  kind: string
  split: string
  row: Record<string, unknown>
  error?: string
  episodes?: TaskEpisode[]
  fields?: { field: string; value: unknown; reaches: Record<string, number> }[]
  judged_by?: Record<string, string>
  diagram?: string
}

async function get<T>(path: string, params: Record<string, string | undefined>): Promise<T> {
  const q = new URLSearchParams(Object.entries(params).filter((kv): kv is [string, string] => kv[1] !== undefined))
  const res = await fetch(`/api/${path}?${q}`)
  if (!res.ok) throw new Error((await res.text()) || `${res.status} from /api/${path}`)
  return res.json() as Promise<T>
}

export const runsQuery = queryOptions({ queryKey: ["runs"], queryFn: () => get<RunSummary[]>("runs", {}) })

export const reportQuery = (run: string, judges?: string) =>
  queryOptions({ queryKey: ["report", run, judges], queryFn: () => get<Report>("report", { run, judges }) })

export const episodesQuery = (run: string, judges?: string) =>
  queryOptions({
    queryKey: ["episodes", run, judges],
    queryFn: () => get<{ profile: string; episodes: EpisodeItem[] }>("episodes", { run, judges }),
  })

export const familyQuery = (run: string, ep: string, judges?: string) =>
  queryOptions({ queryKey: ["family", run, ep, judges], queryFn: () => get<Family>("episode", { run, ep, judges }) })

export const tasksQuery = (suite: string, heldOut: boolean) =>
  queryOptions({
    queryKey: ["tasks", suite, heldOut],
    queryFn: () => get<{ suites: string[]; tasks: TaskItem[] }>("tasks", { suite, test: heldOut ? "held-out" : undefined }),
  })

export const taskQuery = (suite: string, id: string, heldOut: boolean) =>
  queryOptions({
    queryKey: ["task", suite, id, heldOut],
    queryFn: () => get<TaskRow>("task", { suite, id, test: heldOut ? "held-out" : undefined }),
  })

export const suitesQuery = queryOptions({ queryKey: ["suites"], queryFn: () => get<string[]>("suites", {}) })

export type KindInfo = {
  kind: string
  title: string
  tests: string
  given: string
  kept: string | null
  arms: Record<string, string>
  twin: string | null
  rows: Record<string, number>
  example: string | null
  diagram: string | null
}

export type Kinds = {
  suite: string
  kinds: KindInfo[]
  metrics: Record<string, string>
  prose: Record<string, string>
  judged_by: Record<string, string>
}

export const kindsQuery = (suite: string) =>
  queryOptions({ queryKey: ["kinds", suite], queryFn: () => get<Kinds>("kinds", { suite }), staleTime: Infinity })

export type CompareRun = {
  run: string
  endpoint: string
  model: string
  profile: string
  episodes: {
    ep_id: string
    condition: string
    twin_of: string | null
    status: string
    failure: string | null
    written: { label: string; text: string }[]
    metrics: Record<string, number>
    not_scored: string | null
    judged: { id: string; metric: string; question: string; model: boolean | number | null; unscored: string | null; about: string }[]
  }[]
}

export const compareQuery = (task: string) =>
  queryOptions({ queryKey: ["compare", task], queryFn: () => get<CompareRun[]>("compare", { task }) })

// -- a library world: its situations, their groups of rollouts, and one rollout ------------------

export type WorldSummary = { name: string; split: string | null; situations: number; rollouts: number; profiles: string[] }

export type WorldRollout = {
  ep_id: string
  status: string
  failure: string | null
  words: number
  /** rubric -> what the rollout scored under it */
  scores: Record<string, { metrics: Record<string, number>; not_scored: string | null }>
}

export type WorldGroup = {
  group: string
  size: number | null
  round: string | null
  endpoint: string | null
  model: string | null
  run: string | null
  rollouts: WorldRollout[]
}

export type WorldSituation = {
  name: string
  suite: string
  kind: string
  request: string | null
  seats: Record<string, string>
  rubrics: string[]
  files: string[]
  groups: WorldGroup[]
}

export type World = WorldSummary & { profile: string; rubrics: string[]; situations: WorldSituation[] }

export const worldsQuery = queryOptions({ queryKey: ["worlds"], queryFn: () => get<WorldSummary[]>("worlds", {}) })

export const worldQuery = (world: string, judges?: string) =>
  queryOptions({ queryKey: ["world", world, judges], queryFn: () => get<World>("world", { world, judges }) })

export const rolloutQuery = (world: string, ep: string, judges?: string) =>
  queryOptions({ queryKey: ["rollout", world, ep, judges], queryFn: () => get<Family>("episode", { world, ep, judges }) })
