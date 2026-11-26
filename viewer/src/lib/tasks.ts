// A kind's task list as the Tasks table shows it, filtered and sorted by its URL: the same
// order `j`/`k` walk on a task.
import { useQueries } from "@tanstack/react-query"

import type { TasksSearch } from "@/router"
import { episodesQuery, type EpisodeItem } from "@/lib/api"
import { isProfile, type Run } from "@/lib/view"

export type TaskRowView = {
  task_id: string
  row: Record<string, string | number>
  cells: Map<string, EpisodeItem | undefined>
}

export function useKindEpisodes(runs: Run[], judges?: string) {
  const qs = useQueries({ queries: runs.map((r) => episodesQuery(r.name, judges)) })
  return {
    pending: qs.some((q) => q.isPending),
    byRun: new Map(runs.map((r, i) => [r.name, qs[i].data?.episodes ?? []])),
  }
}

/** The arms and metrics a kind's episodes have in these runs, in the kind's arm order. */
export function choices(byRun: Map<string, EpisodeItem[]>, kind: string, armOrder: string[], arm?: string) {
  const arms = new Set<string>()
  for (const eps of byRun.values()) for (const e of eps) if (e.kind === kind) arms.add(e.condition)
  const armList = [...arms].sort((a, b) => armOrder.indexOf(a) - armOrder.indexOf(b))
  const chosenArm = arm && arms.has(arm) ? arm : armList[0]
  const metrics = new Set<string>()
  for (const eps of byRun.values())
    for (const e of eps) if (e.kind === kind && e.condition === chosenArm) for (const m of Object.keys(e.metrics)) metrics.add(m)
  const metricList = [...metrics].sort((a, b) => Number(isProfile(a)) - Number(isProfile(b)) || a.localeCompare(b))
  return { arms: armList, arm: chosenArm, metrics: metricList }
}

export const valueOf = (e: EpisodeItem | undefined, metric: string) => (e ? e.metrics[metric] : undefined)

export function taskRows(
  byRun: Map<string, EpisodeItem[]>,
  runs: Run[],
  kind: string,
  arm: string | undefined,
  metric: string | undefined,
  s: TasksSearch,
): TaskRowView[] {
  const rows = new Map<string, TaskRowView>()
  for (const r of runs)
    for (const e of byRun.get(r.name) ?? []) {
      if (e.kind !== kind || e.condition !== arm) continue
      const row = rows.get(e.task_id) ?? { task_id: e.task_id, row: e.row, cells: new Map() }
      row.cells.set(r.name, e)
      rows.set(e.task_id, row)
    }
  const q = s.q?.toLowerCase()
  let out = [...rows.values()].filter((t) => !q || `${t.task_id} ${Object.values(t.row).join(" ")}`.toLowerCase().includes(q))
  if (metric) {
    const vals = (t: TaskRowView) => runs.map((r) => valueOf(t.cells.get(r.name), metric))
    if (s.only === "any") out = out.filter((t) => vals(t).some((v) => v !== undefined && v > 0))
    if (s.only === "every") out = out.filter((t) => vals(t).every((v) => v !== undefined && v > 0))
    if (s.only === "unscored") out = out.filter((t) => vals(t).some((v) => v === undefined))
  }
  const sortRun = runs.find((r) => r.name === s.sort)
  out.sort((a, b) => a.task_id.localeCompare(b.task_id, "en", { numeric: true }))
  if (sortRun && metric) {
    const dir = s.desc === false ? 1 : -1
    const v = (t: TaskRowView) => valueOf(t.cells.get(sortRun.name), metric)
    out.sort((a, b) => {
      const x = v(a), y = v(b)
      if (x === undefined) return y === undefined ? 0 : 1
      if (y === undefined) return -1
      return dir * (x - y)
    })
  }
  return out
}
