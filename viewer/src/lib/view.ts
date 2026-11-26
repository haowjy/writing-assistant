// What every screen shares: the scope's runs with their colours and visibility, and how
// metrics are classed and printed.
import { useQuery } from "@tanstack/react-query"
import { useSearch } from "@tanstack/react-router"

import { runsQuery, type RunSummary } from "@/lib/api"

export type Run = RunSummary & { suite: string; split: string; endpoint: string }

/** The colour slot of a run: its place among its scope's runs, in the order they were loaded.
 * A run keeps its colour when another is hidden (colour follows the run, not its rank). */
export const runColor = (slot: number) => `var(--run-${(slot % 8) + 1})`

export function useScopeRuns(suite: string, split: string) {
  const { data = [] } = useQuery(runsQuery)
  const hidden = useHidden()
  const all = data.filter((r): r is Run => r.suite === suite && r.split === split)
  const slot = new Map(all.map((r, i) => [r.name, i]))
  return {
    all,
    visible: all.filter((r) => !hidden.has(r.name)),
    color: (name: string) => runColor(slot.get(name) ?? 0),
  }
}

/** Runs hidden in the sidebar: `?hide=a,b` in the URL, kept across screens. */
export function useHidden(): Set<string> {
  const { hide } = useSearch({ strict: false }) as { hide?: string }
  return new Set(hide ? hide.split(",") : [])
}

export const shortTask = (taskId: string, suite: string, kind: string) =>
  taskId.replace(`${suite}:${kind}:`, "").replace(`${suite}:`, "")

// Metrics folded behind "All metrics": the same for every kind, so they would bury the outcome.
export const isProfile = (m: string) =>
  m.startsWith("readability_") || m.startsWith("overlap_") || m === "output_words"

/** Decimals that print every value of a row or column alike: the fewest (up to 3) that show
 * each value as it would print at 3, so 0, 0.5 and 1 read 0.0, 0.5, 1.0 and 0.479 keeps 3. */
export function digits(values: (number | null | undefined)[]): number {
  const xs = values.filter((x): x is number => typeof x === "number" && !Number.isNaN(x))
  const top = Math.max(0, ...xs.map(Math.abs))
  if (top >= 100) return 0
  const need = (x: number) => [0, 1, 2, 3].find((d) => Math.abs(Number(x.toFixed(d)) - Number(x.toFixed(3))) < 1e-9) ?? 3
  return Math.min(top >= 10 ? 1 : 3, Math.max(0, ...xs.map(need)))
}

export function fmt(x: number | null | undefined, d?: number): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "–"
  const n = d ?? digits([x])
  return x.toLocaleString("en", { minimumFractionDigits: n, maximumFractionDigits: n })
}

export const answerText = (a: boolean | number | null) =>
  a === null ? "unscored" : typeof a === "boolean" ? (a ? "Yes" : "No") : a.toFixed(2)
