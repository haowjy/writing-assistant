// Every view has a URL (design: viewer-navigation.md). Visible runs (`hide`) and the judge
// profile (`judges`) ride along on every link; filters, the metric, the arm, the run, side by
// side and an open verdict are each view's own search params.
import { QueryClient } from "@tanstack/react-query"
import {
  createRootRouteWithContext,
  createRoute,
  createRouter,
  redirect,
  retainSearchParams,
} from "@tanstack/react-router"

import { Frame } from "@/components/frame"
import { episodesQuery, runsQuery, worldsQuery } from "@/lib/api"
import { KindTasks } from "@/screens/kind-tasks"
import { Overview } from "@/screens/overview"
import { TaskView } from "@/screens/task"
import { WorldScreen } from "@/screens/world"
import { NotFound } from "@/screens/not-found"

export const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 30_000, retry: 1 } } })

type RootSearch = { hide?: string; judges?: string; about?: string }
const str = (v: unknown) => (typeof v === "string" && v !== "" ? v : undefined)

const root = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  validateSearch: (s: Record<string, unknown>): RootSearch => ({ hide: str(s.hide), judges: str(s.judges), about: str(s.about) }),
  search: { middlewares: [retainSearchParams(["hide", "judges"])] },
  component: Frame,
  notFoundComponent: NotFound,
})

const index = createRoute({
  getParentRoute: () => root,
  path: "/",
  beforeLoad: async ({ context }) => {
    const runs = await context.queryClient.ensureQueryData(runsQuery)
    const first = runs.find((r) => r.suite && r.split)
    if (!first) {
      // no run loaded: a world, when the page was started on one
      const worlds = await context.queryClient.ensureQueryData(worldsQuery)
      if (worlds.length) throw redirect({ to: "/world/$world", params: { world: worlds[0].name } })
    }
    throw redirect({ to: "/$suite/$split", params: { suite: first?.suite ?? "1a", split: first?.split ?? "dev" } })
  },
})

const scope = createRoute({ getParentRoute: () => root, path: "$suite/$split" })

export const overviewRoute = createRoute({ getParentRoute: () => scope, path: "/", component: Overview })

// A kind's About page is held until Jimmy has seen its wireframe: the kind opens on its tasks.
const kind = createRoute({
  getParentRoute: () => scope,
  path: "kind/$kind",
  beforeLoad: ({ params }) => {
    throw redirect({ to: "/$suite/$split/kind/$kind/tasks", params })
  },
})

export type TasksSearch = {
  arm?: string
  metric?: string
  only?: "any" | "every" | "unscored"
  sort?: string
  desc?: boolean
  q?: string
  /** held-out test rows, listed only on this explicit ask (reading them is a use of the test split) */
  test?: boolean
}
const tasksSearch = (s: Record<string, unknown>): TasksSearch => ({
  arm: str(s.arm),
  metric: str(s.metric),
  only: s.only === "any" || s.only === "every" || s.only === "unscored" ? s.only : undefined,
  sort: str(s.sort),
  desc: s.desc === false ? false : undefined,
  q: str(s.q),
  test: s.test === true || s.test === "held-out" ? true : undefined,
})

export const kindTasksRoute = createRoute({
  getParentRoute: () => scope,
  path: "kind/$kind/tasks",
  validateSearch: tasksSearch,
  component: KindTasks,
})

// A task's definition page is held too: a task opens on its first arm as the first visible run played it.
const task = createRoute({
  getParentRoute: () => scope,
  path: "task/$task",
  validateSearch: tasksSearch,
  beforeLoad: async ({ context, params, search }) => {
    const runs = (await context.queryClient.ensureQueryData(runsQuery)).filter(
      (r) => r.suite === params.suite && r.split === params.split,
    )
    for (const r of runs) {
      const { episodes } = await context.queryClient.ensureQueryData(episodesQuery(r.name))
      const ep = episodes.find((e) => e.task_id === params.task)
      if (ep) throw redirect({ to: "/$suite/$split/task/$task/$arm", params: { ...params, arm: ep.condition }, search: { ...search, run: r.name } })
    }
  },
  component: NotFound,
})

export type TaskSearch = TasksSearch & { run?: string; compare?: boolean; verdict?: string }
export const taskArmRoute = createRoute({
  getParentRoute: () => scope,
  path: "task/$task/$arm",
  validateSearch: (s: Record<string, unknown>): TaskSearch => ({
    ...tasksSearch(s),
    run: str(s.run),
    compare: s.compare === true || s.compare === 1 || s.compare === "1" ? true : undefined,
    verdict: str(s.verdict),
  }),
  component: TaskView,
})

// A library world: its situations and their rollouts; `?ep=` opens one rollout, `?verdict=` a finding.
export type WorldSearch = { ep?: string; verdict?: string }
export const worldRoute = createRoute({
  getParentRoute: () => root,
  path: "world/$world",
  validateSearch: (s: Record<string, unknown>): WorldSearch => ({ ep: str(s.ep), verdict: str(s.verdict) }),
  component: WorldScreen,
})

const tree = root.addChildren([index, worldRoute, scope.addChildren([overviewRoute, kind, kindTasksRoute, task, taskArmRoute])])

export const router = createRouter({ routeTree: tree, context: { queryClient }, defaultPreload: "intent", scrollRestoration: true })

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router
  }
}
