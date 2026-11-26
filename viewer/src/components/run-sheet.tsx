// A run's details (`?about=<run>`): what answered, when, at what cost, scored by what code.
import { useQuery } from "@tanstack/react-query"
import { useNavigate, useSearch } from "@tanstack/react-router"

import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { reportQuery, runsQuery } from "@/lib/api"

export function RunSheet() {
  const { about, judges } = useSearch({ strict: false }) as { about?: string; judges?: string }
  const navigate = useNavigate()
  const { data: runs = [] } = useQuery(runsQuery)
  const run = runs.find((r) => r.name === about)
  const { data: report } = useQuery({ ...reportQuery(about ?? "", judges), enabled: !!run?.suite })
  const close = () => navigate({ to: ".", search: (s: Record<string, unknown>) => ({ ...s, about: undefined }) })
  return (
    <Sheet open={!!run} onOpenChange={(o) => !o && close()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-lg">
        <SheetHeader>
          <SheetTitle>{run?.endpoint ?? run?.name}</SheetTitle>
          <SheetDescription className="font-mono text-xs">{run?.name}</SheetDescription>
        </SheetHeader>
        {run && (
          <div className="space-y-5 px-4 pb-6 text-sm">
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2">
              <dt className="text-muted-foreground">Weights</dt>
              <dd className="font-mono text-xs break-all">{run.model}</dd>
              <dt className="text-muted-foreground">Suite</dt>
              <dd>
                {run.suite} · {run.split}
              </dd>
              <dt className="text-muted-foreground">Started</dt>
              <dd>{run.started ? new Date(run.started).toLocaleString() : "not recorded"}</dd>
              <dt className="text-muted-foreground">Episodes</dt>
              <dd className="tabular-nums">
                {run.episodes} ({run.twins} twins)
              </dd>
              <dt className="text-muted-foreground">Judges</dt>
              <dd>{report?.profile ?? "…"}</dd>
            </dl>
            {report && (
              <ul className="space-y-2 border-t pt-4 text-xs leading-relaxed text-muted-foreground">
                {report.header.map((line) => (
                  <li key={line} className="break-words">
                    {line}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}
