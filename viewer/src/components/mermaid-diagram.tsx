import { useEffect, useId, useState } from "react"

import { useTheme } from "@/lib/theme"

/** Mermaid source rendered to SVG; the library loads on first use, not with the page. */
export function MermaidDiagram({ source }: { source: string }) {
  const { resolved } = useTheme()
  const id = useId().replace(/:/g, "")
  const [svg, setSvg] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let live = true
    import("mermaid").then(async ({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "strict",
        theme: resolved === "dark" ? "dark" : "neutral",
        fontFamily: "inherit",
        flowchart: { htmlLabels: true, wrappingWidth: 220 },
      })
      try {
        const out = await mermaid.render(`mm${id}-${resolved}`, source)
        if (live) setSvg(out.svg)
      } catch (e) {
        if (live) setError(String(e))
      }
    })
    return () => {
      live = false
    }
  }, [source, resolved, id])
  if (error) return <pre className="text-destructive text-sm whitespace-pre-wrap">{error}</pre>
  if (!svg) return <div className="text-muted-foreground text-sm">Drawing the diagram…</div>
  return <div className="overflow-x-auto [&_svg]:mx-auto [&_svg]:h-auto" dangerouslySetInnerHTML={{ __html: svg }} />
}
