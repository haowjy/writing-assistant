import { createContext, useContext, useEffect, useState, type ReactNode } from "react"

type Choice = "light" | "dark" | "system"
const KEY = "wa-bench-view-theme"
const dark = window.matchMedia("(prefers-color-scheme: dark)")

const Theme = createContext<{ choice: Choice; resolved: "light" | "dark"; set: (c: Choice) => void } | null>(null)

/** Light or dark: the system's unless chosen, kept in this browser. */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoice] = useState<Choice>(() => (localStorage.getItem(KEY) as Choice) || "system")
  const [systemDark, setSystemDark] = useState(dark.matches)
  useEffect(() => {
    const on = (e: MediaQueryListEvent) => setSystemDark(e.matches)
    dark.addEventListener("change", on)
    return () => dark.removeEventListener("change", on)
  }, [])
  const resolved = choice === "system" ? (systemDark ? "dark" : "light") : choice
  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolved === "dark")
  }, [resolved])
  const set = (c: Choice) => {
    localStorage.setItem(KEY, c)
    setChoice(c)
  }
  return <Theme.Provider value={{ choice, resolved, set }}>{children}</Theme.Provider>
}

export function useTheme() {
  const t = useContext(Theme)
  if (!t) throw new Error("useTheme outside ThemeProvider")
  return t
}
