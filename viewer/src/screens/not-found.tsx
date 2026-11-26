import { Link } from "@tanstack/react-router"

import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"

export function NotFound() {
  return (
    <Empty className="mt-24">
      <EmptyHeader>
        <EmptyTitle>Nothing here</EmptyTitle>
        <EmptyDescription>
          No loaded run played this, or the address is wrong. <Link to="/">Go to the overview</Link>.
        </EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}
