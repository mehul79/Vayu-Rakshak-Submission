import { Card } from "@/components/ui/card"
import type React from "react"

interface MetricsCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon?: React.ReactNode
}

export function MetricsCard({ title, value, subtitle, icon }: MetricsCardProps) {
  return (
    <Card className="p-4 bg-white shadow-md">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm text-gray-600">{title}</h3>
        {icon && <div className="text-gray-400">{icon}</div>}
      </div>
      <div className="flex items-end justify-between">
        <div>
          <p className="text-2xl font-bold text-gray-800">{value}</p>
          {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
        </div>
      </div>
    </Card>
  )
}



