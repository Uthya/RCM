import type { DashboardSummary } from '../types';

interface Props {
  summary: DashboardSummary | null;
  loading?: boolean;
}

interface CardData {
  label: string;
  value: string | number;
  color: string;
}

export default function StatsCards({ summary, loading }: Props) {
  if (loading || !summary) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-white rounded-lg shadow p-5 animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-24 mb-2" />
            <div className="h-8 bg-gray-200 rounded w-16" />
          </div>
        ))}
      </div>
    );
  }

  const cards: CardData[] = [
    { label: 'Total Claims', value: summary.total_claims, color: 'text-indigo-600' },
    { label: 'Needs Fix (>70%)', value: summary.high_risk_count, color: 'text-red-600' },
    { label: 'Review (30-70%)', value: summary.medium_risk_count, color: 'text-yellow-600' },
    { label: 'Auto Submit (<30%)', value: summary.low_risk_count, color: 'text-green-600' },
    { label: 'Predicted', value: summary.total_predicted, color: 'text-blue-600' },
    { label: 'Remittances', value: summary.total_remittances, color: 'text-purple-600' },
    { label: 'Denial Rate', value: `${(summary.denial_rate * 100).toFixed(1)}%`, color: 'text-red-600' },
    { label: 'Total Billed', value: `$${summary.total_billed.toLocaleString()}`, color: 'text-gray-700' },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((card) => (
        <div key={card.label} className="bg-white rounded-lg shadow p-5">
          <p className="text-sm font-medium text-gray-500">{card.label}</p>
          <p className={`text-2xl font-bold mt-1 ${card.color}`}>{card.value}</p>
        </div>
      ))}
    </div>
  );
}
