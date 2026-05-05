interface Props {
  level: string | null | undefined;
  score?: number | null;
  showAction?: boolean;
}

interface RiskConfig {
  colors: string;
  label: string;
  action: string;
  actionColor: string;
}

function getRiskConfig(score: number | null | undefined, level: string | null | undefined): RiskConfig {
  if (score != null) {
    if (score < 0.3) {
      return {
        colors: 'bg-green-100 text-green-800 border-green-200',
        label: 'LOW RISK',
        action: 'Auto Submit',
        actionColor: 'text-green-700 bg-green-50 border-green-300',
      };
    }
    if (score <= 0.7) {
      return {
        colors: 'bg-yellow-100 text-yellow-800 border-yellow-200',
        label: 'REVIEW',
        action: 'Optional Review',
        actionColor: 'text-yellow-700 bg-yellow-50 border-yellow-300',
      };
    }
    return {
      colors: 'bg-red-100 text-red-800 border-red-200',
      label: 'HIGH RISK',
      action: 'Needs Fix',
      actionColor: 'text-red-700 bg-red-50 border-red-300',
    };
  }

  // Fallback to level string
  const map: Record<string, RiskConfig> = {
    HIGH: { colors: 'bg-red-100 text-red-800 border-red-200', label: 'HIGH RISK', action: 'Needs Fix', actionColor: 'text-red-700 bg-red-50 border-red-300' },
    MEDIUM: { colors: 'bg-yellow-100 text-yellow-800 border-yellow-200', label: 'REVIEW', action: 'Optional Review', actionColor: 'text-yellow-700 bg-yellow-50 border-yellow-300' },
    LOW: { colors: 'bg-green-100 text-green-800 border-green-200', label: 'LOW RISK', action: 'Auto Submit', actionColor: 'text-green-700 bg-green-50 border-green-300' },
  };
  return map[level || 'LOW'] || map.LOW;
}

export default function RiskBadge({ level, score, showAction = false }: Props) {
  if (!level && score == null) {
    return <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600 border border-gray-200">Pending</span>;
  }

  const config = getRiskConfig(score, level);

  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border ${config.colors}`}>
        {score != null ? `${(score * 100).toFixed(0)}%` : config.label}
      </span>
      {showAction && (
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${config.actionColor}`}>
          {config.action}
        </span>
      )}
    </span>
  );
}

export { getRiskConfig };
