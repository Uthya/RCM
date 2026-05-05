import type { RiskFactor } from '../types';

interface Props {
  factors: RiskFactor[];
}

export default function RiskFactors({ factors }: Props) {
  if (!factors || factors.length === 0) return null;

  const maxImpact = Math.max(...factors.map((f) => Math.abs(f.impact)), 0.01);

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">Top Risk Factors</h4>
      {factors.map((factor, i) => {
        const width = Math.round((Math.abs(factor.impact) / maxImpact) * 100);
        const isNegative = factor.impact < 0;
        return (
          <div key={i} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-700">{factor.display_name}</span>
              <span className="text-gray-500 font-mono text-xs">
                {isNegative ? '' : '+'}{factor.impact.toFixed(3)}
              </span>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2.5">
              <div
                className={`h-2.5 rounded-full ${isNegative ? 'bg-green-400' : 'bg-red-400'}`}
                style={{ width: `${width}%` }}
              />
            </div>
            <div className="text-xs text-gray-400">Value: {factor.value}</div>
          </div>
        );
      })}
    </div>
  );
}
