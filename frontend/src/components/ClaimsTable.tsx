import { useNavigate } from 'react-router-dom';
import type { Claim } from '../types';
import RiskBadge from './RiskBadge';

interface Props {
  claims: Claim[];
  loading?: boolean;
}

function getActionLabel(score: number | null): { text: string; cls: string } {
  if (score == null) return { text: 'Pending', cls: 'text-gray-400 bg-gray-50' };
  if (score < 0.3) return { text: 'Auto Submit', cls: 'text-green-700 bg-green-50' };
  if (score <= 0.7) return { text: 'Review', cls: 'text-yellow-700 bg-yellow-50' };
  return { text: 'Needs Fix', cls: 'text-red-700 bg-red-50' };
}

export default function ClaimsTable({ claims, loading }: Props) {
  const navigate = useNavigate();

  if (loading) {
    return <div className="text-center py-12 text-gray-500">Loading claims...</div>;
  }

  if (claims.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No claims found. Upload an 837 file to get started.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Claim ID</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Patient</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Payer</th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Charge</th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CPT</th>
            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Denial Risk</th>
            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Outcome</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {claims.map((claim) => {
            const action = getActionLabel(claim.risk_score);
            return (
              <tr
                key={claim.claim_id}
                onClick={() => navigate(`/claims/${claim.claim_id}`)}
                className="hover:bg-gray-50 cursor-pointer transition-colors"
              >
                <td className="px-4 py-3 text-sm font-medium text-indigo-600">{claim.claim_id}</td>
                <td className="px-4 py-3 text-sm text-gray-700">{claim.patient_name}</td>
                <td className="px-4 py-3 text-sm text-gray-700">{claim.payer_name}</td>
                <td className="px-4 py-3 text-sm text-gray-700 text-right font-mono">
                  ${claim.total_charge.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                </td>
                <td className="px-4 py-3 text-sm text-gray-500">
                  {claim.service_lines.map((sl) => sl.cpt_code).join(', ')}
                </td>
                <td className="px-4 py-3 text-center">
                  <RiskBadge level={claim.risk_level} score={claim.risk_score} />
                </td>
                <td className="px-4 py-3 text-center">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${action.cls}`}>
                    {action.text}
                  </span>
                </td>
                <td className="px-4 py-3 text-center text-sm">
                  {claim.actual_outcome ? (
                    <span className={`font-medium ${claim.actual_outcome === 'denied' ? 'text-red-600' : 'text-green-600'}`}>
                      {claim.actual_outcome.charAt(0).toUpperCase() + claim.actual_outcome.slice(1)}
                    </span>
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
