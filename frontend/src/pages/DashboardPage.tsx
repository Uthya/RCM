import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import StatsCards from '../components/StatsCards';
import { fetchDashboardSummary, fetchRiskDistribution, fetchPayerStats } from '../services/api';
import type { DashboardSummary, RiskBucket, PayerStat } from '../types';

export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [buckets, setBuckets] = useState<RiskBucket[]>([]);
  const [payers, setPayers] = useState<PayerStat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const [s, r, p] = await Promise.all([
          fetchDashboardSummary(),
          fetchRiskDistribution(),
          fetchPayerStats(),
        ]);
        setSummary(s);
        setBuckets(r.buckets);
        setPayers(p.payers);
      } catch (err) {
        console.error('Dashboard load error', err);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const getBarColor = (label: string) => {
    const val = parseFloat(label);
    if (val >= 0.7) return '#ef4444';
    if (val >= 0.4) return '#eab308';
    return '#22c55e';
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-1">Overview of denial prediction pipeline</p>
      </div>

      <StatsCards summary={summary} loading={loading} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution Chart */}
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Risk Score Distribution</h2>
          {buckets.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={buckets}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range_label" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Claims">
                  {buckets.map((entry, index) => (
                    <Cell key={index} fill={getBarColor(entry.range_label)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-gray-400">No prediction data yet</div>
          )}
        </div>

        {/* Payer Denial Rates */}
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Payer Denial Rates</h2>
          {payers.length > 0 ? (
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Payer</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Claims</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Denied</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Denial Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {payers.map((p, i) => (
                  <tr key={i}>
                    <td className="px-3 py-2 font-medium">{p.payer_name}</td>
                    <td className="px-3 py-2 text-right">{p.total_claims}</td>
                    <td className="px-3 py-2 text-right text-red-600">{p.denied_count}</td>
                    <td className="px-3 py-2 text-right font-mono">
                      {(p.denial_rate * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-center py-12 text-gray-400">No remittance data yet</div>
          )}
        </div>
      </div>
    </div>
  );
}
