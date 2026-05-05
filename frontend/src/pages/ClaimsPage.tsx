import { useState, useEffect } from 'react';
import ClaimsTable from '../components/ClaimsTable';
import { fetchClaims } from '../services/api';
import type { Claim } from '../types';

export default function ClaimsPage() {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [skip, setSkip] = useState(0);
  const [riskFilter, setRiskFilter] = useState<string>('');
  const limit = 25;

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const params: Record<string, any> = { skip, limit, sort_by: 'created_at', sort_order: -1 };
        if (riskFilter) params.risk_level = riskFilter;
        const data = await fetchClaims(params);
        setClaims(data.claims);
        setTotal(data.total);
      } catch (err) {
        console.error('Failed to load claims', err);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [skip, riskFilter]);

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(skip / limit) + 1;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Claims</h1>
          <p className="text-sm text-gray-500 mt-1">{total} total claims</p>
        </div>
        <div className="flex items-center space-x-3">
          <label className="text-sm text-gray-600">Filter:</label>
          <select
            value={riskFilter}
            onChange={(e) => { setRiskFilter(e.target.value); setSkip(0); }}
            className="border border-gray-300 rounded-md text-sm px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">All Claims</option>
            <option value="HIGH">Needs Fix (&gt;70%)</option>
            <option value="MEDIUM">Review (30-70%)</option>
            <option value="LOW">Auto Submit (&lt;30%)</option>
          </select>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg overflow-hidden">
        <ClaimsTable claims={claims} loading={loading} />
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <button
            onClick={() => setSkip(Math.max(0, skip - limit))}
            disabled={skip === 0}
            className="px-4 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Previous
          </button>
          <span className="text-sm text-gray-600">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setSkip(skip + limit)}
            disabled={currentPage >= totalPages}
            className="px-4 py-2 text-sm border rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
