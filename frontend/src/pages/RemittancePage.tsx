import { useState, useEffect } from 'react';
import { fetchRemittances } from '../services/api';
import type { Remittance } from '../types';

export default function RemittancePage() {
  const [remittances, setRemittances] = useState<Remittance[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [skip, setSkip] = useState(0);
  const limit = 25;

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const data = await fetchRemittances({ skip, limit });
        setRemittances(data.remittances);
        setTotal(data.total);
      } catch (err) {
        console.error('Failed to load remittances', err);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [skip]);

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(skip / limit) + 1;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Remittances (835)</h1>
        <p className="text-sm text-gray-500 mt-1">{total} total remittance records</p>
      </div>

      <div className="bg-white shadow rounded-lg overflow-hidden">
        {loading ? (
          <div className="text-center py-12 text-gray-500">Loading remittances...</div>
        ) : remittances.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            No remittance data. Upload an 835 file to see payment outcomes.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Claim ID</th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Payer</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Billed</th>
                  <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Paid</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">CARC Codes</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Trace #</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {remittances.map((r) => (
                  <tr key={r.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-medium text-indigo-600">{r.claim_id}</td>
                    <td className="px-4 py-3 text-center">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${
                        r.claim_status === 'denied' ? 'bg-red-100 text-red-800' :
                        r.claim_status === 'paid' ? 'bg-green-100 text-green-800' :
                        'bg-yellow-100 text-yellow-800'
                      }`}>
                        {r.claim_status.charAt(0).toUpperCase() + r.claim_status.slice(1)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-700">{r.payer_name}</td>
                    <td className="px-4 py-3 text-sm text-right font-mono">${r.billed_amount.toFixed(2)}</td>
                    <td className="px-4 py-3 text-sm text-right font-mono">${r.paid_amount.toFixed(2)}</td>
                    <td className="px-4 py-3 text-sm text-gray-500">
                      {r.carc_codes.length > 0 ? r.carc_codes.join(', ') : '-'}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-500">{r.trace_number}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
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
          <span className="text-sm text-gray-600">Page {currentPage} of {totalPages}</span>
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
