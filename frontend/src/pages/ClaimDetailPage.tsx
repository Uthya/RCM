import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { fetchClaim } from '../services/api';
import RiskBadge, { getRiskConfig } from '../components/RiskBadge';
import RiskFactors from '../components/RiskFactors';
import type { Claim } from '../types';

function getActionBanner(score: number): {
  bg: string; border: string; title: string; titleColor: string;
  description: string; icon: string;
} {
  if (score < 0.3) {
    return {
      bg: 'bg-green-50', border: 'border-green-300',
      titleColor: 'text-green-800',
      title: 'Safe to Submit',
      description: 'Low denial risk. This claim can be auto-submitted with confidence.',
      icon: 'check',
    };
  }
  if (score <= 0.7) {
    return {
      bg: 'bg-yellow-50', border: 'border-yellow-300',
      titleColor: 'text-yellow-800',
      title: 'Review Recommended',
      description: 'Moderate denial risk. Review the flagged issues below before submitting.',
      icon: 'warning',
    };
  }
  return {
    bg: 'bg-red-50', border: 'border-red-300',
    titleColor: 'text-red-800',
    title: 'Fix Before Submitting',
    description: 'High denial risk. Address the issues below to reduce the chance of denial.',
    icon: 'error',
  };
}

function getFriendlyFactorTip(feature: string, value: string): string {
  const tips: Record<string, string> = {
    modifier_missing: 'Add the appropriate modifier to service lines that are missing one.',
    total_charge: 'High charge amounts get extra scrutiny. Ensure documentation supports medical necessity.',
    payer_denial_rate: 'This payer has a high historical denial rate. Double-check all requirements.',
    cpt_denial_rate: 'This procedure code is frequently denied. Verify coding accuracy and attach supporting docs.',
    provider_denial_rate: 'This provider has elevated denials. Review claim for common rejection patterns.',
    dx_count: 'Few diagnosis codes may indicate insufficient clinical justification.',
    prior_auth_present: 'No prior authorization found. Verify if this procedure requires one for this payer.',
    patient_age: 'Patient age may affect coverage rules. Confirm eligibility.',
    place_of_service_encoded: 'Verify the place of service matches the procedure performed.',
    charge_per_line: 'Charge per line is high. Verify unit counts and pricing.',
    service_line_count: 'Multiple service lines — ensure all are properly coded with distinct CPTs.',
    has_multiple_cpt: 'Multiple CPT codes present. Verify modifier usage to avoid bundling denials.',
  };
  return tips[feature] || 'Review this factor and verify accuracy before submitting.';
}

export default function ClaimDetailPage() {
  const { claimId } = useParams<{ claimId: string }>();
  const [claim, setClaim] = useState<Claim | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!claimId) return;
    const load = async () => {
      setLoading(true);
      try {
        const data = await fetchClaim(claimId);
        setClaim(data);
      } catch (err: any) {
        setError(err?.response?.data?.detail || 'Failed to load claim');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [claimId]);

  if (loading) return <div className="text-center py-12 text-gray-500">Loading claim...</div>;
  if (error) return <div className="text-center py-12 text-red-500">{error}</div>;
  if (!claim) return <div className="text-center py-12 text-gray-500">Claim not found</div>;

  const banner = claim.risk_score != null ? getActionBanner(claim.risk_score) : null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link to="/claims" className="text-sm text-indigo-600 hover:underline">&larr; Back to Claims</Link>
          <h1 className="text-2xl font-bold text-gray-900 mt-1">Claim {claim.claim_id}</h1>
        </div>
        <RiskBadge level={claim.risk_level} score={claim.risk_score} showAction />
      </div>

      {/* Action Banner */}
      {banner && (
        <div className={`${banner.bg} border ${banner.border} rounded-lg p-4 flex items-start gap-3`}>
          <div className="flex-shrink-0 mt-0.5">
            {banner.icon === 'check' && (
              <svg className="w-6 h-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            {banner.icon === 'warning' && (
              <svg className="w-6 h-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            )}
            {banner.icon === 'error' && (
              <svg className="w-6 h-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </div>
          <div>
            <h3 className={`font-semibold ${banner.titleColor}`}>{banner.title}</h3>
            <p className="text-sm text-gray-600 mt-0.5">{banner.description}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Parsed Claim Data */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Claim Details</h2>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Patient</p>
                <p className="font-medium">{claim.patient_name}</p>
              </div>
              <div>
                <p className="text-gray-500">DOB / Gender</p>
                <p className="font-medium">{claim.patient_dob} / {claim.patient_gender || '-'}</p>
              </div>
              <div>
                <p className="text-gray-500">Payer</p>
                <p className="font-medium">{claim.payer_name}</p>
              </div>
              <div>
                <p className="text-gray-500">Payer ID</p>
                <p className="font-medium">{claim.payer_id}</p>
              </div>
              <div>
                <p className="text-gray-500">Billing Provider</p>
                <p className="font-medium">{claim.billing_provider_name}</p>
              </div>
              <div>
                <p className="text-gray-500">Provider NPI</p>
                <p className="font-medium">{claim.billing_provider_npi}</p>
              </div>
              <div>
                <p className="text-gray-500">Total Charge</p>
                <p className="font-medium">${claim.total_charge.toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
              </div>
              <div>
                <p className="text-gray-500">Place of Service</p>
                <p className="font-medium">{claim.place_of_service}</p>
              </div>
              <div>
                <p className="text-gray-500">Prior Auth</p>
                <p className="font-medium">{claim.prior_auth_number || 'None'}</p>
              </div>
              <div>
                <p className="text-gray-500">Taxonomy</p>
                <p className="font-medium">{claim.provider_taxonomy || '-'}</p>
              </div>
            </div>

            {/* Diagnosis Codes */}
            <div className="mt-4">
              <p className="text-sm text-gray-500 mb-1">Diagnosis Codes (ICD-10)</p>
              <div className="flex flex-wrap gap-2">
                {claim.diagnosis_codes.map((dx, i) => (
                  <span key={i} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                    {dx}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Service Lines */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">Service Lines</h2>
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">CPT</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Modifiers</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Charge</th>
                  <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">Units</th>
                  <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {claim.service_lines.map((sl, i) => (
                  <tr key={i}>
                    <td className="px-3 py-2 font-medium">{sl.cpt_code}</td>
                    <td className="px-3 py-2 text-gray-500">{sl.modifiers.join(', ') || '-'}</td>
                    <td className="px-3 py-2 text-right font-mono">${sl.charge.toFixed(2)}</td>
                    <td className="px-3 py-2 text-right">{sl.units}</td>
                    <td className="px-3 py-2 text-gray-500">{sl.service_date || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Column: Prediction + Outcome */}
        <div className="space-y-6">
          {/* AI Prediction */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">AI Prediction</h2>
            {claim.risk_score != null ? (
              <div className="space-y-4">
                <div className="text-center">
                  <div className={`text-4xl font-bold ${
                    claim.risk_score < 0.3 ? 'text-green-600' :
                    claim.risk_score <= 0.7 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {(claim.risk_score * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-gray-500 mt-1">Denial Probability</p>
                </div>

                {/* Risk score bar with threshold markers */}
                <div className="relative">
                  <div className="w-full bg-gray-200 rounded-full h-3">
                    <div
                      className={`h-3 rounded-full transition-all ${
                        claim.risk_score < 0.3 ? 'bg-green-500' :
                        claim.risk_score <= 0.7 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${claim.risk_score * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between mt-1 text-xs text-gray-400">
                    <span>0%</span>
                    <span className="absolute left-1/3 -translate-x-1/2 border-l border-gray-300 pl-1">30%</span>
                    <span className="absolute left-[70%] -translate-x-1/2 border-l border-gray-300 pl-1">70%</span>
                    <span>100%</span>
                  </div>
                  <div className="flex justify-between mt-0.5 text-xs">
                    <span className="text-green-600 font-medium">Auto Submit</span>
                    <span className="text-yellow-600 font-medium">Review</span>
                    <span className="text-red-600 font-medium">Fix</span>
                  </div>
                </div>

                {/* Risk Factors with actionable tips */}
                {claim.risk_factors && claim.risk_factors.length > 0 && (
                  <div className="space-y-3 mt-4">
                    <h4 className="text-sm font-semibold text-gray-700">What's Driving the Risk</h4>
                    {claim.risk_factors.map((factor, i) => {
                      const maxImpact = Math.max(...claim.risk_factors!.map(f => Math.abs(f.impact)), 0.01);
                      const width = Math.round((Math.abs(factor.impact) / maxImpact) * 100);
                      const isPositive = factor.impact > 0;
                      return (
                        <div key={i} className={`rounded-lg p-3 ${isPositive ? 'bg-red-50 border border-red-100' : 'bg-green-50 border border-green-100'}`}>
                          <div className="flex justify-between text-sm">
                            <span className="font-medium text-gray-800">{factor.display_name}</span>
                            <span className={`text-xs font-mono ${isPositive ? 'text-red-600' : 'text-green-600'}`}>
                              {isPositive ? '+' : ''}{(factor.impact * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-100 rounded-full h-1.5 mt-1.5">
                            <div
                              className={`h-1.5 rounded-full ${isPositive ? 'bg-red-400' : 'bg-green-400'}`}
                              style={{ width: `${width}%` }}
                            />
                          </div>
                          <p className="text-xs text-gray-500 mt-1.5">
                            {getFriendlyFactorTip(factor.feature, factor.value)}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No prediction available</p>
            )}
          </div>

          {/* 835 Outcome */}
          <div className="bg-white shadow rounded-lg p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">835 Outcome</h2>
            {claim.actual_outcome ? (
              <div className="space-y-3 text-sm">
                <div>
                  <p className="text-gray-500">Status</p>
                  <p className={`text-lg font-bold ${claim.actual_outcome === 'denied' ? 'text-red-600' : 'text-green-600'}`}>
                    {claim.actual_outcome.charAt(0).toUpperCase() + claim.actual_outcome.slice(1)}
                  </p>
                </div>
                {claim.paid_amount != null && (
                  <div>
                    <p className="text-gray-500">Paid Amount</p>
                    <p className="font-medium">${claim.paid_amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</p>
                  </div>
                )}
                {claim.carc_codes && claim.carc_codes.length > 0 && (
                  <div>
                    <p className="text-gray-500">CARC Codes</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {claim.carc_codes.map((code, i) => (
                        <span key={i} className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">
                          {code}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No 835 remittance data matched yet</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
