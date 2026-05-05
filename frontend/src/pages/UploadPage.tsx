import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUpload from '../components/FileUpload';
import { upload837, upload835, fetchTrainingStatus, retrainModel, fetchJobStatus } from '../services/api';
import type { TrainingStatus, RetrainResult } from '../services/api';
import type { UploadResponse, ClaimErrorDetail, ClaimIssue } from '../types';

/* ─── helpers ─── */

interface PatientGroup {
  patient_name: string;
  claims: ClaimErrorDetail[];
  total_issues: number;
  max_risk: number;
}

function groupByPatient(items: ClaimErrorDetail[]): PatientGroup[] {
  const map = new Map<string, ClaimErrorDetail[]>();
  for (const ce of items) {
    const name = ce.patient_name || 'Unknown';
    if (!map.has(name)) map.set(name, []);
    map.get(name)!.push(ce);
  }
  const groups: PatientGroup[] = [];
  for (const [patient_name, claims] of map) {
    groups.push({
      patient_name,
      claims: claims.sort((a, b) => b.risk_score - a.risk_score),
      total_issues: claims.reduce((sum, c) => sum + c.issues.length, 0),
      max_risk: Math.max(...claims.map(c => c.risk_score)),
    });
  }
  groups.sort((a, b) => b.max_risk - a.max_risk);
  return groups;
}

const riskEmoji = (level: string) =>
  level === 'HIGH' ? '\u{1F534}' : level === 'MEDIUM' ? '\u{1F7E1}' : '\u{1F7E2}';

const colorOf = (score: number) =>
  score > 0.7 ? 'red' : score >= 0.3 ? 'yellow' : 'green';

/* ═══════════════════════════════════════════════
   File Risk Report
   ═══════════════════════════════════════════════ */

function FileRiskReport({ result }: { result: UploadResponse }) {
  const navigate = useNavigate();
  const s = result.risk_summary;
  if (!s) return null;

  const total = result.claims_parsed || 0;
  const avgPct = Math.round(s.avg_risk_score * 100);
  const c = colorOf(s.avg_risk_score);

  const allItems = s.claim_errors || [];
  const highCount = allItems.filter(x => x.risk_score > 0.7).length;
  const medCount = allItems.filter(x => x.risk_score >= 0.3 && x.risk_score <= 0.7).length;
  const lowCount = total - highCount - medCount;

  // Only medium + high for patient attention section
  const flagged = allItems.filter(x => x.risk_score >= 0.3);
  const patientGroups = groupByPatient(flagged);

  return (
    <div className="space-y-6">

      {/* ╔═══════════════════════════════════════════╗
         ║  CARD 1 — File Score + Reasons + Fixes   ║
         ╚═══════════════════════════════════════════╝ */}
      <div className={`rounded-xl overflow-hidden shadow-sm border ${
        c === 'red' ? 'border-red-300' : c === 'yellow' ? 'border-yellow-300' : 'border-green-300'
      }`}>
        {/* Banner */}
        <div className={`px-6 py-5 ${
          c === 'red' ? 'bg-red-50' : c === 'yellow' ? 'bg-yellow-50' : 'bg-green-50'
        }`}>
          <div className="flex items-start justify-between">
            <div>
              <p className={`text-2xl font-bold ${
                c === 'red' ? 'text-red-700' : c === 'yellow' ? 'text-yellow-700' : 'text-green-700'
              }`}>
                {riskEmoji(s.file_risk_level)} Risk: {s.file_risk_level} ({avgPct}%)
              </p>
              <p className="text-sm text-gray-600 mt-1">
                {total} claims parsed &middot; {result.predictions_made} predicted
              </p>
            </div>
            <div className={`text-5xl font-black opacity-20 ${
              c === 'red' ? 'text-red-400' : c === 'yellow' ? 'text-yellow-400' : 'text-green-400'
            }`}>{avgPct}%</div>
          </div>

          {/* Risk bar */}
          <div className="relative mt-4">
            <div className="w-full bg-white/60 rounded-full h-2.5">
              <div className={`h-2.5 rounded-full transition-all ${
                c === 'red' ? 'bg-red-500' : c === 'yellow' ? 'bg-yellow-500' : 'bg-green-500'
              }`} style={{ width: `${avgPct}%` }} />
            </div>
            <div className="flex justify-between mt-1 text-[10px] text-gray-400">
              <span>0%</span>
              <span className="absolute left-[30%] -translate-x-1/2">30%</span>
              <span className="absolute left-[70%] -translate-x-1/2">70%</span>
              <span>100%</span>
            </div>
          </div>
        </div>

        {/* Reasons + Fixes */}
        <div className="px-6 py-4 bg-white grid grid-cols-1 md:grid-cols-2 gap-4">
          {(s.file_top_reasons?.length ?? 0) > 0 && (
            <div>
              <p className="text-sm font-semibold text-gray-800 mb-2">{'\u{1F4CC}'} Top Reasons</p>
              <ul className="space-y-1">
                {s.file_top_reasons.map((r, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />
                    {r.reason}
                    <span className="text-xs text-gray-400">({r.count} claims)</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {(s.file_top_fixes?.length ?? 0) > 0 && (
            <div>
              <p className="text-sm font-semibold text-gray-800 mb-2">{'\u{1F6E0}\u{FE0F}'} Suggested Fixes</p>
              <ul className="space-y-1">
                {s.file_top_fixes.map((f, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm text-gray-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-400 flex-shrink-0" />
                    {f.fix}
                    <span className="text-xs text-gray-400">({f.count})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Claim count strip */}
        <div className="px-6 py-3 bg-gray-50 border-t border-gray-200 flex items-center gap-6 text-sm">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-full bg-red-500" />
            <span className="font-semibold text-gray-800">{highCount}</span>
            <span className="text-gray-500">High</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-full bg-yellow-400" />
            <span className="font-semibold text-gray-800">{medCount}</span>
            <span className="text-gray-500">Medium</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-3 rounded-full bg-green-500" />
            <span className="font-semibold text-gray-800">{lowCount}</span>
            <span className="text-gray-500">Low</span>
          </span>
          <span className="ml-auto text-xs text-gray-400">
            Score range {Math.round(s.min_risk_score * 100)}% – {Math.round(s.max_risk_score * 100)}%
          </span>
        </div>
      </div>

      {/* ╔═══════════════════════════════════════════╗
         ║  CARD 2 — Patients Requiring Attention   ║
         ╚═══════════════════════════════════════════╝ */}
      {patientGroups.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">
            {'\u{1F464}'} Patients Requiring Attention
            <span className="text-sm font-normal text-gray-500 ml-2">
              ({flagged.length} claims across {patientGroups.length} patients)
            </span>
          </h3>

          {patientGroups.map((group) => {
            const gc = colorOf(group.max_risk);
            return (
              <div key={group.patient_name} className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm">
                {/* Patient header */}
                <div className={`px-5 py-3 flex items-center justify-between border-b ${
                  gc === 'red'    ? 'bg-red-50 border-red-200'       :
                  gc === 'yellow' ? 'bg-yellow-50 border-yellow-200' :
                                    'bg-green-50 border-green-200'
                }`}>
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                      gc === 'red' ? 'bg-red-500' : gc === 'yellow' ? 'bg-yellow-500' : 'bg-green-500'
                    }`}>
                      {group.patient_name.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <p className="font-semibold text-gray-900">{group.patient_name}</p>
                      <p className="text-xs text-gray-500">
                        {group.claims.length} claim{group.claims.length > 1 ? 's' : ''}
                        {group.total_issues > 0 && (
                          <span className="text-red-600 ml-1">&middot; {group.total_issues} issue{group.total_issues > 1 ? 's' : ''}</span>
                        )}
                      </p>
                    </div>
                  </div>
                  <span className={`text-lg font-bold ${
                    gc === 'red' ? 'text-red-600' : gc === 'yellow' ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                    {riskEmoji(group.max_risk > 0.7 ? 'HIGH' : 'MEDIUM')} {Math.round(group.max_risk * 100)}%
                  </span>
                </div>

                {/* Claims */}
                <div className="divide-y divide-gray-100">
                  {group.claims.map((ce) => (
                    <ClaimRow key={ce.claim_id} ce={ce} onNav={() => navigate(`/claims/${ce.claim_id}`)} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ─── Single claim row ─── */

const actionBadge = (action: string, label: string) => {
  const cls =
    action === 'fix_required' ? 'bg-red-100 text-red-800 border-red-300' :
    action === 'review'       ? 'bg-yellow-100 text-yellow-800 border-yellow-300' :
                                'bg-green-100 text-green-800 border-green-300';
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${cls}`}>
      {label || action}
    </span>
  );
};

function ClaimRow({ ce, onNav }: { ce: ClaimErrorDetail; onNav: () => void }) {
  const pct = Math.round(ce.risk_score * 100);
  const c = colorOf(ce.risk_score);

  return (
    <div className="px-5 py-4">
      {/* Claim header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span onClick={onNav} className="font-semibold text-indigo-600 hover:underline cursor-pointer text-sm">
            {ce.claim_id}
          </span>
          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
            c === 'red'    ? 'bg-red-100 text-red-800'       :
            c === 'yellow' ? 'bg-yellow-100 text-yellow-800' :
                             'bg-green-100 text-green-800'
          }`}>
            {riskEmoji(ce.risk_level)} {ce.risk_level} ({pct}%)
          </span>
          {ce.action && actionBadge(ce.action, ce.action_label)}
          {ce.payer_name && (
            <span className="text-xs text-gray-400">{ce.payer_name}</span>
          )}
        </div>
        <button onClick={onNav} className="text-xs text-indigo-500 hover:underline flex-shrink-0">
          View Details &rarr;
        </button>
      </div>

      {/* Issues — each with Reason + Fix */}
      {ce.issues.length > 0 ? (
        <div className="space-y-3">
          {ce.issues.map((iss: ClaimIssue, i: number) => (
            <IssueBlock key={i} issue={iss} color={c} />
          ))}
        </div>
      ) : (
        /* No hard issues but elevated risk — show top factors */
        ce.top_factors.length > 0 && (
          <div className="text-sm text-yellow-700 bg-yellow-50 border border-yellow-200 rounded-lg px-3 py-2">
            <span className="font-semibold">{'\u{1F4CC}'} Risk factors: </span>
            {ce.top_factors.filter(f => f.impact > 0).map(f => f.name).join(', ') || 'Elevated model score'}
          </div>
        )
      )}
    </div>
  );
}

/* ─── Single issue block (Reason + Fix) ─── */

function IssueBlock({ issue, color }: { issue: ClaimIssue; color: string }) {
  const reasonLines = issue.reason.split('\n').filter(Boolean);

  return (
    <div className={`rounded-lg border px-4 py-3 ${
      color === 'red'    ? 'bg-red-50/50 border-red-200'       :
      color === 'yellow' ? 'bg-yellow-50/50 border-yellow-200' :
                           'bg-green-50/50 border-green-200'
    }`}>
      {/* Reason */}
      <div className="mb-2">
        <p className="text-xs font-bold text-gray-600 uppercase tracking-wide mb-1">
          {'\u{1F4CC}'} Reason
        </p>
        {reasonLines.map((line, i) => (
          <p key={i} className={`text-sm ${i === 0 ? 'font-medium text-gray-800' : 'text-gray-600'}`}>
            {line}
          </p>
        ))}
      </div>

      {/* Fix */}
      {issue.fixes.length > 0 && (
        <div>
          <p className="text-xs font-bold text-gray-600 uppercase tracking-wide mb-1">
            {'\u{1F6E0}\u{FE0F}'} Fix
          </p>
          <ul className="space-y-0.5">
            {issue.fixes.map((fix, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-gray-700">
                <span className="mt-1.5 w-1 h-1 rounded-full bg-blue-400 flex-shrink-0" />
                {fix}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════
   Main Page
   ═══════════════════════════════════════════════ */

/* ─── Model Training Panel ─── */

function ModelTrainingPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [result, setResult] = useState<RetrainResult | null>(null);
  const [training, setTraining] = useState(false);

  const loadStatus = () => {
    fetchTrainingStatus().then(setStatus).catch(() => {});
  };

  useEffect(() => { loadStatus(); }, []);

  const handleRetrain = async () => {
    setTraining(true);
    setResult(null);
    try {
      const res = await retrainModel();
      setResult(res);
      loadStatus();
    } catch {
      setResult({ status: 'error', message: 'Failed to retrain model' });
    } finally {
      setTraining(false);
    }
  };

  if (!status) return null;

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-5 py-3 bg-indigo-50 border-b border-indigo-200">
        <h3 className="text-sm font-semibold text-indigo-800">{'\u{1F9E0}'} Model Training</h3>
      </div>
      <div className="p-5 space-y-4">
        {/* Data status */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-xl font-bold text-gray-800">{status.total_claims}</div>
            <div className="text-xs text-gray-500">Total Claims</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-xl font-bold text-gray-800">{status.total_remittances}</div>
            <div className="text-xs text-gray-500">Remittances</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="text-xl font-bold text-indigo-700">{status.matched_claims}</div>
            <div className="text-xs text-gray-500">Matched (with outcome)</div>
          </div>
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex justify-center gap-3">
              <span className="text-sm"><span className="font-bold text-green-600">{status.paid_count}</span> paid</span>
              <span className="text-sm"><span className="font-bold text-red-600">{status.denied_count}</span> denied</span>
            </div>
            <div className="text-xs text-gray-500 mt-1">Outcomes</div>
          </div>
        </div>

        {/* Training readiness */}
        {!status.ready_to_train ? (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg px-4 py-3 text-sm text-yellow-800">
            Need at least <strong>{status.min_required}</strong> matched claims to retrain.
            Currently have <strong>{status.matched_claims}</strong>.
            Upload more 835 files to match outcomes to your 837 claims.
          </div>
        ) : (
          <div className="bg-green-50 border border-green-200 rounded-lg px-4 py-3 text-sm text-green-800">
            {'\u{2705}'} Ready to retrain with <strong>{status.matched_claims}</strong> real matched claims.
          </div>
        )}

        {/* Last training */}
        {status.last_training?.trained_at && (
          <div className="text-xs text-gray-500">
            Last trained: {new Date(status.last_training.trained_at).toLocaleString()}
            {status.last_training.metrics && (
              <span className="ml-2">
                &middot; AUC: {status.last_training.metrics.auc_roc}
                &middot; Precision: {status.last_training.metrics.precision}
                &middot; Recall: {status.last_training.metrics.recall}
              </span>
            )}
          </div>
        )}

        {/* Retrain button */}
        <button
          onClick={handleRetrain}
          disabled={!status.ready_to_train || training}
          className={`w-full py-2.5 rounded-lg text-sm font-semibold transition ${
            status.ready_to_train && !training
              ? 'bg-indigo-600 text-white hover:bg-indigo-700'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          }`}
        >
          {training ? 'Training...' : 'Retrain Model with Real Data'}
        </button>

        {/* Result */}
        {result && (
          <div className={`rounded-lg px-4 py-3 text-sm border ${
            result.status === 'success'
              ? 'bg-green-50 border-green-200 text-green-800'
              : result.status === 'insufficient_data'
              ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
              : 'bg-red-50 border-red-200 text-red-800'
          }`}>
            <p className="font-semibold">{result.message}</p>
            {result.metrics && (
              <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                <div>
                  <div className="font-bold">{result.metrics.auc_roc}</div>
                  <div className="text-xs opacity-70">AUC-ROC</div>
                </div>
                <div>
                  <div className="font-bold">{result.metrics.precision}</div>
                  <div className="text-xs opacity-70">Precision</div>
                </div>
                <div>
                  <div className="font-bold">{result.metrics.recall}</div>
                  <div className="text-xs opacity-70">Recall</div>
                </div>
              </div>
            )}
            {result.feature_importance && (
              <details className="mt-2">
                <summary className="cursor-pointer text-xs opacity-70">Feature importance</summary>
                <ul className="mt-1 space-y-0.5">
                  {Object.entries(result.feature_importance)
                    .sort(([,a], [,b]) => b - a)
                    .map(([name, imp]) => (
                      <li key={name} className="flex justify-between text-xs">
                        <span>{name}</span>
                        <span className="font-mono">{imp}</span>
                      </li>
                    ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════
   Main Page
   ═══════════════════════════════════════════════ */

export default function UploadPage() {
  const [result837, setResult837] = useState<UploadResponse | null>(null);
  const [result835, setResult835] = useState<UploadResponse | null>(null);
  const [polling, setPolling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Clean up polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const pollForResults = useCallback((jobId: string, partialResult: UploadResponse) => {
    setPolling(true);
    // Show partial result immediately (has claims_parsed but no risk_summary)
    setResult837(partialResult);

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetchJobStatus(jobId);
        // If we got back the full result (has risk_summary), stop polling
        if (res.risk_summary) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setPolling(false);
          setResult837(res);
        }
        // If still processing, keep polling
      } catch {
        // job not found or error — stop polling
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        setPolling(false);
      }
    }, 3000); // poll every 3 seconds
  }, []);

  const handle837 = async (file: File) => {
    setResult837(null);
    setPolling(false);
    if (pollRef.current) clearInterval(pollRef.current);

    const res = await upload837(file);

    if (res.job_id && res.status === 'processing') {
      // Large file — background processing, start polling
      pollForResults(res.job_id, res);
    } else {
      setResult837(res);
    }
  };

  const handle835 = async (file: File) => {
    const res = await upload835(file);
    setResult835(res);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Upload EDI Files</h1>
        <p className="mt-1 text-sm text-gray-500">
          Upload 837 (claim) and 835 (remittance) files to parse and analyze.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* 837 Upload */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">837 Professional (Claims)</h2>
          <FileUpload
            label="Drop 837 file here"
            description="Upload .edi, .txt, or .x12 file containing 837P claims"
            onUpload={handle837}
          />
        </div>

        {/* 835 Upload */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-gray-800">835 Remittance Advice</h2>
          <FileUpload
            label="Drop 835 file here"
            description="Upload .edi, .txt, or .x12 file containing 835 remittance data"
            onUpload={handle835}
          />
          {result835 && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 space-y-2">
              <p className="font-medium text-blue-800">{result835.message}</p>
              <div className="text-sm text-blue-700 space-y-1">
                <p>Records parsed: {result835.records_parsed}</p>
                <p>Matched to claims: {result835.matched_to_claims}</p>
                <p>Denied: {result835.denied_count}</p>
                <p>Total paid: ${result835.total_paid?.toLocaleString()}</p>
                {result835.training_records_created != null && (
                  <p>Training records created: {result835.training_records_created}</p>
                )}
                {result835.total_matched_claims != null && (
                  <p>Total matched claims: {result835.total_matched_claims}</p>
                )}
              </div>
              {result835.training_status && (
                <div className={`mt-2 rounded-lg px-3 py-2 text-sm border ${
                  result835.auto_retrain_triggered
                    ? 'bg-indigo-50 border-indigo-200 text-indigo-800'
                    : result835.ready_to_retrain
                    ? 'bg-green-50 border-green-200 text-green-800'
                    : 'bg-gray-50 border-gray-200 text-gray-700'
                }`}>
                  {result835.auto_retrain_triggered ? (
                    <p className="font-semibold">Model retraining in background...</p>
                  ) : result835.ready_to_retrain ? (
                    <p>Ready to retrain model</p>
                  ) : (
                    <p>{result835.records_until_retrain} more training records needed for auto-retrain</p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Processing indicator for background jobs */}
      {polling && result837 && !result837.risk_summary && (
        <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-6 text-center space-y-3">
          <div className="flex items-center justify-center gap-3">
            <svg className="animate-spin h-6 w-6 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <p className="text-lg font-semibold text-indigo-800">
              Processing {result837.claims_parsed?.toLocaleString()} claims...
            </p>
          </div>
          <p className="text-sm text-indigo-600">
            Large file detected. Predictions are running in the background. Results will appear automatically.
          </p>
        </div>
      )}

      {/* File Risk Report */}
      {result837 && result837.risk_summary && (
        <FileRiskReport result={result837} />
      )}

      {/* Model Training Panel */}
      <ModelTrainingPanel />
    </div>
  );
}
