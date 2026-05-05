import axios from 'axios';
import type {
  ClaimListResponse,
  Claim,
  RemittanceListResponse,
  Remittance,
  PredictResponse,
  DashboardSummary,
  RiskDistribution,
  PayerStatsResponse,
  UploadResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
});

// Upload
export async function upload837(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post('/upload/837', form);
  return data;
}

export async function upload835(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post('/upload/835', form);
  return data;
}

// Job status polling
export async function fetchJobStatus(jobId: string): Promise<UploadResponse> {
  const { data } = await api.get(`/upload/status/${jobId}`);
  return data;
}

// Claims
export async function fetchClaims(params: {
  skip?: number;
  limit?: number;
  risk_level?: string;
  sort_by?: string;
  sort_order?: number;
}): Promise<ClaimListResponse> {
  const { data } = await api.get('/claims', { params });
  return data;
}

export async function fetchClaim(claimId: string): Promise<Claim> {
  const { data } = await api.get(`/claims/${claimId}`);
  return data;
}

// Remittances
export async function fetchRemittances(params: {
  skip?: number;
  limit?: number;
}): Promise<RemittanceListResponse> {
  const { data } = await api.get('/remittances', { params });
  return data;
}

export async function fetchRemittance(id: string): Promise<Remittance> {
  const { data } = await api.get(`/remittances/${id}`);
  return data;
}

// Predictions
export async function predictClaim(claimId: string): Promise<PredictResponse> {
  const { data } = await api.post(`/predict/${claimId}`);
  return data;
}

export async function predictBatch(claimIds?: string[]): Promise<{ predicted_count: number; results: PredictResponse[] }> {
  const { data } = await api.post('/predict-batch', { claim_ids: claimIds ?? null });
  return data;
}

// Dashboard
export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  const { data } = await api.get('/dashboard/summary');
  return data;
}

export async function fetchRiskDistribution(): Promise<RiskDistribution> {
  const { data } = await api.get('/dashboard/risk-distribution');
  return data;
}

export async function fetchPayerStats(): Promise<PayerStatsResponse> {
  const { data } = await api.get('/dashboard/payer-stats');
  return data;
}

// Health
export async function fetchHealth(): Promise<{ status: string; model_loaded: boolean; db_connected: boolean }> {
  const { data } = await api.get('/health');
  return data;
}

// Model
export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  const { data } = await api.get('/model/training-status');
  return data;
}

export async function retrainModel(): Promise<RetrainResult> {
  const { data } = await api.post('/model/retrain');
  return data;
}

export interface TrainingStatus {
  total_claims: number;
  total_remittances: number;
  matched_claims: number;
  paid_count: number;
  denied_count: number;
  ready_to_train: boolean;
  min_required: number;
  last_training: {
    trained_at: string | null;
    real_samples: number | null;
    metrics: { auc_roc: number; precision: number; recall: number; auc_real_only: number | null } | null;
  } | null;
}

export interface RetrainResult {
  status: string;
  message: string;
  real_samples?: number;
  synthetic_samples?: number;
  denied_count?: number;
  paid_count?: number;
  denial_rate?: number;
  matched_claims?: number;
  metrics?: { auc_roc: number; precision: number; recall: number; auc_real_only: number | null };
  feature_importance?: Record<string, number>;
  elapsed_seconds?: number;
}
