export interface ServiceLine {
  cpt_code: string;
  modifiers: string[];
  charge: number;
  units: number;
  service_date: string | null;
}

export interface RiskFactor {
  feature: string;
  display_name: string;
  impact: number;
  value: string;
}

export interface Claim {
  claim_id: string;
  patient_name: string;
  payer_name: string;
  payer_id: string;
  total_charge: number;
  diagnosis_codes: string[];
  service_lines: ServiceLine[];
  place_of_service: string;
  billing_provider_name: string;
  billing_provider_npi: string;
  patient_dob: string;
  patient_gender: string;
  provider_taxonomy: string;
  prior_auth_number: string;
  created_at: string | null;
  risk_score: number | null;
  risk_level: string | null;
  risk_factors: RiskFactor[] | null;
  actual_outcome: string | null;
  paid_amount: number | null;
  carc_codes: string[] | null;
}

export interface ClaimListResponse {
  claims: Claim[];
  total: number;
  skip: number;
  limit: number;
}

export interface ServiceLinePayment {
  cpt_code: string;
  billed_amount: number;
  paid_amount: number;
  allowed_amount: number;
  adjustments: Record<string, unknown>[];
}

export interface Remittance {
  id: string;
  claim_id: string;
  payer_control_number: string;
  claim_status: string;
  billed_amount: number;
  paid_amount: number;
  patient_responsibility: number;
  payer_name: string;
  payee_name: string;
  trace_number: string;
  payment_date: string;
  carc_codes: string[];
  rarc_codes: string[];
  adjustments: Record<string, unknown>[];
  service_lines: ServiceLinePayment[];
  created_at: string | null;
}

export interface RemittanceListResponse {
  remittances: Remittance[];
  total: number;
  skip: number;
  limit: number;
}

export interface PredictResponse {
  claim_id: string;
  risk_score: number;
  risk_level: string;
  risk_factors: RiskFactor[];
}

export interface DashboardSummary {
  total_claims: number;
  total_predicted: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  total_remittances: number;
  denial_rate: number;
  total_billed: number;
  total_paid: number;
}

export interface RiskBucket {
  range_label: string;
  count: number;
}

export interface RiskDistribution {
  buckets: RiskBucket[];
}

export interface PayerStat {
  payer_name: string;
  payer_id: string;
  total_claims: number;
  denied_count: number;
  denial_rate: number;
}

export interface PayerStatsResponse {
  payers: PayerStat[];
}

export interface TopRiskClaim {
  claim_id: string;
  risk_score: number;
  risk_level: string;
  top_reason: string;
}

export interface ClaimIssue {
  reason: string;
  fixes: string[];
}

export interface ScoreBreakdown {
  base_score: number;
  payer_weight: number;
  payer_name: string;
  cpt_weight: number;
  cpt_label: string;
  issue_weight: number;
  final_score: number;
}

export interface ClaimErrorDetail {
  claim_id: string;
  patient_name: string;
  payer_name: string;
  risk_score: number;
  risk_level: string;
  action: string;
  action_label: string;
  score_breakdown: ScoreBreakdown;
  issues: ClaimIssue[];
  top_factors: { name: string; impact: number }[];
}

export interface FileRiskSummary {
  avg_risk_score: number;
  max_risk_score: number;
  min_risk_score: number;
  file_risk_level: string;
  auto_submit_count: number;
  review_count: number;
  needs_fix_count: number;
  top_risk_claims: TopRiskClaim[];
  claim_errors: ClaimErrorDetail[];
  file_top_reasons: { reason: string; count: number }[];
  file_top_fixes: { fix: string; count: number }[];
}

export interface UploadResponse {
  message: string;
  claims_parsed?: number;
  claims_stored?: number;
  predictions_made?: number;
  payer_breakdown?: Record<string, number>;
  claim_ids?: string[];
  risk_summary?: FileRiskSummary;
  job_id?: string;
  status?: string;
  records_parsed?: number;
  records_stored?: number;
  matched_to_claims?: number;
  denied_count?: number;
  total_paid?: number;
  training_records_created?: number;
  total_training_records?: number;
  training_status?: string;
  auto_retrain_triggered?: boolean;
  total_matched_claims?: number;
  ready_to_retrain?: boolean;
  records_until_retrain?: number;
}
