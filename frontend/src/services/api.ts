import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  timeout: 120000, // 2 minutes for large file uploads
});

// Types
export interface ModelInfo {
  name: string;
  model_key: string;
  description: string;
  default_threshold: number;
  status: string;
}

export interface HealthResponse {
  status: string;
  models_loaded: Record<string, boolean>;
}

export interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
}

export interface BatchPredictionResponse {
  model_name: string;
  threshold_used: number;
  predictions: number[];
  probabilities: number[];
  total_records: number;
  fire_predicted: number;
  no_fire_predicted: number;
  evaluation_metrics: EvaluationMetrics | null;
  confusion_matrix: number[][] | null;
}

export interface ModelComparisonResult {
  model_name: string;
  threshold_used: number;
  predictions: number[];
  probabilities: number[];
  fire_predicted: number;
  no_fire_predicted: number;
  evaluation_metrics: EvaluationMetrics | null;
  confusion_matrix: number[][] | null;
}

export interface ComparisonResponse {
  total_records: number;
  has_ground_truth: boolean;
  results: ModelComparisonResult[];
  best_model: string | null;
}

export interface ThresholdOptimizationResponse {
  model_name: string;
  optimal_threshold: number;
  f1_score: number;
  accuracy: number;
  precision: number;
  recall: number;
  thresholds_tested: number[];
  f1_scores: number[];
}

export interface FeatureColumnsResponse {
  feature_columns: string[];
  total_features: number;
}

export interface SampleDataResponse {
  total_records: number;
  sample_size: number;
  columns: string[];
  preview: Record<string, unknown>[];
  data: Record<string, unknown>[];
  has_ground_truth: boolean;
}

// API Functions
export async function getHealth(): Promise<HealthResponse> {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
}

export async function getModels(): Promise<{ models: ModelInfo[] }> {
  const response = await api.get<{ models: ModelInfo[] }>('/models');
  return response.data;
}

export async function getFeatureColumns(): Promise<FeatureColumnsResponse> {
  const response = await api.get<FeatureColumnsResponse>('/feature-columns');
  return response.data;
}

export async function getSampleData(size: number = 10000): Promise<SampleDataResponse> {
  const response = await api.get<SampleDataResponse>(`/sample?size=${size}`);
  return response.data;
}

export async function compareModelsWithSample(
  size: number = 10000,
  threshold?: number
): Promise<ComparisonResponse> {
  const params = new URLSearchParams();
  params.append('size', size.toString());
  if (threshold !== undefined) {
    params.append('threshold', threshold.toString());
  }
  
  const response = await api.post<ComparisonResponse>(
    `/compare/sample?${params.toString()}`
  );
  return response.data;
}

export async function predictBatch(
  file: File,
  modelName: string = 'RF',
  threshold?: number
): Promise<BatchPredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  params.append('model_name', modelName);
  if (threshold !== undefined) {
    params.append('threshold', threshold.toString());
  }

  const response = await api.post<BatchPredictionResponse>(
    `/predict/batch?${params.toString()}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}

export async function compareModels(
  file: File,
  threshold?: number
): Promise<ComparisonResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  if (threshold !== undefined) {
    params.append('threshold', threshold.toString());
  }

  const response = await api.post<ComparisonResponse>(
    `/compare?${params.toString()}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}

export async function optimizeThreshold(
  file: File,
  modelName: string = 'RF',
  thresholdMin: number = 0.1,
  thresholdMax: number = 0.9,
  thresholdStep: number = 0.05
): Promise<ThresholdOptimizationResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  params.append('model_name', modelName);
  params.append('threshold_min', thresholdMin.toString());
  params.append('threshold_max', thresholdMax.toString());
  params.append('threshold_step', thresholdStep.toString());

  const response = await api.post<ThresholdOptimizationResponse>(
    `/optimize-threshold?${params.toString()}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}

export default api;
