import { useState, useEffect } from 'react';
import {
  Flame,
  GitCompare,
  Target,
  Server,
  CheckCircle,
  XCircle,
  Loader2,
  AlertCircle,
  RefreshCw,
  Database,
  Upload,
  Shuffle
} from 'lucide-react';
import { Header, Card, Button, Tabs, LoadingSpinner } from './components/ui';
import { FileUpload } from './components/FileUpload';
import { PredictionResults } from './components/PredictionResults';
import { ModelComparison } from './components/ModelComparison';
import { ThresholdOptimizerResults } from './components/ThresholdOptimizerResults';
import { SamplePreview } from './components/SamplePreview';
import {
  getHealth,
  getModels,
  predictBatch,
  predictWithSample,
  compareModels,
  compareModelsWithSample,
  optimizeThreshold,
  optimizeThresholdWithSample,
  getSampleData,
  type HealthResponse,
  type ModelInfo,
  type BatchPredictionResponse,
  type ComparisonResponse,
  type ThresholdOptimizationResponse,
  type SampleDataResponse
} from './services/api';

type TabId = 'predict' | 'compare' | 'optimize';
type DataSource = 'upload' | 'sample';

function App() {
  // State
  const [activeTab, setActiveTab] = useState<TabId>('predict');
  const [dataSource, setDataSource] = useState<DataSource>('sample');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('RF');
  const [customThreshold, setCustomThreshold] = useState<string>('');
  const [sampleSize, setSampleSize] = useState<number>(10000);
  const [isLoading, setIsLoading] = useState(false);
  const [isSampleLoading, setIsSampleLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // API State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [healthLoading, setHealthLoading] = useState(true);
  const [sampleData, setSampleData] = useState<SampleDataResponse | null>(null);

  // Results State
  const [predictionResult, setPredictionResult] = useState<BatchPredictionResponse | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResponse | null>(null);
  const [optimizationResult, setOptimizationResult] = useState<ThresholdOptimizationResponse | null>(null);

  // Load health and models on mount
  useEffect(() => {
    loadHealthAndModels();
  }, []);

  // Load sample data when switching to sample mode
  useEffect(() => {
    if (dataSource === 'sample' && !sampleData) {
      loadSampleData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataSource]);

  const loadHealthAndModels = async () => {
    setHealthLoading(true);
    try {
      const [healthData, modelsData] = await Promise.all([
        getHealth(),
        getModels()
      ]);
      setHealth(healthData);
      setModels(modelsData.models);
    } catch (err) {
      console.error('Failed to load API status:', err);
    } finally {
      setHealthLoading(false);
    }
  };

  const loadSampleData = async () => {
    setIsSampleLoading(true);
    setError(null);
    try {
      const data = await getSampleData(sampleSize);
      setSampleData(data);
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } };
      setError(axiosError.response?.data?.detail || 'Failed to load sample data');
    } finally {
      setIsSampleLoading(false);
    }
  };

  const handleRefreshSample = () => {
    setSampleData(null);
    loadSampleData();
    // Clear results when refreshing sample
    setPredictionResult(null);
    setComparisonResult(null);
    setOptimizationResult(null);
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
    // Clear previous results when new file is selected
    setPredictionResult(null);
    setComparisonResult(null);
    setOptimizationResult(null);
  };

  const handleDataSourceChange = (source: DataSource) => {
    setDataSource(source);
    setError(null);
    // Clear results when switching data source
    setPredictionResult(null);
    setComparisonResult(null);
    setOptimizationResult(null);
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    try {
      const threshold = customThreshold ? parseFloat(customThreshold) : undefined;

      if (dataSource === 'sample') {
        // Use the direct sample prediction endpoint
        const result = await predictWithSample(selectedModel, sampleSize, threshold);
        setPredictionResult(result);
      } else {
        if (!selectedFile) {
          setError('Please select a CSV file first');
          setIsLoading(false);
          return;
        }
        const result = await predictBatch(selectedFile, selectedModel, threshold);
        setPredictionResult(result);
      }
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } };
      setError(axiosError.response?.data?.detail || 'Prediction failed. Please check your file and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCompare = async () => {
    setIsLoading(true);
    setError(null);
    setComparisonResult(null);

    try {
      const threshold = customThreshold ? parseFloat(customThreshold) : undefined;

      if (dataSource === 'sample') {
        // Use the direct sample comparison endpoint
        const result = await compareModelsWithSample(sampleSize, threshold);
        setComparisonResult(result);
      } else {
        if (!selectedFile) {
          setError('Please select a CSV file first');
          setIsLoading(false);
          return;
        }
        const result = await compareModels(selectedFile, threshold);
        setComparisonResult(result);
      }
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } };
      setError(axiosError.response?.data?.detail || 'Comparison failed. Please check your file and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleOptimize = async () => {
    setIsLoading(true);
    setError(null);
    setOptimizationResult(null);

    try {
      if (dataSource === 'sample') {
        // Use the direct sample optimization endpoint
        const result = await optimizeThresholdWithSample(selectedModel, sampleSize);
        setOptimizationResult(result);
      } else {
        if (!selectedFile) {
          setError('Please select a CSV file with ground truth (incendio column)');
          setIsLoading(false);
          return;
        }
        const result = await optimizeThreshold(selectedFile, selectedModel);
        setOptimizationResult(result);
      }
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } };
      setError(axiosError.response?.data?.detail || 'Optimization failed. Make sure your file includes the "incendio" column.');
    } finally {
      setIsLoading(false);
    }
  };

  const tabs = [
    { id: 'predict' as TabId, label: 'Single Model', icon: <Flame size={18} /> },
    { id: 'compare' as TabId, label: 'Compare Models', icon: <GitCompare size={18} /> },
    { id: 'optimize' as TabId, label: 'Optimize Threshold', icon: <Target size={18} /> },
  ];

  const allModelsLoaded = health?.models_loaded && Object.values(health.models_loaded).every(Boolean);

  return (
    <div className="min-h-screen bg-[#f5f5f5]">
      <Header>
        {/* API Status */}
        <div className="mt-6 flex justify-center">
          {healthLoading ? (
            <div className="flex items-center gap-2 bg-white/20 px-4 py-2 rounded-full">
              <LoadingSpinner size="sm" />
              <span>Connecting to API...</span>
            </div>
          ) : health ? (
            <button
              onClick={loadHealthAndModels}
              className="flex items-center gap-2 bg-white/20 hover:bg-white/30 px-4 py-2 rounded-full transition-colors"
            >
              {allModelsLoaded ? (
                <>
                  <CheckCircle size={18} />
                  <span>All Models Ready</span>
                </>
              ) : (
                <>
                  <AlertCircle size={18} />
                  <span>Some Models Loading...</span>
                </>
              )}
              <RefreshCw size={14} className="ml-1" />
            </button>
          ) : (
            <div className="flex items-center gap-2 bg-red-500/20 px-4 py-2 rounded-full">
              <XCircle size={18} />
              <span>API Unavailable</span>
            </div>
          )}
        </div>
      </Header>

      <main className="max-w-6xl mx-auto px-5 py-8">
        {/* Models Status */}
        {health && (
          <Card className="mb-8" hover={false}>
            <div className="flex items-center gap-3 mb-4">
              <Server size={24} className="text-[#A3C6D4]" />
              <h2 className="font-heading text-xl font-semibold">Available Models</h2>
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              {models.map((model) => (
                <div
                  key={model.model_key}
                  className={`p-4 rounded-xl border-2 transition-all ${model.status === 'loaded'
                    ? 'border-[#B4C9A9] bg-[#B4C9A9]/10'
                    : 'border-gray-200 bg-gray-50'
                    }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-[#1a1a1a]">{model.name}</p>
                      <p className="text-sm text-gray-500">{model.description}</p>
                    </div>
                    {model.status === 'loaded' ? (
                      <CheckCircle size={20} className="text-[#B4C9A9]" />
                    ) : (
                      <Loader2 size={20} className="text-gray-400 animate-spin" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <Tabs
            tabs={tabs}
            activeTab={activeTab}
            onTabChange={(id) => setActiveTab(id as TabId)}
          />
        </div>

        {/* File Upload */}
        <Card className="mb-8" hover={false}>
          <h2 className="font-heading text-xl font-semibold mb-4">
            Data Source
          </h2>

          {/* Data Source Toggle */}
          <div className="flex gap-3 mb-6">
            <button
              onClick={() => handleDataSourceChange('sample')}
              disabled={isLoading}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${dataSource === 'sample'
                ? 'bg-mondesa-gradient text-white shadow-md'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              <Database size={18} />
              Use Sample Data
            </button>
            <button
              onClick={() => handleDataSourceChange('upload')}
              disabled={isLoading}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${dataSource === 'upload'
                ? 'bg-mondesa-gradient text-white shadow-md'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              <Upload size={18} />
              Upload CSV
            </button>
          </div>

          {/* Sample Data Section */}
          {dataSource === 'sample' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Sample Size
                    </label>
                    <select
                      value={sampleSize}
                      onChange={(e) => {
                        setSampleSize(Number(e.target.value));
                        setSampleData(null);
                      }}
                      disabled={isLoading || isSampleLoading}
                      className="px-4 py-2 border-2 border-gray-200 rounded-lg focus:border-[#A3C6D4] focus:outline-none"
                    >
                      <option value={1000}>1,000 records</option>
                      <option value={5000}>5,000 records</option>
                      <option value={10000}>10,000 records</option>
                      <option value={25000}>25,000 records</option>
                      <option value={50000}>50,000 records</option>
                    </select>
                  </div>
                  <Button
                    variant="secondary"
                    onClick={handleRefreshSample}
                    disabled={isLoading || isSampleLoading}
                  >
                    <span className="flex items-center gap-2">
                      <Shuffle size={16} className={isSampleLoading ? 'animate-spin' : ''} />
                      {isSampleLoading ? 'Loading...' : 'New Random Sample'}
                    </span>
                  </Button>
                </div>
              </div>

              {isSampleLoading ? (
                <div className="flex items-center justify-center py-8">
                  <LoadingSpinner size="lg" />
                  <span className="ml-3 text-gray-600">Loading sample data...</span>
                </div>
              ) : sampleData ? (
                <SamplePreview data={sampleData} />
              ) : null}
            </div>
          )}

          {/* File Upload Section */}
          {dataSource === 'upload' && (
            <FileUpload
              onFileSelect={handleFileSelect}
              disabled={isLoading}
            />
          )}

          {/* Controls - show when we have data */}
          {((dataSource === 'sample' && sampleData) || (dataSource === 'upload' && selectedFile)) && (
            <div className="mt-6 space-y-4">
              {/* Model Selection (for predict and optimize tabs) */}
              {activeTab !== 'compare' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Select Model
                  </label>
                  <div className="flex gap-3">
                    {['RF', 'MLP', 'XGBoost'].map((model) => (
                      <button
                        key={model}
                        onClick={() => setSelectedModel(model)}
                        disabled={isLoading}
                        className={`px-6 py-3 rounded-lg font-medium transition-all ${selectedModel === model
                          ? 'bg-mondesa-gradient text-white shadow-md'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                          }`}
                      >
                        {model === 'RF' ? 'Random Forest' : model === 'MLP' ? 'MLP' : 'XGBoost'}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Custom Threshold (for predict and compare tabs) */}
              {activeTab !== 'optimize' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Custom Threshold (optional)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    placeholder="Leave empty for default (0.5)"
                    value={customThreshold}
                    onChange={(e) => setCustomThreshold(e.target.value)}
                    disabled={isLoading}
                    className="w-full max-w-xs px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-[#A3C6D4] focus:outline-none transition-colors"
                  />
                </div>
              )}

              {/* Action Button */}
              <div className="pt-4">
                <Button
                  onClick={
                    activeTab === 'predict' ? handlePredict :
                      activeTab === 'compare' ? handleCompare :
                        handleOptimize
                  }
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <span className="flex items-center gap-2">
                      <LoadingSpinner size="sm" />
                      Processing...
                    </span>
                  ) : activeTab === 'predict' ? (
                    <span className="flex items-center gap-2">
                      <Flame size={18} />
                      Run Prediction
                    </span>
                  ) : activeTab === 'compare' ? (
                    <span className="flex items-center gap-2">
                      <GitCompare size={18} />
                      Compare All Models
                    </span>
                  ) : (
                    <span className="flex items-center gap-2">
                      <Target size={18} />
                      Find Optimal Threshold
                    </span>
                  )}
                </Button>
              </div>
            </div>
          )}
        </Card>

        {/* Error Display */}
        {error && (
          <Card className="mb-8 !bg-red-50 border-2 border-red-200" hover={false}>
            <div className="flex items-center gap-3 text-red-600">
              <AlertCircle size={24} />
              <div>
                <p className="font-medium">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </Card>
        )}

        {/* Results */}
        {activeTab === 'predict' && predictionResult && (
          <PredictionResults data={predictionResult} />
        )}

        {activeTab === 'compare' && comparisonResult && (
          <ModelComparison data={comparisonResult} />
        )}

        {activeTab === 'optimize' && optimizationResult && (
          <ThresholdOptimizerResults data={optimizationResult} />
        )}

        {/* Footer Info */}
        <div className="mt-12 text-center text-gray-500 text-sm">
          <p>
            Fire Risk Predictor uses machine learning models trained on historical fire occurrence data.
          </p>
          <p className="mt-2">
            <a
              href="https://github.com/MonDesa/fire-risk-predictor"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#A3C6D4] hover:underline"
            >
              View on GitHub
            </a>
            {' • '}
            <a
              href="https://mondesa.org"
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#A3C6D4] hover:underline"
            >
              MonDesa
            </a>
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;
