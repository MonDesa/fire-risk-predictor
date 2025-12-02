import { Flame, Shield, Target, BarChart3, Download } from 'lucide-react';
import { Card, MetricCard } from './ui';
import type { BatchPredictionResponse } from '../services/api';

interface PredictionResultsProps {
    data: BatchPredictionResponse;
    onDownload?: () => void;
}

export function PredictionResults({ data, onDownload }: PredictionResultsProps) {
    const {
        model_name,
        threshold_used,
        total_records,
        fire_predicted,
        no_fire_predicted,
        evaluation_metrics,
        confusion_matrix
    } = data;

    const modelLabels: Record<string, string> = {
        'RF': 'Random Forest',
        'MLP': 'Multi-Layer Perceptron',
        'XGBoost': 'XGBoost',
    };

    const firePercentage = ((fire_predicted / total_records) * 100).toFixed(1);

    return (
        <div className="space-y-6">
            {/* Summary Header */}
            <Card hover={false}>
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                        <h3 className="font-heading text-xl font-semibold text-[#1a1a1a]">
                            Prediction Results
                        </h3>
                        <p className="text-gray-500">
                            Model: {modelLabels[model_name] || model_name} • Threshold: {threshold_used.toFixed(2)}
                        </p>
                    </div>

                    {onDownload && (
                        <button
                            onClick={onDownload}
                            className="flex items-center gap-2 px-4 py-2 text-[#A3C6D4] border-2 border-[#A3C6D4] rounded-lg hover:bg-[#A3C6D4] hover:text-white transition-all"
                        >
                            <Download size={18} />
                            Download Results
                        </button>
                    )}
                </div>
            </Card>

            {/* Key Metrics */}
            <div className="grid gap-4 md:grid-cols-4">
                <MetricCard
                    label="Total Records"
                    value={total_records.toLocaleString()}
                    icon={<BarChart3 size={24} />}
                    color="blue"
                />
                <MetricCard
                    label="Fire Risk Predicted"
                    value={`${fire_predicted.toLocaleString()} (${firePercentage}%)`}
                    icon={<Flame size={24} />}
                    color="amber"
                />
                <MetricCard
                    label="No Fire Risk"
                    value={no_fire_predicted.toLocaleString()}
                    icon={<Shield size={24} />}
                    color="green"
                />
                <MetricCard
                    label="Threshold Used"
                    value={threshold_used.toFixed(2)}
                    icon={<Target size={24} />}
                    color="blue"
                />
            </div>

            {/* Visualization */}
            <Card hover={false}>
                <h4 className="font-heading text-lg font-semibold text-[#1a1a1a] mb-4">
                    Prediction Distribution
                </h4>

                <div className="space-y-4">
                    {/* Progress bar visualization */}
                    <div className="relative h-12 rounded-xl overflow-hidden bg-gray-100">
                        <div
                            className="absolute left-0 top-0 h-full bg-gradient-to-r from-red-400 to-amber-400 flex items-center justify-end pr-3"
                            style={{ width: `${firePercentage}%` }}
                        >
                            {parseFloat(firePercentage) > 15 && (
                                <span className="text-white font-medium text-sm">
                                    {fire_predicted} Fire
                                </span>
                            )}
                        </div>
                        <div
                            className="absolute right-0 top-0 h-full bg-gradient-to-r from-[#B4C9A9] to-[#8FAF80] flex items-center justify-start pl-3"
                            style={{ width: `${100 - parseFloat(firePercentage)}%` }}
                        >
                            {parseFloat(firePercentage) < 85 && (
                                <span className="text-white font-medium text-sm">
                                    {no_fire_predicted} No Fire
                                </span>
                            )}
                        </div>
                    </div>

                    {/* Legend */}
                    <div className="flex justify-center gap-8">
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-gradient-to-r from-red-400 to-amber-400" />
                            <span className="text-sm text-gray-600">Fire Risk ({firePercentage}%)</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-gradient-to-r from-[#B4C9A9] to-[#8FAF80]" />
                            <span className="text-sm text-gray-600">No Fire Risk ({(100 - parseFloat(firePercentage)).toFixed(1)}%)</span>
                        </div>
                    </div>
                </div>
            </Card>

            {/* Evaluation Metrics (if ground truth available) */}
            {evaluation_metrics && (
                <Card hover={false}>
                    <h4 className="font-heading text-lg font-semibold text-[#1a1a1a] mb-4">
                        Evaluation Metrics
                    </h4>

                    <div className="grid gap-4 md:grid-cols-4">
                        <div className="text-center p-4 bg-[#A3C6D4]/10 rounded-xl">
                            <p className="text-3xl font-bold text-[#7BA3B5]">
                                {(evaluation_metrics.accuracy * 100).toFixed(2)}%
                            </p>
                            <p className="text-sm text-gray-600 mt-1">Accuracy</p>
                        </div>
                        <div className="text-center p-4 bg-[#B4C9A9]/10 rounded-xl">
                            <p className="text-3xl font-bold text-[#8FAF80]">
                                {(evaluation_metrics.precision * 100).toFixed(2)}%
                            </p>
                            <p className="text-sm text-gray-600 mt-1">Precision</p>
                        </div>
                        <div className="text-center p-4 bg-amber-400/10 rounded-xl">
                            <p className="text-3xl font-bold text-amber-500">
                                {(evaluation_metrics.recall * 100).toFixed(2)}%
                            </p>
                            <p className="text-sm text-gray-600 mt-1">Recall</p>
                        </div>
                        <div className="text-center p-4 bg-mondesa-gradient rounded-xl">
                            <p className="text-3xl font-bold text-white">
                                {(evaluation_metrics.f1_score * 100).toFixed(2)}%
                            </p>
                            <p className="text-sm text-white/80 mt-1">F1 Score</p>
                        </div>
                    </div>
                </Card>
            )}

            {/* Confusion Matrix */}
            {confusion_matrix && (
                <Card hover={false}>
                    <h4 className="font-heading text-lg font-semibold text-[#1a1a1a] mb-4">
                        Confusion Matrix
                    </h4>

                    <div className="max-w-md mx-auto">
                        <div className="grid grid-cols-3 gap-2 text-center">
                            {/* Header row */}
                            <div></div>
                            <div className="font-medium text-gray-500 py-2">Predicted: No</div>
                            <div className="font-medium text-gray-500 py-2">Predicted: Yes</div>

                            {/* Actual No row */}
                            <div className="font-medium text-gray-500 py-4 text-right pr-2">Actual: No</div>
                            <div className="bg-green-100 p-4 rounded-xl">
                                <p className="text-2xl font-bold text-green-600">{confusion_matrix[0][0]}</p>
                                <p className="text-sm text-gray-500 mt-1">True Negative</p>
                            </div>
                            <div className="bg-red-100 p-4 rounded-xl">
                                <p className="text-2xl font-bold text-red-400">{confusion_matrix[0][1]}</p>
                                <p className="text-sm text-gray-500 mt-1">False Positive</p>
                            </div>

                            {/* Actual Yes row */}
                            <div className="font-medium text-gray-500 py-4 text-right pr-2">Actual: Yes</div>
                            <div className="bg-red-100 p-4 rounded-xl">
                                <p className="text-2xl font-bold text-red-400">{confusion_matrix[1][0]}</p>
                                <p className="text-sm text-gray-500 mt-1">False Negative</p>
                            </div>
                            <div className="bg-green-100 p-4 rounded-xl">
                                <p className="text-2xl font-bold text-green-600">{confusion_matrix[1][1]}</p>
                                <p className="text-sm text-gray-500 mt-1">True Positive</p>
                            </div>
                        </div>
                    </div>
                </Card>
            )}
        </div>
    );
}
