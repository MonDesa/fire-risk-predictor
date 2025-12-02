import { Trophy, AlertTriangle, CheckCircle } from 'lucide-react';
import { Card, StatusBadge } from './ui';
import type { ComparisonResponse, ModelComparisonResult } from '../services/api';

interface ModelComparisonProps {
    data: ComparisonResponse;
}

export function ModelComparison({ data }: ModelComparisonProps) {
    const { total_records, has_ground_truth, results, best_model } = data;

    return (
        <div className="space-y-6">
            {/* Summary Header */}
            <Card hover={false}>
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                        <h3 className="font-heading text-xl font-semibold text-[#1a1a1a]">
                            Comparison Results
                        </h3>
                        <p className="text-gray-500">
                            {total_records.toLocaleString()} records analyzed across 3 models
                        </p>
                    </div>

                    <div className="flex items-center gap-3">
                        {has_ground_truth ? (
                            <StatusBadge status="success">
                                <CheckCircle size={16} className="mr-1" />
                                Ground Truth Available
                            </StatusBadge>
                        ) : (
                            <StatusBadge status="warning">
                                <AlertTriangle size={16} className="mr-1" />
                                No Ground Truth
                            </StatusBadge>
                        )}
                    </div>
                </div>

                {best_model && (
                    <div className="mt-4 p-4 bg-mondesa-gradient rounded-xl text-white flex items-center gap-3">
                        <Trophy size={24} />
                        <div>
                            <p className="font-medium">Best Performing Model</p>
                            <p className="text-2xl font-bold">{best_model}</p>
                        </div>
                    </div>
                )}
            </Card>

            {/* Individual Model Results */}
            <div className="grid gap-6 md:grid-cols-3">
                {results.map((result) => (
                    <ModelResultCard
                        key={result.model_name}
                        result={result}
                        isBest={result.model_name === best_model}
                        hasGroundTruth={has_ground_truth}
                    />
                ))}
            </div>

            {/* Confusion Matrices */}
            {has_ground_truth && (
                <div className="grid gap-6 md:grid-cols-3">
                    {results.map((result) => (
                        result.confusion_matrix && (
                            <ConfusionMatrixCard
                                key={`cm-${result.model_name}`}
                                modelName={result.model_name}
                                matrix={result.confusion_matrix}
                            />
                        )
                    ))}
                </div>
            )}
        </div>
    );
}

interface ModelResultCardProps {
    result: ModelComparisonResult;
    isBest: boolean;
    hasGroundTruth: boolean;
}

function ModelResultCard({ result, isBest, hasGroundTruth }: ModelResultCardProps) {
    const { model_name, threshold_used, fire_predicted, no_fire_predicted, evaluation_metrics } = result;

    const modelLabels: Record<string, string> = {
        'RF': 'Random Forest',
        'MLP': 'Multi-Layer Perceptron',
        'XGBoost': 'XGBoost',
    };

    return (
        <Card className={isBest ? 'ring-2 ring-[#B4C9A9] ring-offset-2' : ''}>
            <div className="space-y-4">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <h4 className="font-heading text-lg font-semibold text-[#1a1a1a]">
                            {modelLabels[model_name] || model_name}
                        </h4>
                        <p className="text-sm text-gray-500">
                            Threshold: {threshold_used.toFixed(2)}
                        </p>
                    </div>
                    {isBest && (
                        <div className="p-2 bg-[#B4C9A9] rounded-full">
                            <Trophy size={20} className="text-white" />
                        </div>
                    )}
                </div>

                {/* Predictions Summary */}
                <div className="grid grid-cols-2 gap-3">
                    <div className="bg-red-50 p-3 rounded-lg text-center">
                        <p className="text-2xl font-bold text-red-500">{fire_predicted}</p>
                        <p className="text-sm text-gray-600">Fire Risk</p>
                    </div>
                    <div className="bg-green-50 p-3 rounded-lg text-center">
                        <p className="text-2xl font-bold text-green-500">{no_fire_predicted}</p>
                        <p className="text-sm text-gray-600">No Fire Risk</p>
                    </div>
                </div>

                {/* Metrics */}
                {hasGroundTruth && evaluation_metrics && (
                    <div className="space-y-2 pt-4 border-t border-gray-100">
                        <MetricRow label="Accuracy" value={evaluation_metrics.accuracy} />
                        <MetricRow label="Precision" value={evaluation_metrics.precision} />
                        <MetricRow label="Recall" value={evaluation_metrics.recall} />
                        <MetricRow
                            label="F1 Score"
                            value={evaluation_metrics.f1_score}
                            highlight={isBest}
                        />
                    </div>
                )}
            </div>
        </Card>
    );
}

interface MetricRowProps {
    label: string;
    value: number;
    highlight?: boolean;
}

function MetricRow({ label, value, highlight = false }: MetricRowProps) {
    const percentage = (value * 100).toFixed(2);

    return (
        <div className={`flex items-center justify-between py-1 ${highlight ? 'bg-[#B4C9A9]/10 px-2 rounded-lg -mx-2' : ''}`}>
            <span className={`text-sm ${highlight ? 'font-medium text-[#1a1a1a]' : 'text-gray-600'}`}>
                {label}
            </span>
            <div className="flex items-center gap-2">
                <div className="w-24 h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                        className={`h-full rounded-full ${highlight ? 'bg-[#B4C9A9]' : 'bg-[#A3C6D4]'}`}
                        style={{ width: `${percentage}%` }}
                    />
                </div>
                <span className={`text-sm font-medium ${highlight ? 'text-[#8FAF80]' : 'text-[#7BA3B5]'}`}>
                    {percentage}%
                </span>
            </div>
        </div>
    );
}

interface ConfusionMatrixCardProps {
    modelName: string;
    matrix: number[][];
}

function ConfusionMatrixCard({ modelName, matrix }: ConfusionMatrixCardProps) {
    const [[tn, fp], [fn, tp]] = matrix;
    const total = tn + fp + fn + tp;

    return (
        <Card hover={false}>
            <h4 className="font-heading text-lg font-semibold text-[#1a1a1a] mb-4">
                {modelName} - Confusion Matrix
            </h4>

            <div className="grid grid-cols-3 gap-1 text-center text-sm">
                {/* Header row */}
                <div></div>
                <div className="font-medium text-gray-500 py-2">Pred: No</div>
                <div className="font-medium text-gray-500 py-2">Pred: Yes</div>

                {/* Actual No row */}
                <div className="font-medium text-gray-500 py-2 text-right pr-2">Actual: No</div>
                <div className="bg-green-100 p-3 rounded-lg">
                    <p className="text-lg font-bold text-green-600">{tn}</p>
                    <p className="text-xs text-gray-500">TN ({((tn / total) * 100).toFixed(1)}%)</p>
                </div>
                <div className="bg-red-100 p-3 rounded-lg">
                    <p className="text-lg font-bold text-red-400">{fp}</p>
                    <p className="text-xs text-gray-500">FP ({((fp / total) * 100).toFixed(1)}%)</p>
                </div>

                {/* Actual Yes row */}
                <div className="font-medium text-gray-500 py-2 text-right pr-2">Actual: Yes</div>
                <div className="bg-red-100 p-3 rounded-lg">
                    <p className="text-lg font-bold text-red-400">{fn}</p>
                    <p className="text-xs text-gray-500">FN ({((fn / total) * 100).toFixed(1)}%)</p>
                </div>
                <div className="bg-green-100 p-3 rounded-lg">
                    <p className="text-lg font-bold text-green-600">{tp}</p>
                    <p className="text-xs text-gray-500">TP ({((tp / total) * 100).toFixed(1)}%)</p>
                </div>
            </div>
        </Card>
    );
}
