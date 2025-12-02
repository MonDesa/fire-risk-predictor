import { Target, TrendingUp } from 'lucide-react';
import { Card } from './ui';
import type { ThresholdOptimizationResponse } from '../services/api';

interface ThresholdOptimizerResultsProps {
    data: ThresholdOptimizationResponse;
}

export function ThresholdOptimizerResults({ data }: ThresholdOptimizerResultsProps) {
    const {
        model_name,
        optimal_threshold,
        f1_score,
        accuracy,
        precision,
        recall,
        thresholds_tested,
        f1_scores
    } = data;

    const modelLabels: Record<string, string> = {
        'RF': 'Random Forest',
        'MLP': 'Multi-Layer Perceptron',
        'XGBoost': 'XGBoost',
    };

    // Find max F1 for chart scaling
    const maxF1 = Math.max(...f1_scores);
    const minF1 = Math.min(...f1_scores);

    return (
        <div className="space-y-6">
            {/* Optimal Threshold Card */}
            <Card hover={false}>
                <div className="text-center space-y-4">
                    <div className="inline-flex p-4 bg-mondesa-gradient rounded-full">
                        <Target size={32} className="text-white" />
                    </div>

                    <div>
                        <p className="text-gray-500">Optimal Threshold for {modelLabels[model_name] || model_name}</p>
                        <p className="text-5xl font-bold text-mondesa-gradient bg-clip-text text-transparent bg-gradient-to-r from-[#A3C6D4] to-[#B4C9A9]">
                            {optimal_threshold.toFixed(2)}
                        </p>
                    </div>

                    <p className="text-sm text-gray-500">
                        This threshold maximizes F1 Score for your dataset
                    </p>
                </div>
            </Card>

            {/* Metrics at Optimal Threshold */}
            <div className="grid gap-4 md:grid-cols-4">
                <Card hover={false} className="text-center">
                    <p className="text-sm text-gray-500 mb-1">F1 Score</p>
                    <p className="text-3xl font-bold text-[#8FAF80]">
                        {(f1_score * 100).toFixed(2)}%
                    </p>
                </Card>
                <Card hover={false} className="text-center">
                    <p className="text-sm text-gray-500 mb-1">Accuracy</p>
                    <p className="text-3xl font-bold text-[#7BA3B5]">
                        {(accuracy * 100).toFixed(2)}%
                    </p>
                </Card>
                <Card hover={false} className="text-center">
                    <p className="text-sm text-gray-500 mb-1">Precision</p>
                    <p className="text-3xl font-bold text-[#7BA3B5]">
                        {(precision * 100).toFixed(2)}%
                    </p>
                </Card>
                <Card hover={false} className="text-center">
                    <p className="text-sm text-gray-500 mb-1">Recall</p>
                    <p className="text-3xl font-bold text-[#7BA3B5]">
                        {(recall * 100).toFixed(2)}%
                    </p>
                </Card>
            </div>

            {/* F1 Score vs Threshold Chart */}
            <Card hover={false}>
                <div className="flex items-center gap-2 mb-6">
                    <TrendingUp size={24} className="text-[#A3C6D4]" />
                    <h4 className="font-heading text-lg font-semibold text-[#1a1a1a]">
                        F1 Score vs Threshold
                    </h4>
                </div>

                {/* Simple CSS chart */}
                <div className="relative">
                    {/* Y-axis labels */}
                    <div className="absolute left-0 top-0 bottom-8 w-12 flex flex-col justify-between text-xs text-gray-500">
                        <span>{(maxF1 * 100).toFixed(0)}%</span>
                        <span>{((maxF1 + minF1) / 2 * 100).toFixed(0)}%</span>
                        <span>{(minF1 * 100).toFixed(0)}%</span>
                    </div>

                    {/* Chart area */}
                    <div className="ml-14 pr-4">
                        <div className="relative h-48 border-l-2 border-b-2 border-gray-200">
                            {/* Grid lines */}
                            <div className="absolute inset-0 flex flex-col justify-between">
                                {[0, 1, 2, 3].map((i) => (
                                    <div key={i} className="border-t border-gray-100 w-full" />
                                ))}
                            </div>

                            {/* Bars */}
                            <div className="absolute inset-0 flex items-end justify-around px-1">
                                {thresholds_tested.map((threshold, idx) => {
                                    const f1 = f1_scores[idx];
                                    const heightPercent = ((f1 - minF1) / (maxF1 - minF1)) * 100 || 0;
                                    const isOptimal = threshold === optimal_threshold;

                                    return (
                                        <div
                                            key={threshold}
                                            className="relative group flex-1 mx-0.5"
                                        >
                                            <div
                                                className={`w-full rounded-t transition-all duration-300 ${isOptimal
                                                        ? 'bg-gradient-to-t from-[#B4C9A9] to-[#8FAF80]'
                                                        : 'bg-[#A3C6D4] hover:bg-[#7BA3B5]'
                                                    }`}
                                                style={{ height: `${Math.max(heightPercent, 5)}%` }}
                                            />

                                            {/* Tooltip */}
                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                                                <div className="bg-[#1a1a1a] text-white text-xs px-2 py-1 rounded whitespace-nowrap">
                                                    T: {threshold.toFixed(2)} | F1: {(f1 * 100).toFixed(1)}%
                                                </div>
                                            </div>

                                            {/* Optimal marker */}
                                            {isOptimal && (
                                                <div className="absolute -top-6 left-1/2 -translate-x-1/2">
                                                    <div className="bg-[#B4C9A9] text-white text-xs px-2 py-0.5 rounded">
                                                        Best
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* X-axis labels */}
                        <div className="flex justify-around mt-2 text-xs text-gray-500">
                            {thresholds_tested.filter((_, i) => i % 3 === 0 || i === thresholds_tested.length - 1).map((t) => (
                                <span key={t}>{t.toFixed(1)}</span>
                            ))}
                        </div>
                        <p className="text-center text-sm text-gray-500 mt-2">Threshold</p>
                    </div>
                </div>
            </Card>

            {/* Usage Instructions */}
            <Card hover={false}>
                <h4 className="font-heading text-lg font-semibold text-[#1a1a1a] mb-3">
                    How to Use This Threshold
                </h4>
                <div className="space-y-2 text-gray-600">
                    <p>
                        Use the optimal threshold <span className="font-mono bg-gray-100 px-2 py-0.5 rounded">{optimal_threshold.toFixed(2)}</span> when
                        making predictions with the <strong>{modelLabels[model_name] || model_name}</strong> model.
                    </p>
                    <p className="text-sm">
                        This value was determined by testing thresholds from {Math.min(...thresholds_tested).toFixed(1)} to {Math.max(...thresholds_tested).toFixed(1)} and
                        selecting the one that maximizes the F1 Score on your dataset.
                    </p>
                </div>
            </Card>
        </div>
    );
}
