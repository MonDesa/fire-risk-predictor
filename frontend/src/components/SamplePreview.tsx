import { Database, CheckCircle } from 'lucide-react';
import type { SampleDataResponse } from '../services/api';

interface SamplePreviewProps {
    data: SampleDataResponse;
}

export function SamplePreview({ data }: SamplePreviewProps) {
    // Get a subset of columns to display (max 8 for readability)
    const displayColumns = data.columns.slice(0, 8);
    const hasMoreColumns = data.columns.length > 8;

    return (
        <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Database size={20} className="text-[#A3C6D4]" />
                    <span className="font-medium text-gray-700">
                        Sample Preview
                    </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-500">
                    <span>{data.sample_size.toLocaleString()} records</span>
                    <span>{data.columns.length} columns</span>
                    {data.has_ground_truth && (
                        <span className="flex items-center gap-1 text-[#B4C9A9]">
                            <CheckCircle size={14} />
                            Has ground truth
                        </span>
                    )}
                </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="border-b border-gray-200">
                            {displayColumns.map((col) => (
                                <th
                                    key={col}
                                    className="text-left py-2 px-3 font-medium text-gray-600 whitespace-nowrap"
                                >
                                    {col}
                                </th>
                            ))}
                            {hasMoreColumns && (
                                <th className="text-left py-2 px-3 font-medium text-gray-400">
                                    +{data.columns.length - 8} more...
                                </th>
                            )}
                        </tr>
                    </thead>
                    <tbody>
                        {data.preview.map((row, idx) => (
                            <tr
                                key={idx}
                                className={`border-b border-gray-100 ${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}
                            >
                                {displayColumns.map((col) => {
                                    const value = row[col];
                                    const displayValue = typeof value === 'number'
                                        ? Number.isInteger(value) ? value : value.toFixed(4)
                                        : String(value ?? '');

                                    return (
                                        <td
                                            key={col}
                                            className="py-2 px-3 text-gray-700 whitespace-nowrap"
                                        >
                                            {displayValue}
                                        </td>
                                    );
                                })}
                                {hasMoreColumns && (
                                    <td className="py-2 px-3 text-gray-400">...</td>
                                )}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Footer */}
            <div className="mt-3 text-xs text-gray-400 text-center">
                Showing first 10 rows of {data.sample_size.toLocaleString()} • From {data.total_records.toLocaleString()} total records in dataset
            </div>
        </div>
    );
}
