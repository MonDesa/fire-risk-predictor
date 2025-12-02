import { useCallback, useState } from 'react';
import { Upload, FileText, X, AlertCircle } from 'lucide-react';
import { Button } from './ui';

interface FileUploadProps {
    onFileSelect: (file: File) => void;
    accept?: string;
    maxSizeMB?: number;
    disabled?: boolean;
}

export function FileUpload({
    onFileSelect,
    accept = '.csv',
    maxSizeMB = 100,
    disabled = false
}: FileUploadProps) {
    const [dragActive, setDragActive] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [error, setError] = useState<string | null>(null);

    const validateFile = useCallback((file: File): boolean => {
        setError(null);

        // Check file type
        if (!file.name.endsWith('.csv')) {
            setError('Please upload a CSV file');
            return false;
        }

        // Check file size
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > maxSizeMB) {
            setError(`File too large. Maximum size: ${maxSizeMB}MB`);
            return false;
        }

        return true;
    }, [maxSizeMB]);

    const handleFile = useCallback((file: File) => {
        if (validateFile(file)) {
            setSelectedFile(file);
            onFileSelect(file);
        }
    }, [validateFile, onFileSelect]);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (disabled) return;

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, [disabled, handleFile]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const clearFile = () => {
        setSelectedFile(null);
        setError(null);
    };

    const formatFileSize = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    return (
        <div className="w-full">
            {!selectedFile ? (
                <div
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${dragActive
                        ? 'border-[#A3C6D4] bg-[#A3C6D4]/10'
                        : 'border-gray-300 hover:border-[#A3C6D4] hover:bg-gray-50'
                        } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                >
                    <input
                        type="file"
                        accept={accept}
                        onChange={handleChange}
                        disabled={disabled}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                    />

                    <div className="flex flex-col items-center gap-4">
                        <div className={`p-4 rounded-full ${dragActive ? 'bg-[#A3C6D4]' : 'bg-gray-100'} transition-colors`}>
                            <Upload
                                size={32}
                                className={dragActive ? 'text-white' : 'text-[#A3C6D4]'}
                            />
                        </div>

                        <div>
                            <p className="text-lg font-medium text-[#1a1a1a]">
                                {dragActive ? 'Drop your file here' : 'Drag & drop your CSV file'}
                            </p>
                            <p className="text-gray-500 mt-1">
                                or <span className="text-[#A3C6D4] font-medium">browse</span> to select
                            </p>
                        </div>

                        <p className="text-sm text-gray-400">
                            Maximum file size: {maxSizeMB}MB
                        </p>
                    </div>
                </div>
            ) : (
                <div className="border-2 border-[#B4C9A9] bg-[#B4C9A9]/10 rounded-2xl p-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="p-3 bg-[#B4C9A9] rounded-xl">
                                <FileText size={24} className="text-white" />
                            </div>
                            <div>
                                <p className="font-medium text-[#1a1a1a]">{selectedFile.name}</p>
                                <p className="text-sm text-gray-500">{formatFileSize(selectedFile.size)}</p>
                            </div>
                        </div>

                        <Button
                            variant="secondary"
                            onClick={clearFile}
                            className="!p-2 !rounded-full"
                        >
                            <X size={20} />
                        </Button>
                    </div>
                </div>
            )}

            {error && (
                <div className="mt-4 flex items-center gap-2 text-red-500 bg-red-50 p-3 rounded-lg">
                    <AlertCircle size={20} />
                    <span>{error}</span>
                </div>
            )}
        </div>
    );
}
