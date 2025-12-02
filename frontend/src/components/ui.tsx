import type { ReactNode } from 'react';

interface HeaderProps {
    children?: ReactNode;
}

export function Header({ children }: HeaderProps) {
    return (
        <header className="bg-mondesa-gradient text-white py-12 px-5 text-center">
            <div className="max-w-6xl mx-auto">
                <div className="flex items-center justify-center gap-4 mb-4">
                    <img
                        src="https://mondesa.org/logo.svg"
                        alt="MonDesa Logo"
                        className="h-12 w-auto"
                    />
                    <h1 className="font-heading text-4xl md:text-5xl font-bold">
                        Fire Risk Predictor
                    </h1>
                </div>
                <p className="text-lg opacity-95 max-w-2xl mx-auto">
                    ML-based fire risk prediction using Random Forest, MLP, and XGBoost models
                </p>
                {children}
            </div>
        </header>
    );
}

interface CardProps {
    children: ReactNode;
    className?: string;
    hover?: boolean;
}

export function Card({ children, className = '', hover = true }: CardProps) {
    return (
        <div className={`${hover ? 'card' : 'card-static'} ${className}`}>
            {children}
        </div>
    );
}

interface ButtonProps {
    children: ReactNode;
    onClick?: () => void;
    variant?: 'primary' | 'secondary';
    disabled?: boolean;
    className?: string;
    type?: 'button' | 'submit';
}

export function Button({
    children,
    onClick,
    variant = 'primary',
    disabled = false,
    className = '',
    type = 'button'
}: ButtonProps) {
    const baseClass = variant === 'primary' ? 'btn-primary' : 'btn-secondary';

    return (
        <button
            type={type}
            onClick={onClick}
            disabled={disabled}
            className={`${baseClass} ${className}`}
        >
            {children}
        </button>
    );
}

interface StatusBadgeProps {
    status: 'success' | 'warning' | 'error' | 'info';
    children: ReactNode;
}

export function StatusBadge({ status, children }: StatusBadgeProps) {
    const colors = {
        success: 'bg-[#B4C9A9] text-white',
        warning: 'bg-amber-400 text-white',
        error: 'bg-red-400 text-white',
        info: 'bg-[#A3C6D4] text-white',
    };

    return (
        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${colors[status]}`}>
            {children}
        </span>
    );
}

interface LoadingSpinnerProps {
    size?: 'sm' | 'md' | 'lg';
}

export function LoadingSpinner({ size = 'md' }: LoadingSpinnerProps) {
    const sizes = {
        sm: 'w-4 h-4',
        md: 'w-8 h-8',
        lg: 'w-12 h-12',
    };

    return (
        <div className={`${sizes[size]} animate-spin rounded-full border-4 border-[#A3C6D4] border-t-transparent`} />
    );
}

interface ProgressBarProps {
    value: number;
    max?: number;
    label?: string;
}

export function ProgressBar({ value, max = 100, label }: ProgressBarProps) {
    const percentage = Math.min((value / max) * 100, 100);

    return (
        <div className="w-full">
            {label && (
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>{label}</span>
                    <span>{percentage.toFixed(1)}%</span>
                </div>
            )}
            <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                <div
                    className="h-full bg-mondesa-gradient-45 transition-all duration-300"
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
}

interface TabsProps {
    tabs: { id: string; label: string; icon?: ReactNode }[];
    activeTab: string;
    onTabChange: (tabId: string) => void;
}

export function Tabs({ tabs, activeTab, onTabChange }: TabsProps) {
    return (
        <div className="flex gap-2 p-1 bg-gray-100 rounded-xl">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => onTabChange(tab.id)}
                    className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-300 ${activeTab === tab.id
                        ? 'bg-white text-[#1a1a1a] shadow-md'
                        : 'text-gray-600 hover:text-[#1a1a1a] hover:bg-white/50'
                        }`}
                >
                    {tab.icon}
                    {tab.label}
                </button>
            ))}
        </div>
    );
}

interface MetricCardProps {
    label: string;
    value: string | number;
    icon?: ReactNode;
    color?: 'blue' | 'green' | 'amber' | 'red';
}

export function MetricCard({ label, value, icon, color = 'blue' }: MetricCardProps) {
    const colors = {
        blue: 'border-l-[#A3C6D4]',
        green: 'border-l-[#B4C9A9]',
        amber: 'border-l-amber-400',
        red: 'border-l-red-400',
    };

    return (
        <div className={`bg-white rounded-xl p-4 shadow-sm border-l-4 ${colors[color]}`}>
            <div className="flex items-center gap-3">
                {icon && <div className="text-[#A3C6D4]">{icon}</div>}
                <div>
                    <p className="text-sm text-gray-500">{label}</p>
                    <p className="text-2xl font-bold text-[#1a1a1a]">
                        {typeof value === 'number' ? value.toFixed(4) : value}
                    </p>
                </div>
            </div>
        </div>
    );
}
