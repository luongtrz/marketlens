import React from 'react';

type CoinLogoProps = {
    symbol: string;
    className?: string;
    title?: string;
};

/** Branded BTC / ETH logos (SVG); other symbols fall back to a colored initial pill. */
const CoinLogo: React.FC<CoinLogoProps> = ({ symbol, className = 'w-6 h-6', title }) => {
    const s = symbol.toUpperCase();
    const label = title ?? symbol;

    if (s === 'BTC') {
        return (
            <svg
                className={className}
                viewBox="0 0 32 32"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden
                role="img"
            >
                <title>{label}</title>
                <circle cx="16" cy="16" r="16" fill="#F7931A" />
                <path
                    fill="#fff"
                    d="M22.207 14.147c.26-1.748-1.068-2.686-2.882-3.315l.59-2.357-1.436-.358-.574 2.296c-.377-.095-.765-.185-1.152-.273l.577-2.31-1.435-.358-.59 2.357c-.312-.07-.618-.14-.915-.213l.002-.007-1.979-.494-.383 1.534s1.068.245 1.046.26c.584.146.689.534.671.84l-.671 2.687c.04.01.092.025.15.048l-.152-.038-.94 3.768c-.071.178-.252.445-.66.344.015.022-1.046-.261-1.046-.261l-.715 1.648 1.868.465c.347.087.687.18 1.021.265l-.596 2.392 1.434.358.59-2.357c.392.106.772.204 1.144.295l-.588 2.354 1.435.358.596-2.39c2.455.465 4.301.277 5.077-1.942.625-1.784-.031-2.811-1.328-3.48.945-.218 1.656-.838 1.846-2.12zm-3.302 4.626c-.444 1.781-3.456.818-4.429.577l.79-3.166c.973.243 4.092.725 3.639 2.589zm.443-4.644c-.406 1.628-2.924.801-3.737.598l.717-2.872c.813.203 3.437.58 3.02 2.274z"
                />
            </svg>
        );
    }

    if (s === 'ETH') {
        return (
            <svg
                className={className}
                viewBox="0 0 32 32"
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden
                role="img"
            >
                <title>{label}</title>
                <circle cx="16" cy="16" r="16" fill="#627EEA" />
                <path fill="#fff" d="M16 8L8 17.25h16L16 8z" />
                <path fill="#C5CCF9" fillOpacity=".9" d="M8 17.25L16 25l8-7.75H8z" />
                <path fill="#EDEEFC" fillOpacity=".55" d="M16 8v6.18l8 3.06L16 8z" />
                <path fill="#8D9CEC" fillOpacity=".75" d="M16 25v-11.76L8 17.25 16 25z" />
                <path fill="#fff" fillOpacity=".45" d="M24 17.25L16 14.18V8l8 9.25z" />
            </svg>
        );
    }

    const bg =
        s === 'SOL'
            ? 'bg-purple-600'
            : 'bg-slate-700 dark:bg-slate-600';

    return (
        <div
            className={`rounded-full flex items-center justify-center text-[10px] font-bold text-white shadow-sm shrink-0 ${bg} ${className}`}
            title={label}
            aria-hidden
        >
            {symbol[0] ?? '?'}
        </div>
    );
};

export default CoinLogo;
