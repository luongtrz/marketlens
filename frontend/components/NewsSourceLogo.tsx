import React, { useEffect, useState } from 'react';
import { Globe } from 'lucide-react';
import { normalizeNewsSourceDomain, publisherLogoUrl } from '../utils/newsSourceLogo';

export type NewsSourceLogoProps = {
    /** Hostname-ish value from ``NewsArticle.source`` */
    source: string;
    /** Article URL fallback when ``source`` is not a hostname */
    articleUrl?: string;
    /** Lucide Globe size when logo unavailable */
    iconSize?: number;
    globeClassName?: string;
    /** Extra classes on ``<img>`` */
    imgClassName?: string;
    /** Fill parent box edge-to-edge (e.g. modal tile); uses ``object-cover``. */
    fillSquare?: boolean;
};

/**
 * Publisher favicon when domain resolves (Google favicon API, then DuckDuckGo); else globe.
 */
const NewsSourceLogo: React.FC<NewsSourceLogoProps> = ({
    source,
    articleUrl,
    iconSize = 24,
    globeClassName = 'text-indigo-600 dark:text-indigo-400 shrink-0',
    imgClassName,
    fillSquare = false,
}) => {
    const domain = normalizeNewsSourceDomain(source, articleUrl ?? null);
    /** 0 = primary CDN, 1 = secondary CDN, ≥2 show globe */
    const [step, setStep] = useState(0);

    const resolvedImgClass =
        imgClassName ??
        (fillSquare
            ? 'h-full w-full min-h-0 min-w-0 object-cover object-center select-none'
            : 'max-h-[28px] max-w-[28px] w-auto h-auto object-contain rounded-md select-none');

    useEffect(() => {
        setStep(0);
    }, [domain, source, articleUrl]);

    if (!domain || step >= 2) {
        if (fillSquare) {
            return (
                <div className="flex h-full w-full items-center justify-center bg-slate-50 text-indigo-600 dark:bg-slate-900 dark:text-indigo-400">
                    <Globe size={iconSize} className={globeClassName} aria-hidden />
                </div>
            );
        }
        return <Globe size={iconSize} className={globeClassName} aria-hidden />;
    }

    const src =
        step === 0
            ? publisherLogoUrl(domain)
            : `https://icons.duckduckgo.com/ip3/${encodeURIComponent(domain)}.ico`;

    return (
        <img
            src={src}
            alt=""
            referrerPolicy="no-referrer"
            loading="lazy"
            decoding="async"
            draggable={false}
            className={resolvedImgClass}
            onError={() => setStep((s) => Math.min(s + 1, 2))}
        />
    );
};

export default NewsSourceLogo;
