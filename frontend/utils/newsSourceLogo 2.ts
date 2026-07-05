/** Map publisher domain → direct logo URL when favicons look poor. */

const PUBLISHER_LOGO_OVERRIDE: Partial<Record<string, string>> = {};

function coerceHostname(host: string): string | null {
    const x = host.trim().replace(/^www\./i, '').toLowerCase();
    if (!x || x === 'unknown' || !x.includes('.')) return null;
    return x;
}

/** Derive canonical domain for favicon lookups from API ``source`` and/or article ``url``. */
export function normalizeNewsSourceDomain(source: string | undefined | null, articleUrl?: string | null): string | null {
    const raw = (source || '').trim();
    if (!raw && !articleUrl) return null;

    const lc = raw.toLowerCase();

    if (lc.startsWith('http://') || lc.startsWith('https://')) {
        try {
            return coerceHostname(new URL(lc).hostname);
        } catch {
            /* fall through */
        }
    }

    if (lc && !lc.includes('/') && !lc.includes(' ')) {
        try {
            return coerceHostname(new URL(`https://${lc}`).hostname);
        } catch {
            /* fall through */
        }
    }

    if (articleUrl) {
        try {
            return coerceHostname(new URL(articleUrl).hostname);
        } catch {
            /* noop */
        }
    }

    return null;
}

/** Resolved icon URL for a normalized domain (override or favicon CDN). */
export function publisherLogoUrl(domain: string): string {
    const o = PUBLISHER_LOGO_OVERRIDE[domain];
    if (o) return o;
    return `https://www.google.com/s2/favicons?sz=128&domain=${encodeURIComponent(domain)}`;
}
