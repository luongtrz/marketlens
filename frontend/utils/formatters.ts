export const formatSource = (source: string): string => {
    if (!source) return 'Unknown Source';

    // If it looks like a URL, extract the domain
    if (source.startsWith('http')) {
        try {
            const url = new URL(source);
            // Get hostname (e.g., www.coindesk.com)
            let hostname = url.hostname;
            // Remove www.
            hostname = hostname.replace(/^www\./, '');
            // Remove TLD (.com, .org, etc.) - simple approach, take first part
            const namePart = hostname.split('.')[0];
            // Capitalize
            return namePart.charAt(0).toUpperCase() + namePart.slice(1);
        } catch (e) {
            // Fallback to normal processing if URL parsing fails
        }
    }

    // Remove "RSS Feed", "RSS", "Feed" (case insensitive)
    let formatted = source
        .replace(/rss\s*feed/gi, '')
        .replace(/rss/gi, '')
        .replace(/feed/gi, '');

    // Replace underscores/dashes with space
    formatted = formatted.replace(/[_-]/g, ' ');

    // Trim and Title Case
    return formatted
        .trim()
        .split(' ')
        .filter(word => word.length > 0)
        .map(word => {
            // Keep acronyms like BTC, ETH, AI, USD if they are already uppercase
            if (word === word.toUpperCase() && word.length > 1) return word;
            return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
        })
        .join(' ');
};

export const formatDateToUTC7 = (dateStr: string): string => {
    if (!dateStr) return '';
    try {
        const date = new Date(dateStr);

        // Check if valid date
        if (isNaN(date.getTime())) return dateStr;

        // Compact format: 14 Jan, 21:00
        return new Intl.DateTimeFormat('en-GB', {
            timeZone: 'Asia/Bangkok', // UTC+7
            day: 'numeric',
            month: 'short',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        }).format(date);
    } catch (e) {
        return dateStr;
    }
};

export const formatDateToLocalWithOffset = (dateStr: string): string => {
    if (!dateStr) return '';
    try {
        const date = new Date(dateStr);

        if (isNaN(date.getTime())) return dateStr;

        const offsetMinutes = -date.getTimezoneOffset();
        const sign = offsetMinutes >= 0 ? '+' : '-';
        const absMinutes = Math.abs(offsetMinutes);
        const hours = Math.floor(absMinutes / 60)
            .toString()
            .padStart(2, '0');
        const minutes = (absMinutes % 60)
            .toString()
            .padStart(2, '0');
        const offsetLabel = `UTC${sign}${hours}:${minutes}`;

        const formatted = new Intl.DateTimeFormat(undefined, {
            day: 'numeric',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
        }).format(date);

        return `${formatted} (${offsetLabel})`;
    } catch (e) {
        return dateStr;
    }
};
