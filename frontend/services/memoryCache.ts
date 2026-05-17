type CacheEntry<T> = {
    value: T;
    expiresAt: number;
};

const entries = new Map<string, CacheEntry<unknown>>();
const pending = new Map<string, Promise<unknown>>();

const clone = <T>(value: T): T => {
    if (typeof structuredClone === 'function') {
        return structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value)) as T;
};

export const getCached = <T>(key: string): T | null => {
    const entry = entries.get(key);
    if (!entry) {
        console.debug('[memory-cache] miss', key);
        return null;
    }
    if (Date.now() >= entry.expiresAt) {
        entries.delete(key);
        console.debug('[memory-cache] expired', key);
        return null;
    }
    console.info('[memory-cache] hit', key);
    return clone(entry.value as T);
};

export const setCached = <T>(key: string, value: T, ttlMs: number): T => {
    if (ttlMs > 0) {
        entries.set(key, {
            value: clone(value),
            expiresAt: Date.now() + ttlMs,
        });
        console.debug('[memory-cache] set', key, `ttl=${ttlMs}ms`);
    }
    return value;
};

export const getOrFetchCached = async <T>(
    key: string,
    ttlMs: number,
    loader: () => Promise<T>,
): Promise<T> => {
    const cached = getCached<T>(key);
    if (cached !== null) return cached;

    const inFlight = pending.get(key);
    if (inFlight) {
        console.info('[memory-cache] in-flight hit', key);
        return clone(await inFlight) as T;
    }

    const promise = loader();
    pending.set(key, promise);
    try {
        const value = await promise;
        setCached(key, value, ttlMs);
        return clone(value);
    } finally {
        pending.delete(key);
    }
};
