/**
 * CallWhisper LocalStorage Cache
 *
 * Based on LibV2 advanced-react-mastery patterns:
 * - Offline-first with localStorage
 * - Cache invalidation with TTL
 * - LRU eviction when storage is full
 * - Graceful fallback when localStorage unavailable
 *
 * Usage:
 *   TranscriptionCache.set(recordingId, transcriptionData);
 *   const data = TranscriptionCache.get(recordingId);
 */

const TranscriptionCache = {
    // Configuration
    config: {
        prefix: 'callwhisper_cache_',
        indexKey: 'callwhisper_cache_index',
        maxEntries: 50,          // Maximum cached transcriptions
        maxSizeMB: 5,            // Max total cache size in MB
        ttlMs: 7 * 24 * 60 * 60 * 1000,  // 7 days default TTL
    },

    // In-memory index for fast lookups
    _index: null,

    /**
     * Initialize cache
     */
    init() {
        this._loadIndex();
        this._cleanup();
        console.log('[Cache] Initialized with', Object.keys(this._index.entries).length, 'entries');
    },

    /**
     * Get cached transcription
     *
     * @param {string} recordingId
     * @returns {Object|null} Cached data or null
     */
    get(recordingId) {
        if (!this._isAvailable()) {
            return null;
        }

        const entry = this._index.entries[recordingId];
        if (!entry) {
            return null;
        }

        // Check TTL
        if (Date.now() - entry.cachedAt > this.config.ttlMs) {
            this.clear(recordingId);
            return null;
        }

        try {
            const key = this.config.prefix + recordingId;
            const data = localStorage.getItem(key);

            if (!data) {
                // Index out of sync, clean up
                delete this._index.entries[recordingId];
                this._saveIndex();
                return null;
            }

            // Update access time for LRU
            entry.lastAccessed = Date.now();
            entry.accessCount = (entry.accessCount || 0) + 1;
            this._saveIndex();

            return JSON.parse(data);

        } catch (error) {
            console.error('[Cache] Get error:', error);
            return null;
        }
    },

    /**
     * Cache transcription data
     *
     * @param {string} recordingId
     * @param {Object} data - Data to cache
     */
    set(recordingId, data) {
        if (!this._isAvailable()) {
            return false;
        }

        try {
            const key = this.config.prefix + recordingId;
            const serialized = JSON.stringify({
                ...data,
                _cachedAt: new Date().toISOString(),
            });

            // Check if we need to make room
            const entrySize = serialized.length;
            this._ensureSpace(entrySize);

            // Store data
            localStorage.setItem(key, serialized);

            // Update index
            this._index.entries[recordingId] = {
                cachedAt: Date.now(),
                lastAccessed: Date.now(),
                accessCount: 0,
                size: entrySize,
            };
            this._saveIndex();

            console.log('[Cache] Cached recording:', recordingId);
            return true;

        } catch (error) {
            if (error.name === 'QuotaExceededError') {
                // Storage full, try to evict and retry
                this._evictOldest();
                try {
                    return this.set(recordingId, data);
                } catch (retryError) {
                    console.error('[Cache] Storage full, cannot cache');
                }
            }
            console.error('[Cache] Set error:', error);
            return false;
        }
    },

    /**
     * Clear specific entry from cache
     *
     * @param {string} recordingId
     */
    clear(recordingId) {
        if (!this._isAvailable()) {
            return;
        }

        try {
            const key = this.config.prefix + recordingId;
            localStorage.removeItem(key);

            delete this._index.entries[recordingId];
            this._saveIndex();

            console.log('[Cache] Cleared:', recordingId);

        } catch (error) {
            console.error('[Cache] Clear error:', error);
        }
    },

    /**
     * Clear all cached data
     */
    clearAll() {
        if (!this._isAvailable()) {
            return 0;
        }

        let count = 0;
        try {
            for (const recordingId of Object.keys(this._index.entries)) {
                const key = this.config.prefix + recordingId;
                localStorage.removeItem(key);
                count++;
            }

            localStorage.removeItem(this.config.indexKey);
            this._index = { entries: {}, createdAt: Date.now() };

            console.log('[Cache] Cleared all:', count, 'entries');

        } catch (error) {
            console.error('[Cache] Clear all error:', error);
        }

        return count;
    },

    /**
     * Check if data is cached
     *
     * @param {string} recordingId
     * @returns {boolean}
     */
    has(recordingId) {
        if (!this._isAvailable()) {
            return false;
        }

        const entry = this._index.entries[recordingId];
        if (!entry) {
            return false;
        }

        // Check TTL
        if (Date.now() - entry.cachedAt > this.config.ttlMs) {
            this.clear(recordingId);
            return false;
        }

        return true;
    },

    /**
     * Get cache metadata for entry
     *
     * @param {string} recordingId
     * @returns {Object|null}
     */
    getMeta(recordingId) {
        if (!this._index.entries[recordingId]) {
            return null;
        }

        const entry = this._index.entries[recordingId];
        return {
            cachedAt: new Date(entry.cachedAt).toISOString(),
            lastAccessed: new Date(entry.lastAccessed).toISOString(),
            accessCount: entry.accessCount || 0,
            size: entry.size || 0,
            ttlRemaining: Math.max(0, this.config.ttlMs - (Date.now() - entry.cachedAt)),
        };
    },

    /**
     * Get cache statistics
     */
    getStats() {
        const entries = Object.keys(this._index.entries);
        let totalSize = 0;

        for (const entry of Object.values(this._index.entries)) {
            totalSize += entry.size || 0;
        }

        return {
            entryCount: entries.length,
            maxEntries: this.config.maxEntries,
            totalSizeBytes: totalSize,
            totalSizeMB: (totalSize / (1024 * 1024)).toFixed(2),
            maxSizeMB: this.config.maxSizeMB,
            available: this._isAvailable(),
        };
    },

    /**
     * List all cached recording IDs
     */
    listCached() {
        return Object.keys(this._index.entries).map(recordingId => ({
            recordingId,
            ...this.getMeta(recordingId),
        }));
    },

    // --- Private methods ---

    /**
     * Check if localStorage is available
     */
    _isAvailable() {
        try {
            const test = '__cache_test__';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    },

    /**
     * Load index from localStorage
     */
    _loadIndex() {
        try {
            const stored = localStorage.getItem(this.config.indexKey);
            if (stored) {
                this._index = JSON.parse(stored);
            } else {
                this._index = { entries: {}, createdAt: Date.now() };
            }
        } catch (error) {
            console.error('[Cache] Index load error:', error);
            this._index = { entries: {}, createdAt: Date.now() };
        }
    },

    /**
     * Save index to localStorage
     */
    _saveIndex() {
        try {
            localStorage.setItem(this.config.indexKey, JSON.stringify(this._index));
        } catch (error) {
            console.error('[Cache] Index save error:', error);
        }
    },

    /**
     * Cleanup expired entries
     */
    _cleanup() {
        const now = Date.now();
        let cleanedCount = 0;

        for (const [recordingId, entry] of Object.entries(this._index.entries)) {
            if (now - entry.cachedAt > this.config.ttlMs) {
                this.clear(recordingId);
                cleanedCount++;
            }
        }

        if (cleanedCount > 0) {
            console.log('[Cache] Cleaned up', cleanedCount, 'expired entries');
        }
    },

    /**
     * Ensure there's space for new entry
     */
    _ensureSpace(neededBytes) {
        const entries = Object.keys(this._index.entries);

        // Check entry count
        while (entries.length >= this.config.maxEntries) {
            this._evictOldest();
        }

        // Check total size
        let totalSize = Object.values(this._index.entries)
            .reduce((sum, e) => sum + (e.size || 0), 0);

        const maxBytes = this.config.maxSizeMB * 1024 * 1024;
        while (totalSize + neededBytes > maxBytes && Object.keys(this._index.entries).length > 0) {
            const evictedSize = this._evictOldest();
            totalSize -= evictedSize;
        }
    },

    /**
     * Evict least recently accessed entry
     * Returns size of evicted entry
     */
    _evictOldest() {
        const entries = Object.entries(this._index.entries);
        if (entries.length === 0) {
            return 0;
        }

        // Find least recently accessed
        let oldest = null;
        let oldestTime = Infinity;

        for (const [recordingId, entry] of entries) {
            if (entry.lastAccessed < oldestTime) {
                oldestTime = entry.lastAccessed;
                oldest = recordingId;
            }
        }

        if (oldest) {
            const size = this._index.entries[oldest]?.size || 0;
            this.clear(oldest);
            console.log('[Cache] Evicted LRU:', oldest);
            return size;
        }

        return 0;
    },

    /**
     * Prefetch transcription from server and cache it
     */
    async prefetch(recordingId) {
        if (this.has(recordingId)) {
            console.log('[Cache] Already cached:', recordingId);
            return this.get(recordingId);
        }

        try {
            const response = await fetch(`/api/recording/${recordingId}`);
            if (!response.ok) {
                throw new Error(`Fetch failed: ${response.status}`);
            }

            const data = await response.json();
            this.set(recordingId, data);
            return data;

        } catch (error) {
            console.error('[Cache] Prefetch error:', error);
            return null;
        }
    },

    /**
     * Get with fallback to server fetch
     */
    async getOrFetch(recordingId) {
        // Try cache first
        const cached = this.get(recordingId);
        if (cached) {
            return { data: cached, fromCache: true };
        }

        // Fetch from server
        try {
            const response = await fetch(`/api/recording/${recordingId}`);
            if (!response.ok) {
                throw new Error(`Fetch failed: ${response.status}`);
            }

            const data = await response.json();

            // Cache for next time
            this.set(recordingId, data);

            return { data, fromCache: false };

        } catch (error) {
            console.error('[Cache] Fetch error:', error);
            return { data: null, fromCache: false, error: error.message };
        }
    },
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => TranscriptionCache.init());
} else {
    TranscriptionCache.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TranscriptionCache;
}
