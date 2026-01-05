/**
 * CallWhisper Auto-Save with Debounce
 *
 * Based on LibV2 advanced-react-mastery patterns:
 * - Debounced auto-save for transcript edits
 * - Visual status indicator
 * - Conflict detection
 * - Offline queue for failed saves
 *
 * Usage:
 *   AutoSave.init({ saveEndpoint: '/api/transcription/save' });
 *   AutoSave.track(textarea, recordingId);
 */

const AutoSave = {
    // Configuration
    config: {
        debounceMs: 500,         // Wait 500ms after last change
        maxWaitMs: 5000,         // Force save after 5s even if still typing
        saveEndpoint: '/api/transcription/save',
        retryAttempts: 3,
        retryDelayMs: 1000,
    },

    // State
    _pendingChanges: new Map(),  // recordingId -> { content, timer, lastChange }
    _saveQueue: [],              // Offline queue
    _lastSaved: new Map(),       // recordingId -> { content, timestamp }
    _status: 'idle',             // idle, saving, saved, error
    _listeners: [],

    /**
     * Initialize auto-save
     */
    init(options = {}) {
        Object.assign(this.config, options);

        // Process offline queue when online
        window.addEventListener('online', () => this._processQueue());

        console.log('[AutoSave] Initialized with', this.config.debounceMs, 'ms debounce');
    },

    /**
     * Track changes on an element
     *
     * @param {HTMLElement} element - Input/textarea to track
     * @param {string} recordingId - Associated recording ID
     */
    track(element, recordingId) {
        if (!element || !recordingId) {
            console.error('[AutoSave] track() requires element and recordingId');
            return;
        }

        // Set initial content as "saved"
        this._lastSaved.set(recordingId, {
            content: element.value || element.textContent,
            timestamp: Date.now(),
        });

        // Input handler with debounce
        const handleInput = () => {
            const content = element.value || element.textContent;
            this._schedulesSave(recordingId, content);
        };

        element.addEventListener('input', handleInput);

        // Save on blur (immediate)
        element.addEventListener('blur', () => {
            if (this.hasPendingChanges(recordingId)) {
                this._cancelScheduled(recordingId);
                this.saveNow(recordingId);
            }
        });

        // Prevent navigation with unsaved changes
        window.addEventListener('beforeunload', (e) => {
            if (this.hasPendingChanges()) {
                e.preventDefault();
                e.returnValue = 'You have unsaved changes.';
            }
        });

        console.log('[AutoSave] Tracking element for recording:', recordingId);
    },

    /**
     * Schedule a debounced save
     */
    _schedulesSave(recordingId, content) {
        // Cancel existing timer
        this._cancelScheduled(recordingId);

        const pending = {
            content,
            lastChange: Date.now(),
            timer: null,
            maxTimer: null,
        };

        // Set debounce timer
        pending.timer = setTimeout(() => {
            this._doSave(recordingId);
        }, this.config.debounceMs);

        // Set max wait timer (save even if still typing)
        const existing = this._pendingChanges.get(recordingId);
        if (!existing) {
            pending.maxTimer = setTimeout(() => {
                if (this._pendingChanges.has(recordingId)) {
                    this._cancelScheduled(recordingId);
                    this._doSave(recordingId);
                }
            }, this.config.maxWaitMs);
        } else {
            pending.maxTimer = existing.maxTimer;
        }

        this._pendingChanges.set(recordingId, pending);
        this._setStatus('pending');
    },

    /**
     * Cancel scheduled save
     */
    _cancelScheduled(recordingId) {
        const pending = this._pendingChanges.get(recordingId);
        if (pending) {
            clearTimeout(pending.timer);
            // Don't clear maxTimer - it should still fire
        }
    },

    /**
     * Execute save immediately
     */
    async saveNow(recordingId = null) {
        if (recordingId) {
            await this._doSave(recordingId);
        } else {
            // Save all pending
            const recordingIds = Array.from(this._pendingChanges.keys());
            for (const id of recordingIds) {
                await this._doSave(id);
            }
        }
    },

    /**
     * Perform the actual save
     */
    async _doSave(recordingId) {
        const pending = this._pendingChanges.get(recordingId);
        if (!pending) {
            return;
        }

        // Clear from pending
        clearTimeout(pending.timer);
        clearTimeout(pending.maxTimer);
        this._pendingChanges.delete(recordingId);

        // Check if content actually changed
        const lastSaved = this._lastSaved.get(recordingId);
        if (lastSaved && lastSaved.content === pending.content) {
            console.log('[AutoSave] No changes to save for:', recordingId);
            this._updateStatusIfIdle();
            return;
        }

        this._setStatus('saving');

        try {
            await this._sendSave(recordingId, pending.content);

            // Update last saved
            this._lastSaved.set(recordingId, {
                content: pending.content,
                timestamp: Date.now(),
            });

            this._setStatus('saved');

            // Reset status after delay
            setTimeout(() => this._updateStatusIfIdle(), 2000);

        } catch (error) {
            console.error('[AutoSave] Save failed:', error);

            // Queue for retry if offline
            if (!navigator.onLine) {
                this._queueForRetry(recordingId, pending.content);
            }

            this._setStatus('error');
        }
    },

    /**
     * Send save request to server
     */
    async _sendSave(recordingId, content, attempt = 1) {
        const response = await fetch(this.config.saveEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recording_id: recordingId,
                content: content,
                timestamp: Date.now(),
            }),
        });

        if (!response.ok) {
            if (attempt < this.config.retryAttempts && response.status >= 500) {
                // Retry with exponential backoff
                await this._sleep(this.config.retryDelayMs * attempt);
                return this._sendSave(recordingId, content, attempt + 1);
            }
            throw new Error(`Save failed: ${response.status}`);
        }

        return response.json();
    },

    /**
     * Queue save for when back online
     */
    _queueForRetry(recordingId, content) {
        // Remove existing entry for same recording
        this._saveQueue = this._saveQueue.filter(q => q.recordingId !== recordingId);

        this._saveQueue.push({
            recordingId,
            content,
            queuedAt: Date.now(),
        });

        console.log('[AutoSave] Queued for retry:', recordingId);
    },

    /**
     * Process queued saves
     */
    async _processQueue() {
        if (this._saveQueue.length === 0) {
            return;
        }

        console.log('[AutoSave] Processing queue:', this._saveQueue.length, 'items');

        while (this._saveQueue.length > 0) {
            const item = this._saveQueue[0];

            try {
                await this._sendSave(item.recordingId, item.content);
                this._saveQueue.shift(); // Remove successful

                this._lastSaved.set(item.recordingId, {
                    content: item.content,
                    timestamp: Date.now(),
                });

            } catch (error) {
                console.error('[AutoSave] Queue item failed:', error);
                break; // Stop processing on failure
            }
        }

        this._updateStatusIfIdle();
    },

    /**
     * Check if there are pending changes
     */
    hasPendingChanges(recordingId = null) {
        if (recordingId) {
            return this._pendingChanges.has(recordingId);
        }
        return this._pendingChanges.size > 0 || this._saveQueue.length > 0;
    },

    /**
     * Get current status
     */
    getStatus() {
        return this._status;
    },

    /**
     * Subscribe to status changes
     */
    onStatusChange(callback) {
        this._listeners.push(callback);
        return () => {
            this._listeners = this._listeners.filter(cb => cb !== callback);
        };
    },

    /**
     * Set status and notify listeners
     */
    _setStatus(status) {
        if (this._status === status) {
            return;
        }

        this._status = status;

        for (const listener of this._listeners) {
            try {
                listener(status);
            } catch (error) {
                console.error('[AutoSave] Listener error:', error);
            }
        }

        // Update visual indicator
        this._updateIndicator(status);
    },

    /**
     * Update status if no pending operations
     */
    _updateStatusIfIdle() {
        if (!this.hasPendingChanges() && this._status !== 'idle') {
            this._setStatus('idle');
        }
    },

    /**
     * Update visual status indicator
     */
    _updateIndicator(status) {
        let indicator = document.getElementById('autosave-indicator');

        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'autosave-indicator';
            indicator.setAttribute('role', 'status');
            indicator.setAttribute('aria-live', 'polite');
            indicator.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 9999;
                transition: opacity 0.3s ease, background 0.3s ease;
            `;
            document.body.appendChild(indicator);
        }

        const statusConfig = {
            idle: { text: '', bg: 'transparent', opacity: '0' },
            pending: { text: 'Unsaved changes...', bg: '#fff3cd', opacity: '1' },
            saving: { text: 'Saving...', bg: '#cce5ff', opacity: '1' },
            saved: { text: 'Saved', bg: '#d4edda', opacity: '1' },
            error: { text: 'Save failed', bg: '#f8d7da', opacity: '1' },
        };

        const config = statusConfig[status] || statusConfig.idle;
        indicator.textContent = config.text;
        indicator.style.background = config.bg;
        indicator.style.opacity = config.opacity;
    },

    /**
     * Utility: sleep for ms
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    },

    /**
     * Get save statistics
     */
    getStats() {
        return {
            pendingCount: this._pendingChanges.size,
            queuedCount: this._saveQueue.length,
            savedCount: this._lastSaved.size,
            status: this._status,
        };
    },

    /**
     * Discard pending changes
     */
    discard(recordingId = null) {
        if (recordingId) {
            this._cancelScheduled(recordingId);
            this._pendingChanges.delete(recordingId);
        } else {
            for (const id of this._pendingChanges.keys()) {
                this._cancelScheduled(id);
            }
            this._pendingChanges.clear();
        }

        this._updateStatusIfIdle();
    },
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AutoSave;
}
