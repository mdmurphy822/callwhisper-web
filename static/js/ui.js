/**
 * UI Management for CallWhisper
 * Handles DOM updates and user interactions
 *
 * Based on LibV2 accessibility patterns:
 * - Accessible toast notifications (replaces alert())
 * - ARIA live region announcements
 * - Focus management
 */

const UI = {
    // DOM Element references
    elements: {
        statusDot: null,
        statusText: null,
        timer: null,
        deviceSelect: null,
        refreshDevices: null,
        ticketId: null,
        progressSection: null,
        progressFill: null,
        progressText: null,
        btnStart: null,
        btnStop: null,
        btnOpenFolder: null,
        btnReset: null,
        recordingsList: null,
        connectionDot: null,
        connectionText: null,
        notificationContainer: null,
        statusAnnouncer: null,
        deviceError: null,
        ticketError: null,
        // Transcript panel elements
        transcriptPanel: null,
        transcriptInfo: null,
        transcriptContent: null,
        transcriptClose: null,
        transcriptDownload: null,
    },

    // Toast notification counter for unique IDs
    toastCounter: 0,

    /**
     * Initialize UI
     */
    init() {
        // Get DOM references
        this.elements.statusDot = document.querySelector('.status-dot');
        this.elements.statusText = document.getElementById('status-text');
        this.elements.timer = document.getElementById('timer');
        this.elements.deviceSelect = document.getElementById('device-select');
        this.elements.refreshDevices = document.getElementById('refresh-devices');
        this.elements.ticketId = document.getElementById('ticket-id');
        this.elements.progressSection = document.getElementById('progress-section');
        this.elements.progressFill = document.getElementById('progress-fill');
        this.elements.progressText = document.getElementById('progress-text');
        this.elements.btnStart = document.getElementById('btn-start');
        this.elements.btnStop = document.getElementById('btn-stop');
        this.elements.btnOpenFolder = document.getElementById('btn-open-folder');
        this.elements.btnReset = document.getElementById('btn-reset');
        this.elements.recordingsList = document.getElementById('recordings-list');
        this.elements.connectionDot = document.querySelector('.connection-dot');
        this.elements.connectionText = document.getElementById('connection-text');

        // Accessibility elements
        this.elements.notificationContainer = document.getElementById('notification-container');
        this.elements.statusAnnouncer = document.getElementById('status-announcer');
        this.elements.deviceError = document.getElementById('device-error');
        this.elements.ticketError = document.getElementById('ticket-error');

        // Transcript panel elements
        this.elements.transcriptPanel = document.getElementById('transcript-panel');
        this.elements.transcriptInfo = document.getElementById('transcript-info');
        this.elements.transcriptContent = document.getElementById('transcript-content');
        this.elements.transcriptClose = document.getElementById('transcript-close');
        this.elements.transcriptDownload = document.getElementById('transcript-download');
        this.elements.transcriptCopy = document.getElementById('transcript-copy');

        // Search elements
        this.elements.recordingsSearch = document.getElementById('recordings-search');

        // Queue panel elements
        this.elements.queueSection = document.getElementById('queue-section');
        this.elements.queueCount = document.getElementById('queue-count');
        this.elements.queueList = document.getElementById('queue-list');
        this.elements.statQueued = document.getElementById('stat-queued');
        this.elements.statProcessing = document.getElementById('stat-processing');
        this.elements.statCompleted = document.getElementById('stat-completed');
        this.elements.statFailed = document.getElementById('stat-failed');
        this.elements.queueClearHistory = document.getElementById('queue-clear-history');
    },

    /**
     * Update status display
     */
    setStatus(state) {
        const { statusDot, statusText } = this.elements;

        // Remove all state classes
        statusDot.className = 'status-dot';
        statusDot.classList.add(state);

        // Update text
        const stateLabels = {
            idle: 'Idle',
            recording: 'Recording',
            processing: 'Processing',
            done: 'Done',
            error: 'Error',
        };
        statusText.textContent = stateLabels[state] || state;
    },

    /**
     * Update timer display
     */
    setTimer(formatted) {
        this.elements.timer.textContent = formatted;
    },

    /**
     * Reset timer
     */
    resetTimer() {
        this.elements.timer.textContent = '00:00';
    },

    /**
     * Update device dropdown
     */
    setDevices(devices) {
        const select = this.elements.deviceSelect;
        select.innerHTML = '';

        if (devices.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No devices found';
            select.appendChild(option);
            select.disabled = true;
            return;
        }

        // Filter to only safe devices
        const safeDevices = devices.filter(d => d.safe);

        if (safeDevices.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No safe devices found';
            select.appendChild(option);
            select.disabled = true;
            return;
        }

        // Add safe devices
        for (const device of safeDevices) {
            const option = document.createElement('option');
            option.value = device.name;
            option.textContent = device.name;
            select.appendChild(option);
        }

        select.disabled = false;
    },

    /**
     * Get selected device
     */
    getSelectedDevice() {
        return this.elements.deviceSelect.value;
    },

    /**
     * Get ticket ID
     */
    getTicketId() {
        return this.elements.ticketId.value.trim();
    },

    /**
     * Show/hide progress bar
     */
    showProgress(show) {
        this.elements.progressSection.style.display = show ? 'block' : 'none';
    },

    /**
     * Update progress
     */
    setProgress(percent, text) {
        this.elements.progressFill.style.width = `${percent}%`;
        if (text) {
            this.elements.progressText.textContent = text;
        }
    },

    /**
     * Update partial transcript preview during transcription
     * Shows real-time transcript text as it becomes available
     *
     * @param {string} text - The partial transcript text
     * @param {boolean} isFinal - Whether this is the final transcript
     */
    updatePartialTranscript(text, isFinal = false) {
        const section = document.getElementById('partial-transcript-section');
        const textEl = document.getElementById('partial-transcript-text');

        if (!section || !textEl) return;

        if (!text || text.trim() === '') {
            // Hide if no text
            section.style.display = 'none';
            return;
        }

        // Show the section
        section.style.display = 'block';

        // Truncate to last 500 chars for preview
        const preview = text.length > 500 ? '...' + text.slice(-500) : text;

        if (isFinal) {
            // Final transcript - remove cursor
            textEl.innerHTML = this.escapeHtml(preview);
        } else {
            // Still transcribing - show with blinking cursor
            textEl.innerHTML = this.escapeHtml(preview) + '<span class="blinking-cursor" aria-hidden="true"></span>';
        }

        // Auto-scroll to bottom
        textEl.scrollTop = textEl.scrollHeight;
    },

    /**
     * Hide partial transcript preview
     */
    hidePartialTranscript() {
        const section = document.getElementById('partial-transcript-section');
        if (section) {
            section.style.display = 'none';
        }
    },

    /**
     * Update button states based on app state
     * Also checks SetupWizard._recordingBlocked to disable recording when no virtual audio
     */
    setButtonStates(state) {
        const { btnStart, btnStop, deviceSelect, ticketId } = this.elements;

        // Check if recording is blocked due to missing virtual audio
        const recordingBlocked = typeof SetupWizard !== 'undefined' && SetupWizard._recordingBlocked;

        switch (state) {
            case 'idle':
                // Block start button if no virtual audio
                if (recordingBlocked) {
                    btnStart.disabled = true;
                    btnStart.title = 'Virtual audio required - install VB-Cable to enable recording';
                } else {
                    btnStart.disabled = !deviceSelect.value;
                    btnStart.title = '';
                }
                btnStop.disabled = true;
                deviceSelect.disabled = false;
                ticketId.disabled = false;
                break;

            case 'recording':
                btnStart.disabled = true;
                btnStart.title = '';
                btnStop.disabled = false;
                deviceSelect.disabled = true;
                ticketId.disabled = true;
                break;

            case 'processing':
                btnStart.disabled = true;
                btnStart.title = '';
                btnStop.disabled = true;
                deviceSelect.disabled = true;
                ticketId.disabled = true;
                break;

            case 'done':
            case 'error':
                // Block start button if no virtual audio
                if (recordingBlocked) {
                    btnStart.disabled = true;
                    btnStart.title = 'Virtual audio required - install VB-Cable to enable recording';
                } else {
                    btnStart.disabled = !deviceSelect.value;
                    btnStart.title = '';
                }
                btnStop.disabled = true;
                deviceSelect.disabled = false;
                ticketId.disabled = false;
                break;
        }
    },

    /**
     * Add recording to list
     * @param {Object} recording - Recording data
     * @param {string} searchQuery - Optional search query for highlighting
     */
    addRecording(recording, searchQuery = '') {
        const list = this.elements.recordingsList;

        // Remove empty message if present
        const emptyMsg = list.querySelector('.empty-message');
        if (emptyMsg) {
            emptyMsg.remove();
        }

        // Create recording item
        const item = document.createElement('div');
        item.className = 'recording-item';
        item.dataset.id = recording.id;

        const duration = this.formatDuration(recording.duration_seconds);

        // Apply search highlighting if query provided
        const displayId = searchQuery
            ? this.highlightSearch(recording.id, searchQuery)
            : this.escapeHtml(recording.id);
        const displayTicket = searchQuery
            ? this.highlightSearch(recording.ticket_id || 'No ticket', searchQuery)
            : this.escapeHtml(recording.ticket_id || 'No ticket');

        item.innerHTML = `
            <div class="recording-info">
                <div class="recording-id">${displayId}</div>
                <div class="recording-meta">${duration} - ${displayTicket}</div>
            </div>
            <div class="recording-actions">
                <button type="button"
                        class="btn-view"
                        data-recording-id="${recording.id}"
                        aria-label="View transcript for ${recording.id}">
                    View
                </button>
                <div class="export-dropdown">
                    <button type="button"
                            class="btn-export"
                            data-recording-id="${recording.id}"
                            aria-label="Export options for ${recording.id}"
                            aria-haspopup="true"
                            aria-expanded="false">
                        Export
                    </button>
                    <div class="export-menu" role="menu" style="display: none;">
                        <button type="button" data-format="json" role="menuitem">JSON</button>
                        <button type="button" data-format="vtt" role="menuitem">VTT (Subtitles)</button>
                        <button type="button" data-format="csv" role="menuitem">CSV</button>
                        <button type="button" data-format="pdf" role="menuitem">PDF</button>
                        <button type="button" data-format="docx" role="menuitem">Word (DOCX)</button>
                    </div>
                </div>
                <a href="/api/recordings/${recording.id}/download"
                   class="btn-download"
                   download="${recording.id}.vtb">
                    VTB
                </a>
            </div>
        `;

        // Add to top of list
        list.insertBefore(item, list.firstChild);
    },

    /**
     * Update recordings list from array
     * @param {Array} recordings - List of recording objects
     * @param {string} searchQuery - Optional search query for highlighting
     */
    setRecordings(recordings, searchQuery = '') {
        const list = this.elements.recordingsList;
        list.innerHTML = '';

        if (recordings.length === 0) {
            const message = searchQuery
                ? '<p class="empty-message">No recordings match your search</p>'
                : '<p class="empty-message">No recordings yet</p>';
            list.innerHTML = message;
            return;
        }

        for (const recording of recordings) {
            this.addRecording(recording, searchQuery);
        }
    },

    /**
     * Highlight search query in text
     * @param {string} text - Text to highlight
     * @param {string} query - Search query
     * @returns {string} - HTML with highlighted matches
     */
    highlightSearch(text, query) {
        if (!query || !text) return this.escapeHtml(text);

        const escaped = this.escapeHtml(text);
        const queryLower = query.toLowerCase();
        const textLower = text.toLowerCase();

        let result = '';
        let lastIndex = 0;

        let index = textLower.indexOf(queryLower);
        while (index !== -1) {
            // Add text before match
            result += escaped.slice(lastIndex, index);
            // Add highlighted match
            const match = escaped.slice(index, index + query.length);
            result += `<span class="search-highlight">${match}</span>`;
            lastIndex = index + query.length;
            index = textLower.indexOf(queryLower, lastIndex);
        }

        // Add remaining text
        result += escaped.slice(lastIndex);
        return result;
    },

    /**
     * Format duration in seconds to MM:SS
     */
    formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    },

    /**
     * Update connection status
     */
    setConnectionStatus(connected) {
        const { connectionDot, connectionText } = this.elements;

        connectionDot.className = 'connection-dot';
        connectionDot.classList.add(connected ? 'connected' : 'disconnected');
        connectionText.textContent = connected ? 'Connected' : 'Disconnected';
    },

    /**
     * Show error message (accessible toast notification)
     * Based on LibV2 accessibility patterns - replaces alert()
     */
    showError(message) {
        this.showNotification(message, 'error');
    },

    /**
     * Show success message
     */
    showSuccess(message) {
        this.showNotification(message, 'success');
    },

    /**
     * Show warning message
     */
    showWarning(message) {
        this.showNotification(message, 'warning');
    },

    /**
     * Show info message
     */
    showInfo(message) {
        this.showNotification(message, 'info');
    },

    /**
     * Show accessible toast notification
     * @param {string} message - The message to display
     * @param {string} type - 'error', 'success', 'warning', or 'info'
     * @param {number} duration - Auto-dismiss duration in ms (0 = manual dismiss)
     */
    showNotification(message, type = 'info', duration = 5000) {
        const container = this.elements.notificationContainer;
        if (!container) return;

        const toastId = `toast-${++this.toastCounter}`;

        // Icon mapping
        const icons = {
            error: '\u26D4',    // No entry
            success: '\u2713',  // Check mark
            warning: '\u26A0',  // Warning
            info: '\u2139'      // Info
        };

        // Create toast element
        const toast = document.createElement('div');
        toast.id = toastId;
        toast.className = `toast toast-${type}`;
        toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
        toast.setAttribute('aria-live', type === 'error' ? 'assertive' : 'polite');

        toast.innerHTML = `
            <span class="toast-icon" aria-hidden="true">${icons[type] || icons.info}</span>
            <span class="toast-message">${this.escapeHtml(message)}</span>
            <button class="toast-close"
                    type="button"
                    aria-label="Dismiss notification"
                    onclick="UI.dismissNotification('${toastId}')">
                <span aria-hidden="true">&times;</span>
            </button>
        `;

        container.appendChild(toast);

        // Focus the close button for keyboard users on error
        if (type === 'error') {
            const closeBtn = toast.querySelector('.toast-close');
            if (closeBtn) {
                closeBtn.focus();
            }
        }

        // Auto-dismiss after duration (unless duration is 0)
        if (duration > 0) {
            setTimeout(() => {
                this.dismissNotification(toastId);
            }, duration);
        }

        return toastId;
    },

    /**
     * Dismiss a notification
     * @param {string} toastId - The ID of the toast to dismiss
     */
    dismissNotification(toastId) {
        const toast = document.getElementById(toastId);
        if (!toast) return;

        // Add removing animation class
        toast.classList.add('removing');

        // Remove after animation completes
        setTimeout(() => {
            toast.remove();
        }, 300);
    },

    /**
     * Dismiss all visible notifications
     */
    dismissAllNotifications() {
        const container = this.elements.notificationContainer;
        if (!container) return;

        const toasts = container.querySelectorAll('.notification-toast');
        toasts.forEach(toast => {
            this.dismissNotification(toast.id);
        });
    },

    /**
     * Announce message to screen readers via ARIA live region
     * @param {string} message - Message to announce
     */
    announceStatus(message) {
        const announcer = this.elements.statusAnnouncer;
        if (!announcer) return;

        // Clear and set to trigger announcement
        announcer.textContent = '';
        setTimeout(() => {
            announcer.textContent = message;
        }, 100);
    },

    /**
     * Escape HTML to prevent XSS
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Set field validation error
     * @param {string} fieldId - 'device' or 'ticket'
     * @param {string} message - Error message (empty to clear)
     */
    setFieldError(fieldId, message) {
        const errorEl = fieldId === 'device'
            ? this.elements.deviceError
            : this.elements.ticketError;

        const inputEl = fieldId === 'device'
            ? this.elements.deviceSelect
            : this.elements.ticketId;

        if (!errorEl || !inputEl) return;

        errorEl.textContent = message;
        inputEl.setAttribute('aria-invalid', message ? 'true' : 'false');
    },

    /**
     * Clear all field errors
     */
    clearFieldErrors() {
        this.setFieldError('device', '');
        this.setFieldError('ticket', '');
    },

    /**
     * Disable all controls during loading
     */
    setLoading(loading) {
        const { btnStart, btnStop, deviceSelect, ticketId, refreshDevices } = this.elements;

        if (loading) {
            btnStart.disabled = true;
            btnStop.disabled = true;
            deviceSelect.disabled = true;
            ticketId.disabled = true;
            refreshDevices.disabled = true;
        } else {
            refreshDevices.disabled = false;
            // Other states will be set by setButtonStates
        }
    },

    /**
     * Show transcript panel with content
     * @param {string} id - Recording ID
     * @param {string} text - Transcript text content
     * @param {Object} metadata - Recording metadata (duration, ticket_id, word_count)
     */
    showTranscript(id, text, metadata = {}) {
        const { transcriptPanel, transcriptInfo, transcriptContent, transcriptDownload } = this.elements;
        if (!transcriptPanel) return;

        // Build info line
        const duration = metadata.duration_seconds
            ? this.formatDuration(metadata.duration_seconds)
            : '';
        const ticket = metadata.ticket_id || 'No ticket';
        const words = metadata.word_count || 0;

        transcriptInfo.innerHTML = `
            <strong>${id}</strong>
            <span class="transcript-meta">${duration} &bull; ${ticket} &bull; ${words} words</span>
        `;

        // Set transcript text
        transcriptContent.textContent = text || '[No transcript available]';

        // Update download link
        transcriptDownload.href = `/api/recordings/${id}/download`;
        transcriptDownload.download = `${id}.vtb`;

        // Show panel with animation
        transcriptPanel.style.display = 'flex';

        // Mark active recording in list
        const items = document.querySelectorAll('.recording-item');
        items.forEach(item => item.classList.remove('active'));
        const activeItem = document.querySelector(`.recording-item[data-id="${id}"]`);
        if (activeItem) {
            activeItem.classList.add('active');
        }

        // Announce to screen readers
        this.announceStatus(`Showing transcript for ${id}`);
    },

    /**
     * Close transcript panel
     */
    closeTranscript() {
        const { transcriptPanel } = this.elements;
        if (!transcriptPanel) return;

        transcriptPanel.style.display = 'none';

        // Remove active state from recordings
        const items = document.querySelectorAll('.recording-item');
        items.forEach(item => item.classList.remove('active'));

        this.announceStatus('Transcript panel closed');
    },

    /**
     * Copy transcript text to clipboard
     */
    async copyTranscript() {
        const { transcriptContent, transcriptCopy } = this.elements;
        if (!transcriptContent || !transcriptCopy) return;

        const text = transcriptContent.textContent;
        if (!text || text === '[No transcript available]') {
            this.showWarning('No transcript to copy');
            return;
        }

        try {
            await navigator.clipboard.writeText(text);

            // Visual feedback
            const originalText = transcriptCopy.textContent;
            transcriptCopy.textContent = 'Copied!';
            transcriptCopy.disabled = true;

            setTimeout(() => {
                transcriptCopy.textContent = originalText;
                transcriptCopy.disabled = false;
            }, 2000);

            this.showSuccess('Transcript copied to clipboard');
            this.announceStatus('Transcript copied to clipboard');
        } catch (err) {
            // Fallback for older browsers
            this.fallbackCopy(text);
        }
    },

    /**
     * Fallback copy method for browsers without clipboard API
     */
    fallbackCopy(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        document.body.appendChild(textarea);
        textarea.select();

        try {
            document.execCommand('copy');
            this.showSuccess('Transcript copied to clipboard');
        } catch (err) {
            this.showError('Failed to copy transcript');
        }

        document.body.removeChild(textarea);
    },

    /**
     * Filter recordings list by search query
     * Searches ID, ticket ID, and transcript preview
     */
    filterRecordings(query) {
        const { recordingsList } = this.elements;
        if (!recordingsList) return;

        const items = recordingsList.querySelectorAll('.recording-item');
        const lowerQuery = query.toLowerCase().trim();

        if (!lowerQuery) {
            // Show all if no query
            items.forEach(item => {
                item.style.display = '';
                item.classList.remove('search-hidden');
            });
            return;
        }

        let visibleCount = 0;
        items.forEach(item => {
            const id = item.dataset.id || '';
            const info = item.querySelector('.recording-info');
            const text = info ? info.textContent.toLowerCase() : '';

            // Also search transcript preview if available
            const preview = item.dataset.transcriptPreview || '';

            const matches = id.toLowerCase().includes(lowerQuery) ||
                           text.includes(lowerQuery) ||
                           preview.toLowerCase().includes(lowerQuery);

            if (matches) {
                item.style.display = '';
                item.classList.remove('search-hidden');
                visibleCount++;
            } else {
                item.style.display = 'none';
                item.classList.add('search-hidden');
            }
        });

        // Announce results to screen readers
        this.announceStatus(`${visibleCount} recording${visibleCount !== 1 ? 's' : ''} found`);
    },

    /**
     * Clear search and show all recordings
     */
    clearSearch() {
        const { recordingsSearch } = this.elements;
        if (recordingsSearch) {
            recordingsSearch.value = '';
        }
        this.filterRecordings('');
    },

    // ========================================================================
    // Queue UI Methods
    // ========================================================================

    /**
     * Show or hide the queue panel
     * @param {boolean} show - Whether to show the panel
     */
    showQueue(show) {
        const { queueSection } = this.elements;
        if (queueSection) {
            queueSection.style.display = show ? 'block' : 'none';
        }
    },

    /**
     * Update queue status display
     * @param {Object} status - Queue status from API
     */
    updateQueueStatus(status) {
        const {
            queueSection, queueCount, queueList,
            statQueued, statProcessing, statCompleted, statFailed
        } = this.elements;

        if (!status || !status.counts) return;

        // Update counts
        if (statQueued) statQueued.textContent = status.counts.queued || 0;
        if (statProcessing) statProcessing.textContent = status.counts.processing || 0;
        if (statCompleted) statCompleted.textContent = status.counts.completed || 0;
        if (statFailed) statFailed.textContent = status.counts.failed || 0;
        if (queueCount) queueCount.textContent = status.counts.queued || 0;

        // Show/hide section based on activity
        const hasActivity = (status.counts.queued > 0) ||
                           (status.counts.processing > 0) ||
                           (status.counts.completed > 0 && status.counts.completed < 10) ||
                           (status.counts.failed > 0);
        this.showQueue(hasActivity);

        // Render queue list
        if (queueList) {
            queueList.innerHTML = '';

            // Processing job (show first)
            if (status.processing) {
                queueList.appendChild(this.createQueueItem(status.processing, 'processing'));
            }

            // Queued jobs
            if (status.queued) {
                for (const job of status.queued) {
                    queueList.appendChild(this.createQueueItem(job, 'queued'));
                }
            }

            // Recent failed jobs
            if (status.failed && status.failed.length > 0) {
                for (const job of status.failed.slice(-3)) {
                    queueList.appendChild(this.createQueueItem(job, 'failed'));
                }
            }
        }
    },

    /**
     * Create a queue item element
     * @param {Object} job - Job data
     * @param {string} type - 'queued', 'processing', or 'failed'
     * @returns {HTMLElement}
     */
    createQueueItem(job, type) {
        const item = document.createElement('div');
        item.className = `queue-item queue-item-${type}`;
        item.dataset.jobId = job.job_id;

        let statusText = '';
        let progressHtml = '';

        switch (type) {
            case 'processing':
                statusText = `Processing... ${job.progress || 0}%`;
                progressHtml = `
                    <div class="queue-item-progress">
                        <div class="queue-item-progress-fill" style="width: ${job.progress || 0}%"></div>
                    </div>
                `;
                break;
            case 'queued':
                statusText = 'Queued';
                break;
            case 'failed':
                statusText = `Failed: ${job.error_message || 'Unknown error'}`;
                break;
        }

        item.innerHTML = `
            <div class="queue-item-info">
                <span class="queue-item-name">${this.escapeHtml(job.original_filename || job.job_id)}</span>
                <span class="queue-item-status queue-item-status-${type}">${statusText}</span>
                ${progressHtml}
            </div>
            ${type === 'queued' ? `
                <button class="queue-item-cancel"
                        data-job-id="${job.job_id}"
                        type="button"
                        aria-label="Cancel job ${job.job_id}">
                    &times;
                </button>
            ` : ''}
        `;

        return item;
    },

    /**
     * Handle batch upload - upload multiple files
     * @param {FileList} files - Files to upload
     * @returns {Promise<Object>} Upload result
     */
    async uploadBatch(files) {
        if (!files || files.length === 0) {
            this.showWarning('No files selected');
            return null;
        }

        // Show uploading notification
        const uploadingToast = this.showNotification(
            `Uploading ${files.length} file${files.length > 1 ? 's' : ''}...`,
            'info',
            0 // Don't auto-dismiss
        );

        try {
            const formData = new FormData();

            for (const file of files) {
                formData.append('files', file);
            }

            // Add ticket prefix if set
            const ticketId = this.getTicketId();
            if (ticketId) {
                formData.append('ticket_prefix', ticketId);
            }

            const response = await fetch('/api/recordings/batch-upload', {
                method: 'POST',
                body: formData
            });

            // Dismiss uploading notification
            this.dismissNotification(uploadingToast);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const result = await response.json();

            this.showSuccess(`${result.jobs_queued} file${result.jobs_queued > 1 ? 's' : ''} queued for transcription`);
            this.announceStatus(`${result.jobs_queued} files added to queue`);

            // Refresh queue status
            await this.refreshQueueStatus();

            return result;

        } catch (err) {
            this.dismissNotification(uploadingToast);
            this.showError(`Upload failed: ${err.message}`);
            throw err;
        }
    },

    /**
     * Refresh queue status from API
     */
    async refreshQueueStatus() {
        try {
            const response = await fetch('/api/queue/status');
            if (response.ok) {
                const status = await response.json();
                this.updateQueueStatus(status);
            }
        } catch (err) {
            console.error('Failed to refresh queue status:', err);
        }
    },

    /**
     * Cancel a queued job
     * @param {string} jobId - Job ID to cancel
     */
    async cancelQueuedJob(jobId) {
        try {
            const response = await fetch(`/api/queue/jobs/${jobId}`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Cancel failed');
            }

            this.showSuccess('Job cancelled');
            await this.refreshQueueStatus();

        } catch (err) {
            this.showError(`Failed to cancel job: ${err.message}`);
        }
    },

    /**
     * Clear queue history (completed and failed)
     */
    async clearQueueHistory() {
        try {
            const response = await fetch('/api/queue/clear-history', {
                method: 'POST'
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Clear failed');
            }

            this.showSuccess('Queue history cleared');
            await this.refreshQueueStatus();

        } catch (err) {
            this.showError(`Failed to clear history: ${err.message}`);
        }
    },

    // ========================================================================
    // Export Dropdown Methods
    // ========================================================================

    /**
     * Set up export dropdown event handlers
     * Uses event delegation for dynamically added dropdowns
     */
    setupExportDropdowns() {
        // Use event delegation on the recordings list
        document.addEventListener('click', (e) => {
            // Toggle dropdown when clicking export button
            if (e.target.classList.contains('btn-export')) {
                e.stopPropagation();
                const dropdown = e.target.closest('.export-dropdown');
                const menu = dropdown?.querySelector('.export-menu');

                if (menu) {
                    // Close all other dropdowns first
                    document.querySelectorAll('.export-menu').forEach(m => {
                        if (m !== menu) {
                            m.style.display = 'none';
                            m.closest('.export-dropdown')
                                ?.querySelector('.btn-export')
                                ?.setAttribute('aria-expanded', 'false');
                        }
                    });

                    // Toggle this dropdown
                    const isOpen = menu.style.display === 'block';
                    menu.style.display = isOpen ? 'none' : 'block';
                    e.target.setAttribute('aria-expanded', !isOpen);
                }
                return;
            }

            // Handle format selection
            const menuButton = e.target.closest('.export-menu button');
            if (menuButton) {
                e.stopPropagation();
                const format = menuButton.dataset.format;
                const dropdown = menuButton.closest('.export-dropdown');
                const recordingId = dropdown?.querySelector('.btn-export')?.dataset.recordingId;

                if (format && recordingId) {
                    this.exportRecording(recordingId, format);
                }

                // Close the menu
                const menu = menuButton.closest('.export-menu');
                if (menu) {
                    menu.style.display = 'none';
                    dropdown?.querySelector('.btn-export')?.setAttribute('aria-expanded', 'false');
                }
                return;
            }

            // Close all dropdowns when clicking outside
            document.querySelectorAll('.export-menu').forEach(menu => {
                menu.style.display = 'none';
                menu.closest('.export-dropdown')
                    ?.querySelector('.btn-export')
                    ?.setAttribute('aria-expanded', 'false');
            });
        });

        // Handle keyboard navigation for accessibility
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.export-menu').forEach(menu => {
                    menu.style.display = 'none';
                    menu.closest('.export-dropdown')
                        ?.querySelector('.btn-export')
                        ?.setAttribute('aria-expanded', 'false');
                });
            }
        });
    },

    /**
     * Export a recording in the specified format
     * @param {string} recordingId - Recording ID
     * @param {string} format - Export format (json, vtt, csv, pdf, docx)
     */
    async exportRecording(recordingId, format) {
        try {
            // Show loading notification
            this.showInfo(`Exporting ${format.toUpperCase()}...`);

            // Trigger download via new window/tab
            const url = `/api/recordings/${recordingId}/export/${format}`;
            window.open(url, '_blank');

            // Log for debugging
            console.log(`[UI] Export triggered: ${recordingId} -> ${format}`);

        } catch (err) {
            console.error('[UI] Export failed:', err);
            this.showError(`Export failed: ${err.message}`);
        }
    }
};

/**
 * Theme Manager for CallWhisper
 * Handles light/dark mode switching with system preference detection
 */
const ThemeManager = {
    STORAGE_KEY: 'callwhisper-theme',

    /**
     * Initialize theme manager
     * - Detects system preference
     * - Loads saved preference
     * - Sets up event listeners
     */
    init() {
        // Check system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');

        // Load saved preference or use system default
        const saved = localStorage.getItem(this.STORAGE_KEY);
        const theme = saved || (prefersDark.matches ? 'dark' : 'light');

        this.setTheme(theme, false); // Don't announce on initial load

        // Listen for system preference changes (only if no saved preference)
        prefersDark.addEventListener('change', (e) => {
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.setTheme(e.matches ? 'dark' : 'light', true);
            }
        });

        // Set up toggle button handler
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggle());
        }

        console.log('[ThemeManager] Initialized with theme:', theme);
    },

    /**
     * Set theme
     * @param {string} theme - 'light' or 'dark'
     * @param {boolean} announce - Whether to announce change to screen readers
     */
    setTheme(theme, announce = true) {
        document.documentElement.setAttribute('data-theme', theme);

        // Update toggle button aria state
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            toggleBtn.setAttribute('aria-label',
                theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'
            );
        }

        // Announce to screen readers
        if (announce && typeof UI !== 'undefined' && UI.announceStatus) {
            UI.announceStatus(`Switched to ${theme} mode`);
        }
    },

    /**
     * Toggle between light and dark themes
     */
    toggle() {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';

        this.setTheme(next, true);
        localStorage.setItem(this.STORAGE_KEY, next);

        console.log('[ThemeManager] Theme toggled to:', next);
    },

    /**
     * Get current theme
     * @returns {string} Current theme ('light' or 'dark')
     */
    getTheme() {
        return document.documentElement.getAttribute('data-theme') || 'dark';
    }
};

/**
 * Transcript Editor for CallWhisper
 * Handles editing and saving transcript text
 */
const TranscriptEditor = {
    _originalText: '',
    _currentRecordingId: null,
    _isEditing: false,
    _hasChanges: false,

    /**
     * Initialize transcript editor
     * Sets up event handlers for edit/save/cancel
     */
    init() {
        const editBtn = document.getElementById('transcript-edit-btn');
        const saveBtn = document.getElementById('transcript-save');
        const cancelBtn = document.getElementById('transcript-cancel');
        const textarea = document.getElementById('transcript-textarea');

        if (editBtn) {
            editBtn.addEventListener('click', () => this.enterEditMode());
        }

        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.save());
        }

        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.cancel());
        }

        if (textarea) {
            textarea.addEventListener('input', () => this.onTextChange());
        }

        console.log('[TranscriptEditor] Initialized');
    },

    /**
     * Set the current recording ID for editing
     * @param {string} recordingId - Recording ID
     */
    setRecording(recordingId) {
        this._currentRecordingId = recordingId;
        this._isEditing = false;
        this._hasChanges = false;
    },

    /**
     * Enter edit mode
     */
    enterEditMode() {
        if (!this._currentRecordingId) {
            UI.showError('No recording selected');
            return;
        }

        this._isEditing = true;
        this._originalText = document.getElementById('transcript-content').textContent;

        const textarea = document.getElementById('transcript-textarea');
        if (textarea) {
            textarea.value = this._originalText;
        }

        // Toggle visibility
        document.getElementById('transcript-content').style.display = 'none';
        document.getElementById('transcript-editor').style.display = 'flex';
        document.getElementById('transcript-view-actions').style.display = 'none';
        document.getElementById('transcript-edit-actions').style.display = 'flex';

        // Focus textarea
        textarea?.focus();

        UI.announceStatus('Edit mode activated. Make changes and save when done.');
        console.log('[TranscriptEditor] Edit mode entered for:', this._currentRecordingId);
    },

    /**
     * Exit edit mode without saving
     */
    exitEditMode() {
        this._isEditing = false;
        this._hasChanges = false;

        // Toggle visibility back
        document.getElementById('transcript-content').style.display = 'block';
        document.getElementById('transcript-editor').style.display = 'none';
        document.getElementById('transcript-view-actions').style.display = 'flex';
        document.getElementById('transcript-edit-actions').style.display = 'none';

        // Hide unsaved indicator
        document.getElementById('transcript-unsaved').style.display = 'none';

        console.log('[TranscriptEditor] Edit mode exited');
    },

    /**
     * Handle text changes in textarea
     */
    onTextChange() {
        const textarea = document.getElementById('transcript-textarea');
        const current = textarea?.value || '';
        this._hasChanges = current !== this._originalText;

        // Show/hide unsaved indicator
        const indicator = document.getElementById('transcript-unsaved');
        if (indicator) {
            indicator.style.display = this._hasChanges ? 'inline' : 'none';
        }
    },

    /**
     * Save changes to server
     */
    async save() {
        const textarea = document.getElementById('transcript-textarea');
        const text = textarea?.value || '';

        if (!this._currentRecordingId) {
            UI.showError('No recording selected');
            return;
        }

        try {
            UI.showInfo('Saving changes...');

            const response = await fetch(`/api/recordings/${this._currentRecordingId}/transcript`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Save failed');
            }

            const result = await response.json();

            // Update the read-only view with new text
            const content = document.getElementById('transcript-content');
            if (content) {
                content.textContent = text;
            }

            // Update original text reference
            this._originalText = text;
            this._hasChanges = false;

            // Exit edit mode
            this.exitEditMode();

            UI.showSuccess(`Transcript saved (${result.word_count} words)`);
            console.log('[TranscriptEditor] Saved successfully:', result);

        } catch (err) {
            console.error('[TranscriptEditor] Save failed:', err);
            UI.showError(`Failed to save: ${err.message}`);
        }
    },

    /**
     * Cancel editing with confirmation if there are unsaved changes
     */
    cancel() {
        if (this._hasChanges) {
            if (!confirm('Discard unsaved changes?')) {
                return;
            }
        }

        // Revert textarea to original
        const textarea = document.getElementById('transcript-textarea');
        if (textarea) {
            textarea.value = this._originalText;
        }

        this.exitEditMode();
        UI.announceStatus('Edit cancelled');
    },

    /**
     * Check if currently in edit mode
     * @returns {boolean}
     */
    isEditing() {
        return this._isEditing;
    },

    /**
     * Check if there are unsaved changes
     * @returns {boolean}
     */
    hasUnsavedChanges() {
        return this._hasChanges;
    }
};
