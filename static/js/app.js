/**
 * CallWhisper Application
 * Main application logic
 */

const App = {
    state: 'idle',
    currentRecordingId: null,
    selectedRecordingId: null,  // Recording currently being viewed in transcript panel
    _isStarting: false,  // Guard against multiple concurrent startRecording calls

    // Filter/search state
    _filters: {
        query: '',
        dateFrom: '',
        dateTo: '',
        ticketPrefix: '',
        sort: 'newest',
        page: 1,
        pageSize: 20,
    },
    _totalPages: 1,
    _totalRecordings: 0,

    /**
     * Initialize application
     */
    async init() {
        // Initialize UI first
        UI.init();

        // Initialize theme manager (light/dark mode)
        if (typeof ThemeManager !== 'undefined') {
            ThemeManager.init();
        }

        // Initialize transcript editor
        if (typeof TranscriptEditor !== 'undefined') {
            TranscriptEditor.init();
        }

        // Run pre-flight startup checks (shows overlay with system status)
        if (typeof StartupCheck !== 'undefined') {
            await StartupCheck.run();
        }

        // Check first-run setup (VB-Cable installation)
        // This shows the setup wizard if virtual audio is not detected
        if (typeof SetupWizard !== 'undefined') {
            await SetupWizard.checkPrerequisites();
        }

        // Set up event handlers
        this.setupEventHandlers();

        // Connect WebSocket
        wsClient.connect();

        // Set up WebSocket event listeners
        this.setupWebSocketListeners();

        // Load initial data
        await this.loadDevices();
        await this.loadRecordings();

        // Load queue status
        await UI.refreshQueueStatus();

        // Check for incomplete jobs from previous session
        await this.checkIncompleteJobs();

        // Set up queue event handlers
        this.setupQueueEventHandlers();

        // Set up export dropdown handlers
        UI.setupExportDropdowns();
    },

    /**
     * Refresh devices (called after VB-Cable installation)
     */
    async refreshDevices() {
        await this.loadDevices();
    },

    /**
     * Update button states based on current app state
     * Called by SetupWizard when recording blocked state changes
     */
    updateButtonStates() {
        UI.setButtonStates(this.state);
    },

    /**
     * Set up UI event handlers
     */
    setupEventHandlers() {
        // Start recording button
        UI.elements.btnStart.addEventListener('click', () => {
            this.startRecording();
        });

        // Stop recording button
        UI.elements.btnStop.addEventListener('click', () => {
            this.stopRecording();
        });

        // Refresh devices button
        UI.elements.refreshDevices.addEventListener('click', () => {
            this.loadDevices();
        });

        // Reset button
        UI.elements.btnReset.addEventListener('click', () => {
            this.reset();
        });

        // Device selection change
        UI.elements.deviceSelect.addEventListener('change', () => {
            UI.setButtonStates(this.state);
        });

        // Open folder button
        UI.elements.btnOpenFolder.addEventListener('click', async () => {
            if (!this.selectedRecordingId) {
                UI.showWarning('No recording selected');
                return;
            }
            try {
                const response = await fetch(`/api/recordings/${this.selectedRecordingId}/open-folder`, {
                    method: 'POST'
                });
                if (!response.ok) {
                    const error = await response.json();
                    UI.showError(error.detail || 'Failed to open folder');
                }
            } catch (err) {
                UI.showError('Failed to open folder: ' + err.message);
            }
        });

        // Transcript panel close button
        if (UI.elements.transcriptClose) {
            UI.elements.transcriptClose.addEventListener('click', () => {
                UI.closeTranscript();
            });
        }

        // Transcript copy button
        if (UI.elements.transcriptCopy) {
            UI.elements.transcriptCopy.addEventListener('click', () => {
                UI.copyTranscript();
            });
        }

        // Recordings search input
        if (UI.elements.recordingsSearch) {
            // Debounced server-side search
            let searchTimeout;
            UI.elements.recordingsSearch.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this._filters.query = e.target.value;
                    this._filters.page = 1;  // Reset to first page on search
                    this.loadRecordings();
                }, 300);
            });

            // Clear search on Escape
            UI.elements.recordingsSearch.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    e.target.value = '';
                    this._filters.query = '';
                    this._filters.page = 1;
                    this.loadRecordings();
                    e.target.blur();
                }
            });
        }

        // Filter toggle button
        const filterToggle = document.getElementById('filter-toggle');
        const filterPanel = document.getElementById('filter-panel');
        if (filterToggle && filterPanel) {
            filterToggle.addEventListener('click', () => {
                const isExpanded = filterToggle.getAttribute('aria-expanded') === 'true';
                filterToggle.setAttribute('aria-expanded', !isExpanded);
                filterPanel.style.display = isExpanded ? 'none' : 'block';
            });
        }

        // Filter apply button
        const filterApply = document.getElementById('filter-apply');
        if (filterApply) {
            filterApply.addEventListener('click', () => {
                this._filters.dateFrom = document.getElementById('filter-date-from')?.value || '';
                this._filters.dateTo = document.getElementById('filter-date-to')?.value || '';
                this._filters.ticketPrefix = document.getElementById('filter-ticket')?.value || '';
                this._filters.sort = document.getElementById('filter-sort')?.value || 'newest';
                this._filters.page = 1;
                this.loadRecordings();
            });
        }

        // Filter clear button
        const filterClear = document.getElementById('filter-clear');
        if (filterClear) {
            filterClear.addEventListener('click', () => {
                // Clear filter inputs
                const dateFrom = document.getElementById('filter-date-from');
                const dateTo = document.getElementById('filter-date-to');
                const ticketInput = document.getElementById('filter-ticket');
                const sortSelect = document.getElementById('filter-sort');

                if (dateFrom) dateFrom.value = '';
                if (dateTo) dateTo.value = '';
                if (ticketInput) ticketInput.value = '';
                if (sortSelect) sortSelect.value = 'newest';

                // Clear filter state
                this._filters.dateFrom = '';
                this._filters.dateTo = '';
                this._filters.ticketPrefix = '';
                this._filters.sort = 'newest';
                this._filters.page = 1;
                this.loadRecordings();
            });
        }

        // Pagination buttons
        const paginationPrev = document.getElementById('pagination-prev');
        const paginationNext = document.getElementById('pagination-next');

        if (paginationPrev) {
            paginationPrev.addEventListener('click', () => {
                if (this._filters.page > 1) {
                    this._filters.page--;
                    this.loadRecordings();
                }
            });
        }

        if (paginationNext) {
            paginationNext.addEventListener('click', () => {
                if (this._filters.page < this._totalPages) {
                    this._filters.page++;
                    this.loadRecordings();
                }
            });
        }

        // View transcript button (event delegation on recordings list)
        UI.elements.recordingsList.addEventListener('click', (e) => {
            const viewBtn = e.target.closest('.btn-view');
            if (viewBtn) {
                const recordingId = viewBtn.dataset.recordingId;
                if (recordingId) {
                    this.viewTranscript(recordingId);
                }
            }
        });

        // File upload drag-and-drop
        this.setupFileUpload();
    },

    /**
     * Set up WebSocket event listeners
     */
    setupWebSocketListeners() {
        // Connection status
        wsClient.on('connection', (data) => {
            UI.setConnectionStatus(data.connected);

            if (data.connected) {
                // Refresh state after reconnection
                wsClient.requestState();
            }
        });

        // Initial connection with state
        wsClient.on('connected', (data) => {
            this.updateState(data.state);
        });

        // State changes
        wsClient.on('state_change', (data) => {
            this.updateState(data.state);

            if (data.recording_id) {
                this.currentRecordingId = data.recording_id;
            }
        });

        // Timer updates
        wsClient.on('timer', (data) => {
            UI.setTimer(data.formatted);
        });

        // Processing progress
        wsClient.on('processing_progress', (data) => {
            UI.setProgress(data.percent, data.stage);
        });

        // Partial transcript updates (real-time preview)
        wsClient.on('partial_transcript', (data) => {
            UI.updatePartialTranscript(data.text, data.is_final);
        });

        // Recording complete
        wsClient.on('recording_complete', (data) => {
            UI.showProgress(false);
            UI.hidePartialTranscript(); // Hide preview when complete
            UI.showSuccess('Recording complete! Transcript is ready for download.');
            this.loadRecordings(); // Refresh recordings list
            UI.refreshQueueStatus(); // Update queue display
        });

        // Queue status updates
        wsClient.on('queue_status', (data) => {
            UI.updateQueueStatus(data);
        });

        // Error
        wsClient.on('error', (data) => {
            this.updateState('error');
            UI.hidePartialTranscript(); // Hide preview on error
            UI.showError(data.message);
        });

        // State response
        wsClient.on('state', (data) => {
            this.updateState(data.state);
            if (data.elapsed_formatted) {
                UI.setTimer(data.elapsed_formatted);
            }
        });
    },

    /**
     * Update application state
     */
    updateState(newState) {
        const previousState = this.state;
        this.state = newState;
        UI.setStatus(newState);
        UI.setButtonStates(newState);

        // Show/hide progress based on state
        if (newState === 'processing') {
            UI.showProgress(true);
        } else if (newState !== 'recording') {
            UI.showProgress(false);
        }

        // Reset timer when idle
        if (newState === 'idle') {
            UI.resetTimer();
        }

        // Announce state changes to screen readers (LibV2 accessibility pattern)
        if (newState !== previousState) {
            const announcements = {
                recording: 'Recording started',
                processing: 'Processing recording, please wait',
                done: 'Recording complete and ready for download',
                error: 'An error occurred',
                idle: previousState ? 'Ready to record' : null,
            };

            const announcement = announcements[newState];
            if (announcement) {
                UI.announceStatus(announcement);
            }
        }
    },

    /**
     * Load audio devices
     */
    async loadDevices() {
        UI.setLoading(true);

        try {
            const response = await fetch('/api/devices');
            if (!response.ok) {
                throw new Error(`Failed to load devices: ${response.status}`);
            }
            const data = await response.json();
            UI.setDevices(data.devices);
        } catch (error) {
            console.error('Failed to load devices:', error);
            UI.showError('Failed to load audio devices');
        } finally {
            UI.setLoading(false);
            UI.setButtonStates(this.state);
        }
    },

    /**
     * Load recordings list with filters and pagination
     */
    async loadRecordings() {
        try {
            // Build query string with filters
            const params = new URLSearchParams();

            if (this._filters.query) {
                params.set('query', this._filters.query);
            }
            if (this._filters.dateFrom) {
                params.set('date_from', this._filters.dateFrom);
            }
            if (this._filters.dateTo) {
                params.set('date_to', this._filters.dateTo);
            }
            if (this._filters.ticketPrefix) {
                params.set('ticket_prefix', this._filters.ticketPrefix);
            }
            if (this._filters.sort) {
                params.set('sort', this._filters.sort);
            }
            params.set('page', this._filters.page.toString());
            params.set('page_size', this._filters.pageSize.toString());

            const response = await fetch(`/api/recordings/search?${params.toString()}`);
            if (!response.ok) {
                throw new Error(`Failed to load recordings: ${response.status}`);
            }
            const data = await response.json();

            // Update pagination state
            this._totalPages = data.total_pages;
            this._totalRecordings = data.total;

            // Render recordings with search highlighting
            UI.setRecordings(data.recordings, this._filters.query);

            // Update pagination UI
            this.updatePaginationUI();
        } catch (error) {
            console.error('Failed to load recordings:', error);
        }
    },

    /**
     * Update pagination UI based on current state
     */
    updatePaginationUI() {
        const paginationControls = document.getElementById('pagination-controls');
        const paginationPrev = document.getElementById('pagination-prev');
        const paginationNext = document.getElementById('pagination-next');
        const paginationInfo = document.getElementById('pagination-info');

        if (!paginationControls) return;

        // Show/hide pagination based on total pages
        if (this._totalPages <= 1 && this._totalRecordings <= this._filters.pageSize) {
            paginationControls.style.display = 'none';
            return;
        }

        paginationControls.style.display = 'flex';

        // Update button states
        if (paginationPrev) {
            paginationPrev.disabled = this._filters.page <= 1;
        }
        if (paginationNext) {
            paginationNext.disabled = this._filters.page >= this._totalPages;
        }

        // Update info text
        if (paginationInfo) {
            paginationInfo.textContent = `Page ${this._filters.page} of ${this._totalPages} (${this._totalRecordings} recordings)`;
        }
    },

    /**
     * View transcript for a recording
     */
    async viewTranscript(recordingId) {
        try {
            const response = await fetch(`/api/recordings/${recordingId}/transcript`);
            if (!response.ok) {
                if (response.status === 404) {
                    UI.showError('Recording not found');
                } else {
                    throw new Error(`Failed to load transcript: ${response.status}`);
                }
                return;
            }

            const data = await response.json();
            this.selectedRecordingId = recordingId;
            UI.elements.btnOpenFolder.disabled = false;
            UI.showTranscript(recordingId, data.text, {
                duration_seconds: data.duration_seconds,
                ticket_id: data.ticket_id,
                word_count: data.word_count,
            });

            // Set up transcript editor for this recording
            if (typeof TranscriptEditor !== 'undefined') {
                TranscriptEditor.setRecording(recordingId);
            }
        } catch (error) {
            console.error('Failed to load transcript:', error);
            UI.showError('Failed to load transcript');
        }
    },

    /**
     * Check system health before recording
     */
    async checkHealthBeforeRecording(device) {
        try {
            const response = await fetch(`/api/health/detailed?device=${encodeURIComponent(device)}`);
            const data = await response.json();

            if (!data.healthy) {
                // Find unhealthy checks
                const issues = data.checks
                    .filter(c => !c.healthy)
                    .map(c => c.message);

                UI.showWarning(`System issues detected: ${issues.join(', ')}`);
                return false;
            }

            return true;
        } catch (error) {
            console.error('Health check failed:', error);
            // Don't block recording if health check endpoint fails
            UI.showWarning('Could not verify system health - proceeding anyway');
            return true;
        }
    },

    /**
     * Start recording
     */
    async startRecording() {
        // Guard against multiple concurrent calls
        if (this._isStarting) {
            return;
        }

        const device = UI.getSelectedDevice();
        const ticketId = UI.getTicketId();

        if (!device) {
            UI.showError('Please select a recording device');
            return;
        }

        this._isStarting = true;
        UI.setLoading(true);

        // Check system health before recording
        const healthOk = await this.checkHealthBeforeRecording(device);
        if (!healthOk) {
            UI.setLoading(false);
            this._isStarting = false;
            return;
        }

        try {
            const response = await fetch('/api/recording/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    device: device,
                    ticket_id: ticketId || null,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start recording');
            }

            const data = await response.json();
            this.currentRecordingId = data.recording_id;

        } catch (error) {
            console.error('Failed to start recording:', error);
            UI.showError(error.message);
        } finally {
            UI.setLoading(false);
            this._isStarting = false;
        }
    },

    /**
     * Stop recording
     */
    async stopRecording() {
        UI.setLoading(true);

        try {
            const response = await fetch('/api/recording/stop', {
                method: 'POST',
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to stop recording');
            }

            // State will be updated via WebSocket

        } catch (error) {
            console.error('Failed to stop recording:', error);
            UI.showError(error.message);
        } finally {
            UI.setLoading(false);
        }
    },

    /**
     * Reset to idle state
     */
    async reset() {
        try {
            const response = await fetch('/api/reset', {
                method: 'POST',
            });

            if (response.ok) {
                this.updateState('idle');
                this.selectedRecordingId = null;
                UI.resetTimer();
                UI.showProgress(false);
                UI.closeTranscript();
                UI.elements.btnOpenFolder.disabled = true;
            }
        } catch (error) {
            console.error('Failed to reset:', error);
        }
    },

    /**
     * Set up file upload drag-and-drop handlers
     */
    setupFileUpload() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        if (!dropZone || !fileInput) {
            console.warn('[App] Drop zone elements not found');
            return;
        }

        // Click to browse
        dropZone.addEventListener('click', () => {
            if (!dropZone.classList.contains('uploading')) {
                fileInput.click();
            }
        });

        // Keyboard support
        dropZone.addEventListener('keydown', (e) => {
            if ((e.key === 'Enter' || e.key === ' ') && !dropZone.classList.contains('uploading')) {
                e.preventDefault();
                fileInput.click();
            }
        });

        // Drag events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!dropZone.classList.contains('uploading')) {
                dropZone.classList.add('drag-over');
            }
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');

            if (dropZone.classList.contains('uploading')) {
                return;
            }

            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                this.handleFileUpload(files);
            }
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            const files = e.target.files;
            if (files && files.length > 0) {
                this.handleFileUpload(files);
            }
            // Reset input so same file can be selected again
            fileInput.value = '';
        });

        // Prevent default drag behavior on window
        window.addEventListener('dragover', (e) => e.preventDefault());
        window.addEventListener('drop', (e) => e.preventDefault());
    },

    /**
     * Upload audio file for transcription
     */
    async uploadFile(file) {
        const dropZone = document.getElementById('drop-zone');
        const dropText = dropZone?.querySelector('.drop-text');

        // Client-side validation
        if (!file.type.startsWith('audio/')) {
            UI.showError('Please select an audio file (WAV, MP3, OGG, M4A, FLAC)');
            return;
        }

        const maxSize = 500 * 1024 * 1024; // 500MB
        if (file.size > maxSize) {
            UI.showError(`File too large. Maximum size is ${maxSize / 1024 / 1024}MB`);
            return;
        }

        // Check if already processing
        if (this.state === 'recording' || this.state === 'processing') {
            UI.showWarning('Please wait until the current operation completes');
            return;
        }

        const ticketId = UI.getTicketId();
        const formData = new FormData();
        formData.append('file', file);
        if (ticketId) {
            formData.append('ticket_id', ticketId);
        }

        // Update UI to uploading state
        if (dropZone) {
            dropZone.classList.add('uploading');
        }
        if (dropText) {
            dropText.textContent = `Uploading ${file.name}...`;
        }

        this.updateState('processing');
        UI.showProgress(true);
        UI.setProgress(0, 'Uploading...');
        UI.announceStatus(`Uploading ${file.name}`);

        try {
            const response = await fetch('/api/recordings/upload', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }

            const data = await response.json();
            this.currentRecordingId = data.recording_id;
            UI.showSuccess(`Processing ${file.name}...`);

            // Progress updates will come via WebSocket

        } catch (error) {
            console.error('[App] Upload failed:', error);
            UI.showError(error.message);
            this.updateState('error');
        } finally {
            // Reset drop zone appearance
            if (dropZone) {
                dropZone.classList.remove('uploading');
            }
            if (dropText) {
                dropText.textContent = 'Drop audio file here or click to browse';
            }
        }
    },

    /**
     * Check for incomplete jobs from previous session
     */
    async checkIncompleteJobs() {
        try {
            const response = await fetch('/api/jobs/incomplete');

            if (!response.ok) {
                console.log('[App] Incomplete jobs endpoint not available');
                return;
            }

            const data = await response.json();

            if (data.count > 0 && data.jobs.length > 0) {
                this.showRecoveryDialog(data.jobs[0]);
            }
        } catch (error) {
            console.error('[App] Failed to check incomplete jobs:', error);
        }
    },

    /**
     * Escape HTML special characters to prevent XSS
     */
    _escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    /**
     * Show job recovery dialog
     */
    showRecoveryDialog(job) {
        const progress = job.total_chunks > 0
            ? Math.round((job.chunks_completed / job.total_chunks) * 100)
            : 0;

        const statusText = {
            'recording': 'Recording in progress',
            'processing': 'Processing',
            'chunk_0': 'Transcribing (starting)',
        }[job.status] || `Processing (${this._escapeHtml(job.status)})`;

        const audioFile = job.audio_path ? job.audio_path.split('/').pop() : 'Unknown';

        // Escape all user-provided values
        const safeJobId = this._escapeHtml(job.job_id);
        const safeAudioFile = this._escapeHtml(audioFile);
        const safeDeviceName = this._escapeHtml(job.device_name);
        const safeTicketId = this._escapeHtml(job.ticket_id);

        const overlayHtml = `
            <div id="recovery-overlay" class="recovery-overlay" role="dialog" aria-modal="true" aria-labelledby="recovery-title">
                <div class="recovery-modal">
                    <h2 id="recovery-title" class="recovery-title">
                        <span class="recovery-title-icon" aria-hidden="true">!</span>
                        Incomplete Transcription Found
                    </h2>

                    <div class="recovery-job">
                        <div class="recovery-job-id">Job: ${safeJobId}</div>
                        <div class="recovery-job-meta">
                            <div>Status: ${statusText}</div>
                            <div>Audio: ${safeAudioFile}</div>
                            ${job.device_name ? `<div>Device: ${safeDeviceName}</div>` : ''}
                            ${job.ticket_id ? `<div>Ticket: ${safeTicketId}</div>` : ''}
                        </div>
                        ${job.total_chunks > 0 ? `
                        <div class="recovery-job-progress">
                            <div class="recovery-job-progress-bar">
                                <div class="recovery-job-progress-fill" style="width: ${progress}%"></div>
                            </div>
                            <div class="recovery-job-progress-text">
                                ${progress}% complete (${job.chunks_completed}/${job.total_chunks} chunks)
                            </div>
                        </div>
                        ` : ''}
                    </div>

                    <div class="recovery-actions">
                        <button id="recovery-resume-btn"
                                class="recovery-btn recovery-btn-primary">
                            Resume Transcription
                        </button>
                        <button id="recovery-discard-btn"
                                class="recovery-btn recovery-btn-secondary">
                            Discard and Start Fresh
                        </button>
                        <button id="recovery-later-btn"
                                class="recovery-btn recovery-btn-tertiary">
                            Remind Me Later
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', overlayHtml);

        // Add event listeners (safer than inline onclick with unescaped values)
        const resumeBtn = document.getElementById('recovery-resume-btn');
        const discardBtn = document.getElementById('recovery-discard-btn');
        const laterBtn = document.getElementById('recovery-later-btn');

        if (resumeBtn) {
            resumeBtn.addEventListener('click', () => this.resumeJob(job.job_id));
            resumeBtn.focus();
        }
        if (discardBtn) {
            discardBtn.addEventListener('click', () => this.discardJob(job.job_id));
        }
        if (laterBtn) {
            laterBtn.addEventListener('click', () => this.dismissRecovery());
        }

        // Announce to screen readers
        UI.announceStatus('Incomplete transcription found from previous session. Choose to resume, discard, or remind later.');
    },

    /**
     * Resume an incomplete job
     */
    async resumeJob(jobId) {
        const resumeBtn = document.getElementById('recovery-resume-btn');
        if (resumeBtn) {
            resumeBtn.disabled = true;
            resumeBtn.textContent = 'Resuming...';
        }

        try {
            const response = await fetch(`/api/jobs/${jobId}/resume`, {
                method: 'POST',
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to resume job');
            }

            this.dismissRecovery();
            UI.showSuccess('Resuming transcription...');

            // State will be updated via WebSocket

        } catch (error) {
            console.error('[App] Failed to resume job:', error);
            UI.showError(error.message);

            if (resumeBtn) {
                resumeBtn.disabled = false;
                resumeBtn.textContent = 'Resume Transcription';
            }
        }
    },

    /**
     * Discard an incomplete job
     */
    async discardJob(jobId) {
        try {
            const response = await fetch(`/api/jobs/${jobId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to discard job');
            }

            this.dismissRecovery();
            UI.showInfo('Job discarded. Ready for new recording.');

        } catch (error) {
            console.error('[App] Failed to discard job:', error);
            UI.showError(error.message);
        }
    },

    /**
     * Dismiss recovery dialog (remind later)
     */
    dismissRecovery() {
        const overlay = document.getElementById('recovery-overlay');
        if (overlay) {
            overlay.remove();
        }
    },

    // ========================================================================
    // Batch Processing Methods
    // ========================================================================

    /**
     * Handle file upload - single or multiple files
     * @param {FileList} files - Files to upload
     */
    async handleFileUpload(files) {
        if (!files || files.length === 0) {
            return;
        }

        // For single file, use the existing flow
        if (files.length === 1) {
            await this.uploadFile(files[0]);
            return;
        }

        // For multiple files, use batch upload
        await UI.uploadBatch(files);
    },

    /**
     * Set up queue event handlers
     */
    setupQueueEventHandlers() {
        // Queue list click delegation (for cancel buttons)
        const queueList = document.getElementById('queue-list');
        if (queueList) {
            queueList.addEventListener('click', async (e) => {
                const cancelBtn = e.target.closest('.queue-item-cancel');
                if (cancelBtn) {
                    const jobId = cancelBtn.dataset.jobId;
                    if (jobId) {
                        cancelBtn.disabled = true;
                        await UI.cancelQueuedJob(jobId);
                    }
                }
            });
        }

        // Clear history button
        const clearHistoryBtn = document.getElementById('queue-clear-history');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', async () => {
                clearHistoryBtn.disabled = true;
                await UI.clearQueueHistory();
                clearHistoryBtn.disabled = false;
            });
        }

        // Start queue status polling when queue has items
        this.startQueuePolling();
    },

    /**
     * Queue status polling interval ID
     */
    _queuePollInterval: null,

    /**
     * Start polling for queue status updates
     * (Fallback for when WebSocket updates aren't available)
     */
    startQueuePolling() {
        // Poll every 2 seconds when queue is active
        const pollInterval = 2000;

        this._queuePollInterval = setInterval(async () => {
            // Only poll if queue section is visible
            const queueSection = document.getElementById('queue-section');
            if (queueSection && queueSection.style.display !== 'none') {
                await UI.refreshQueueStatus();
            }
        }, pollInterval);
    },

    /**
     * Stop queue status polling
     */
    stopQueuePolling() {
        if (this._queuePollInterval) {
            clearInterval(this._queuePollInterval);
            this._queuePollInterval = null;
        }
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
