/**
 * CallWhisper Pre-Flight Startup Check
 *
 * Shows system status on startup, catches issues before user tries to record.
 * Based on LibV2 orchestrator-architecture patterns for health checks.
 *
 * Usage:
 *   StartupCheck.run();  // Call before App.init()
 */

const StartupCheck = {
    // Configuration
    config: {
        apiEndpoint: '/api/health/detailed',
        autoDismissDelay: 2500,  // ms to wait before auto-dismiss when all pass
        minDisplayTime: 1000,    // minimum time to show overlay
    },

    // State
    _overlayElement: null,
    _checkResults: null,
    _startTime: null,

    /**
     * Run pre-flight checks and show status overlay
     * @returns {Promise<boolean>} True if all critical checks passed
     */
    async run() {
        this._startTime = Date.now();
        this._showOverlay();

        try {
            const response = await fetch(this.config.apiEndpoint);

            if (!response.ok) {
                // API not available - show warning but don't block
                this._updateOverlay({
                    healthy: true,
                    checks: [{
                        name: 'api',
                        healthy: true,
                        message: 'Health check API not available - continuing anyway'
                    }]
                });
                await this._autoDismiss();
                return true;
            }

            const data = await response.json();
            this._checkResults = data;
            this._updateOverlay(data);

            // User clicks Continue button to proceed (no auto-dismiss)
            // Return true if healthy, false otherwise
            return data.healthy;

        } catch (error) {
            console.error('[StartupCheck] Error:', error);
            // Network error - show warning but don't block
            this._updateOverlay({
                healthy: true,
                checks: [{
                    name: 'network',
                    healthy: false,
                    message: 'Could not reach server - is the backend running?'
                }]
            });
            return false;
        }
    },

    /**
     * Show initial loading overlay
     */
    _showOverlay() {
        const overlayHtml = `
            <div id="startup-overlay" class="startup-overlay" role="dialog" aria-modal="true" aria-labelledby="startup-title">
                <div class="startup-modal">
                    <h2 id="startup-title" class="startup-title">
                        <span class="startup-logo" aria-hidden="true"></span>
                        CallWhisper Starting...
                    </h2>

                    <div id="startup-checks" class="startup-checks" aria-live="polite">
                        <div class="startup-check startup-check-loading">
                            <span class="startup-check-icon" aria-hidden="true"></span>
                            <span class="startup-check-text">Checking system requirements...</span>
                        </div>
                    </div>

                    <div id="startup-actions" class="startup-actions" style="display: none;">
                        <button id="startup-continue-btn"
                                class="startup-btn startup-btn-primary"
                                onclick="StartupCheck.dismiss()">
                            Continue Anyway
                        </button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', overlayHtml);
        this._overlayElement = document.getElementById('startup-overlay');
    },

    /**
     * Update overlay with check results
     */
    _updateOverlay(data) {
        const checksContainer = document.getElementById('startup-checks');
        const actionsContainer = document.getElementById('startup-actions');

        if (!checksContainer) return;

        // Map check names to friendly labels
        const checkLabels = {
            ffmpeg: 'FFmpeg',
            whisper_model: 'Transcription Model',
            disk_space: 'Disk Space',
            memory: 'Memory',
            recordings_dir: 'Recordings Directory',
            audio_device: 'Audio Device',
            api: 'Server Connection',
            network: 'Network'
        };

        // Build check items HTML
        let checksHtml = '';
        let hasIssues = false;
        let hasCriticalIssue = false;

        for (const check of data.checks) {
            const label = checkLabels[check.name] || check.name;
            const isSkipped = check.details?.skipped;
            const isCritical = ['ffmpeg', 'whisper_model'].includes(check.name);

            let statusClass = 'startup-check-pass';
            let statusIcon = '\u2713';  // Check mark
            let statusText = check.message;

            if (!check.healthy && !isSkipped) {
                hasIssues = true;
                statusClass = isCritical ? 'startup-check-fail' : 'startup-check-warn';
                statusIcon = isCritical ? '\u2717' : '\u26A0';  // X or warning

                if (isCritical) {
                    hasCriticalIssue = true;
                }

                // Add actionable details for specific failures
                statusText = this._getActionableMessage(check);
            } else if (isSkipped) {
                statusClass = 'startup-check-skip';
                statusIcon = '\u2014';  // Dash
            }

            // Format details for disk/memory
            if (check.healthy && check.details) {
                if (check.name === 'disk_space' && check.details.free_gb) {
                    statusText = `${check.details.free_gb.toFixed(1)} GB available`;
                }
                if (check.name === 'memory' && check.details.available_mb) {
                    statusText = `${Math.round(check.details.available_mb)} MB available`;
                }
            }

            checksHtml += `
                <div class="startup-check ${statusClass}">
                    <span class="startup-check-icon" aria-hidden="true">${statusIcon}</span>
                    <div class="startup-check-content">
                        <span class="startup-check-label">${label}</span>
                        <span class="startup-check-text">${statusText}</span>
                        ${this._getActionButton(check)}
                    </div>
                </div>
            `;
        }

        checksContainer.innerHTML = checksHtml;

        // Update title based on status
        const titleEl = document.getElementById('startup-title');
        if (titleEl) {
            if (hasCriticalIssue) {
                titleEl.innerHTML = '<span class="startup-logo startup-logo-error" aria-hidden="true"></span>Setup Required';
            } else if (hasIssues) {
                titleEl.innerHTML = '<span class="startup-logo startup-logo-warn" aria-hidden="true"></span>Ready with Warnings';
            } else {
                titleEl.innerHTML = '<span class="startup-logo startup-logo-ok" aria-hidden="true"></span>Ready to Go!';
            }
        }

        // Show actions - always show when results are ready
        if (actionsContainer) {
            actionsContainer.style.display = 'block';
            const continueBtn = document.getElementById('startup-continue-btn');
            if (continueBtn) {
                if (hasCriticalIssue) {
                    continueBtn.textContent = 'Continue Without Recording';
                } else {
                    continueBtn.textContent = 'Continue';
                }
                continueBtn.focus();
            }

            // Allow clicking overlay background to dismiss when all checks pass
            if (!hasIssues && this._overlayElement) {
                this._overlayElement.style.cursor = 'pointer';
                this._overlayElement.addEventListener('click', (e) => {
                    if (e.target === this._overlayElement) {
                        this.dismiss();
                    }
                }, { once: true });
            }
        }
    },

    /**
     * Get actionable message for a failed check
     */
    _getActionableMessage(check) {
        const messages = {
            ffmpeg: 'FFmpeg not found. Required for audio processing.',
            whisper_model: 'No transcription model found.',
            disk_space: `Low disk space: ${check.details?.free_gb?.toFixed(1) || '?'} GB free`,
            memory: `Low memory: ${Math.round(check.details?.available_mb || 0)} MB available`,
            recordings_dir: 'Cannot write to recordings directory',
            audio_device: check.message
        };

        return messages[check.name] || check.message;
    },

    /**
     * Get action button HTML for fixable issues
     */
    _getActionButton(check) {
        if (check.healthy || check.details?.skipped) {
            return '';
        }

        const actions = {
            ffmpeg: {
                label: 'Download Instructions',
                action: 'StartupCheck.showFfmpegHelp()'
            },
            whisper_model: {
                label: 'Download Model',
                action: 'StartupCheck.showModelHelp()'
            },
            audio_device: {
                label: 'Setup Audio',
                action: 'StartupCheck.showAudioHelp()'
            }
        };

        const action = actions[check.name];
        if (!action) return '';

        return `
            <button class="startup-check-action" onclick="${action}">
                ${action.label}
            </button>
        `;
    },

    /**
     * Auto-dismiss overlay after delay (when all checks pass)
     */
    async _autoDismiss() {
        const elapsed = Date.now() - this._startTime;
        const remaining = Math.max(0, this.config.minDisplayTime - elapsed);

        // Wait for minimum display time
        await new Promise(resolve => setTimeout(resolve, remaining));

        // Then wait for auto-dismiss delay
        await new Promise(resolve => setTimeout(resolve, this.config.autoDismissDelay));

        this.dismiss();
    },

    /**
     * Dismiss the overlay
     */
    dismiss() {
        if (this._overlayElement) {
            this._overlayElement.classList.add('startup-overlay-dismiss');

            setTimeout(() => {
                this._overlayElement.remove();
                this._overlayElement = null;
            }, 300);
        }
    },

    /**
     * Show FFmpeg download help
     */
    showFfmpegHelp() {
        const modal = document.querySelector('.startup-modal');
        if (!modal) return;

        modal.innerHTML = `
            <h2 class="startup-title">FFmpeg Required</h2>

            <div class="startup-help-content">
                <p>FFmpeg is required for audio processing. To install:</p>

                <div class="startup-help-section">
                    <h3>Option 1: Run Download Script</h3>
                    <code class="startup-code">scripts/download-vendor.ps1</code>
                    <p>This will download FFmpeg and place it in the vendor folder.</p>
                </div>

                <div class="startup-help-section">
                    <h3>Option 2: Manual Download</h3>
                    <ol>
                        <li>Download from <a href="https://github.com/BtbN/FFmpeg-Builds/releases" target="_blank" rel="noopener">FFmpeg Builds</a></li>
                        <li>Extract ffmpeg.exe to the <code>vendor/</code> folder</li>
                        <li>Restart CallWhisper</li>
                    </ol>
                </div>

                <div class="startup-help-section">
                    <h3>Option 3: IT Deployment</h3>
                    <p>Contact your IT department - FFmpeg can be deployed via SCCM/Intune.</p>
                </div>
            </div>

            <div class="startup-actions">
                <button class="startup-btn startup-btn-secondary" onclick="StartupCheck.goBack()">
                    Back
                </button>
                <button class="startup-btn startup-btn-primary" onclick="StartupCheck.dismiss()">
                    Continue Without Recording
                </button>
            </div>
        `;
    },

    /**
     * Show Whisper model download help
     */
    showModelHelp() {
        const modal = document.querySelector('.startup-modal');
        if (!modal) return;

        modal.innerHTML = `
            <h2 class="startup-title">Transcription Model Required</h2>

            <div class="startup-help-content">
                <p>A Whisper model file is required for transcription. To install:</p>

                <div class="startup-help-section">
                    <h3>Option 1: Run Download Script</h3>
                    <code class="startup-code">scripts/download-vendor.ps1</code>
                    <p>This will download the recommended model (ggml-medium.en.bin).</p>
                </div>

                <div class="startup-help-section">
                    <h3>Option 2: Manual Download</h3>
                    <ol>
                        <li>Download from <a href="https://huggingface.co/ggerganov/whisper.cpp/tree/main" target="_blank" rel="noopener">Hugging Face</a></li>
                        <li>Choose ggml-medium.en.bin (1.5 GB) for best results</li>
                        <li>Place in the <code>models/</code> folder</li>
                        <li>Restart CallWhisper</li>
                    </ol>
                </div>

                <p class="startup-help-note">
                    <strong>Note:</strong> Smaller models (tiny, base, small) are faster but less accurate.
                    The medium.en model is recommended for call transcription.
                </p>
            </div>

            <div class="startup-actions">
                <button class="startup-btn startup-btn-secondary" onclick="StartupCheck.goBack()">
                    Back
                </button>
                <button class="startup-btn startup-btn-primary" onclick="StartupCheck.dismiss()">
                    Continue Without Recording
                </button>
            </div>
        `;
    },

    /**
     * Show audio device setup help
     */
    showAudioHelp() {
        const modal = document.querySelector('.startup-modal');
        if (!modal) return;

        modal.innerHTML = `
            <h2 class="startup-title">Audio Device Setup</h2>

            <div class="startup-help-content">
                <p>A virtual audio device is required to capture call audio.</p>

                <div class="startup-help-section">
                    <h3>Option 1: VB-Cable (Recommended)</h3>
                    <ol>
                        <li>Download from <a href="https://vb-audio.com/Cable/" target="_blank" rel="noopener">vb-audio.com</a></li>
                        <li>Run installer as Administrator</li>
                        <li>Restart computer</li>
                        <li>Set Jabber/Finesse speaker to "CABLE Input"</li>
                    </ol>
                </div>

                <div class="startup-help-section">
                    <h3>Option 2: Windows Stereo Mix</h3>
                    <ol>
                        <li>Right-click speaker icon in taskbar</li>
                        <li>Select "Sounds" then "Recording" tab</li>
                        <li>Right-click empty area, enable "Show Disabled Devices"</li>
                        <li>Right-click "Stereo Mix" and select "Enable"</li>
                    </ol>
                    <p class="startup-help-note">Note: Stereo Mix may not be available on all systems.</p>
                </div>

                <div class="startup-help-section">
                    <h3>Option 3: VoiceMeeter (Free Alternative)</h3>
                    <p>Download from <a href="https://vb-audio.com/Voicemeeter/" target="_blank" rel="noopener">vb-audio.com/Voicemeeter</a></p>
                </div>
            </div>

            <div class="startup-actions">
                <button class="startup-btn startup-btn-secondary" onclick="StartupCheck.goBack()">
                    Back
                </button>
                <button class="startup-btn startup-btn-primary" onclick="StartupCheck.recheckAudio()">
                    Check Again
                </button>
            </div>
        `;
    },

    /**
     * Go back to main status view
     */
    async goBack() {
        const modal = document.querySelector('.startup-modal');
        if (!modal) return;

        // Re-run the check
        modal.innerHTML = `
            <h2 class="startup-title">Checking...</h2>
            <div class="startup-checks">
                <div class="startup-check startup-check-loading">
                    <span class="startup-check-icon" aria-hidden="true"></span>
                    <span class="startup-check-text">Re-checking system requirements...</span>
                </div>
            </div>
        `;

        try {
            const response = await fetch(this.config.apiEndpoint);
            const data = await response.json();
            this._checkResults = data;

            // Rebuild the modal
            modal.innerHTML = `
                <h2 id="startup-title" class="startup-title">
                    <span class="startup-logo" aria-hidden="true"></span>
                    Checking...
                </h2>
                <div id="startup-checks" class="startup-checks" aria-live="polite"></div>
                <div id="startup-actions" class="startup-actions" style="display: none;">
                    <button id="startup-continue-btn"
                            class="startup-btn startup-btn-primary"
                            onclick="StartupCheck.dismiss()">
                        Continue Anyway
                    </button>
                </div>
            `;

            this._updateOverlay(data);

        } catch (error) {
            console.error('[StartupCheck] Recheck error:', error);
        }
    },

    /**
     * Re-check audio devices specifically
     */
    async recheckAudio() {
        await this.goBack();

        // If audio is now detected, auto-continue
        if (this._checkResults?.healthy) {
            setTimeout(() => this.dismiss(), 1500);
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StartupCheck;
}
