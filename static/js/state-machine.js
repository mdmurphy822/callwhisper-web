/**
 * CallWhisper UI State Machine
 *
 * Based on LibV2 advanced-react-mastery patterns:
 * - Explicit state transitions prevent invalid UI states
 * - State machine validates all transitions
 * - Logs invalid transition attempts for debugging
 *
 * States:
 *   IDLE -> RECORDING -> STOPPING -> PROCESSING -> COMPLETED/ERROR
 *   ERROR -> IDLE (via reset)
 *   COMPLETED -> IDLE (via reset or new recording)
 */

const RecordingStateMachine = {
    // State definitions with allowed transitions
    states: {
        IDLE: {
            transitions: ['RECORDING'],
            label: 'Ready',
            canStart: true,
            canStop: false,
        },
        RECORDING: {
            transitions: ['STOPPING', 'ERROR'],
            label: 'Recording',
            canStart: false,
            canStop: true,
        },
        STOPPING: {
            transitions: ['PROCESSING', 'ERROR'],
            label: 'Stopping',
            canStart: false,
            canStop: false,
        },
        PROCESSING: {
            transitions: ['COMPLETED', 'ERROR'],
            label: 'Processing',
            canStart: false,
            canStop: false,
        },
        COMPLETED: {
            transitions: ['IDLE', 'RECORDING'],
            label: 'Completed',
            canStart: true,
            canStop: false,
        },
        ERROR: {
            transitions: ['IDLE'],
            label: 'Error',
            canStart: false,
            canStop: false,
        },
    },

    // Current state
    _currentState: 'IDLE',

    // Transition history for debugging
    _history: [],
    _maxHistory: 50,

    /**
     * Get current state
     */
    get current() {
        return this._currentState;
    },

    /**
     * Get state info object
     */
    get info() {
        return this.states[this._currentState] || this.states.IDLE;
    },

    /**
     * Check if transition is valid
     */
    canTransition(to) {
        const currentInfo = this.states[this._currentState];
        if (!currentInfo) {
            return false;
        }
        return currentInfo.transitions.includes(to);
    },

    /**
     * Attempt to transition to a new state
     *
     * @param {string} nextState - The target state
     * @param {Object} metadata - Optional metadata for logging
     * @returns {boolean} - True if transition succeeded
     */
    transition(nextState, metadata = {}) {
        const normalizedNext = nextState.toUpperCase();

        // Validate target state exists
        if (!this.states[normalizedNext]) {
            console.error(
                `[StateMachine] Invalid state: ${nextState}. ` +
                `Valid states: ${Object.keys(this.states).join(', ')}`
            );
            return false;
        }

        // Check if transition is allowed
        if (!this.canTransition(normalizedNext)) {
            console.error(
                `[StateMachine] Invalid transition: ${this._currentState} -> ${normalizedNext}. ` +
                `Allowed: ${this.states[this._currentState].transitions.join(', ')}`
            );

            // Record invalid attempt in history
            this._recordHistory({
                from: this._currentState,
                to: normalizedNext,
                success: false,
                reason: 'invalid_transition',
                metadata,
            });

            return false;
        }

        // Execute transition
        const previousState = this._currentState;
        this._currentState = normalizedNext;

        // Record successful transition
        this._recordHistory({
            from: previousState,
            to: normalizedNext,
            success: true,
            metadata,
        });

        console.log(
            `[StateMachine] Transition: ${previousState} -> ${normalizedNext}`
        );

        // Dispatch event for listeners
        this._dispatchTransition(previousState, normalizedNext, metadata);

        return true;
    },

    /**
     * Force state (use sparingly, for recovery/reset)
     */
    forceState(state, reason = 'forced') {
        const normalizedState = state.toUpperCase();

        if (!this.states[normalizedState]) {
            console.error(`[StateMachine] Cannot force invalid state: ${state}`);
            return false;
        }

        const previousState = this._currentState;
        this._currentState = normalizedState;

        this._recordHistory({
            from: previousState,
            to: normalizedState,
            success: true,
            forced: true,
            reason,
        });

        console.warn(
            `[StateMachine] Forced transition: ${previousState} -> ${normalizedState} (${reason})`
        );

        this._dispatchTransition(previousState, normalizedState, { forced: true, reason });

        return true;
    },

    /**
     * Reset to IDLE state
     */
    reset() {
        return this.forceState('IDLE', 'user_reset');
    },

    /**
     * Map server state string to state machine state
     */
    mapServerState(serverState) {
        const mapping = {
            'idle': 'IDLE',
            'recording': 'RECORDING',
            'stopping': 'STOPPING',
            'processing': 'PROCESSING',
            'done': 'COMPLETED',
            'completed': 'COMPLETED',
            'error': 'ERROR',
        };

        return mapping[serverState.toLowerCase()] || 'IDLE';
    },

    /**
     * Sync with server state
     */
    syncWithServer(serverState) {
        const mappedState = this.mapServerState(serverState);

        // If already in this state, no action needed
        if (this._currentState === mappedState) {
            return true;
        }

        // Try normal transition first
        if (this.canTransition(mappedState)) {
            return this.transition(mappedState, { source: 'server_sync' });
        }

        // Force sync if server is authoritative
        return this.forceState(mappedState, 'server_sync');
    },

    /**
     * Check if action is allowed in current state
     */
    canPerformAction(action) {
        const currentInfo = this.states[this._currentState];
        if (!currentInfo) {
            return false;
        }

        switch (action) {
            case 'start':
                return currentInfo.canStart;
            case 'stop':
                return currentInfo.canStop;
            case 'reset':
                return this._currentState !== 'IDLE';
            default:
                return false;
        }
    },

    /**
     * Get transition history
     */
    getHistory() {
        return [...this._history];
    },

    /**
     * Clear history
     */
    clearHistory() {
        this._history = [];
    },

    /**
     * Record transition in history
     */
    _recordHistory(entry) {
        this._history.push({
            ...entry,
            timestamp: new Date().toISOString(),
        });

        // Trim history if too long
        if (this._history.length > this._maxHistory) {
            this._history = this._history.slice(-this._maxHistory);
        }
    },

    /**
     * Event listeners for transitions
     */
    _listeners: [],

    /**
     * Subscribe to state transitions
     */
    onTransition(callback) {
        this._listeners.push(callback);

        // Return unsubscribe function
        return () => {
            this._listeners = this._listeners.filter(cb => cb !== callback);
        };
    },

    /**
     * Dispatch transition event
     */
    _dispatchTransition(from, to, metadata) {
        for (const callback of this._listeners) {
            try {
                callback({ from, to, metadata });
            } catch (error) {
                console.error('[StateMachine] Listener error:', error);
            }
        }
    },

    /**
     * Get debug info
     */
    getDebugInfo() {
        return {
            currentState: this._currentState,
            stateInfo: this.states[this._currentState],
            allowedTransitions: this.states[this._currentState]?.transitions || [],
            historyLength: this._history.length,
            lastTransition: this._history[this._history.length - 1] || null,
        };
    },
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RecordingStateMachine;
}
