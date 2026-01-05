"""
Windows Process Monitor using WMI.

Watches for target process start/stop events.
This is Tier 2 in the call detection architecture.
"""

import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

if sys.platform != "win32":
    raise ImportError("ProcessMonitor is only available on Windows")

import wmi

from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessEvent:
    """Event fired when a target process starts or stops."""

    event_type: str  # "started" or "stopped"
    process_name: str
    process_id: int
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"Process({self.process_name}[{self.process_id}]: {self.event_type})"


class ProcessMonitor:
    """
    Monitors process lifecycle for target executables using WMI.

    Uses WMI event subscriptions:
    - Win32_ProcessStartTrace (for process creation)
    - Win32_ProcessStopTrace (for process termination)

    Also provides polling-based fallback for environments where
    WMI event subscriptions require elevated privileges.
    """

    def __init__(
        self,
        target_processes: List[str],
        poll_interval: float = 2.0,
        use_wmi_events: bool = False,  # WMI events require admin, use polling by default
    ):
        """
        Initialize the process monitor.

        Args:
            target_processes: List of process names to monitor (e.g., ["CiscoJabber.exe"])
            poll_interval: How often to poll for process changes (seconds)
            use_wmi_events: Whether to use WMI event subscriptions (requires admin)
        """
        self._targets = [p.lower() for p in target_processes]
        self._poll_interval = poll_interval
        self._use_wmi_events = use_wmi_events
        self._callbacks: List[Callable[[ProcessEvent], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._wmi: Optional[wmi.WMI] = None

        # Track known processes: {process_name: set of PIDs}
        self._known_processes: Dict[str, Set[int]] = {}

        logger.info(
            "process_monitor_initialized",
            targets=self._targets,
            poll_interval=poll_interval,
            use_wmi_events=use_wmi_events,
        )

    def add_callback(self, callback: Callable[[ProcessEvent], None]) -> None:
        """Register a callback for process events."""
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ProcessEvent], None]) -> None:
        """Remove a registered callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def start(self) -> None:
        """Start monitoring processes."""
        if self._running:
            return

        self._running = True

        # Initialize WMI connection
        try:
            self._wmi = wmi.WMI()
        except Exception as e:
            logger.error("wmi_init_error", error=str(e))
            raise RuntimeError(f"Failed to initialize WMI: {e}")

        # Take initial snapshot
        self._snapshot_processes()

        # Start monitoring thread
        self._thread = threading.Thread(
            target=self._monitor_loop, name="ProcessMonitor", daemon=True
        )
        self._thread.start()
        logger.info("process_monitor_started")

    def stop(self) -> None:
        """Stop monitoring processes."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._known_processes.clear()
        self._wmi = None
        logger.info("process_monitor_stopped")

    def is_process_running(self, name: str) -> bool:
        """Check if a target process is currently running."""
        name_lower = name.lower()
        with self._lock:
            return (
                name_lower in self._known_processes
                and len(self._known_processes[name_lower]) > 0
            )

    def get_running_processes(self) -> Dict[str, List[int]]:
        """Get all running target processes and their PIDs."""
        with self._lock:
            return {k: list(v) for k, v in self._known_processes.items()}

    def get_all_processes(self) -> List[Dict]:
        """Get all running processes (for debugging)."""
        if not self._wmi:
            return []

        processes = []
        try:
            for proc in self._wmi.Win32_Process():
                processes.append(
                    {
                        "name": proc.Name,
                        "pid": proc.ProcessId,
                        "command_line": proc.CommandLine or "",
                    }
                )
        except Exception as e:
            logger.error("enumerate_processes_error", error=str(e))

        return processes

    def _monitor_loop(self) -> None:
        """Main monitoring loop running in a separate thread."""
        if self._use_wmi_events:
            self._wmi_event_loop()
        else:
            self._poll_loop()

    def _poll_loop(self) -> None:
        """Polling-based process monitoring (no admin required)."""
        while self._running:
            try:
                self._check_processes()
            except Exception as e:
                logger.error("process_poll_error", error=str(e))

            time.sleep(self._poll_interval)

    def _wmi_event_loop(self) -> None:
        """
        WMI event-based monitoring (requires admin privileges).

        Note: This is more responsive but requires elevated permissions.
        """
        try:
            # Create a separate WMI connection for events
            wmi_events = wmi.WMI()

            # Watch for process creation
            process_watcher = wmi_events.Win32_Process.watch_for(
                notification_type="creation", delay_secs=1
            )

            while self._running:
                try:
                    # Check for new process (with timeout)
                    new_process = process_watcher(timeout_ms=1000)
                    if new_process:
                        name = new_process.Name.lower()
                        if name in self._targets:
                            self._fire_event(
                                ProcessEvent(
                                    event_type="started",
                                    process_name=new_process.Name,
                                    process_id=new_process.ProcessId,
                                )
                            )
                            with self._lock:
                                if name not in self._known_processes:
                                    self._known_processes[name] = set()
                                self._known_processes[name].add(new_process.ProcessId)

                except wmi.x_wmi_timed_out:
                    # Normal timeout, check for stopped processes
                    self._check_stopped_processes()
                except Exception as e:
                    logger.error("wmi_event_error", error=str(e))

        except Exception as e:
            logger.error("wmi_event_loop_failed", error=str(e), fallback="polling")
            # Fall back to polling
            self._poll_loop()

    def _snapshot_processes(self) -> None:
        """Take initial snapshot of running processes."""
        if not self._wmi:
            return

        with self._lock:
            self._known_processes.clear()

            for target in self._targets:
                self._known_processes[target] = set()

            try:
                for proc in self._wmi.Win32_Process():
                    name = proc.Name.lower()
                    if name in self._targets:
                        self._known_processes[name].add(proc.ProcessId)

                        # Fire initial started event for existing processes
                        self._fire_event(
                            ProcessEvent(
                                event_type="started",
                                process_name=proc.Name,
                                process_id=proc.ProcessId,
                            )
                        )

            except Exception as e:
                logger.error("process_snapshot_error", error=str(e))

        logger.info(
            "process_snapshot_complete",
            processes={k: len(v) for k, v in self._known_processes.items()},
        )

    def _check_processes(self) -> None:
        """Check for process changes (polling mode)."""
        if not self._wmi:
            return

        current_processes: Dict[str, Set[int]] = {t: set() for t in self._targets}

        try:
            for proc in self._wmi.Win32_Process():
                name = proc.Name.lower()
                if name in self._targets:
                    current_processes[name].add(proc.ProcessId)

        except Exception as e:
            logger.error("process_check_error", error=str(e))
            return

        with self._lock:
            for name in self._targets:
                old_pids = self._known_processes.get(name, set())
                new_pids = current_processes.get(name, set())

                # Check for started processes
                for pid in new_pids - old_pids:
                    self._fire_event(
                        ProcessEvent(
                            event_type="started", process_name=name, process_id=pid
                        )
                    )

                # Check for stopped processes
                for pid in old_pids - new_pids:
                    self._fire_event(
                        ProcessEvent(
                            event_type="stopped", process_name=name, process_id=pid
                        )
                    )

            # Update known processes
            self._known_processes = current_processes

    def _check_stopped_processes(self) -> None:
        """Check for processes that have stopped (WMI event mode)."""
        if not self._wmi:
            return

        current_pids: Dict[str, Set[int]] = {t: set() for t in self._targets}

        try:
            for proc in self._wmi.Win32_Process():
                name = proc.Name.lower()
                if name in self._targets:
                    current_pids[name].add(proc.ProcessId)

        except Exception as e:
            logger.error("check_stopped_error", error=str(e))
            return

        with self._lock:
            for name in self._targets:
                old_pids = self._known_processes.get(name, set())
                new_pids = current_pids.get(name, set())

                # Fire stopped events for missing PIDs
                for pid in old_pids - new_pids:
                    self._fire_event(
                        ProcessEvent(
                            event_type="stopped", process_name=name, process_id=pid
                        )
                    )

            # Update known processes
            self._known_processes = current_pids

    def _fire_event(self, event: ProcessEvent) -> None:
        """Fire event to all registered callbacks."""
        logger.debug(
            "process_event",
            event_type=event.event_type,
            process=event.process_name,
            pid=event.process_id,
        )

        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    "process_callback_error",
                    error=str(e),
                    callback=(
                        callback.__name__
                        if hasattr(callback, "__name__")
                        else str(callback)
                    ),
                )

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def target_processes(self) -> List[str]:
        """Get list of target processes."""
        return self._targets.copy()
