"""Progress tracking utilities for long-running operations."""

from __future__ import annotations

from time import perf_counter
from typing import Optional

from loguru import logger


def _format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Examples:
        45 seconds -> "45s"
        125 seconds -> "2m 5s"
        3725 seconds -> "1h 2m"
        90125 seconds -> "1d 1h 2m"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    minutes = int(seconds / 60)
    if minutes < 60:
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
    
    hours = int(minutes / 60)
    mins = int(minutes % 60)
    if hours < 24:
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
    
    days = int(hours / 24)
    hrs = int(hours % 24)
    return f"{days}d {hrs}h" if hrs > 0 else f"{days}d"


class ProgressTracker:
    """
    Tracks progress with stages, percentages, and timing.
    
    Features:
    - Stage-based tracking (e.g., "Stage 1/3: Chunking documents")
    - Percentage milestones (10%, 20%, 30%... 100%)
    - Item counts (processed/total)
    - Timing: elapsed time, ETA, items/sec
    - Adaptive logging (only at percentage thresholds)
    
    Example:
        progress = ProgressTracker(
            total=10000,
            stage_name="Processing documents",
            total_stages=3,
            current_stage=1
        )
        
        for item in items:
            # ... process item ...
            progress.update()
    """
    
    def __init__(
        self,
        total: int,
        stage_name: str,
        total_stages: int = 1,
        current_stage: int = 1,
        log_interval: int = 10,
    ):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            stage_name: Name of the current stage (e.g., "Processing documents")
            total_stages: Total number of stages in the operation
            current_stage: Current stage number (1-indexed)
            log_interval: Percentage interval for logging (default: 10 for 10%, 20%, etc.)
        """
        self.total = max(total, 1)  # Avoid division by zero
        self.current = 0
        self.stage_name = stage_name
        self.total_stages = total_stages
        self.current_stage = current_stage
        self.log_interval = log_interval
        self.start_time = perf_counter()
        self.last_logged_percentage = -log_interval
        
    def update(self, count: int = 1) -> None:
        """
        Update progress and log at percentage milestones.
        
        Args:
            count: Number of items processed in this update (default: 1)
        """
        self.current += count
        percentage = int((self.current / self.total) * 100)
        
        # Log only at configured intervals (10%, 20%, 30%... 100%)
        if percentage >= self.last_logged_percentage + self.log_interval:
            self._log_progress(percentage)
            self.last_logged_percentage = percentage
    
    def force_log(self) -> None:
        """Force log current progress regardless of percentage threshold."""
        percentage = int((self.current / self.total) * 100)
        self._log_progress(percentage)
        self.last_logged_percentage = percentage
            
    def _log_progress(self, percentage: int) -> None:
        """Log detailed progress information."""
        elapsed = perf_counter() - self.start_time
        items_per_sec = self.current / elapsed if elapsed > 0 else 0
        remaining_items = self.total - self.current
        eta_seconds = remaining_items / items_per_sec if items_per_sec > 0 else 0
        
        logger.info(
            "Stage {}/{}: {} | {}% complete | {}/{} items | "
            "Elapsed: {} | ETA: {} | Speed: {:.2f} items/sec",
            self.current_stage,
            self.total_stages,
            self.stage_name,
            percentage,
            self.current,
            self.total,
            _format_duration(elapsed),
            _format_duration(eta_seconds),
            items_per_sec,
        )
    
    def complete(self) -> None:
        """Mark progress as complete and log final statistics."""
        elapsed = perf_counter() - self.start_time
        items_per_sec = self.current / elapsed if elapsed > 0 else 0
        
        logger.info(
            "Stage {}/{}: {} | COMPLETE | {}/{} items | "
            "Total time: {} | Avg speed: {:.2f} items/sec",
            self.current_stage,
            self.total_stages,
            self.stage_name,
            self.current,
            self.total,
            _format_duration(elapsed),
            items_per_sec,
        )


class MultiStageProgress:
    """
    Wrapper for multi-stage operations with automatic stage transitions.
    
    Example:
        stages = MultiStageProgress([
            ("Loading documents", 1000),
            ("Chunking text", 5000),
            ("Extracting relations", 5000),
        ])
        
        # Stage 1
        for doc in docs:
            # ... process ...
            stages.update()
        
        stages.next_stage()
        
        # Stage 2
        for chunk in chunks:
            # ... process ...
            stages.update()
    """
    
    def __init__(self, stages: list[tuple[str, int]], log_interval: int = 10):
        """
        Initialize multi-stage progress tracker.
        
        Args:
            stages: List of (stage_name, total_items) tuples
            log_interval: Percentage interval for logging
        """
        self.stages = stages
        self.log_interval = log_interval
        self.current_stage_idx = 0
        self.total_stages = len(stages)
        self.tracker: Optional[ProgressTracker] = None
        self._start_stage()
    
    def _start_stage(self) -> None:
        """Start tracking the current stage."""
        if self.current_stage_idx < len(self.stages):
            stage_name, total = self.stages[self.current_stage_idx]
            self.tracker = ProgressTracker(
                total=total,
                stage_name=stage_name,
                total_stages=self.total_stages,
                current_stage=self.current_stage_idx + 1,
                log_interval=self.log_interval,
            )
    
    def update(self, count: int = 1) -> None:
        """Update progress for the current stage."""
        if self.tracker:
            self.tracker.update(count)
    
    def next_stage(self) -> None:
        """Complete current stage and move to the next one."""
        if self.tracker:
            self.tracker.complete()
        
        self.current_stage_idx += 1
        self._start_stage()
    
    def complete(self) -> None:
        """Complete the final stage."""
        if self.tracker:
            self.tracker.complete()

