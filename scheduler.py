"""
Periodic re-optimization scheduler for territory optimization.

Uses APScheduler to trigger optimization pipeline on a configurable
cron schedule, with on-demand trigger support.
"""
import logging
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class OptimizationScheduler:
    """Manages periodic re-optimization scheduling."""

    def __init__(self, config: Dict[str, Any], pipeline_runner: Optional[Callable] = None):
        """
        Initialize scheduler.

        Args:
            config: Scheduler configuration
            pipeline_runner: Callable that runs the optimization pipeline
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.cron_hour = config.get('cron_hour', 2)
        self.cron_minute = config.get('cron_minute', 0)
        self.max_concurrent = config.get('max_concurrent_jobs', 1)
        self.pipeline_runner = pipeline_runner
        self.scheduler = None
        self._running_jobs = 0
        self._lock = threading.Lock()
        self.last_run_result = None

    def start(self):
        """Start the scheduler."""
        if not self.enabled:
            logger.info("Scheduler is disabled in configuration")
            return

        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger

            self.scheduler = BackgroundScheduler()
            trigger = CronTrigger(hour=self.cron_hour, minute=self.cron_minute)

            self.scheduler.add_job(
                self._run_optimization,
                trigger=trigger,
                id='territory_optimization',
                name='Territory Optimization',
                replace_existing=True,
                max_instances=self.max_concurrent
            )

            self.scheduler.start()
            logger.info(f"Scheduler started - optimization runs daily at "
                         f"{self.cron_hour:02d}:{self.cron_minute:02d}")

        except ImportError:
            logger.warning("APScheduler not installed - scheduler disabled. "
                           "Install with: pip install apscheduler")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    def trigger_now(self, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Trigger an immediate optimization run.

        Args:
            parameters: Optional override parameters

        Returns:
            Pipeline result dictionary
        """
        logger.info("Manual optimization trigger requested")
        return self._run_optimization(parameters)

    def _run_optimization(self, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute optimization pipeline."""
        with self._lock:
            if self._running_jobs >= self.max_concurrent:
                msg = "Max concurrent jobs reached, skipping"
                logger.warning(msg)
                return {'status': 'SKIPPED', 'reason': msg}
            self._running_jobs += 1

        try:
            logger.info(f"Starting scheduled optimization at {datetime.now().isoformat()}")

            if self.pipeline_runner:
                result = self.pipeline_runner(parameters)
            else:
                # Import here to avoid circular imports
                from pipeline import OptimizationPipeline
                pipeline = OptimizationPipeline()
                result = pipeline.run(parameters)

            self.last_run_result = {
                'timestamp': datetime.now().isoformat(),
                'result': result
            }

            logger.info(f"Scheduled optimization completed: {result.get('status', 'UNKNOWN')}")
            return result

        except Exception as e:
            logger.error(f"Scheduled optimization failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}

        finally:
            with self._lock:
                self._running_jobs -= 1

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        status = {
            'enabled': self.enabled,
            'schedule': f"Daily at {self.cron_hour:02d}:{self.cron_minute:02d}",
            'running_jobs': self._running_jobs,
            'last_run': self.last_run_result
        }

        if self.scheduler:
            jobs = self.scheduler.get_jobs()
            status['next_run'] = str(jobs[0].next_run_time) if jobs else None

        return status
