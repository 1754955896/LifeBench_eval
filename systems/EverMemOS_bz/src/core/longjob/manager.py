"""
Long job manager implementation.
长任务管理器实现。
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.longjob.interfaces import LongJobInterface, LongJobStatus
from core.di import service, get_beans_by_type
import os


@service(name="long_job_manager", primary=True)
class LongJobManager:
    """
    长任务管理器
    负责管理多个长任务的启动、停止、监控等功能
    """

    def __init__(self):
        """
        初始化长任务管理器
        从环境变量读取配置
        """
        # 从环境变量读取配置
        self.max_concurrent_jobs = int(os.getenv('LONGJOB_MAX_CONCURRENT_JOBS', '10'))
        self.auto_discover = (
            os.getenv('LONGJOB_AUTO_DISCOVER', 'true').lower() == 'true'
        )
        self.auto_start_mode = os.getenv('LONGJOB_AUTO_START_MODE', 'all').lower()
        self.job_whitelist = self._parse_job_list(
            os.getenv('LONGJOB_JOB_WHITELIST', '')
        )
        self.job_blacklist = self._parse_job_list(
            os.getenv('LONGJOB_JOB_BLACKLIST', '')
        )
        self.startup_timeout = float(os.getenv('LONGJOB_STARTUP_TIMEOUT', '60.0'))
        self.shutdown_timeout = float(os.getenv('LONGJOB_SHUTDOWN_TIMEOUT', '30.0'))
        self.wait_for_current_task = (
            os.getenv('LONGJOB_WAIT_FOR_CURRENT_TASK', 'true').lower() == 'true'
        )
        self.log_level = os.getenv('LONGJOB_LOG_LEVEL', 'INFO')
        self.log_startup_details = (
            os.getenv('LONGJOB_LOG_STARTUP_DETAILS', 'true').lower() == 'true'
        )
        self.log_job_lifecycle = (
            os.getenv('LONGJOB_LOG_JOB_LIFECYCLE', 'true').lower() == 'true'
        )

        self.logger = logging.getLogger(__name__)

        # 设置日志级别
        if hasattr(logging, self.log_level):
            self.logger.setLevel(getattr(logging, self.log_level))

        # 任务存储
        self._jobs: Dict[str, LongJobInterface] = {}
        self._job_tasks: Dict[str, asyncio.Task] = {}

        # 管理器状态
        self._is_shutdown = False
        self._shutdown_event = asyncio.Event()

        # 统计信息
        self.stats = {
            'total_jobs_started': 0,
            'total_jobs_stopped': 0,
            'total_jobs_failed': 0,
            'manager_start_time': datetime.now(),
        }

        self.logger.info(
            "LongJobManager initialized with max_concurrent_jobs=%d",
            self.max_concurrent_jobs,
        )

        # 自动发现标志
        self._auto_discovered = False

    def _parse_job_list(self, job_list_str: str) -> List[str]:
        """
        解析任务列表字符串

        Args:
            job_list_str: 逗号分隔的任务ID字符串

        Returns:
            List[str]: 任务ID列表
        """
        if not job_list_str:
            return []
        return [job.strip() for job in job_list_str.split(',') if job.strip()]

    def should_start_job(self, job_id: str) -> bool:
        """
        判断是否应该启动指定的任务

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否应该启动
        """
        if self.auto_start_mode == 'none':
            return False
        elif self.auto_start_mode == 'all':
            return True
        elif self.auto_start_mode == 'whitelist':
            return job_id in self.job_whitelist
        elif self.auto_start_mode == 'blacklist':
            return job_id not in self.job_blacklist
        else:
            return True

    async def add_job(self, job: LongJobInterface) -> bool:
        """
        添加一个长任务到管理器

        Args:
            job: 要添加的长任务实例

        Returns:
            bool: 是否添加成功
        """
        if self._is_shutdown:
            self.logger.warning(
                "Cannot add job %s: manager is shutting down", job.job_id
            )
            return False

        if job.job_id in self._jobs:
            self.logger.warning("Job %s already exists in manager", job.job_id)
            return False

        if len(self._jobs) >= self.max_concurrent_jobs:
            self.logger.warning(
                "Cannot add job %s: maximum concurrent jobs (%d) reached",
                job.job_id,
                self.max_concurrent_jobs,
            )
            return False

        self._jobs[job.job_id] = job
        self.logger.info("Job %s added to manager", job.job_id)
        return True

    async def discover_and_add_jobs(self) -> Dict[str, bool]:
        """
        自动发现并添加所有 LongJobInterface 实现

        Returns:
            Dict[str, bool]: 每个任务的添加结果
        """
        if self._auto_discovered:
            self.logger.info("Jobs already auto-discovered, skipping")
            return {}

        self.logger.info("Starting auto-discovery of LongJobInterface implementations")

        try:
            # 从 DI 容器中获取所有 LongJobInterface 实现
            job_implementations = get_beans_by_type(LongJobInterface)

            results = {}
            for job in job_implementations:
                # 跳过 LongJobManager 自身
                if isinstance(job, LongJobManager):
                    continue

                try:
                    success = await self.add_job(job)
                    results[job.job_id] = success
                    if success:
                        self.logger.info(
                            "Auto-discovered and added job: %s", job.job_id
                        )
                    else:
                        self.logger.warning(
                            "Failed to add auto-discovered job: %s", job.job_id
                        )
                except Exception as e:
                    self.logger.error(
                        "Error adding auto-discovered job %s: %s",
                        getattr(job, 'job_id', 'unknown'),
                        str(e),
                    )
                    results[getattr(job, 'job_id', 'unknown')] = False

            self._auto_discovered = True
            self.logger.info(
                "Auto-discovery completed. Found %d job implementations", len(results)
            )
            return results

        except Exception as e:
            self.logger.error(
                "Error during job auto-discovery: %s", str(e), exc_info=True
            )
            return {}

    async def default_start(
        self, auto_discover: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        默认启动流程：自动发现并启动环境变量配置允许的长任务

        Args:
            auto_discover: 是否自动发现任务（None则使用环境变量设置）

        Returns:
            Dict[str, Any]: 启动结果统计
        """
        if self.log_startup_details:
            config_info = {
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'auto_discover': self.auto_discover,
                'auto_start_mode': self.auto_start_mode,
                'job_whitelist': self.job_whitelist,
                'job_blacklist': self.job_blacklist,
            }
            self.logger.info(
                "Starting default startup process with config: %s", config_info
            )
        else:
            self.logger.info("Starting default startup process")

        results = {
            'discovered_jobs': {},
            'started_jobs': {},
            'filtered_jobs': {},
            'total_discovered': 0,
            'total_filtered': 0,
            'total_started': 0,
            'errors': [],
        }

        try:
            # 自动发现任务
            should_auto_discover = (
                auto_discover if auto_discover is not None else self.auto_discover
            )
            if should_auto_discover:
                discovered = await self.discover_and_add_jobs()
                results['discovered_jobs'] = discovered
                results['total_discovered'] = len(discovered)

            # 根据环境变量配置过滤要启动的任务
            jobs_to_start = {}
            for job_id, job in self._jobs.items():
                if self.should_start_job(job_id):
                    jobs_to_start[job_id] = job
                else:
                    results['filtered_jobs'][job_id] = 'excluded_by_config'
                    if self.log_startup_details:
                        self.logger.info("Job %s excluded by configuration", job_id)

            results['total_filtered'] = len(results['filtered_jobs'])

            # 启动过滤后的任务
            if jobs_to_start:
                started = {}
                for job_id in jobs_to_start:
                    try:
                        success = await self.start_job(job_id)
                        started[job_id] = success
                        if success and self.log_job_lifecycle:
                            self.logger.info("Successfully started job: %s", job_id)
                    except Exception as e:
                        started[job_id] = False
                        error_msg = f"Failed to start job {job_id}: {str(e)}"
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)

                results['started_jobs'] = started
                results['total_started'] = sum(
                    1 for success in started.values() if success
                )
            else:
                self.logger.warning("No jobs found to start after filtering")

            if self.log_startup_details:
                self.logger.info(
                    "Default startup completed: discovered=%d, filtered=%d, started=%d",
                    results['total_discovered'],
                    results['total_filtered'],
                    results['total_started'],
                )
            else:
                self.logger.info(
                    "Default startup completed: started=%d jobs",
                    results['total_started'],
                )

        except Exception as e:
            error_msg = f"Error during default startup: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            results['errors'].append(error_msg)

        return results

    async def start_job(self, job_id: str) -> bool:
        """
        启动指定的长任务

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否启动成功
        """
        if self._is_shutdown:
            self.logger.warning("Cannot start job %s: manager is shutting down", job_id)
            return False

        if job_id not in self._jobs:
            self.logger.error("Job %s not found in manager", job_id)
            return False

        job = self._jobs[job_id]

        if job.is_running():
            self.logger.warning("Job %s is already running", job_id)
            return False

        try:
            # 创建任务并启动
            task = asyncio.create_task(self._run_job_with_monitoring(job))
            self._job_tasks[job_id] = task

            self.stats['total_jobs_started'] += 1
            self.logger.info("Job %s started successfully", job_id)
            return True

        except Exception as e:
            self.stats['total_jobs_failed'] += 1
            self.logger.error(
                "Failed to start job %s: %s", job_id, str(e), exc_info=True
            )
            return False

    async def stop_job(
        self, job_id: str, timeout: float = 30.0, wait_for_current_task: bool = True
    ) -> bool:
        """
        停止指定的长任务

        Args:
            job_id: 任务ID
            timeout: 停止超时时间（秒）
            wait_for_current_task: 是否等待当前任务完成

        Returns:
            bool: 是否停止成功
        """
        if job_id not in self._jobs:
            self.logger.error("Job %s not found in manager", job_id)
            return False

        job = self._jobs[job_id]

        if not job.is_running():
            self.logger.warning("Job %s is not running", job_id)
            return True

        try:
            # 停止任务（支持优雅停机）
            if (
                hasattr(job, 'shutdown')
                and 'wait_for_current_task' in job.shutdown.__code__.co_varnames
            ):
                await asyncio.wait_for(
                    job.shutdown(
                        timeout=timeout, wait_for_current_task=wait_for_current_task
                    ),
                    timeout=timeout + 5.0,  # 给一点额外时间
                )
            else:
                await asyncio.wait_for(job.shutdown(), timeout=timeout)

            # 等待任务完成
            if job_id in self._job_tasks:
                task = self._job_tasks[job_id]
                if not task.done():
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except asyncio.TimeoutError:
                        self.logger.warning(
                            "Task for job %s did not complete, cancelling", job_id
                        )
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                del self._job_tasks[job_id]

            self.stats['total_jobs_stopped'] += 1
            self.logger.info("Job %s stopped successfully", job_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to stop job %s: %s", job_id, str(e), exc_info=True
            )
            return False

    async def remove_job(self, job_id: str) -> bool:
        """
        从管理器中移除指定的长任务

        Args:
            job_id: 任务ID

        Returns:
            bool: 是否移除成功
        """
        if job_id not in self._jobs:
            self.logger.error("Job %s not found in manager", job_id)
            return False

        job = self._jobs[job_id]

        # 如果任务正在运行，先停止它
        if job.is_running():
            success = await self.stop_job(job_id)
            if not success:
                self.logger.error("Failed to stop job %s before removal", job_id)
                return False

        # 移除任务
        del self._jobs[job_id]
        self.logger.info("Job %s removed from manager", job_id)
        return True

    async def start_all_jobs(self) -> Dict[str, bool]:
        """
        启动所有未运行的任务

        Returns:
            Dict[str, bool]: 每个任务的启动结果
        """
        results = {}

        for job_id, job in self._jobs.items():
            if not job.is_running():
                results[job_id] = await self.start_job(job_id)
            else:
                results[job_id] = True  # 已经在运行

        return results

    async def stop_all_jobs(self, timeout: float = 30.0) -> Dict[str, bool]:
        """
        停止所有运行中的任务

        Args:
            timeout: 每个任务的停止超时时间（秒）

        Returns:
            Dict[str, bool]: 每个任务的停止结果
        """
        results = {}

        # 并发停止所有任务
        stop_tasks = []
        running_jobs = []

        for job_id, job in self._jobs.items():
            if job.is_running():
                stop_tasks.append(self.stop_job(job_id, timeout))
                running_jobs.append(job_id)
            else:
                results[job_id] = True  # 已经停止

        if stop_tasks:
            stop_results = await asyncio.gather(*stop_tasks, return_exceptions=True)

            for job_id, result in zip(running_jobs, stop_results):
                if isinstance(result, Exception):
                    self.logger.error(
                        "Exception stopping job %s: %s", job_id, str(result)
                    )
                    results[job_id] = False
                else:
                    results[job_id] = result

        return results

    async def shutdown(self, timeout: float = 60.0) -> None:
        """
        关闭管理器，停止所有任务

        Args:
            timeout: 总的关闭超时时间（秒）
        """
        if self._is_shutdown:
            self.logger.warning("Manager is already shutting down")
            return

        self.logger.info("Starting manager shutdown")
        self._is_shutdown = True

        try:
            # 停止所有任务
            await asyncio.wait_for(
                self.stop_all_jobs(timeout=timeout / 2), timeout=timeout
            )

        except asyncio.TimeoutError:
            self.logger.warning(
                "Manager shutdown timeout, some jobs may not have stopped gracefully"
            )

        except Exception as e:
            self.logger.error(
                "Error during manager shutdown: %s", str(e), exc_info=True
            )

        finally:
            # 取消所有剩余的任务
            for job_id, task in self._job_tasks.items():
                if not task.done():
                    self.logger.warning("Cancelling task for job %s", job_id)
                    task.cancel()

            # 等待所有任务取消完成
            if self._job_tasks:
                await asyncio.gather(*self._job_tasks.values(), return_exceptions=True)

            self._job_tasks.clear()
            self._shutdown_event.set()

            self.logger.info("Manager shutdown completed")

    def get_job_status(self, job_id: str) -> Optional[LongJobStatus]:
        """
        获取指定任务的状态

        Args:
            job_id: 任务ID

        Returns:
            Optional[LongJobStatus]: 任务状态，如果任务不存在返回None
        """
        if job_id not in self._jobs:
            return None

        return self._jobs[job_id].get_status()

    def get_all_jobs_status(self) -> Dict[str, LongJobStatus]:
        """
        获取所有任务的状态

        Returns:
            Dict[str, LongJobStatus]: 任务ID到状态的映射
        """
        return {job_id: job.get_status() for job_id, job in self._jobs.items()}

    def get_running_jobs(self) -> List[str]:
        """
        获取所有运行中的任务ID列表

        Returns:
            List[str]: 运行中的任务ID列表
        """
        return [job_id for job_id, job in self._jobs.items() if job.is_running()]

    def get_job_stats(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        获取指定任务的统计信息

        Args:
            job_id: 任务ID

        Returns:
            Optional[Dict[str, Any]]: 任务统计信息，如果任务不存在或不支持统计返回None
        """
        if job_id not in self._jobs:
            return None

        job = self._jobs[job_id]

        # 检查任务是否支持统计信息
        if hasattr(job, 'get_stats'):
            return job.get_stats()

        return {
            'job_id': job_id,
            'status': job.get_status().value,
            'is_running': job.is_running(),
        }

    def get_manager_stats(self) -> Dict[str, Any]:
        """
        获取管理器统计信息

        Returns:
            Dict[str, Any]: 管理器统计信息
        """
        stats = self.stats.copy()
        stats.update(
            {
                'total_jobs': len(self._jobs),
                'running_jobs': len(self.get_running_jobs()),
                'is_shutdown': self._is_shutdown,
                'uptime': (
                    datetime.now() - stats['manager_start_time']
                ).total_seconds(),
            }
        )

        return stats

    async def _run_job_with_monitoring(self, job: LongJobInterface) -> None:
        """
        运行任务并进行监控

        Args:
            job: 要运行的任务
        """
        job_id = job.job_id

        try:
            self.logger.info("Starting monitoring for job %s", job_id)

            # 启动任务
            await job.start()

            # 监控任务状态，直到任务停止或管理器关闭
            while not self._is_shutdown and job.is_running():
                await asyncio.sleep(1.0)

            # 如果管理器正在关闭，确保任务也停止
            if self._is_shutdown and job.is_running():
                self.logger.info("Shutting down job %s due to manager shutdown", job_id)
                await job.shutdown()

        except Exception as e:
            self.stats['total_jobs_failed'] += 1
            self.logger.error("Job %s failed: %s", job_id, str(e), exc_info=True)

            # 尝试清理任务
            try:
                if job.is_running():
                    await job.shutdown()
            except Exception as cleanup_error:
                self.logger.error(
                    "Error during cleanup of failed job %s: %s",
                    job_id,
                    str(cleanup_error),
                    exc_info=True,
                )

        finally:
            self.logger.info("Monitoring ended for job %s", job_id)

    def __len__(self) -> int:
        """返回管理器中的任务数量"""
        return len(self._jobs)

    def __contains__(self, job_id: str) -> bool:
        """检查任务是否存在于管理器中"""
        return job_id in self._jobs
