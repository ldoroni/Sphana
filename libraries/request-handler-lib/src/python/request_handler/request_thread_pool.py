import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

class RequestThreadPool:
    __thread_pool: Optional[ThreadPoolExecutor] = None

    @staticmethod
    def init(max_workers: int):
        RequestThreadPool.__thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    @staticmethod
    def run_async(func, *args, **kwargs):
        if not RequestThreadPool.__thread_pool:
            raise RuntimeError("ThreadPool not initialized. Call ThreadPool.init() first.")

        loop = asyncio.get_event_loop()
        return loop.run_in_executor(RequestThreadPool.__thread_pool, lambda: func(*args, **kwargs))
