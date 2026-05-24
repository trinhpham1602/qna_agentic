"""Parallel multi-agent crawl với Firecrawl streaming.

Module này implement plan trong PLAN_PARALLEL_CRAWL_AGENT.md:
- Nhiều CrawlAgent stream song song từ home URL qua Firecrawl watcher
- JudgeConsumer chấm content theo từng page, fire early-answer khi đủ match
- BackgroundIngest tiếp tục lưu phần còn lại vào pgvector
- CacheChecker check trước khi crawl: URL đã crawl <1h + sim đủ thì dùng DB luôn
"""

from vietjet.crawl_parallel.coordinator import CrawlCoordinator, Event
from vietjet.crawl_parallel.frontier import URLFrontier

__all__ = ["CrawlCoordinator", "Event", "URLFrontier"]
