"""Process-level resource observations for data preparation reports."""
import resource
import sys
import time


def resource_observation(started):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {'elapsed_seconds': time.perf_counter() - started,
            'peak_process_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024),
            'rss_scope': 'process lifetime high-water mark; excludes child processes/Spark JVM; DuckDB memory_limit is not a total RSS cap'}
