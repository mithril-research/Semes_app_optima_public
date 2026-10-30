"""Compress large trend responses only; keep other API/stream behavior intact."""
from starlette.middleware.gzip import GZipMiddleware


class TrendResponseCompression:
    paths = frozenset(('/api/scatter/differences', '/api/scatter/step-times'))

    def __init__(self, app):
        self.app = app
        # Level 1 favors first-query CPU latency over maximum compression.
        self.compressed = GZipMiddleware(app, minimum_size=2048, compresslevel=1)

    async def __call__(self, scope, receive, send):
        app = self.compressed if scope['type'] == 'http' and scope.get('path') in self.paths else self.app
        await app(scope, receive, send)
