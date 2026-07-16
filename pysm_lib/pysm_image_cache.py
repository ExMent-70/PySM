"""Public PySM API for reusable GUI image caching.

The API provides metadata-based source invalidation, a byte-bounded memory
LRU, atomic persistent derivatives and asynchronous ``QImage`` loading.
``QPixmap`` creation remains the responsibility of the GUI thread.

Typical setup::

    from pysm_lib.pysm_image_cache import ImageRequest, QtImageCache

    cache = QtImageCache(cache_root, memory_limit_bytes=256 * 1024 * 1024)
    request = ImageRequest(photo_path, (320, 240), mode="fill")
    key = cache.key_for(request)
"""

from .gui.image_cache import (
    AsyncImageLoader,
    AsyncImageLoaderStats,
    AsyncImageResult,
    ByteLRUCache,
    CacheStats,
    DiskImageCache,
    ImageCache,
    ImageCacheKey,
    ImageRequest,
    QtImageCache,
    QtImageCacheStats,
)

__all__ = [
    "AsyncImageLoader",
    "AsyncImageLoaderStats",
    "AsyncImageResult",
    "ByteLRUCache",
    "CacheStats",
    "DiskImageCache",
    "ImageCache",
    "ImageCacheKey",
    "ImageRequest",
    "QtImageCache",
    "QtImageCacheStats",
]
