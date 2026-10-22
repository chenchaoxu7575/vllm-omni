# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capacity checks before touching POSIX shared-memory pages."""

import os


class SharedMemoryCapacityError(MemoryError):
    """Insufficient tmpfs space; retrying a larger representation cannot help."""


def require_shared_memory(nbytes: int) -> None:
    """Fail before allocation/copy can SIGBUS; this is a check, not a reservation."""
    space = os.statvfs("/dev/shm")
    available = space.f_bavail * space.f_frsize
    if nbytes > available:
        raise SharedMemoryCapacityError(
            f"Insufficient /dev/shm space: required {nbytes} bytes, available {available} bytes. "
            "Free shared memory or increase its size before retrying."
        )
