"""
Stub for missing silence_fd_output function.
This provides a context manager to suppress output during tests.
"""

import contextlib
import os
import sys


@contextlib.contextmanager 
def silence_fd_output():
    """Context manager to suppress stdout and stderr output."""
    # Save current stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    try:
        # Redirect to devnull
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
