from typing import List
from bindings import native_invoke_accelergy

from io import StringIO
import logging
import sys


def invoke_accelergy(input_files: List[str], out_prefix: str,
                     out_dir: str, log_level=logging.INFO):
    logger = logging.getLogger(__name__ + '.'
                               + invoke_accelergy.__name__)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = captured_stdout = StringIO()
        sys.stderr = captured_stderr = StringIO()
        native_invoke_accelergy(input_files, out_prefix, out_dir)
    finally:
        sys.stdout = old_stderr
        sys.stderr = old_stderr
    if captured_stdout.getvalue():
        logger.info(captured_stdout.getvalue())
    if captured_stderr.getvalue():
        logger.error(captured_stderr.getvalue())
