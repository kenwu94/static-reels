"""Allows `python -m staticreels ...` as an alternative to the `staticreels` command."""

import sys

from .cli import main

sys.exit(main())
