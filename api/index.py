"""Vercel Serverless Entrypoint for Jev BTC Trader."""

import sys
import os
from pathlib import Path

# Add root directory to sys.path so all imports work seamlessly in Vercel serverless environment
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server import DashboardRequestHandler


class handler(DashboardRequestHandler):
    """Vercel Serverless HTTP Request Handler.
    
    Inherits complete production-tested REST API, paper trading engine,
    multi-asset portfolio accounting, and client-side page routing from
    DashboardRequestHandler without code divergence.
    """
    pass
