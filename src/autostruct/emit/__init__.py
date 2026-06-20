"""Output emitters: Kaitai .ksy, HTML structure map, terminal summary."""

from .html import render_html
from .ksy import render_ksy
from .summary import render_summary

__all__ = ["render_ksy", "render_html", "render_summary"]
