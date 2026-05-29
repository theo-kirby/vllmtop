"""Curses colors matching btop's default theme, with graceful fallbacks.

Colors are taken verbatim from btop's built-in ``Default`` theme. On a
truecolor-capable terminal we redefine palette slots to the exact RGB via
``init_color``; on a 256-color terminal we map each hex to the nearest xterm
palette index; otherwise we fall back to the 8 basic colors.
"""

from __future__ import annotations

import curses

# --- color pair ids -------------------------------------------------------
PAIR_DEFAULT = 0
PAIR_TITLE = 1
PAIR_GREEN = 2
PAIR_YELLOW = 3
PAIR_RED = 4
PAIR_CYAN = 5
PAIR_DIM = 6  # normal label text (btop main_fg)
PAIR_MAGENTA = 7
PAIR_HI = 8  # highlight accent (btop hi_fg) — superscript panel numbers
PAIR_DIV = 9  # box outline / divider line
PAIR_INACTIVE = 10  # de-emphasised text
PAIR_BOX_GPU = 11
PAIR_BOX_THRU = 12
PAIR_BOX_REQ = 13
PAIR_BOX_LAT = 14
PAIR_BOX_CACHE = 15

# btop's value gradient (green -> yellow -> red) is rendered as a run of color
# pairs starting here, so graphs/meters can fade smoothly by position/value
# instead of snapping between three threshold colors.
GRAD_STEPS = 48
GRAD_PAIR_BASE = 16  # gradient pairs occupy [16, 16 + GRAD_STEPS)

# btop "Default" theme hex values, keyed by the role each pair plays here.
_PALETTE = {
    "title": "#ee",        # title
    "green": "#77ca9b",    # cpu_start  (good / low)
    "yellow": "#cbc06c",   # cpu_mid    (warn)
    "red": "#dc4c4c",      # cpu_end    (crit)
    "cyan": "#74e6fc",     # cached_mid
    "text": "#cc",         # main_fg
    "magenta": "#d9626d",  # used_mid
    "hi": "#b54040",       # hi_fg
    "div": "#30",          # div_line
    "inactive": "#40",     # inactive_fg
    "box_gpu": "#556d59",  # cpu_box
    "box_thru": "#6c6c4b", # mem_box
    "box_req": "#5c588d",  # net_box
    "box_lat": "#805252",  # proc_box
    "box_cache": "#556d59",
}

_PAIR_ROLE = {
    PAIR_TITLE: "title",
    PAIR_GREEN: "green",
    PAIR_YELLOW: "yellow",
    PAIR_RED: "red",
    PAIR_CYAN: "cyan",
    PAIR_DIM: "text",
    PAIR_MAGENTA: "magenta",
    PAIR_HI: "hi",
    PAIR_DIV: "div",
    PAIR_INACTIVE: "inactive",
    PAIR_BOX_GPU: "box_gpu",
    PAIR_BOX_THRU: "box_thru",
    PAIR_BOX_REQ: "box_req",
    PAIR_BOX_LAT: "box_lat",
    PAIR_BOX_CACHE: "box_cache",
}

# 8-color fallback for terminals without 256 colors.
_BASIC = {
    "title": curses.COLOR_WHITE,
    "text": curses.COLOR_WHITE,
    "inactive": curses.COLOR_WHITE,
    "div": curses.COLOR_WHITE,
    "green": curses.COLOR_GREEN,
    "yellow": curses.COLOR_YELLOW,
    "red": curses.COLOR_RED,
    "cyan": curses.COLOR_CYAN,
    "magenta": curses.COLOR_MAGENTA,
    "hi": curses.COLOR_RED,
    "box_gpu": curses.COLOR_GREEN,
    "box_thru": curses.COLOR_YELLOW,
    "box_req": curses.COLOR_BLUE,
    "box_lat": curses.COLOR_RED,
    "box_cache": curses.COLOR_CYAN,
}


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int],
          t: float) -> tuple[int, int, int]:
    return tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(3))  # type: ignore[return-value]


def _build_gradient(n: int) -> list[tuple[int, int, int]]:
    """`n` RGB steps fading green->yellow->red, mid color at the midpoint.

    Mirrors btop's cpu gradient (start ``#77ca9b``, mid ``#cbc06c``, end
    ``#dc4c4c``): the first half interpolates start->mid, the second mid->end.
    """
    start, mid, end = (0x77, 0xCA, 0x9B), (0xCB, 0xC0, 0x6C), (0xDC, 0x4C, 0x4C)
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        f = i / (n - 1) if n > 1 else 0.0
        if f <= 0.5:
            out.append(_lerp(start, mid, f / 0.5))
        else:
            out.append(_lerp(mid, end, (f - 0.5) / 0.5))
    return out


def _parse_hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 2:  # grayscale shorthand, e.g. "#cc" -> #cccccc
        v = int(h, 16)
        return v, v, v
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _scale(v: int) -> int:
    return round(v / 255 * 1000)


_CUBE = (0, 95, 135, 175, 215, 255)


def _to_256(r: int, g: int, b: int) -> int:
    """Nearest xterm-256 palette index for an RGB triple."""

    def cube(v: int) -> tuple[int, int]:
        i = min(range(6), key=lambda k: abs(_CUBE[k] - v))
        return i, _CUBE[i]

    ri, rv = cube(r)
    gi, gv = cube(g)
    bi, bv = cube(b)
    cube_idx = 16 + 36 * ri + 6 * gi + bi
    cube_err = (rv - r) ** 2 + (gv - g) ** 2 + (bv - b) ** 2

    gray = round((r + g + b) / 3)
    gi2 = min(23, max(0, round((gray - 8) / 10)))
    gv2 = 8 + 10 * gi2
    gray_err = sum((gv2 - c) ** 2 for c in (r, g, b))
    gray_idx = 232 + gi2

    return cube_idx if cube_err <= gray_err else gray_idx


class Theme:
    def __init__(self) -> None:
        self.has_color = False
        self._grad_pairs: list[int] = []

    def init(self) -> None:
        if not curses.has_colors():
            self.has_color = False
            return
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK

        colors = getattr(curses, "COLORS", 8)
        try:
            can_change = curses.can_change_color() and colors >= 256
        except curses.error:
            can_change = False

        fg: dict[str, int] = {}
        if can_change:
            slot = 16
            for name, hexv in _PALETTE.items():
                r, g, b = _parse_hex(hexv)
                try:
                    curses.init_color(slot, _scale(r), _scale(g), _scale(b))
                    fg[name] = slot
                    slot += 1
                except curses.error:
                    fg[name] = _BASIC[name]
        elif colors >= 256:
            fg = {n: _to_256(*_parse_hex(h)) for n, h in _PALETTE.items()}
        else:
            fg = dict(_BASIC)

        for pair_id, role in _PAIR_ROLE.items():
            try:
                curses.init_pair(pair_id, fg[role], bg)
            except curses.error:
                pass

        # btop-style value gradient. Allocate it all-or-nothing so the frac->
        # index mapping in grad_attr() stays aligned; a basic terminal keeps an
        # empty list and falls back to the three threshold colors.
        self._grad_pairs = []
        grad = _build_gradient(GRAD_STEPS)
        try:
            if can_change:
                base_slot = 16 + len(_PALETTE)
                for i, (r, g, b) in enumerate(grad):
                    curses.init_color(base_slot + i, _scale(r), _scale(g),
                                      _scale(b))
                    curses.init_pair(GRAD_PAIR_BASE + i, base_slot + i, bg)
                    self._grad_pairs.append(GRAD_PAIR_BASE + i)
            elif colors >= 256:
                for i, (r, g, b) in enumerate(grad):
                    curses.init_pair(GRAD_PAIR_BASE + i, _to_256(r, g, b), bg)
                    self._grad_pairs.append(GRAD_PAIR_BASE + i)
        except curses.error:
            self._grad_pairs = []

        self.has_color = True

    def attr(self, pair: int, bold: bool = False, dim: bool = False) -> int:
        a = curses.color_pair(pair) if self.has_color else curses.A_NORMAL
        if bold:
            a |= curses.A_BOLD
        if dim:
            a |= curses.A_DIM
        return a

    def grad_attr(self, frac: float, bold: bool = False) -> int:
        """Attribute for a point ``frac`` (0..1) along the green->red gradient.

        Used for btop-style positional coloring: a graph's vertical position or
        a meter cell's spot along its length. Falls back to the three threshold
        colors when no smooth gradient could be allocated.
        """
        if not self.has_color:
            return curses.A_BOLD if bold else curses.A_NORMAL
        pairs = self._grad_pairs
        if pairs:
            idx = max(0, min(len(pairs) - 1, round(frac * (len(pairs) - 1))))
            a = curses.color_pair(pairs[idx])
        else:
            pair = (PAIR_RED if frac >= 0.85 else
                    PAIR_YELLOW if frac >= 0.5 else PAIR_GREEN)
            a = curses.color_pair(pair)
        if bold:
            a |= curses.A_BOLD
        return a

    def threshold(self, value: float, warn: float, crit: float) -> int:
        """Pick the btop cpu-gradient green/yellow/red by a low-good threshold."""
        if value >= crit:
            return PAIR_RED
        if value >= warn:
            return PAIR_YELLOW
        return PAIR_GREEN
