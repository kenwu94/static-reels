"""Floor plans for the spectrogram and blueprint segments, drawn with Pillow as a line mask.

The apartment is always the same layout (bedroom west, bath and kitchen north,
living room south-east). Only the bedroom changes: as `plan.bedroom` goes from
1 to 0 the west wall thickens (drawn hatched) and swallows the room, until the
unit is a studio. A dashed person outline can lie in the bed or inside the wall.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

from .story import Plan
from .visuals import load_font


def _dashed(d: ImageDraw.ImageDraw, points: list[tuple[float, float]], dash: float = 9, gap: float = 7,
            width: int = 2, fill: int = 255) -> None:
    """Draw a closed polyline as dashes (Pillow has no dashed lines)."""
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]):
        seg = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            if on:
                a, b = pos / seg, (pos + step) / seg
                d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=fill, width=width)
            pos += step
            left -= step
            if left <= 1e-6:
                on = not on
                left = dash if on else gap


def _ellipse(cx: float, cy: float, rx: float, ry: float, k: int = 48) -> list[tuple[float, float]]:
    return [(cx + rx * math.cos(2 * math.pi * i / k), cy + ry * math.sin(2 * math.pi * i / k)) for i in range(k)]


def draw_plan(width: int, height: int, plan: Plan, font_path: Path) -> tuple[Image.Image, tuple[float, float]]:
    """Return (line mask, point next to the person outline). Lines are 255 on 0."""
    img = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(img)
    m = 0.07
    X0, Y0, X1, Y1 = width * m, height * m, width * (1 - m), height * (1 - m)
    IW, IH = X1 - X0, Y1 - Y0
    ext, inner = max(4, int(width * 0.008)), max(2, int(width * 0.004))
    BW = 0.38 * IW  # full bedroom width
    xp, yb = X0 + BW, Y0 + 0.62 * IH  # bedroom's east and south walls
    wall_in = X0 + (1 - plan.bedroom) * BW  # inner face of the (thickening) west wall
    door = 0.09 * IW
    small = load_font(str(font_path), max(12, int(height * 0.026)))

    def window(xa: float, xb: float, y: float) -> None:
        d.rectangle([xa, y - ext, xb, y + ext], fill=0)
        for yy in (y - ext, y - ext // 2, y + ext // 2, y + ext):
            d.line([(xa, yy), (xb, yy)], fill=255, width=1)

    def swing(hx: float, hy: float, r: float, start: int, end: int, leaf_to: tuple[float, float]) -> None:
        d.arc([hx - r, hy - r, hx + r, hy + r], start, end, fill=200, width=1)
        d.line([(hx, hy), leaf_to], fill=255, width=2)

    # Exterior walls, entrance door (south) and windows.
    d.rectangle([X0, Y0, X1, Y1], outline=255, width=ext)
    ex = X1 - 0.22 * IW
    d.rectangle([ex, Y1 - ext, ex + door, Y1 + ext], fill=0)
    swing(ex, Y1, door, 270, 360, (ex, Y1 - door))
    window(X0 + 0.48 * IW, X0 + 0.68 * IW, Y1)
    window(xp + 0.32 * IW, xp + 0.5 * IW, Y0)

    # The west wall swallowing the bedroom: hatched solid.
    hatch_w, hatch_h = int(wall_in - X0), int(yb - Y0)
    if hatch_w > 2:
        hatch = Image.new("L", (hatch_w, hatch_h), 0)
        hd = ImageDraw.Draw(hatch)
        for k in range(-hatch_h, hatch_w, 14):
            hd.line([(k, 0), (k + hatch_h, hatch_h)], fill=150, width=2)
        img.paste(hatch, (int(X0), int(Y0)), hatch)
        d.rectangle([X0, Y0, wall_in, yb], outline=255, width=inner)

    # Bedroom walls: east partition, south wall with a door (gone once it's a studio).
    d.line([(xp, Y0), (xp, yb)], fill=255, width=inner)
    if plan.bedroom > 0:
        d.line([(X0, yb), (xp - door - 10, yb)], fill=255, width=inner)
        d.line([(xp - 10, yb), (xp, yb)], fill=255, width=inner)
        swing(xp - 10, yb, door, 180, 270, (xp - 10, yb - door))
        if plan.bedroom > 0.4:
            window(wall_in + 0.25 * (xp - wall_in), wall_in + 0.65 * (xp - wall_in), Y0)
    else:
        d.line([(X0, yb), (xp, yb)], fill=255, width=inner)

    # Bath (with door) and kitchen counter along the north wall.
    xb, yk = xp + 0.25 * IW, Y0 + 0.35 * IH
    d.line([(xb, Y0), (xb, yk)], fill=255, width=inner)
    d.line([(xp, yk), (xp + 0.04 * IW, yk)], fill=255, width=inner)
    d.line([(xp + 0.04 * IW + door, yk), (xb, yk)], fill=255, width=inner)
    swing(xp + 0.04 * IW, yk, door, 270, 360, (xp + 0.04 * IW, yk - door))
    d.line([(xb + 0.04 * IW, Y0 + 0.08 * IH), (X1 - 0.03 * IW, Y0 + 0.08 * IH)], fill=220, width=2)

    # Bed (and nightstand) against the west wall, squeezed as the room shrinks.
    bed_w, bed_l = 0.27 * BW, 0.5 * (yb - Y0)
    bx, by = wall_in + 6, Y0 + 0.12 * (yb - Y0)
    if plan.bedroom > 0:
        bed_w = min(bed_w, xp - 6 - bx)
        d.rectangle([bx, by, bx + bed_w, by + bed_l], outline=210, width=2)
        d.rectangle([bx + 6, by + 6, bx + bed_w - 6, by + 0.16 * bed_l], outline=210, width=2)
        ns = 0.1 * BW
        if bx + bed_w + 8 + ns < xp - 4:
            d.rectangle([bx + bed_w + 8, by, bx + bed_w + 8 + ns, by + ns], outline=190, width=2)

    # Dashed person outline: lying in the bed, or inside the wall.
    anchor = (xp + 0.06 * IW, Y0 + 0.3 * IH)
    if plan.outline == "bed" and plan.bedroom > 0:
        cx, top = bx + bed_w / 2, by + 0.05 * bed_l
    elif plan.outline == "wall":
        cx, top = (X0 + max(wall_in, X0 + 0.27 * BW)) / 2, by + 0.05 * bed_l
    else:
        cx = None
    if cx is not None:
        pw = 0.27 * BW
        _dashed(d, _ellipse(cx, top + 0.09 * bed_l, 0.07 * bed_l * 0.8, 0.07 * bed_l), fill=235)
        _dashed(d, _ellipse(cx, top + 0.55 * bed_l, 0.32 * pw, 0.38 * bed_l), fill=235)
        anchor = (cx, top + 0.4 * bed_l)

    # Labels and the bedroom's dimension line.
    d.text(((xp + xb) / 2, (Y0 + yk) / 2), "BATH", font=small, anchor="mm", fill=200)
    d.text(((xb + X1) / 2, (Y0 + yk) / 2 + 0.04 * IH), "KITCHEN", font=small, anchor="mm", fill=200)
    d.text((xp + 0.38 * IW, yk + 0.38 * IH), "STUDIO" if plan.bedroom == 0 else "LIVING", font=small, anchor="mm", fill=200)
    if plan.bedroom > 0.45:
        d.text(((wall_in + xp) / 2, by + bed_l + 0.12 * (yb - Y0)), "BEDROOM", font=small, anchor="mm", fill=200)
    if plan.label and plan.bedroom > 0:
        y = yb - 0.07 * (yb - Y0)
        a, b = wall_in + 6, xp - 6
        d.line([(a, y), (b, y)], fill=255, width=2)
        for x in (a, b):
            d.line([(x, y - 9), (x, y + 9)], fill=255, width=2)
        big = load_font(str(font_path), max(14, int(height * 0.045)))
        d.text(((a + b) / 2, y - 8), plan.label, font=big, anchor="mb", fill=255)
    return img, anchor
