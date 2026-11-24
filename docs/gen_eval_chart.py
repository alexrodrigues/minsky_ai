"""Regenerate docs/eval-chart-{light,dark}.svg from real `minsky eval` numbers.

Update ROWS below after re-running `minsky eval` / `minsky eval --multi-query`,
then run this script from the repo root: `python3 docs/gen_eval_chart.py`.
Colors are the project's validated accessible palette (see the dataviz method
these numbers were charted with); keep light/dark in sync by editing both
branches of `build()` together.
"""

import math
import os

ROWS = [
    ("hit@1", 0.71, 0.67),
    ("hit@3", 0.88, 0.75),
    ("hit@5", 0.92, 0.92),
    ("hit@10", 0.96, 0.96),
    ("MRR", 0.784, 0.741),
    ("nDCG@3", 0.784, 0.692),
]

W, H = 640, 400
CHART_LEFT = 130
CHART_RIGHT = 560
ROW_Y0 = 96
ROW_DY = 42
AXIS_Y = ROW_Y0 + ROW_DY * (len(ROWS) - 1) + 28
FOOTER_Y = AXIS_Y + 38
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
HERE = os.path.dirname(os.path.abspath(__file__))


def x_of(v):
    return CHART_LEFT + v * (CHART_RIGHT - CHART_LEFT)


def build(mode: str) -> str:
    if mode == "light":
        surface = "#fcfcfb"
        primary = "#0b0b0b"
        secondary = "#52514e"
        muted = "#898781"
        gridline = "#e1e0d9"
        baseline = "#c3c2b7"
        blue = "#2a78d6"
        gray = "#898781"
        delta_down = "#d03b3b"
        border = "rgba(11,11,11,0.10)"
    else:
        surface = "#1a1a19"
        primary = "#ffffff"
        secondary = "#c3c2b7"
        muted = "#898781"
        gridline = "#2c2c2a"
        baseline = "#383835"
        blue = "#3987e5"
        gray = "#9b9992"
        delta_down = "#e66767"
        border = "rgba(255,255,255,0.10)"

    parts = []
    parts.append(
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="{FONT}">'
    )
    parts.append(
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="10" '
        f'fill="{surface}" stroke="{border}" stroke-width="1"/>'
    )

    parts.append(
        f'<text x="24" y="32" font-size="16" font-weight="600" fill="{primary}">'
        f"Measured impact of --multi-query</text>"
    )
    parts.append(
        f'<text x="24" y="50" font-size="11.5" fill="{secondary}">'
        f"minsky_ai's own 24-case evalset.toml, 2026-11-24 "
        f"&#8226; axis: score 0&#8211;1</text>"
    )

    leg_y = 68
    parts.append(f'<circle cx="28" cy="{leg_y - 4}" r="5" fill="{blue}"/>')
    parts.append(
        f'<text x="40" y="{leg_y}" font-size="11.5" fill="{secondary}">'
        f"vector only (default)</text>"
    )
    parts.append(f'<circle cx="210" cy="{leg_y - 4}" r="5" fill="{gray}"/>')
    parts.append(
        f'<text x="222" y="{leg_y}" font-size="11.5" fill="{secondary}">'
        f"--multi-query (RRF fusion)</text>"
    )

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gx = x_of(frac)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{ROW_Y0 - 16}" x2="{gx:.1f}" y2="{AXIS_Y}" '
            f'stroke="{gridline}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{AXIS_Y + 16}" font-size="10.5" fill="{muted}" '
            f'text-anchor="middle">{frac:.2f}</text>'
        )
    parts.append(
        f'<line x1="{CHART_LEFT}" y1="{AXIS_Y}" x2="{CHART_RIGHT}" y2="{AXIS_Y}" '
        f'stroke="{baseline}" stroke-width="1"/>'
    )

    for i, (label, v_vec, v_multi) in enumerate(ROWS):
        y = ROW_Y0 + i * ROW_DY
        parts.append(
            f'<text x="{CHART_LEFT - 14}" y="{y + 4}" font-size="12.5" fill="{secondary}" '
            f'text-anchor="end">{label}</text>'
        )

        tied = math.isclose(v_vec, v_multi, abs_tol=1e-9)
        xv = x_of(v_vec)
        xm = x_of(v_multi)

        if tied:
            # Stack vertically so both marks stay visible without faking an x-gap.
            parts.append(
                f'<line x1="{xv:.1f}" y1="{y - 6:.1f}" x2="{xv:.1f}" y2="{y + 6:.1f}" '
                f'stroke="{baseline}" stroke-width="2" stroke-linecap="round"/>'
            )
            parts.append(
                f'<circle cx="{xv:.1f}" cy="{y - 6:.1f}" r="6" fill="{gray}" '
                f'stroke="{surface}" stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{xv:.1f}" cy="{y + 6:.1f}" r="6" fill="{blue}" '
                f'stroke="{surface}" stroke-width="2"/>'
            )
            parts.append(
                f'<text x="{xv + 14:.1f}" y="{y + 4}" font-size="11.5" fill="{primary}">'
                f"{v_vec:.2f} (tied)</text>"
            )
        else:
            x0, x1 = min(xv, xm), max(xv, xm)
            parts.append(
                f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" '
                f'stroke="{baseline}" stroke-width="2" stroke-linecap="round"/>'
            )
            parts.append(
                f'<circle cx="{xm:.1f}" cy="{y}" r="6" fill="{gray}" '
                f'stroke="{surface}" stroke-width="2"/>'
            )
            parts.append(
                f'<circle cx="{xv:.1f}" cy="{y}" r="6" fill="{blue}" '
                f'stroke="{surface}" stroke-width="2"/>'
            )

            label_above = y - 11
            label_below = y + 15
            if xv > xm:
                parts.append(
                    f'<text x="{xv:.1f}" y="{label_above}" font-size="11.5" fill="{primary}" '
                    f'text-anchor="middle">{v_vec:.2f}</text>'
                )
                parts.append(
                    f'<text x="{xm:.1f}" y="{label_below}" font-size="11.5" fill="{secondary}" '
                    f'text-anchor="middle">{v_multi:.2f}</text>'
                )
            else:
                parts.append(
                    f'<text x="{xv:.1f}" y="{label_below}" font-size="11.5" fill="{primary}" '
                    f'text-anchor="middle">{v_vec:.2f}</text>'
                )
                parts.append(
                    f'<text x="{xm:.1f}" y="{label_above}" font-size="11.5" fill="{secondary}" '
                    f'text-anchor="middle">{v_multi:.2f}</text>'
                )

            delta = v_multi - v_vec
            parts.append(
                f'<text x="{CHART_RIGHT + 16}" y="{y + 4}" font-size="11.5" '
                f'fill="{delta_down}" text-anchor="start">{delta:+.3f}</text>'
            )

    parts.append(
        f'<text x="24" y="{FOOTER_Y}" font-size="10.5" fill="{muted}">'
        f"hit@k shown as a fraction (0.71 = 71%) to share one axis with MRR/nDCG.</text>"
    )
    parts.append(
        f'<text x="24" y="{FOOTER_Y + 14}" font-size="10.5" fill="{muted}">'
        f"Right column: --multi-query's delta vs the default.</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    with open(os.path.join(HERE, "eval-chart-light.svg"), "w") as f:
        f.write(build("light"))
    with open(os.path.join(HERE, "eval-chart-dark.svg"), "w") as f:
        f.write(build("dark"))
    print("wrote docs/eval-chart-light.svg and docs/eval-chart-dark.svg")
