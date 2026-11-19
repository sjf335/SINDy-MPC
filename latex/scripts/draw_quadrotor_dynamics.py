"""Original vector schematic. Design references inspected: Lee et al.,
CDC 2010 Fig. 1 (arXiv:1003.2005v1), Faessler et al., RA-L 2018 Fig. 2.
See research/dynamics_schematic_review.md. Rotor geometry is illustrative.
"""
from pathlib import Path
import json
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mplcache_dynamics"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import numpy as np
from PIL import Image, ImageOps

OUT = ROOT / "figures"
NAME = "section2a_quadrotor_dynamics"
WIDTH = 251.65 / 72
X_LIM, Y_LIM = (-1.78, 1.78), (-1.09, 1.38)
HEIGHT = WIDTH * (Y_LIM[1]-Y_LIM[0]) / (X_LIM[1]-X_LIM[0])
INK, MUTED = "#30363D", "#747C84"
FRAME, FORCE = "#35566F", "#8E3B32"
STRUCTURE, ARM_FILL, DISK_FILL = "#68747E", "#C8CFD5", "#F7F8F9"
BLADE_FILL, HUB_FILL = "#C9D0D6", "#6A747D"
DOLLAR = "$"
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8, "text.usetex": False,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "mathtext.cal": "Times New Roman:italic", "mathtext.fallback": None,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def rot(axis, degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R = rot("y", -14) @ rot("x", -9)
AZ, EL = np.deg2rad([20, 30])
P = np.array([[np.cos(AZ), -np.sin(AZ), 0],
              [-np.sin(EL)*np.sin(AZ), -np.sin(EL)*np.cos(AZ), np.cos(EL)]])
CENTER, ORIGIN = np.array([.40, .34]), np.array([-1.05, -.80])
F_BODY, G_WORLD = np.array([.70, -.04, 1.00]), np.array([0., 0., -1.35])


def project(body_points):
    return np.asarray(body_points) @ R.T @ P.T + CENTER


def arrow(ax, start, end, color=INK, lw=.8, head=6, ls="-", z=8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head,
                                linewidth=lw, color=color, linestyle=ls,
                                capstyle="round", joinstyle="miter",
                                shrinkA=0, shrinkB=0, zorder=z))


def label(ax, xy, text, color=INK, size=8):
    return ax.text(*xy, text, fontsize=size, color=color, ha="center", va="center",
                   zorder=15, bbox=dict(facecolor="white", edgecolor="none", pad=.15))


def polygon(ax, xyz, color, edge=MUTED, lw=.55, z=4):
    ax.add_patch(Polygon(project(xyz), closed=True, facecolor=color,
                         edgecolor=edge, linewidth=lw, joinstyle="round", zorder=z))


def draw():
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_aspect("equal")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.axis("off")
    # Position joins the two frame origins exactly.
    arrow(ax, ORIGIN, CENTER, color=MUTED, lw=.65, head=5, z=1)
    label(ax, (-.59, -.54), r"$\mathbf{p}$", size=8)

    # Flat engineering linework: rotor envelopes, paired blades, and hubs.
    motors = np.array([[.69, -.69, 0], [.69, .69, 0],
                       [-.69, .69, 0], [-.69, -.69, 0]])
    for motor in motors:
        unit = motor[:2]/np.linalg.norm(motor[:2])
        normal = .031*np.array([-unit[1], unit[0]])
        corners = np.array([normal, motor[:2]+normal, motor[:2]-normal, -normal])
        polygon(ax, np.c_[corners, np.zeros(4)], ARM_FILL, edge=STRUCTURE, lw=.48, z=2)
    theta = np.linspace(0, 2*np.pi, 160)
    for motor, blade_angle in zip(motors, [18, 108, 18, 108]):
        disk = motor + np.c_[.26*np.cos(theta), .26*np.sin(theta),
                             np.full(theta.size, .025)]
        polygon(ax, disk, DISK_FILL, edge=STRUCTURE, lw=.55, z=4)
        phase = np.deg2rad(blade_angle)
        direction = np.array([np.cos(phase), np.sin(phase)])
        normal = np.array([-direction[1], direction[0]])
        for side in (-1, 1):
            xy = (side*.105 + .115*np.cos(theta))[:, None]*direction
            xy += (.039*np.sin(theta))[:, None]*normal
            blade = motor + np.c_[xy, np.full(theta.size, .026)]
            polygon(ax, blade, BLADE_FILL, edge=STRUCTURE, lw=.35, z=5)
        hub = motor + np.c_[.034*np.cos(theta), .034*np.sin(theta),
                            np.full(theta.size, .027)]
        polygon(ax, hub, HUB_FILL, edge=STRUCTURE, lw=.35, z=6)
    chassis_xy = np.array([[-.13, -.07], [-.07, -.13], [.07, -.13], [.13, -.07],
                           [.13, .07], [.07, .13], [-.07, .13], [-.13, .07]])
    polygon(ax, np.c_[chassis_xy, np.full(8, .015)], "#DDE2E6",
            edge=STRUCTURE, lw=.55, z=5)

    # Dashed world frame and solid body frame remain distinct in grayscale.
    world_offsets = [(0.07, -.045), (-.07, -.035), (-.07, .06)]
    for i, (name, offset) in enumerate(zip("xyz", world_offsets)):
        tip = ORIGIN + .40*P[:, i]
        arrow(ax, ORIGIN, tip, color=MUTED, lw=.60, head=4.6, ls=(0, (3, 2)), z=9)
        label(ax, tip+offset, rf"{DOLLAR}{name}_{{\mathcal{{W}}}}{DOLLAR}", color=MUTED, size=8)
    ax.plot(*ORIGIN, "o", ms=2, color=MUTED, zorder=10)
    label(ax, ORIGIN+[-.16, .01], r"$\mathcal{W}$", color=MUTED, size=8)
    body_offsets = [(0.08, -.04), (-.08, -.07), (-.06, .08)]
    for i, (name, offset, length) in enumerate(zip("xyz", body_offsets, [1.13, 1.08, .88])):
        tip = project(length*np.eye(3)[i])
        arrow(ax, CENTER, tip, color=FRAME, lw=.78, head=5.5, z=9)
        label(ax, tip+offset, rf"{DOLLAR}{name}_{{\mathcal{{B}}}}{DOLLAR}", color=FRAME, size=8)
    label(ax, CENTER+[-.20, .02], r"$\mathcal{B}$", color=FRAME, size=8)

    # Commands are labels rather than physical thrust arrows.
    offsets = [(.15, .22), (.08, -.22), (-.33, -.06), (-.13, .24)]
    for i, (motor, offset) in enumerate(zip(motors, offsets), 1):
        label(ax, project(motor+[0, 0, .025])+offset, rf"$u_{i}$", size=8)

    # The drawn acceleration vectors are both in world coordinates.
    force_tip = project(F_BODY)
    gravity_tip = CENTER+P@G_WORLD
    arrow(ax, CENTER, force_tip, color=FORCE, lw=1.0, head=6.5, z=10)
    label(ax, force_tip+[.05, .10], r"$R(\mathbf{q})\mathbf{f}_{\mathcal{B}}$", color=FORCE, size=8)
    arrow(ax, CENTER, gravity_tip, color=FORCE, lw=1.0, head=6.5, z=10)
    label(ax, gravity_tip+[.11, -.07], r"$\mathbf{g}_{\mathcal{W}}$", color=FORCE, size=8)
    ax.plot(*CENTER, "o", ms=2.7, color=INK, zorder=12)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT/f"{NAME}.{ext}", dpi=600, facecolor="white")
    with Image.open(OUT/f"{NAME}.png") as im:
        ImageOps.grayscale(im).save(OUT/f"{NAME}_grayscale.png")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, collisions = [], []
    for text in ax.texts:
        box = text.get_window_extent(renderer)
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            outside.append(text.get_text())
    for i, first in enumerate(ax.texts):
        for second in ax.texts[i+1:]:
            if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                collisions.append([first.get_text(), second.get_text()])
    report = {
        "size_inches": [WIDTH, HEIGHT], "png_dpi": 600,
        "vector_outputs": [f"{NAME}.pdf", f"{NAME}.svg"],
        "style": "Muted navy coordinates, burgundy accelerations, neutral gray engineering linework.",
        "palette": {"body_frame": FRAME, "accelerations": FORCE,
                    "world_frame": MUTED, "structure": STRUCTURE,
                    "rotor_envelope": DISK_FILL, "blades": BLADE_FILL},
        "line_width_pt": {"world_axes": .60, "body_axes": .78, "accelerations": 1.0},
        "rotor_blades": "Schematic paired blades; their static orientation does not encode spin direction.",
        "design_references_viewed": [
            {"citation": "Lee et al., CDC 2010", "version": "arXiv:1003.2005v1", "figure": 1, "page": 2},
            {"citation": "Faessler et al., RA-L 2018", "figure": 2, "page": 2}],
        "rotation_determinant": float(np.linalg.det(R)),
        "rotation_orthogonality_error": float(np.linalg.norm(R.T@R-np.eye(3))),
        "projection_row_orthogonality_error": float(np.linalg.norm(P@P.T-np.eye(2))),
        "force_projection_error": float(np.linalg.norm(force_tip-CENTER-P@R@F_BODY)),
        "gravity_parallel_to_negative_world_z": bool(np.allclose(gravity_tip-CENTER, -1.35*P[:, 2])),
        "specific_force_not_assumed_parallel_to_body_z": bool(np.linalg.norm(np.cross(F_BODY, [0, 0, 1])) > 0),
        "world_and_body_axes_differ_by_line_style": True,
        "labels_outside_canvas": outside, "overlapping_label_boxes": collisions,
        "rotor_locations": "schematic; not a Pelican motor-channel geometry assertion",
    }
    (OUT/f"{NAME}_verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    plt.close(fig)
    if outside or collisions:
        raise RuntimeError(f"Label verification failed: outside={outside}, overlaps={collisions}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    draw()
