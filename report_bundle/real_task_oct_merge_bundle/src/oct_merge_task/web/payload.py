from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_preview_web_payload(preview_run_dir: str | Path, output_dir: str | Path) -> dict:
    preview_run_dir = Path(preview_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((preview_run_dir / "task_simulation_summary.json").read_text(encoding="utf-8"))
    preview_run = summary["preview_run"]
    stitched_bricks = preview_run_dir / "preview_run" / "stitched_bricks"
    if not stitched_bricks.exists():
        stitched_bricks = preview_run_dir / "stitched_bricks"

    case_dir = preview_run_dir.parent / "case"
    if not case_dir.exists():
        case_dir = preview_run_dir / "case"
    case_summary = json.loads((case_dir / "case_summary.json").read_text(encoding="utf-8"))
    preview_a = np.load(case_dir / "preview_volume_a.npy").astype(np.float32)
    preview_b = np.load(case_dir / "preview_volume_b.npy").astype(np.float32)

    slices = _build_slice_payload(stitched_bricks, preview_run["stitched_shape"])
    point_clouds = {
        "volume_a": _build_volume_point_cloud(preview_a, name="Volume A", base_color="#51d6ff"),
        "volume_b": _build_volume_point_cloud(preview_b, name="Volume B", base_color="#ff9a4d"),
        "stitched": _build_stitched_point_cloud(stitched_bricks, preview_run["stitched_shape"], preview_a, preview_b, int(round(preview_run["estimated_transform"]["tx"]))),
    }
    payload = {
        "preview_shape": summary["preview_shape"],
        "shape_a": case_summary["shape_a"],
        "shape_b": case_summary["shape_b"],
        "same_size_pair": case_summary["same_size_pair"],
        "overlap_voxels": case_summary["overlap_voxels"],
        "overlap_ratio": case_summary["overlap_ratio"],
        "stitched_shape": preview_run["stitched_shape"],
        "transform": preview_run["estimated_transform"],
        "brick_count": preview_run["brick_count"],
        "benchmark": summary["benchmark"],
        "slices": slices,
        "point_clouds": point_clouds,
    }
    (output_dir / "payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_slice_payload(stitched_bricks_dir: Path, stitched_shape: list[int]) -> dict:
    layout = json.loads((stitched_bricks_dir / "layout.json").read_text(encoding="utf-8"))
    plane_z = np.zeros((stitched_shape[0], stitched_shape[1]), dtype=np.float32)
    z_index = stitched_shape[2] // 2
    for task in layout["tasks"]:
        origin = task["origin"]
        shape = task["shape"]
        if not (origin[2] <= z_index < origin[2] + shape[2]):
            continue
        brick = np.load(stitched_bricks_dir / "bricks" / f"brick_{task['brick_id'][0]}_{task['brick_id'][1]}_{task['brick_id'][2]}.npy")
        local_index = z_index - origin[2]
        plane_z[
            origin[0] : origin[0] + shape[0],
            origin[1] : origin[1] + shape[1],
        ] = brick[:, :, local_index]
    return {
        "axial_mid": _encode_plane(plane_z),
    }


def _encode_plane(plane: np.ndarray) -> dict:
    plane = np.asarray(plane, dtype=np.float32)
    plane = plane - float(plane.min())
    max_value = float(plane.max())
    if max_value > 0.0:
        plane = plane / max_value
    image = (plane * 255.0).astype(np.uint8)
    return {
        "shape": list(image.shape),
        "data": image.flatten().tolist(),
    }


def _build_stitched_point_cloud(stitched_bricks_dir: Path, stitched_shape: list[int], preview_a: np.ndarray, preview_b: np.ndarray, tx: int) -> dict:
    layout = json.loads((stitched_bricks_dir / "layout.json").read_text(encoding="utf-8"))
    points = []
    values = []
    labels = []
    threshold = 0.35
    max_points = 12000
    for task in layout["tasks"]:
        brick = np.load(stitched_bricks_dir / "bricks" / f"brick_{task['brick_id'][0]}_{task['brick_id'][1]}_{task['brick_id'][2]}.npy")
        coords = np.argwhere(brick >= threshold)
        if len(coords) == 0:
            continue
        origin = np.array(task["origin"], dtype=np.int32)
        for coord in coords:
            point = coord + origin
            points.append(point.tolist())
            values.append(float(brick[tuple(coord)]))
            in_a = point[0] < preview_a.shape[0] and point[1] < preview_a.shape[1] and point[2] < preview_a.shape[2]
            in_b = tx <= point[0] < tx + preview_b.shape[0] and point[1] < preview_b.shape[1] and point[2] < preview_b.shape[2]
            if in_a and in_b:
                labels.append("Overlap")
            elif in_a:
                labels.append("A-only")
            else:
                labels.append("B-only")
    if len(points) > max_points:
        order = np.argsort(values)[-max_points:]
        points = [points[i] for i in order]
        values = [values[i] for i in order]
        labels = [labels[i] for i in order]
    return {
        "name": "Stitched",
        "legend": [
            {"label": "A-only", "color": "#51d6ff"},
            {"label": "B-only", "color": "#ff9a4d"},
            {"label": "Overlap", "color": "#f6ff7e"},
        ],
        "count": len(points),
        "points": points,
        "values": values,
        "labels": labels,
        "shape": stitched_shape,
    }


def _build_volume_point_cloud(volume: np.ndarray, name: str, base_color: str) -> dict:
    coords = np.argwhere(volume >= 0.35)
    values = volume[coords[:, 0], coords[:, 1], coords[:, 2]] if len(coords) else np.array([], dtype=np.float32)
    labels = []
    legend = [
        {"label": "Layered Background", "color": "#2a6f97"},
        {"label": "Reflective Block", "color": base_color},
        {"label": "Tubular Structure", "color": "#ff6ad5"},
        {"label": "Spherical Structure", "color": "#ffe66d"},
        {"label": "Slanted Band", "color": "#7ef7c5"},
    ]
    max_points = 12000
    for coord, value in zip(coords, values):
        x, y, z = coord.tolist()
        if value < 0.4:
            labels.append("Layered Background")
        elif y < volume.shape[1] // 4 and z > volume.shape[2] // 2:
            labels.append("Tubular Structure")
        elif y > int(volume.shape[1] * 0.55) and z < int(volume.shape[2] * 0.4):
            labels.append("Spherical Structure")
        elif y > volume.shape[1] // 5 and z > volume.shape[2] // 5 and x > volume.shape[0] // 3:
            labels.append("Slanted Band")
        else:
            labels.append("Reflective Block")
    if len(coords) > max_points:
        order = np.argsort(values)[-max_points:]
        coords = coords[order]
        values = values[order]
        labels = [labels[i] for i in order]
    return {
        "name": name,
        "legend": legend,
        "count": int(len(coords)),
        "points": coords.tolist(),
        "values": values.tolist(),
        "labels": labels,
        "shape": list(volume.shape),
        "color": base_color,
    }
