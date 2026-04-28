"""STL viewer with scene-file support, dynamic transforms, and capture tools.

Controls:
- Left mouse drag: rotate
- Mouse wheel: zoom
- Arrow keys / WASD: rotate camera or move selected model in edit mode
- Q/E: roll camera or move selected model on Z in edit mode
- X: toggle axes
- Z: toggle triangle edges
- F: toggle depth sorting
- C: toggle backface culling for selected model
- Tab: select next model
- Space: toggle selected model
- M: toggle selected-model edit mode
- P: toggle animation playback
- F12: save a screenshot PNG
- Ctrl+S: export the current scene state as JSON
- V: start or stop GIF recording
- T/R/Y: switch gizmo mode to move, rotate, or scale while editing
- Shift-click the model list to multi-select
- G: parent selected models under the active model
- U: clear parents for the selected models
- Shift+R: reset the view while editing
- 1-9: toggle model visibility by index
- R: reset view, or choose rotate gizmo mode while editing

Usage:
    python 3d_viewer.py
    python 3d_viewer.py path/to/scene.json
    python 3d_viewer.py path/to/model.stl
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pygame
from PIL import Image
import trimesh

Color = Tuple[int, int, int]

DEFAULT_AXES_VISIBLE = True
DEFAULT_MODEL_COLOR: Color = (180, 180, 220)
DEFAULT_SCENE_FILE = "scene.json"
DEFAULT_MAX_FACES_PER_MODEL = 40000
DEFAULT_DOWNSAMPLE_STEP = 1
DEFAULT_DEDUPLICATE_VERTICES = True
DEFAULT_DRAW_EDGES = False
DEFAULT_DEPTH_SORT = False
DEFAULT_BACKFACE_CULLING = True
DEFAULT_GIZMO_LENGTH = 0.65
DEFAULT_GIZMO_HIT_RADIUS = 12.0
DEFAULT_GIZMO_RING_RADIUS = 0.75
DEFAULT_GIZMO_RING_SAMPLES = 64
DEFAULT_GIZMO_CENTER_HANDLE_RADIUS = 14.0
DEFAULT_GIZMO_MOVE_SENSITIVITY = 0.01
DEFAULT_GIZMO_ROTATE_SENSITIVITY = 0.45
DEFAULT_GIZMO_SCALE_SENSITIVITY = 0.01

GIZMO_MOVE = "move"
GIZMO_ROTATE = "rotate"
GIZMO_SCALE = "scale"
GIZMO_MODES = (GIZMO_MOVE, GIZMO_ROTATE, GIZMO_SCALE)


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class SceneModel:
    name: str
    source_path: Path
    vertices: np.ndarray
    faces: np.ndarray
    color: Color
    visible: bool
    original_face_count: int
    backface_culling: bool
    position: np.ndarray
    rotation_deg: np.ndarray
    scale: float
    animation: "ModelAnimation | None"
    parent_index: int | None = None


@dataclass
class KeyframeTrack:
    times: np.ndarray
    values: np.ndarray


@dataclass
class ModelAnimation:
    enabled: bool
    loop: bool
    speed: float
    position: KeyframeTrack | None
    rotation_deg: KeyframeTrack | None
    scale: KeyframeTrack | None


@dataclass
class GizmoDragState:
    active: bool = False
    axis_index: int | None = None
    mode: str = GIZMO_MOVE
    start_mouse: tuple[int, int] = (0, 0)
    start_world_matrices: dict[int, np.ndarray] | None = None
    selection_pivot_world: np.ndarray | None = None
    selection_pivot_screen: np.ndarray | None = None
    start_mouse_angle: float | None = None
    start_mouse_distance: float | None = None
    axis_screen_direction: np.ndarray | None = None


@dataclass
class Scene:
    models: List[SceneModel]
    axes_visible: bool
    scene_source: Path | None
    draw_edges: bool
    depth_sort: bool
    render_center: np.ndarray
    render_scale: float


def clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def parse_vector(value: Any, default: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Expected a 3-element vector")
    if len(value) != 3:
        raise ValueError("Expected a 3-element vector")
    return np.array([float(component) for component in value], dtype=float)


def parse_color(value: Any, default: Color = DEFAULT_MODEL_COLOR) -> Color:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            return tuple(int(text[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
        parts = [part.strip() for part in text.split(",")]
        if len(parts) == 3:
            return tuple(clamp_channel(float(part)) for part in parts)  # type: ignore[return-value]
        raise ValueError(f"Unsupported color string: {value!r}")
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError("Expected a 3-element color")
    return tuple(clamp_channel(float(component)) for component in value)  # type: ignore[return-value]


def parse_keyframe_track(raw_track: Any, expected_length: int) -> KeyframeTrack | None:
    if raw_track is None:
        return None
    if not isinstance(raw_track, Sequence) or isinstance(raw_track, (str, bytes)):
        raise ValueError("Animation tracks must be a list of keyframes")

    times: List[float] = []
    values: List[Sequence[float]] = []
    for entry in raw_track:
        if isinstance(entry, dict):
            time_value = entry.get("time")
            key_value = entry.get("value", entry.get("values"))
        elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)) and len(entry) == 2:
            time_value, key_value = entry
        else:
            raise ValueError("Each keyframe must be a [time, value] pair or an object with time/value fields")

        if expected_length == 1 and not isinstance(key_value, Sequence):
            parsed_value = [float(key_value)]
        else:
            if not isinstance(key_value, Sequence) or isinstance(key_value, (str, bytes)):
                raise ValueError("Animation keyframe values must be a sequence")
            if len(key_value) != expected_length:
                raise ValueError(f"Expected keyframe values of length {expected_length}")
            parsed_value = [float(component) for component in key_value]

        times.append(float(time_value))
        values.append(parsed_value)

    if not times:
        return None

    order = np.argsort(np.array(times, dtype=float))
    sorted_times = np.array(times, dtype=np.float32)[order]
    sorted_values = np.array(values, dtype=np.float32)[order]
    return KeyframeTrack(times=sorted_times, values=sorted_values)


def parse_model_animation(raw_animation: Any) -> ModelAnimation | None:
    if raw_animation is None:
        return None
    if not isinstance(raw_animation, dict):
        raise ValueError("Model animation must be an object")

    return ModelAnimation(
        enabled=bool(raw_animation.get("enabled", True)),
        loop=bool(raw_animation.get("loop", True)),
        speed=float(raw_animation.get("speed", 1.0)),
        position=parse_keyframe_track(raw_animation.get("position"), 3),
        rotation_deg=parse_keyframe_track(raw_animation.get("rotation_deg", raw_animation.get("rotation")), 3),
        scale=parse_keyframe_track(raw_animation.get("scale"), 1),
    )


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rzm @ rym @ rxm


def build_transform_matrix(position: np.ndarray, rotation_deg: np.ndarray, scale: float) -> np.ndarray:
    rotation = rotation_matrix(*(math.radians(component) for component in rotation_deg))
    linear = rotation * float(scale)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = linear.astype(np.float32)
    matrix[:3, 3] = position.astype(np.float32)
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogenous = np.ones((len(points), 4), dtype=np.float32)
    homogenous[:, :3] = points.astype(np.float32)
    transformed = homogenous @ matrix.T
    return transformed[:, :3]


def matrix_translation(matrix: np.ndarray) -> np.ndarray:
    return matrix[:3, 3].astype(np.float32, copy=True)


def decompose_transform_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    position = matrix[:3, 3].astype(np.float32, copy=True)
    linear = matrix[:3, :3].astype(np.float32, copy=True)
    scale = float(np.mean([np.linalg.norm(linear[:, axis]) for axis in range(3)]))
    if scale <= 1e-12:
        return position, np.zeros(3, dtype=np.float32), 1.0

    rotation = linear / scale
    sy = float(-rotation[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1e-6:
        rx = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        rz = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        rx = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        rz = 0.0
    return position, np.array([math.degrees(rx), math.degrees(ry), math.degrees(rz)], dtype=np.float32), scale


def collect_selected_root_indices(selected_indices: set[int], models: Sequence[SceneModel]) -> list[int]:
    selected = set(selected_indices)
    roots: list[int] = []
    for index in sorted(selected):
        ancestor = models[index].parent_index
        include = True
        while ancestor is not None:
            if ancestor in selected:
                include = False
                break
            ancestor = models[ancestor].parent_index
        if include:
            roots.append(index)
    return roots


def selection_pivot_world(
    selected_indices: set[int],
    models: Sequence[SceneModel],
    animation_time: float,
    play_animations: bool,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if not selected_indices:
        return np.zeros(3, dtype=np.float32), {}

    world_matrices = {}
    root_indices = collect_selected_root_indices(selected_indices, models)
    pivots = []
    for index in root_indices:
        world_matrix = resolve_model_world_matrix(index, models, animation_time, play_animations, world_matrices)
        pivots.append(matrix_translation(world_matrix))
    if not pivots:
        return np.zeros(3, dtype=np.float32), world_matrices
    return np.mean(np.vstack(pivots), axis=0).astype(np.float32), world_matrices


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / length).astype(np.float32)


def project_world_point(
    point: np.ndarray,
    rotation: np.ndarray,
    width: int,
    height: int,
    camera_distance: float,
    zoom: float,
    scene_center: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    transformed = ((point - scene_center) * scene_scale).astype(np.float32)
    view_point = transformed @ rotation.T
    return project_points(view_point[None, :], width, height, camera_distance, zoom)[0]


def project_world_points(
    points: np.ndarray,
    rotation: np.ndarray,
    width: int,
    height: int,
    camera_distance: float,
    zoom: float,
    scene_center: np.ndarray,
    scene_scale: float,
) -> np.ndarray:
    transformed = ((points - scene_center) * scene_scale).astype(np.float32)
    view_points = transformed @ rotation.T
    return project_points(view_points, width, height, camera_distance, zoom)


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    segment_length_sq = float(np.dot(segment, segment))
    if segment_length_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / segment_length_sq)
    projection = max(0.0, min(1.0, projection))
    nearest = start + projection * segment
    return float(np.linalg.norm(point - nearest))


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return float("inf")
    distances = [point_to_segment_distance(point, start.astype(float), end.astype(float)) for start, end in zip(polyline, np.roll(polyline, -1, axis=0))]
    return float(min(distances)) if distances else float("inf")


def hit_test_gizmo_axis(mouse_pos: tuple[int, int], pivot_screen: np.ndarray, axis_screen_points: Sequence[np.ndarray], hit_radius: float) -> int | None:
    mouse = np.array(mouse_pos, dtype=float)
    closest_axis = None
    closest_distance = hit_radius
    for axis_index, handle_point in enumerate(axis_screen_points):
        distance = point_to_segment_distance(mouse, pivot_screen.astype(float), handle_point.astype(float))
        if distance <= closest_distance:
            closest_distance = distance
            closest_axis = axis_index
    return closest_axis


def hit_test_gizmo_ring(mouse_pos: tuple[int, int], ring_points: Sequence[np.ndarray], hit_radius: float) -> int | None:
    mouse = np.array(mouse_pos, dtype=float)
    closest_ring: int | None = None
    closest_distance = hit_radius
    for ring_index, ring in enumerate(ring_points):
        distance = point_to_polyline_distance(mouse, np.asarray(ring, dtype=float))
        if distance <= closest_distance:
            closest_distance = distance
            closest_ring = ring_index
    return closest_ring


def hit_test_gizmo_center(mouse_pos: tuple[int, int], pivot_screen: np.ndarray, hit_radius: float) -> bool:
    mouse = np.array(mouse_pos, dtype=float)
    return float(np.linalg.norm(mouse - pivot_screen.astype(float))) <= hit_radius


def axis_to_world_delta(axis_index: int, drag_pixels: float, scene_scale: float, zoom: float) -> float:
    return drag_pixels * DEFAULT_GIZMO_MOVE_SENSITIVITY / max(scene_scale * max(zoom, 0.2), 1e-6)


def axis_to_rotation_delta(drag_pixels: float) -> float:
    return drag_pixels * DEFAULT_GIZMO_ROTATE_SENSITIVITY


def axis_to_scale_factor(drag_pixels: float) -> float:
    return 1.0 + drag_pixels * DEFAULT_GIZMO_SCALE_SENSITIVITY


def angle_between_points(center: np.ndarray, point: tuple[int, int]) -> float:
    delta = np.array(point, dtype=float) - center.astype(float)
    return math.atan2(float(delta[1]), float(delta[0]))


def distance_between_points(center: np.ndarray, point: tuple[int, int]) -> float:
    delta = np.array(point, dtype=float) - center.astype(float)
    return float(np.linalg.norm(delta))


def gizmo_mode_label(mode: str) -> str:
    if mode == GIZMO_MOVE:
        return "move"
    if mode == GIZMO_ROTATE:
        return "rotate"
    if mode == GIZMO_SCALE:
        return "scale"
    return "move"


def world_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )


@dataclass
class GizmoGeometry:
    pivot_world: np.ndarray
    pivot_screen: np.ndarray
    axis_world: tuple[np.ndarray, np.ndarray, np.ndarray]
    axis_screen_points: tuple[np.ndarray, np.ndarray, np.ndarray]
    axis_screen_directions: tuple[np.ndarray, np.ndarray, np.ndarray]
    rotation_ring_points: tuple[np.ndarray, np.ndarray, np.ndarray]
    scale_handle_points: tuple[np.ndarray, np.ndarray, np.ndarray]


def compute_gizmo_geometry(
    selected_indices: set[int],
    models: Sequence[SceneModel],
    animation_time: float,
    play_animations: bool,
    rotation: np.ndarray,
    width: int,
    height: int,
    camera_distance: float,
    zoom: float,
    scene_center: np.ndarray,
    scene_scale: float,
) -> GizmoGeometry:
    pivot_world, _ = selection_pivot_world(selected_indices, models, animation_time, play_animations)
    pivot_screen = project_world_point(pivot_world, rotation, width, height, camera_distance, zoom, scene_center, scene_scale)
    axis_world = world_axes()
    axis_length = DEFAULT_GIZMO_LENGTH / max(scene_scale, 1e-6)
    ring_radius = DEFAULT_GIZMO_RING_RADIUS / max(scene_scale, 1e-6)
    scale_handle_length = axis_length * 1.15

    axis_screen_points: list[np.ndarray] = []
    axis_screen_directions: list[np.ndarray] = []
    rotation_ring_points: list[np.ndarray] = []
    scale_handle_points: list[np.ndarray] = []

    ring_bases = (
        (np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    )
    for axis_vector in axis_world:
        handle_world = pivot_world + axis_vector * axis_length
        handle_screen = project_world_point(handle_world, rotation, width, height, camera_distance, zoom, scene_center, scene_scale)
        axis_screen_points.append(handle_screen)
        axis_screen_directions.append(normalize_vector(handle_screen.astype(np.float32) - pivot_screen.astype(np.float32)))

    for basis_u, basis_v in ring_bases:
        ring_points: list[np.ndarray] = []
        for sample_index in range(DEFAULT_GIZMO_RING_SAMPLES):
            angle = (sample_index / DEFAULT_GIZMO_RING_SAMPLES) * math.tau
            point_world = pivot_world + (basis_u * math.cos(angle) + basis_v * math.sin(angle)) * ring_radius
            ring_points.append(project_world_point(point_world, rotation, width, height, camera_distance, zoom, scene_center, scene_scale))
        rotation_ring_points.append(np.asarray(ring_points, dtype=np.float32))

    for axis_vector in axis_world:
        handle_world = pivot_world + axis_vector * scale_handle_length
        scale_handle_points.append(project_world_point(handle_world, rotation, width, height, camera_distance, zoom, scene_center, scene_scale))

    return GizmoGeometry(
        pivot_world=pivot_world,
        pivot_screen=pivot_screen,
        axis_world=axis_world,
        axis_screen_points=(axis_screen_points[0], axis_screen_points[1], axis_screen_points[2]),
        axis_screen_directions=(axis_screen_directions[0], axis_screen_directions[1], axis_screen_directions[2]),
        rotation_ring_points=(rotation_ring_points[0], rotation_ring_points[1], rotation_ring_points[2]),
        scale_handle_points=(scale_handle_points[0], scale_handle_points[1], scale_handle_points[2]),
    )


def apply_gizmo_drag(
    models: Sequence[SceneModel],
    drag_state: GizmoDragState,
    current_mouse: tuple[int, int],
    selected_indices: set[int],
    scene_scale: float,
    zoom: float,
    animation_time: float,
    play_animations: bool,
) -> None:
    if (
        drag_state.axis_index is None
        or drag_state.start_world_matrices is None
        or drag_state.selection_pivot_world is None
    ):
        return
    root_indices = collect_selected_root_indices(selected_indices, models)
    axis_vectors = world_axes()

    if drag_state.mode == GIZMO_MOVE:
        if drag_state.axis_screen_direction is None:
            return
        mouse_delta = np.array(current_mouse, dtype=float) - np.array(drag_state.start_mouse, dtype=float)
        axis_component = float(np.dot(mouse_delta, drag_state.axis_screen_direction))
        delta_vector = axis_vectors[drag_state.axis_index] * axis_to_world_delta(drag_state.axis_index, axis_component, scene_scale, zoom)
        group_matrix = build_transform_matrix(delta_vector, np.zeros(3, dtype=np.float32), 1.0)
    elif drag_state.mode == GIZMO_ROTATE:
        if drag_state.selection_pivot_screen is None or drag_state.start_mouse_angle is None:
            return
        current_angle = angle_between_points(drag_state.selection_pivot_screen, current_mouse)
        angle_delta = math.degrees(current_angle - drag_state.start_mouse_angle)
        delta_degrees = np.zeros(3, dtype=np.float32)
        delta_degrees[drag_state.axis_index] = angle_delta
        rotation = build_transform_matrix(np.zeros(3, dtype=np.float32), delta_degrees, 1.0)
        translate_to_pivot = build_transform_matrix(drag_state.selection_pivot_world, np.zeros(3, dtype=np.float32), 1.0)
        translate_back = build_transform_matrix(-drag_state.selection_pivot_world, np.zeros(3, dtype=np.float32), 1.0)
        group_matrix = translate_to_pivot @ rotation @ translate_back
    else:
        if drag_state.selection_pivot_screen is None or drag_state.start_mouse_distance is None:
            return
        current_distance = distance_between_points(drag_state.selection_pivot_screen, current_mouse)
        if drag_state.axis_index == 3:
            scale_factor = max(0.05, current_distance / max(drag_state.start_mouse_distance, 1e-6))
            scale_matrix = build_transform_matrix(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), scale_factor)
        else:
            scale_factor = max(0.05, current_distance / max(drag_state.start_mouse_distance, 1e-6))
            axis_scale = np.ones(3, dtype=np.float32)
            axis_scale[drag_state.axis_index] = scale_factor
            scale_matrix = np.eye(4, dtype=np.float32)
            scale_matrix[0, 0] = axis_scale[0]
            scale_matrix[1, 1] = axis_scale[1]
            scale_matrix[2, 2] = axis_scale[2]
        translate_to_pivot = build_transform_matrix(drag_state.selection_pivot_world, np.zeros(3, dtype=np.float32), 1.0)
        translate_back = build_transform_matrix(-drag_state.selection_pivot_world, np.zeros(3, dtype=np.float32), 1.0)
        group_matrix = translate_to_pivot @ scale_matrix @ translate_back

    world_cache = {}
    for root_index in root_indices:
        start_world_matrix = drag_state.start_world_matrices.get(root_index)
        if start_world_matrix is None:
            continue
        new_world_matrix = group_matrix @ start_world_matrix
        model = models[root_index]
        if model.parent_index is not None:
            parent_world = resolve_model_world_matrix(model.parent_index, models, animation_time, play_animations, world_cache)
            local_matrix = np.linalg.inv(parent_world) @ new_world_matrix
        else:
            local_matrix = new_world_matrix

        position, rotation_deg, scale = decompose_transform_matrix(local_matrix)
        model.position = position
        model.rotation_deg = rotation_deg
        model.scale = scale


def draw_gizmo(
    screen: pygame.Surface,
    font: pygame.font.Font,
    selected_indices: set[int],
    models: Sequence[SceneModel],
    animation_time: float,
    play_animations: bool,
    rotation: np.ndarray,
    width: int,
    height: int,
    camera_distance: float,
    zoom: float,
    scene_center: np.ndarray,
    scene_scale: float,
    mode: str,
) -> None:
    geometry = compute_gizmo_geometry(
        selected_indices,
        models,
        animation_time,
        play_animations,
        rotation,
        width,
        height,
        camera_distance,
        zoom,
        scene_center,
        scene_scale,
    )
    pivot_screen = geometry.pivot_screen
    colors = ((240, 90, 90), (90, 220, 90), (90, 150, 255))
    labels = ("X", "Y", "Z")

    if mode == GIZMO_ROTATE:
        for ring_points, color, label in zip(geometry.rotation_ring_points, colors, labels):
            if len(ring_points) >= 2:
                pygame.draw.lines(screen, color, True, ring_points.tolist(), 2)
            label_point = ring_points[len(ring_points) // 4] if len(ring_points) else pivot_screen
            label_surface = font.render(label, True, color)
            screen.blit(label_surface, (int(label_point[0]) + 8, int(label_point[1]) - 8))
        pygame.draw.circle(screen, (230, 220, 80), pivot_screen, 8)
    elif mode == GIZMO_SCALE:
        center_color = (230, 220, 80)
        pygame.draw.rect(screen, center_color, pygame.Rect(int(pivot_screen[0]) - 7, int(pivot_screen[1]) - 7, 14, 14), border_radius=3)
        for handle_screen, color, label in zip(geometry.scale_handle_points, colors, labels):
            pygame.draw.line(screen, color, pivot_screen, handle_screen, 4)
            pygame.draw.rect(screen, color, pygame.Rect(int(handle_screen[0]) - 5, int(handle_screen[1]) - 5, 10, 10), border_radius=2)
            label_surface = font.render(label, True, color)
            screen.blit(label_surface, (int(handle_screen[0]) + 8, int(handle_screen[1]) - 8))
    else:
        for handle_screen, color, label in zip(geometry.axis_screen_points, colors, labels):
            pygame.draw.line(screen, color, pivot_screen, handle_screen, 4)
            pygame.draw.circle(screen, color, handle_screen, 6)
            label_surface = font.render(label, True, color)
            screen.blit(label_surface, (int(handle_screen[0]) + 8, int(handle_screen[1]) - 8))
        pygame.draw.circle(screen, (230, 220, 80), pivot_screen, 7)

    mode_surface = font.render(f"Gizmo: {gizmo_mode_label(mode)}", True, (240, 240, 240))
    screen.blit(mode_surface, (int(pivot_screen[0]) + 12, int(pivot_screen[1]) + 12))

    return None


def apply_model_transform(vertices: np.ndarray, position: np.ndarray, rotation_deg: np.ndarray, scale: float) -> np.ndarray:
    rotation = rotation_matrix(*(math.radians(component) for component in rotation_deg))
    return ((vertices * scale) @ rotation.T) + position


def sample_track(track: KeyframeTrack | None, time_value: float, expected_length: int) -> np.ndarray:
    if track is None or len(track.times) == 0:
        if expected_length == 1:
            return np.array([1.0], dtype=np.float32)
        return np.zeros(expected_length, dtype=np.float32)

    if len(track.times) == 1 or time_value <= float(track.times[0]):
        return track.values[0].astype(np.float32, copy=False)
    if time_value >= float(track.times[-1]):
        return track.values[-1].astype(np.float32, copy=False)

    right = int(np.searchsorted(track.times, time_value, side="right"))
    left = right - 1
    left_time = float(track.times[left])
    right_time = float(track.times[right])
    span = max(1e-9, right_time - left_time)
    factor = (time_value - left_time) / span
    return (track.values[left] * (1.0 - factor) + track.values[right] * factor).astype(np.float32, copy=False)


def sample_model_animation(animation: ModelAnimation | None, animation_time: float) -> tuple[np.ndarray, np.ndarray, float]:
    if animation is None or not animation.enabled:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0

    all_times: List[np.ndarray] = []
    for track in (animation.position, animation.rotation_deg, animation.scale):
        if track is not None and len(track.times) > 0:
            all_times.append(track.times)

    if not all_times:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0

    duration = float(max(times[-1] for times in all_times))
    sample_time = animation_time * animation.speed
    if animation.loop and duration > 0.0:
        sample_time = sample_time % duration
    else:
        sample_time = min(sample_time, duration)

    position = sample_track(animation.position, sample_time, 3)
    rotation = sample_track(animation.rotation_deg, sample_time, 3)
    scale_track = sample_track(animation.scale, sample_time, 1)
    scale = float(scale_track[0]) if len(scale_track) else 1.0
    return position, rotation, scale


def _looks_like_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    return expected_size == len(data)


def _load_binary_stl(data: bytes) -> Mesh:
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    vertices = np.empty((triangle_count * 3, 3), dtype=np.float32)
    faces = np.empty((triangle_count, 3), dtype=np.int32)
    offset = 84

    for tri_index in range(triangle_count):
        offset += 12  # normal vector, ignored
        base = tri_index * 3
        for local_index in range(3):
            x, y, z = struct.unpack_from("<fff", data, offset)
            offset += 12
            vertices[base + local_index] = (x, y, z)
        faces[tri_index] = (base, base + 1, base + 2)
        offset += 2  # attribute byte count

    return Mesh(vertices=vertices, faces=faces)


def _load_ascii_stl(text: str) -> Mesh:
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    current: List[int] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("vertex"):
            parts = line.split()
            if len(parts) != 4:
                continue
            vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
            vertices.append(vertex)
            current.append(len(vertices) - 1)
        elif line.startswith("endfacet"):
            if len(current) >= 3:
                faces.append(tuple(current[-3:]))
            current = []

    if not vertices or not faces:
        raise ValueError("No triangles found in STL file")

    return Mesh(np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32))


def load_stl(path: Path) -> Mesh:
    data = path.read_bytes()
    if _looks_like_binary_stl(data):
        return _load_binary_stl(data)
    return _load_ascii_stl(data.decode("utf-8", errors="ignore"))


def _load_trimesh_mesh(path: Path) -> Mesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        vertices = np.asarray(loaded.vertices, dtype=np.float32)
        faces = np.asarray(loaded.faces, dtype=np.int32)
    elif isinstance(loaded, trimesh.Scene):
        combined = loaded.dump(concatenate=True)
        if not isinstance(combined, trimesh.Trimesh):
            raise ValueError(f"Unsupported mesh scene: {path}")
        vertices = np.asarray(combined.vertices, dtype=np.float32)
        faces = np.asarray(combined.faces, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported 3D model file: {path}")

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"No mesh faces found in {path}")

    return Mesh(vertices=vertices, faces=faces)


def load_mesh(path: Path) -> Mesh:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return load_stl(path)
    return _load_trimesh_mesh(path)


def deduplicate_mesh_vertices(mesh: Mesh) -> Mesh:
    if len(mesh.vertices) == 0:
        return mesh
    unique_vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
    remapped_faces = inverse[mesh.faces]
    return Mesh(unique_vertices.astype(np.float32), remapped_faces.astype(np.int32))


def downsample_faces(faces: np.ndarray, max_faces: int | None, step: int) -> np.ndarray:
    reduced = faces
    safe_step = max(1, int(step))
    if safe_step > 1 and len(reduced) > 0:
        reduced = reduced[::safe_step]

    if max_faces is not None and max_faces > 0 and len(reduced) > max_faces:
        indices = np.linspace(0, len(reduced) - 1, num=max_faces, dtype=int)
        reduced = reduced[indices]

    return reduced.astype(np.int32, copy=False)


def compute_scene_frame(models: Sequence[SceneModel]) -> tuple[np.ndarray, float]:
    if not models:
        return np.zeros(3, dtype=np.float32), 1.0

    world_matrices = {}
    transformed_vertices = [
        transform_points(model.vertices, resolve_model_world_matrix(index, models, 0.0, False, world_matrices))
        for index, model in enumerate(models)
    ]
    all_vertices = np.vstack(transformed_vertices)
    center = all_vertices.mean(axis=0)
    bounds = all_vertices.max(axis=0) - all_vertices.min(axis=0)
    longest = float(np.max(bounds))
    scale = 1.0 if longest == 0 else 2.0 / longest
    return center.astype(np.float32), float(scale)


def load_scene_source(input_path: Path | None) -> tuple[Path, Path]:
    if input_path is not None:
        resolved = input_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved, resolved.parent

    cwd_scene = Path.cwd() / DEFAULT_SCENE_FILE
    if cwd_scene.exists():
        return cwd_scene.resolve(), cwd_scene.parent.resolve()

    script_scene = Path(__file__).with_name(DEFAULT_SCENE_FILE)
    if script_scene.exists():
        return script_scene.resolve(), script_scene.parent.resolve()

    raise FileNotFoundError(
        f"No input file was provided and {DEFAULT_SCENE_FILE} was not found in the current directory or next to the viewer."
    )


def prepare_scene_for_render(models: Sequence[SceneModel]) -> List[np.ndarray]:
    render_center, render_scale = compute_scene_frame(models)
    return [((model.vertices - render_center) * render_scale).astype(np.float32) for model in models]


def load_scene(input_path: Path | None) -> Scene:
    source_path, base_dir = load_scene_source(input_path)

    if source_path.suffix.lower() != ".json":
        mesh = deduplicate_mesh_vertices(load_mesh(source_path))
        faces = downsample_faces(mesh.faces, DEFAULT_MAX_FACES_PER_MODEL, DEFAULT_DOWNSAMPLE_STEP)
        models = [
            SceneModel(
                name=source_path.stem,
                source_path=source_path,
                vertices=mesh.vertices,
                faces=faces,
                color=DEFAULT_MODEL_COLOR,
                visible=True,
                original_face_count=len(mesh.faces),
                backface_culling=DEFAULT_BACKFACE_CULLING,
                position=np.zeros(3, dtype=np.float32),
                rotation_deg=np.zeros(3, dtype=np.float32),
                scale=1.0,
                animation=None,
                parent_index=None,
            )
        ]
        render_center, render_scale = compute_scene_frame(models)
        return Scene(
            models=models,
            axes_visible=DEFAULT_AXES_VISIBLE,
            scene_source=None,
            draw_edges=DEFAULT_DRAW_EDGES,
            depth_sort=DEFAULT_DEPTH_SORT,
            render_center=render_center,
            render_scale=render_scale,
        )

    scene_data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(scene_data, dict):
        raise ValueError("Scene file must contain a JSON object")

    axes_visible = bool(scene_data.get("axes_visible", DEFAULT_AXES_VISIBLE))
    perf = scene_data.get("performance", {})
    if not isinstance(perf, dict):
        raise ValueError("Scene file 'performance' must be an object")

    default_max_faces = int(perf.get("max_faces_per_model", DEFAULT_MAX_FACES_PER_MODEL))
    default_step = int(perf.get("downsample_step", DEFAULT_DOWNSAMPLE_STEP))
    default_dedup = bool(perf.get("deduplicate_vertices", DEFAULT_DEDUPLICATE_VERTICES))
    draw_edges = bool(perf.get("draw_edges", DEFAULT_DRAW_EDGES))
    depth_sort = bool(perf.get("depth_sort", DEFAULT_DEPTH_SORT))
    default_backface_culling = bool(perf.get("backface_culling", DEFAULT_BACKFACE_CULLING))

    raw_models = scene_data.get("models", [])
    if not isinstance(raw_models, list):
        raise ValueError("Scene file 'models' must be a list")

    models: List[SceneModel] = []
    pending_parents: List[Any] = []
    for index, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            raise ValueError("Each model entry must be a JSON object")

        model_path_value = raw_model.get("path")
        if not isinstance(model_path_value, str) or not model_path_value.strip():
            raise ValueError("Each model entry needs a non-empty 'path'")

        model_path = (base_dir / model_path_value).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        mesh = load_mesh(model_path)
        if bool(raw_model.get("deduplicate_vertices", default_dedup)):
            mesh = deduplicate_mesh_vertices(mesh)

        max_faces = raw_model.get("max_faces", default_max_faces)
        max_faces_int = int(max_faces) if max_faces is not None else None
        downsample_step_value = int(raw_model.get("downsample_step", default_step))
        reduced_faces = downsample_faces(mesh.faces, max_faces_int, downsample_step_value)

        offset = parse_vector(raw_model.get("offset", raw_model.get("position")))
        rotation_deg = parse_vector(raw_model.get("rotation_deg", raw_model.get("rotation")))
        scale = float(raw_model.get("scale", 1.0))
        visible = bool(raw_model.get("visible", True))
        backface_culling = bool(raw_model.get("backface_culling", default_backface_culling))
        color = parse_color(raw_model.get("color"), DEFAULT_MODEL_COLOR)
        name = str(raw_model.get("name") or model_path.stem or f"model_{index}")
        animation = parse_model_animation(raw_model.get("animation"))
        parent_ref = raw_model.get("parent")

        models.append(
            SceneModel(
                name=name,
                source_path=model_path,
                vertices=mesh.vertices,
                faces=reduced_faces,
                color=color,
                visible=visible,
                original_face_count=len(mesh.faces),
                backface_culling=backface_culling,
                position=offset.astype(np.float32),
                rotation_deg=rotation_deg.astype(np.float32),
                scale=scale,
                animation=animation,
                parent_index=None,
            )
        )
        pending_parents.append(parent_ref)

    name_to_index = {model.name: index for index, model in enumerate(models)}
    path_to_index = {str(model.source_path): index for index, model in enumerate(models)}
    stem_to_index = {model.source_path.stem: index for index, model in enumerate(models)}

    for index, parent_ref in enumerate(pending_parents):
        if parent_ref is None:
            continue
        parent_index: int | None = None
        if isinstance(parent_ref, int) and 0 <= parent_ref < len(models):
            parent_index = parent_ref
        elif isinstance(parent_ref, str):
            parent_index = name_to_index.get(parent_ref)
            if parent_index is None:
                parent_index = path_to_index.get(parent_ref)
            if parent_index is None:
                parent_index = stem_to_index.get(parent_ref)
        if parent_index is not None and parent_index != index:
            models[index].parent_index = parent_index

    render_center, render_scale = compute_scene_frame(models)
    return Scene(
        models=models,
        axes_visible=axes_visible,
        scene_source=source_path,
        draw_edges=draw_edges,
        depth_sort=depth_sort,
        render_center=render_center,
        render_scale=render_scale,
    )


def project_points(points: np.ndarray, width: int, height: int, camera_distance: float, zoom: float) -> np.ndarray:
    denominator = np.maximum(0.1, camera_distance + points[:, 2])
    factor = zoom * camera_distance / denominator
    projection_scale = min(width, height) / 2

    projected = np.empty((len(points), 2), dtype=np.int32)
    projected[:, 0] = (width / 2 + points[:, 0] * factor * projection_scale).astype(np.int32)
    projected[:, 1] = (height / 2 - points[:, 1] * factor * projection_scale).astype(np.int32)
    return projected


def tint_colors(base_color: Color, shades: np.ndarray) -> np.ndarray:
    base = np.array(base_color, dtype=np.float32)
    tinted = np.clip(base[None, :] * shades[:, None], 0, 255)
    return tinted.astype(np.uint8)


def draw_text_block(screen: pygame.Surface, font: pygame.font.Font, lines: Sequence[str], x: int, y: int) -> None:
    for index, text in enumerate(lines):
        surface = font.render(text, True, (235, 235, 235))
        screen.blit(surface, (x, y + index * 22))


def wrap_text_lines(font: pygame.font.Font, text: str, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


@dataclass
class UiLayout:
    instructions_rect: pygame.Rect
    instructions_lines: List[str]
    models_rect: pygame.Rect
    model_rows: List[tuple[int, pygame.Rect]]


def build_ui_layout(
    font: pygame.font.Font,
    width: int,
    height: int,
    models: Sequence[SceneModel],
    selected_indices: set[int],
    active_index: int,
    mode_label: str,
    gizmo_label: str,
    animation_label: str,
    recording_label: str,
) -> UiLayout:
    panel_padding = 12
    panel_gap = 12
    left_width = min(560, max(360, width // 2 - panel_gap * 2))
    right_width = min(520, max(360, width // 2 - panel_gap * 2))

    instructions_lines = []
    for line in [
        f"Scene: {mode_label} | Gizmo: {gizmo_label} | Animations: {animation_label} | {recording_label}",
        "Drag empty space to rotate the camera. Wheel zooms. Ctrl+R reframes the current models.",
        "Edit mode: click a gizmo handle and drag it. T/R/Y switch move, rotate, and scale.",
        "Multi-select: shift-click model rows to add/remove. G parents selected models to the active one. U clears parents.",
        "P pauses animation. F12 saves a PNG. Ctrl+S exports the current scene JSON. V records a GIF.",
        "Tab changes the active model. Space toggles its visibility. C flips its culling. R resets the camera in view mode.",
    ]:
        instructions_lines.extend(wrap_text_lines(font, line, left_width - panel_padding * 2 - 12))

    instructions_height = panel_padding * 2 + len(instructions_lines) * 20 + 18
    instructions_rect = pygame.Rect(panel_padding, panel_padding, left_width, instructions_height)

    models_rect_width = right_width
    models_rect_height = min(height - 24, max(260, 40 + min(len(models), 14) * 22 + 18))
    models_rect = pygame.Rect(width - models_rect_width - panel_padding, panel_padding, models_rect_width, models_rect_height)

    model_rows: List[tuple[int, pygame.Rect]] = []
    row_y = models_rect.y + 34
    visible_models = models[: min(len(models), 14)]
    for index, _model in enumerate(visible_models):
        row_rect = pygame.Rect(models_rect.x + 10, row_y + index * 22, models_rect.width - 20, 20)
        model_rows.append((index, row_rect))

    return UiLayout(
        instructions_rect=instructions_rect,
        instructions_lines=instructions_lines,
        models_rect=models_rect,
        model_rows=model_rows,
    )


def draw_panel(screen: pygame.Surface, rect: pygame.Rect, fill_color: tuple[int, int, int, int], border_color: tuple[int, int, int]) -> None:
    panel = pygame.Surface(rect.size, pygame.SRCALPHA)
    panel.fill(fill_color)
    screen.blit(panel, rect.topleft)
    pygame.draw.rect(screen, border_color, rect, 1, border_radius=10)


def draw_model_panel(
    screen: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    models: Sequence[SceneModel],
    selected_indices: set[int],
    active_index: int,
) -> None:
    draw_panel(screen, rect, (10, 12, 18, 190), (90, 100, 120))
    title = font.render(f"Models ({len(models)})", True, (245, 245, 245))
    screen.blit(title, (rect.x + 10, rect.y + 8))

    visible_models = models[: min(len(models), 14)]
    for index, model in enumerate(visible_models):
        is_active = index == active_index
        is_selected = index in selected_indices
        state = "on" if model.visible else "off"
        reduced = f"{len(model.faces)}/{model.original_face_count} faces"
        parent = f" <- {models[model.parent_index].name}" if model.parent_index is not None else ""
        prefix = ">" if is_active else ("*" if is_selected else " ")
        line = f"{prefix} {index + 1:>2}. [{state}] {model.name}{parent} | {reduced} | cull:{'on' if model.backface_culling else 'off'}"
        color = (255, 255, 220) if is_active else ((220, 240, 220) if is_selected else (220, 220, 220))
        screen.blit(font.render(line, True, color), (rect.x + 10, rect.y + 34 + index * 22))

    if len(models) > len(visible_models):
        extra_line = font.render(f"... and {len(models) - len(visible_models)} more models", True, (180, 180, 180))
        screen.blit(extra_line, (rect.x + 10, rect.bottom - 22))


def draw_instructions_panel(screen: pygame.Surface, font: pygame.font.Font, rect: pygame.Rect, lines: Sequence[str]) -> None:
    draw_panel(screen, rect, (10, 12, 18, 190), (90, 100, 120))
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, (235, 235, 235)), (rect.x + 10, rect.y + 10 + index * 20))


def is_ancestor(ancestor_index: int, descendant_index: int, models: Sequence[SceneModel]) -> bool:
    current = models[descendant_index].parent_index
    while current is not None:
        if current == ancestor_index:
            return True
        current = models[current].parent_index
    return False


def select_single_model(model_index: int) -> set[int]:
    return {model_index}


def toggle_model_selection(selected_indices: set[int], model_index: int) -> set[int]:
    updated = set(selected_indices)
    if model_index in updated and len(updated) > 1:
        updated.remove(model_index)
    else:
        updated.add(model_index)
    if not updated:
        updated.add(model_index)
    return updated


def group_selected_models(models: Sequence[SceneModel], selected_indices: set[int], active_index: int) -> None:
    for index in selected_indices:
        if index == active_index:
            continue
        if not is_ancestor(index, active_index, models):
            models[index].parent_index = active_index


def unparent_selected_models(models: Sequence[SceneModel], selected_indices: set[int]) -> None:
    for index in selected_indices:
        models[index].parent_index = None


def slugify(text: str) -> str:
    slug = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in text)
    return slug.strip("_") or "scene"


def save_screenshot(screen: pygame.Surface, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(screen, str(output_path))


def export_scene_snapshot(scene: Scene, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "axes_visible": scene.axes_visible,
        "performance": {
            "draw_edges": scene.draw_edges,
            "depth_sort": scene.depth_sort,
        },
        "models": [],
    }
    for model in scene.models:
        data["models"].append(
            {
                "path": str(model.source_path),
                "name": model.name,
                "offset": [float(component) for component in model.position],
                "rotation_deg": [float(component) for component in model.rotation_deg],
                "scale": float(model.scale),
                "color": list(model.color),
                "visible": model.visible,
                "backface_culling": model.backface_culling,
                "parent": scene.models[model.parent_index].name if model.parent_index is not None else None,
            }
        )
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def surface_to_image(screen: pygame.Surface) -> Image.Image:
    raw_pixels = pygame.image.tostring(screen, "RGB")
    return Image.frombytes("RGB", screen.get_size(), raw_pixels)


def save_recording(frames: Sequence[Image.Image], output_path: Path, frame_duration_ms: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    first_frame, *rest_frames = frames
    first_frame.save(
        output_path,
        save_all=True,
        append_images=rest_frames,
        duration=frame_duration_ms,
        loop=0,
    )


def resolve_model_local_transform(model: SceneModel, animation_time: float, play_animations: bool) -> tuple[np.ndarray, np.ndarray, float]:
    position = model.position.astype(np.float32, copy=True)
    rotation_deg = model.rotation_deg.astype(np.float32, copy=True)
    scale = float(model.scale)

    if play_animations:
        animated_position, animated_rotation, animated_scale = sample_model_animation(model.animation, animation_time)
        position = position + animated_position
        rotation_deg = rotation_deg + animated_rotation
        scale = max(1e-6, scale * animated_scale)

    return position, rotation_deg, scale


def resolve_model_local_matrix(model: SceneModel, animation_time: float, play_animations: bool) -> np.ndarray:
    position, rotation_deg, scale = resolve_model_local_transform(model, animation_time, play_animations)
    return build_transform_matrix(position, rotation_deg, scale)


def resolve_model_world_matrix(
    index: int,
    models: Sequence[SceneModel],
    animation_time: float,
    play_animations: bool,
    cache: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    if cache is None:
        cache = {}
    if index in cache:
        return cache[index]

    model = models[index]
    local_matrix = resolve_model_local_matrix(model, animation_time, play_animations)
    if model.parent_index is None:
        cache[index] = local_matrix
        return local_matrix

    parent_matrix = resolve_model_world_matrix(model.parent_index, models, animation_time, play_animations, cache)
    world_matrix = parent_matrix @ local_matrix
    cache[index] = world_matrix
    return world_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="View STL files or a scene JSON file with pygame")
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Optional STL or scene JSON path. Defaults to scene.json if present.",
    )
    args = parser.parse_args()

    scene = load_scene(args.input_path)

    pygame.init()
    scene_label = scene.scene_source.name if scene.scene_source is not None else (args.input_path.name if args.input_path else DEFAULT_SCENE_FILE)
    pygame.display.set_caption(f"STL Viewer - {scene_label}")
    width, height = 1200, 900
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)
    capture_root = Path("captures")
    scene_slug = slugify(scene_label)

    rotation_x = -0.4
    rotation_y = 0.6
    rotation_z = 0.0
    zoom = 0.9
    camera_distance = 3.0
    dragging = False
    last_mouse_pos = (0, 0)
    running = True
    active_model_index = 0
    selected_indices: set[int] = {0} if scene.models else set()
    edit_mode = False
    transform_mode = GIZMO_MOVE
    gizmo_drag = GizmoDragState(mode=transform_mode)
    play_animations = True
    animation_time = 0.0
    recording = False
    recorded_frames: List[Image.Image] = []
    recording_output_path: Path | None = None
    recording_started_at = ""

    axis_length = 1.4
    axes = [
        (np.array([0.0, 0.0, 0.0]), np.array([axis_length, 0.0, 0.0]), (220, 70, 70), "X"),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, axis_length, 0.0]), (70, 220, 70), "Y"),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, axis_length]), (70, 140, 255), "Z"),
    ]

    light_dir = np.array([0.3, 0.4, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    while running:
        delta_seconds = clock.tick(60) / 1000.0
        rot = rotation_matrix(rotation_x, rotation_y, rotation_z).astype(np.float32)
        if scene.models:
            active_model_index = max(0, min(active_model_index, len(scene.models) - 1))
            selected_indices = {index for index in selected_indices if 0 <= index < len(scene.models)} or {active_model_index}
            selected_indices.add(active_model_index)
        mode_label = "edit" if edit_mode else "camera"
        animation_label = "playing" if play_animations else "paused"
        recording_label = f"recording -> {recording_output_path.name}" if recording and recording_output_path is not None else "recording off"
        ui_layout = build_ui_layout(
            font,
            width,
            height,
            scene.models,
            selected_indices,
            active_model_index,
            mode_label,
            gizmo_mode_label(transform_mode),
            animation_label,
            recording_label,
        )
        selection_center_world, selection_world_matrices = selection_pivot_world(selected_indices, scene.models, animation_time, play_animations)
        gizmo_geometry = compute_gizmo_geometry(
            selected_indices,
            scene.models,
            animation_time,
            play_animations,
            rot,
            width,
            height,
            camera_distance,
            zoom,
            scene.render_center,
            scene.render_scale,
        ) if scene.models else None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = max(400, event.w), max(300, event.h)
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if ui_layout.models_rect.collidepoint(event.pos):
                        for row_index, row_rect in ui_layout.model_rows:
                            if row_rect.collidepoint(event.pos):
                                active_model_index = row_index
                                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                    selected_indices = toggle_model_selection(selected_indices, row_index)
                                else:
                                    selected_indices = select_single_model(row_index)
                                break
                        else:
                            pass
                    elif edit_mode and scene.models and gizmo_geometry is not None:
                        start_world_matrices = {index: selection_world_matrices[index] for index in collect_selected_root_indices(selected_indices, scene.models) if index in selection_world_matrices}
                        hit_ring = hit_test_gizmo_ring(event.pos, gizmo_geometry.rotation_ring_points, DEFAULT_GIZMO_HIT_RADIUS)
                        hit_scale = hit_test_gizmo_center(event.pos, gizmo_geometry.pivot_screen, DEFAULT_GIZMO_CENTER_HANDLE_RADIUS)
                        hit_axis = hit_test_gizmo_axis(event.pos, gizmo_geometry.pivot_screen, gizmo_geometry.axis_screen_points, DEFAULT_GIZMO_HIT_RADIUS)

                        if hit_ring is not None:
                            gizmo_drag = GizmoDragState(
                                active=True,
                                axis_index=hit_ring,
                                mode=GIZMO_ROTATE,
                                start_mouse=event.pos,
                                start_world_matrices=start_world_matrices,
                                selection_pivot_world=gizmo_geometry.pivot_world,
                                selection_pivot_screen=gizmo_geometry.pivot_screen,
                                start_mouse_angle=angle_between_points(gizmo_geometry.pivot_screen, event.pos),
                            )
                            dragging = False
                        elif hit_scale:
                            gizmo_drag = GizmoDragState(
                                active=True,
                                axis_index=3,
                                mode=GIZMO_SCALE,
                                start_mouse=event.pos,
                                start_world_matrices=start_world_matrices,
                                selection_pivot_world=gizmo_geometry.pivot_world,
                                selection_pivot_screen=gizmo_geometry.pivot_screen,
                                start_mouse_distance=distance_between_points(gizmo_geometry.pivot_screen, event.pos),
                            )
                            dragging = False
                        elif hit_axis is not None:
                            gizmo_drag = GizmoDragState(
                                active=True,
                                axis_index=hit_axis,
                                mode=transform_mode,
                                start_mouse=event.pos,
                                start_world_matrices=start_world_matrices,
                                selection_pivot_world=gizmo_geometry.pivot_world,
                                selection_pivot_screen=gizmo_geometry.pivot_screen,
                                axis_screen_direction=gizmo_geometry.axis_screen_directions[hit_axis],
                            )
                            dragging = False
                        else:
                            dragging = True
                            last_mouse_pos = event.pos
                    else:
                        dragging = True
                        last_mouse_pos = event.pos
                elif event.button == 4:
                    zoom *= 1.08
                elif event.button == 5:
                    zoom /= 1.08
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
                    gizmo_drag.active = False
                    gizmo_drag.axis_index = None
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                rotation_y += dx * 0.01
                rotation_x += dy * 0.01
                last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEMOTION and gizmo_drag.active and scene.models:
                apply_gizmo_drag(
                    scene.models,
                    gizmo_drag,
                    event.pos,
                    selected_indices,
                    scene.render_scale,
                    zoom,
                    animation_time,
                    play_animations,
                )
            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                if edit_mode and event.key == pygame.K_t:
                    transform_mode = GIZMO_MOVE
                    gizmo_drag.mode = transform_mode
                elif edit_mode and event.key == pygame.K_r and mods & pygame.KMOD_SHIFT:
                    rotation_x, rotation_y, rotation_z = -0.4, 0.6, 0.0
                    zoom = 0.9
                elif event.key == pygame.K_r and mods & pygame.KMOD_CTRL:
                    scene.render_center, scene.render_scale = compute_scene_frame(scene.models)
                elif edit_mode and event.key == pygame.K_r:
                    transform_mode = GIZMO_ROTATE
                    gizmo_drag.mode = transform_mode
                elif edit_mode and event.key == pygame.K_y:
                    transform_mode = GIZMO_SCALE
                    gizmo_drag.mode = transform_mode
                elif edit_mode and event.key == pygame.K_g and scene.models:
                    group_selected_models(scene.models, selected_indices, active_model_index)
                    scene.render_center, scene.render_scale = compute_scene_frame(scene.models)
                elif edit_mode and event.key == pygame.K_u and scene.models:
                    unparent_selected_models(scene.models, selected_indices)
                    scene.render_center, scene.render_scale = compute_scene_frame(scene.models)
                elif event.key == pygame.K_r:
                    rotation_x, rotation_y, rotation_z = -0.4, 0.6, 0.0
                    zoom = 0.9
                elif event.key == pygame.K_x:
                    scene.axes_visible = not scene.axes_visible
                elif event.key == pygame.K_z:
                    scene.draw_edges = not scene.draw_edges
                elif event.key == pygame.K_f:
                    scene.depth_sort = not scene.depth_sort
                elif event.key == pygame.K_m:
                    edit_mode = not edit_mode
                elif event.key == pygame.K_p:
                    play_animations = not play_animations
                elif event.key == pygame.K_s and mods & pygame.KMOD_CTRL:
                    snapshot_path = capture_root / scene_slug / f"{scene_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    export_scene_snapshot(scene, snapshot_path)
                elif event.key == pygame.K_F12:
                    screenshot_path = capture_root / scene_slug / f"{scene_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    save_screenshot(screen, screenshot_path)
                elif event.key == pygame.K_v:
                    if recording:
                        recording = False
                        if recording_output_path is not None:
                            save_recording(recorded_frames, recording_output_path, 33)
                        recorded_frames.clear()
                        recording_output_path = None
                    else:
                        recording_started_at = datetime.now().strftime('%Y%m%d_%H%M%S')
                        recording_dir = capture_root / scene_slug / recording_started_at
                        recording_output_path = recording_dir / f"{scene_slug}_{recording_started_at}.gif"
                        recording_dir.mkdir(parents=True, exist_ok=True)
                        recorded_frames.clear()
                        recording = True
                elif event.key == pygame.K_c and scene.models:
                    scene.models[active_model_index].backface_culling = not scene.models[active_model_index].backface_culling
                elif event.key == pygame.K_TAB and scene.models:
                    active_model_index = (active_model_index + 1) % len(scene.models)
                    selected_indices.add(active_model_index)
                elif event.key == pygame.K_SPACE and scene.models:
                    scene.models[active_model_index].visible = not scene.models[active_model_index].visible
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    model_index = event.key - pygame.K_1
                    if model_index < len(scene.models):
                        scene.models[model_index].visible = not scene.models[model_index].visible

        if scene.models and active_model_index >= len(scene.models):
            active_model_index = 0
        if scene.models and not selected_indices:
            selected_indices = {active_model_index}

        keys = pygame.key.get_pressed()
        mods = pygame.key.get_mods()
        if edit_mode and scene.models:
            selected_model = scene.models[active_model_index]
            move_step = 0.02 / max(scene.render_scale, 1e-6)
            rotate_step = 45.0 * delta_seconds
            scale_step = 0.6 * delta_seconds
            speed_multiplier = 4.0 if mods & pygame.KMOD_SHIFT else 1.0
            move_step *= speed_multiplier
            rotate_step *= speed_multiplier
            scale_step *= speed_multiplier

            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                selected_model.position[0] -= move_step
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                selected_model.position[0] += move_step
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                selected_model.position[1] += move_step
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                selected_model.position[1] -= move_step
            if keys[pygame.K_q]:
                selected_model.position[2] -= move_step
            if keys[pygame.K_e]:
                selected_model.position[2] += move_step
            if keys[pygame.K_j]:
                selected_model.rotation_deg[1] -= rotate_step
            if keys[pygame.K_l]:
                selected_model.rotation_deg[1] += rotate_step
            if keys[pygame.K_i]:
                selected_model.rotation_deg[0] -= rotate_step
            if keys[pygame.K_k]:
                selected_model.rotation_deg[0] += rotate_step
            if keys[pygame.K_u]:
                selected_model.rotation_deg[2] -= rotate_step
            if keys[pygame.K_o]:
                selected_model.rotation_deg[2] += rotate_step
            if keys[pygame.K_LEFTBRACKET]:
                selected_model.scale = max(0.05, selected_model.scale * (1.0 - scale_step))
            if keys[pygame.K_RIGHTBRACKET]:
                selected_model.scale = selected_model.scale * (1.0 + scale_step)
        else:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                rotation_y -= 0.02
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                rotation_y += 0.02
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                rotation_x -= 0.02
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                rotation_x += 0.02
            if keys[pygame.K_q]:
                rotation_z -= 0.02
            if keys[pygame.K_e]:
                rotation_z += 0.02

        if play_animations:
            animation_time += delta_seconds

        screen.fill((18, 20, 28))

        rendered_triangles = []
        visible_triangle_count = 0

        if scene.axes_visible:
            for start, end, color, label in axes:
                axis_points = np.vstack([
                    (start - scene.render_center) * scene.render_scale,
                    (end - scene.render_center) * scene.render_scale,
                ]).astype(np.float32)
                axis_points = axis_points @ rot.T
                projected = project_points(axis_points, width, height, camera_distance, zoom)
                pygame.draw.line(screen, color, projected[0], projected[1], 3)
                label_surface = font.render(label, True, color)
                screen.blit(label_surface, (projected[1][0] + 6, projected[1][1] + 2))

        world_cache: dict[int, np.ndarray] = {}
        for index, model in enumerate(scene.models):
            if not model.visible or len(model.faces) == 0:
                continue

            world_matrix = resolve_model_world_matrix(index, scene.models, animation_time, play_animations, world_cache)
            transformed_vertices = transform_points(model.vertices, world_matrix)
            transformed_vertices = (transformed_vertices - scene.render_center) * scene.render_scale
            transformed_vertices = transformed_vertices @ rot.T
            face_vertices = transformed_vertices[model.faces]

            edge_1 = face_vertices[:, 1] - face_vertices[:, 0]
            edge_2 = face_vertices[:, 2] - face_vertices[:, 0]
            normals = np.cross(edge_1, edge_2)
            lengths = np.linalg.norm(normals, axis=1)
            valid = lengths > 1e-12
            normals[valid] /= lengths[valid, None]

            if model.backface_culling:
                visible_mask = valid & (normals[:, 2] < 0.0)
            else:
                visible_mask = valid
            if not np.any(visible_mask):
                continue

            visible_faces = face_vertices[visible_mask]
            visible_normals = normals[visible_mask]
            shades = np.maximum(0.15, -(visible_normals @ light_dir))
            colors = tint_colors(model.color, shades)
            depths = visible_faces[:, :, 2].mean(axis=1)
            projected_faces = project_points(
                visible_faces.reshape(-1, 3), width, height, camera_distance, zoom
            ).reshape(-1, 3, 2)

            visible_triangle_count += len(projected_faces)

            if scene.depth_sort:
                for depth, points, color in zip(depths, projected_faces, colors):
                    rendered_triangles.append((float(depth), points, tuple(int(c) for c in color)))
            else:
                for points, color in zip(projected_faces, colors):
                    fill_color = tuple(int(c) for c in color)
                    pygame.draw.polygon(screen, fill_color, points)
                    if scene.draw_edges:
                        pygame.draw.polygon(screen, (15, 15, 15), points, 1)

        if scene.depth_sort:
            rendered_triangles.sort(key=lambda item: item[0], reverse=True)
            for _, points, color in rendered_triangles:
                pygame.draw.polygon(screen, color, points)
                if scene.draw_edges:
                    pygame.draw.polygon(screen, (15, 15, 15), points, 1)

        if edit_mode and scene.models:
            draw_gizmo(
                screen,
                font,
                selected_indices,
                scene.models,
                animation_time,
                play_animations,
                rot,
                width,
                height,
                camera_distance,
                zoom,
                scene.render_center,
                scene.render_scale,
                transform_mode,
            )

        draw_instructions_panel(screen, font, ui_layout.instructions_rect, ui_layout.instructions_lines)
        draw_model_panel(screen, font, ui_layout.models_rect, scene.models, selected_indices, active_model_index)

        fps = clock.get_fps()

        pygame.display.flip()
        if recording:
            recorded_frames.append(surface_to_image(screen))

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
