"""STL viewer with scene-file support and coordinate axes using pygame.

Controls:
- Left mouse drag: rotate
- Mouse wheel: zoom
- Arrow keys / WASD: rotate
- Q/E: roll
- X: toggle axes
- Z: toggle triangle edges
- F: toggle depth sorting
- C: toggle backface culling for selected model
- Tab: select next model
- Space: toggle selected model
- 1-9: toggle model visibility by index
- R: reset view

Usage:
    python stl_viewer.py
    python stl_viewer.py path/to/scene.json
    python stl_viewer.py path/to/model.stl
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import pygame

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


@dataclass
class Scene:
    models: List[SceneModel]
    axes_visible: bool
    scene_source: Path | None
    draw_edges: bool
    depth_sort: bool


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


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rzm @ rym @ rxm


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
    if not models:
        return []

    all_vertices = np.vstack([model.vertices for model in models])
    center = all_vertices.mean(axis=0)
    bounds = all_vertices.max(axis=0) - all_vertices.min(axis=0)
    longest = float(np.max(bounds))
    scale = 1.0 if longest == 0 else 2.0 / longest

    return [((model.vertices - center) * scale).astype(np.float32) for model in models]


def load_scene(input_path: Path | None) -> Scene:
    source_path, base_dir = load_scene_source(input_path)

    if source_path.suffix.lower() != ".json":
        mesh = deduplicate_mesh_vertices(load_stl(source_path))
        faces = downsample_faces(mesh.faces, DEFAULT_MAX_FACES_PER_MODEL, DEFAULT_DOWNSAMPLE_STEP)
        return Scene(
            models=[
                SceneModel(
                    name=source_path.stem,
                    source_path=source_path,
                    vertices=mesh.vertices,
                    faces=faces,
                    color=DEFAULT_MODEL_COLOR,
                    visible=True,
                    original_face_count=len(mesh.faces),
                    backface_culling=DEFAULT_BACKFACE_CULLING,
                )
            ],
            axes_visible=DEFAULT_AXES_VISIBLE,
            scene_source=None,
            draw_edges=DEFAULT_DRAW_EDGES,
            depth_sort=DEFAULT_DEPTH_SORT,
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
    for index, raw_model in enumerate(raw_models, start=1):
        if not isinstance(raw_model, dict):
            raise ValueError("Each model entry must be a JSON object")

        model_path_value = raw_model.get("path")
        if not isinstance(model_path_value, str) or not model_path_value.strip():
            raise ValueError("Each model entry needs a non-empty 'path'")

        model_path = (base_dir / model_path_value).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        mesh = load_stl(model_path)
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

        rotation = rotation_matrix(*(math.radians(component) for component in rotation_deg))
        transformed_vertices = ((mesh.vertices * scale) @ rotation.T) + offset

        models.append(
            SceneModel(
                name=name,
                source_path=model_path,
                vertices=transformed_vertices.astype(np.float32),
                faces=reduced_faces,
                color=color,
                visible=visible,
                original_face_count=len(mesh.faces),
                backface_culling=backface_culling,
            )
        )

    return Scene(
        models=models,
        axes_visible=axes_visible,
        scene_source=source_path,
        draw_edges=draw_edges,
        depth_sort=depth_sort,
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
    normalized_model_vertices = prepare_scene_for_render(scene.models)

    pygame.init()
    scene_label = scene.scene_source.name if scene.scene_source is not None else (args.input_path.name if args.input_path else DEFAULT_SCENE_FILE)
    pygame.display.set_caption(f"STL Viewer - {scene_label}")
    width, height = 1200, 900
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    rotation_x = -0.4
    rotation_y = 0.6
    rotation_z = 0.0
    zoom = 0.9
    camera_distance = 3.0
    dragging = False
    last_mouse_pos = (0, 0)
    running = True
    selected_model_index = 0

    axis_length = 1.4
    axes = [
        (np.array([0.0, 0.0, 0.0]), np.array([axis_length, 0.0, 0.0]), (220, 70, 70), "X"),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, axis_length, 0.0]), (70, 220, 70), "Y"),
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, axis_length]), (70, 140, 255), "Z"),
    ]

    light_dir = np.array([0.3, 0.4, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = max(400, event.w), max(300, event.h)
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse_pos = event.pos
                elif event.button == 4:
                    zoom *= 1.08
                elif event.button == 5:
                    zoom /= 1.08
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                rotation_y += dx * 0.01
                rotation_x += dy * 0.01
                last_mouse_pos = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    rotation_x, rotation_y, rotation_z = -0.4, 0.6, 0.0
                    zoom = 0.9
                elif event.key == pygame.K_x:
                    scene.axes_visible = not scene.axes_visible
                elif event.key == pygame.K_z:
                    scene.draw_edges = not scene.draw_edges
                elif event.key == pygame.K_f:
                    scene.depth_sort = not scene.depth_sort
                elif event.key == pygame.K_c and scene.models:
                    selected_model = scene.models[selected_model_index]
                    selected_model.backface_culling = not selected_model.backface_culling
                elif event.key == pygame.K_TAB and scene.models:
                    selected_model_index = (selected_model_index + 1) % len(scene.models)
                elif event.key == pygame.K_SPACE and scene.models:
                    scene.models[selected_model_index].visible = not scene.models[selected_model_index].visible
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    model_index = event.key - pygame.K_1
                    if model_index < len(scene.models):
                        scene.models[model_index].visible = not scene.models[model_index].visible

        if scene.models and selected_model_index >= len(scene.models):
            selected_model_index = 0

        keys = pygame.key.get_pressed()
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

        screen.fill((18, 20, 28))

        rot = rotation_matrix(rotation_x, rotation_y, rotation_z).astype(np.float32)
        rendered_triangles = []
        visible_triangle_count = 0

        if scene.axes_visible:
            for start, end, color, label in axes:
                axis_points = np.vstack([start @ rot.T, end @ rot.T]).astype(np.float32)
                projected = project_points(axis_points, width, height, camera_distance, zoom)
                pygame.draw.line(screen, color, projected[0], projected[1], 3)
                label_surface = font.render(label, True, color)
                screen.blit(label_surface, (projected[1][0] + 6, projected[1][1] + 2))

        for model, base_vertices in zip(scene.models, normalized_model_vertices):
            if not model.visible or len(model.faces) == 0:
                continue

            transformed_vertices = base_vertices @ rot.T
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

        fps = clock.get_fps()
        top_left_lines = [
            f"Scene: {scene_label}",
            f"Models: {len(scene.models)} | Visible triangles: {visible_triangle_count} | FPS: {fps:0.1f}",
            "Drag: rotate | Wheel: zoom | R: reset | X: axes | Z: edges | F: depth sort | C: cull",
            "Tab/Space: select & show/hide | 1-9: toggle models",
        ]
        draw_text_block(screen, font, top_left_lines, 12, 12)

        model_start_y = 100
        for index, model in enumerate(scene.models[:12], start=1):
            is_selected = index - 1 == selected_model_index
            state = "on" if model.visible else "off"
            reduced = f"{len(model.faces)}/{model.original_face_count} faces"
            cull_state = "cull:on" if model.backface_culling else "cull:off"
            prefix = ">" if is_selected else " "
            model_line = f"{prefix} {index:>2}. [{state}] {model.name} ({model.source_path.name}) {reduced} {cull_state}"
            color = (255, 255, 220) if is_selected else (220, 220, 220)
            surface = font.render(model_line, True, color)
            screen.blit(surface, (12, model_start_y + (index - 1) * 20))

        if len(scene.models) > 12:
            extra_line = font.render(f"... and {len(scene.models) - 12} more models", True, (200, 200, 200))
            screen.blit(extra_line, (12, model_start_y + 12 * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
