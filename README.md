# Robotics Pose Visualiser

A small Python 3D model viewer built with `pygame`. It supports loading single meshes or JSON scene files that can position multiple models with per-model offsets, rotations, visibility, color, live transforms, and simple keyframe animation.

## Requirements

- Python 3.12 or newer
- `numpy`
- `pygame`
- `Pillow`
- `trimesh`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

The viewer looks for a `scene.json` file automatically when launched with no arguments.

```bash
python 3d_viewer.py
```

You can also pass a specific scene or supported mesh file:

```bash
python 3d_viewer.py path/to/scene.json
python 3d_viewer.py path/to/model.stl
python 3d_viewer.py path/to/model.obj
python 3d_viewer.py path/to/model.ply
```

If you are using the virtual environment in this workspace, run:

```bash
.venv/bin/python 3d_viewer.py
```

## Scene File

Scene files are standard JSON. Put all of your model parts in one file and set their offsets, rotations, and optional animation tracks there.

Use [sample_scene.json](sample_scene.json) as a starting point and replace the STL paths with your own files.

Example structure:

```json
{
	"axes_visible": true,
	"performance": {
		"max_faces_per_model": 30000,
		"downsample_step": 1,
		"deduplicate_vertices": true,
		"draw_edges": false,
		"depth_sort": false,
		"backface_culling": true
	},
	"models": [
		{
			"path": "models/base.stl",
			"name": "base",
			"offset": [0, 0, 0],
			"rotation_deg": [0, 0, 0],
			"scale": 1.0,
			"color": [220, 90, 90],
			"visible": true,
			"max_faces": 20000,
			"backface_culling": true
		},
		{
			"path": "models/arm.stl",
			"name": "arm",
			"offset": [0, 0, 35],
			"rotation_deg": [0, 0, 90],
			"color": "#66ccff",
			"visible": true,
			"max_faces": 15000,
			"backface_culling": true,
			"animation": {
				"enabled": true,
				"loop": true,
				"speed": 1.0,
				"rotation_deg": [
					[0.0, [0, 0, 0]],
					[1.0, [0, 0, 30]],
					[2.0, [0, 0, 0]]
				],
				"position": [
					[0.0, [0, 0, 0]],
					[1.0, [0, 12, 0]],
					[2.0, [0, 0, 0]]
				]
			}
		}
	]
}
```

The viewer resolves model paths relative to the scene file, so you can keep the JSON and model files together.

Supported mesh formats come from `trimesh`, which includes common formats such as STL, OBJ, PLY, GLB/GLTF, OFF, DAE, and others depending on installed extras.

## Controls

- Left mouse drag: rotate the camera
- Mouse wheel: zoom in and out
- Arrow keys or `WASD`: rotate the camera when not editing a model
- `Q` / `E`: roll the camera when not editing a model
- `X`: toggle coordinate axes
- `Z`: toggle triangle edge outlines
- `F`: toggle depth sorting (better visuals, slower)
- `C`: toggle backface culling for the selected model
- `Tab`: select the next model in the scene
- `Space`: toggle the selected model on or off
- `M`: toggle selected-model edit mode
- `P`: play or pause built-in animations
- `F12`: save a screenshot PNG to `captures/`
- `Ctrl+S`: export the current scene state to JSON
- `V`: record a GIF animation to `captures/`
- `1` to `9`: toggle individual models by index
- `R`: reset the view
- Shift-click a model in the list to add or remove it from the multi-selection.
- `G`: parent the selected models under the active model.
- `U`: clear the parent for the selected models.
- In edit mode, click and drag the gizmo handles to move, rotate, or scale the selected model.
- `T`, `R`, and `Y`: switch the gizmo to move, rotate, or scale mode while editing.
- `Shift+R`: reset the view while in edit mode.

## Performance Tuning

- `performance.max_faces_per_model`: global triangle cap per model.
- `performance.downsample_step`: keep every Nth face (for example `2` keeps half).
- `performance.deduplicate_vertices`: reuses identical STL vertices to reduce transform work.
- `performance.draw_edges`: triangle outlines are helpful but slower.
- `performance.depth_sort`: painter sorting improves overlap quality but costs CPU.
- `performance.backface_culling`: hides triangles facing away from the camera.
- `models[].max_faces`: per-model face cap override for heavy parts.
- `models[].backface_culling`: per-model override for meshes with inconsistent winding.

Start with `depth_sort: false`, `draw_edges: false`, and lower `max_faces` on the heaviest STL first.

If you see holes or missing patches, first try `depth_sort: true`, then set `backface_culling: false` on the affected model.

## Notes

- The viewer supports both ASCII and binary STL files.
- Scene transforms stay editable after load, so you can move parts around without reopening the file.
- The model list in the viewer shows visibility and face reduction counts so you can tune performance quickly.
- Press `Ctrl+R` to reframe the camera around the current model positions after large edits.