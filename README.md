# Robotics Pose Visualiser

A small Python STL viewer built with `pygame`. It supports loading either a single STL or a JSON scene file that can position multiple STLs with per-model offsets, rotations, visibility, and color.

## Requirements

- Python 3.12 or newer
- `numpy`
- `pygame`

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

The viewer looks for a `scene.json` file automatically when launched with no arguments.

```bash
python stl_viewer.py
```

You can also pass a specific scene or STL file:

```bash
python stl_viewer.py path/to/scene.json
python stl_viewer.py path/to/model.stl
```

If you are using the virtual environment in this workspace, run:

```bash
.venv/bin/python stl_viewer.py
```

## Scene File

Scene files are standard JSON. Put all of your STL parts in one file and set their offsets and rotations there.

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
			"backface_culling": true
		}
	]
}
```

The viewer resolves model paths relative to the scene file, so you can keep the JSON and STL files together.

## Controls

- Left mouse drag: rotate the view
- Mouse wheel: zoom in and out
- Arrow keys or `WASD`: rotate the view
- `Q` / `E`: roll the view
- `X`: toggle coordinate axes
- `Z`: toggle triangle edge outlines
- `F`: toggle depth sorting (better visuals, slower)
- `C`: toggle backface culling for the selected model
- `Tab`: select the next model in the scene
- `Space`: toggle the selected model on or off
- `1` to `9`: toggle individual models by index
- `R`: reset the view

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
- Scene transforms are applied when the scene loads, so the relative model layout stays fixed while you rotate the camera.
- The model list in the viewer shows visibility and face reduction counts so you can tune performance quickly.