# Easy Nodes and MG_SuperSimple

MagicNodes provides simplified “Easy” variants that expose only high‑value controls while relying on preset files for the rest. These are grouped under the UI category `MagicNodes/Easy`.

- Presets live in `pressets/mg_cade25.cfg` and `pressets/mg_controlfusion.cfg` with INI‑like sections `Step 1..4` and simple `key: value` pairs. The token `$(ROOT)` is supported in paths and is substituted at load time.
- Loader: `mod/easy/preset_loader.py` caches by mtime and does light type parsing.
- The Step+Custom scheme keeps UI and presets in sync: choose a Step to load defaults, then optionally toggle Custom to override only the visible controls, leaving hidden parameters from the Step preset intact.

## MG_SuperSimple (Easy)

Single node that reproduces the 2/3/4‑step CADE+ControlFusion pipeline with minimal surface.

Category: `MagicNodes/Easy`

Inputs
- `model` (MODEL)
- `positive` (CONDITIONING), `negative` (CONDITIONING)
- `vae` (VAE)
- `latent` (LATENT)
- `control_net` (CONTROL_NET) — required by ControlFusion
- `reference_image` (IMAGE, optional) — forwarded to CADE
- `clip_vision` (CLIP_VISION, optional) — forwarded to CADE

Controls
- `step_count` int (1..4): how many steps to run
- `custom` toggle: when On, the visible CADE controls below override the Step presets across all steps; when Off, all CADE values come from presets
- `seed` int with `control_after_generate`
- `steps` int (default 25) — applies to steps 2..4
- `cfg` float (default 4.5)
- `denoise` float (default 0.65, clamped 0.45..0.9) — applies to steps 2..4
- `sampler_name` (default `ddim`)
- `scheduler` (default `MGHybrid`)
- `clipseg_text` string (default `hand, feet, face`)

Behavior
- Step 1 runs CADE with `Step 1` preset and forces `denoise=1.0` (single exception to the override rule). All other visible fields follow the Step+Custom logic described above.
- For steps 2..N: ControlFusion (with `Step N` preset) updates `positive/negative` based on the current image, then CADE (with `Step N` preset) refines the latent/image.
- Initial `positive/negative` come from the node inputs; subsequent steps use the latest CF outputs. `latent` is always taken from the previous CADE.
- When `custom` is Off, UI values are ignored entirely; presets define all CADE parameters.
- ControlFusion inside this node always relies on presets (no additional CF UI here) to keep the surface minimal.

Outputs
- `(LATENT, IMAGE)` from the final executed step (e.g., step 2 if `step_count=2`). No preview outputs.

Quickstart
1) Drop `MG_SuperSimple` into your graph under `MagicNodes/Easy`.
2) Connect `model/positive/negative/vae/latent`, and a `control_net` module; optionally connect `reference_image` and `clip_vision`.
3) Choose `step_count` (2/3/4). Leave `custom` Off to use pure presets, or enable it to apply your `seed/steps/cfg/denoise/sampler/scheduler/clipseg_text` across all steps (with Step 1 `denoise=1.0`).
4) Run. The node returns the final `(LATENT, IMAGE)` for the chosen depth.

Notes
- Presets are read from `pressets/mg_cade25.cfg` and `pressets/mg_controlfusion.cfg`. Keep them in UTF‑8 and prefer `$(ROOT)` over absolute paths.
- `seed` is shared across all steps for determinism; if per‑step offsets are desired later, this can be added as an option without breaking current behavior.

