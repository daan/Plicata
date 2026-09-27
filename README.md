# Cable-actuated Hoberman ring

    uv sync
    uv run jupyter lab hoberman_ring.ipynb

Package `src/plicata/`:

- `hoberman.py`   geometry core (kinematics, cable routing, envelopes, DXF export). No UI dependency.
- `views.py`      shared Model + RingView, LinkView, EnvelopeView (ipycanvas); `viewer.py` older all-in-one viewer.
- `actuation.py`  servo lookup tables; `drums.py` single-motor shaped-drum alternative.
- `dynamixels.py`, `cli.py`  servo bus and the `plicata` command.
- `hoberman_ring.py`, `hoberman_3scissors.py`  standalone DXF generators: `uv run python -m plicata.hoberman_ring`.

`hoberman_ring.ipynb` wiring, length plots, export. `drawings/` generated DXF/PDF drawings.

Cable definition: a `Cable` is a list of `Waypoint(kind, unit, side, radius)` with kind in
`Jo` (outer joint), `Ji` (inner joint), `P` (centre joint). First and last are anchors.
`contraction_cable()` and `expansion_cable()` build the two standard routings.

## Running the ring (two Dynamixels, U2D2)

    uv run plicata status
    uv run plicata off           # then close the ring by hand until the links touch, take up slack
    uv run plicata set-home      # once per power-up
    uv run plicata jog con 90    # check directions: positive winds the cable in
    uv run plicata expand
    uv run plicata contract
    uv run plicata goto --dia 450

Contraction servo id 3, expansion servo id 2, 7.5 mm drums. Settings at the top of `plicata.py`.
