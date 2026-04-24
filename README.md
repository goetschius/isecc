# isecc
Icosahedral Subparticle Extraction and Correlated Classification

Project transfer from GitLab underway

Versions of ISECC_recombine are currently out-of-date and should not be used.

BUG NOTICE:
ISECC_star_subparticle_subtract will fail if the input star already has rlnOriginalImageName. 
-- i.e., from a relion subtract job
-- current work-around is to remove that column using awk
-- long-term fix is to avoid repurposing that metadata item with ISECC

GUI wrapper:
`volume_viewer_gui.py` provides a PySide6 + pyvistaqt desktop wrapper for the two GIF renderers:
- `volume_viewer_pyvista_chain_palettes_gif.py`
- `volume_viewer_pyvista_ken10_experimental_gif.py`

Run it with:

```bash
python volume_viewer_gui.py
```

The GUI gives you:
- file pickers for MRC, mmCIF, and output GIF paths
- live mesh preview in a Qt PyVista viewport
- subprocess-based GIF export using the existing CLI renderers
- an export log pane so renderer output stays visible
