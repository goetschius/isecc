# isecc
Icosahedral Subparticle Extraction and Correlated Classification

Project transfer from GitLab underway

Versions of `ISECC_recombine*.py` are out-of-date, should not be used, and have
been moved to `LEGACY/`.

Versions of `ISECC_local_motions*.py` are historical and have been moved to
`LEGACY/`.

Versions of `ISECC_correlated_classification*.py` and
`ISECC_csparc_subparticle_create.py` are historical and have been moved to
`LEGACY/`.

## CryoSPARC WalkAlongZ Pathway

Use `ISECC_csparc_symmetryexpand.py --walk-along-z` when you want to preserve
one output particle per input particle, keep the existing poses, and translate
each particle along its own local Z axis. This replaces the old
`--no-symmetry-expand` wording, which still works as a legacy alias.

`--subparticle-distance` gives the local-Z distance in Angstroms. Negative
values walk in the opposite direction, which is useful for reversing a previous
+Z subparticle shift.

Example from the P4/J55 check:

```bash
/home/daniel/coding/ISECC/isecc/.venv/bin/python \
  /home/daniel/coding/ISECC/isecc/ISECC_csparc_symmetryexpand.py \
  /mnt/csparc_storage/CS-psa53/J55/particles_0004.cs \
  --walk-along-z \
  --passthrough /mnt/csparc_storage/CS-psa53/J54/J54_passthrough_particles_class_2.cs \
  --subparticle-distance -117 \
  --output /mnt/csparc_storage/CS-psa53/J55/J55_particles_0004_WalkAlongZMinus117.cs
```

## CryoSPARC 3DVA ClusterMatch Pathway

3DVA ClusterMatch: rotate one or more 3DVA particle clusters into a shared pose
frame, without changing particle translations or defocus.

Use `ISECC_3dva_cluster_match.py` for this pathway. It rotates only
`alignments3D/pose`, preserving `alignments3D/shift` and CTF defocus fields.
The old `rotate_cs_particles_about_z.py` filename remains as a legacy wrapper.

Example from the P4/J63 cluster-match run:

```bash
/home/daniel/coding/ISECC/isecc/.venv/bin/python \
  /home/daniel/coding/ISECC/isecc/ISECC_3dva_cluster_match.py \
  --passthrough /mnt/csparc_storage/CS-psa53/J63/J63_passthrough_particles_all_clusters.cs \
  --output /mnt/csparc_storage/CS-psa53/J63/J63_clusters_000_001_rotZ_0_30deg_combined_particles.cs \
  --input-rotation /mnt/csparc_storage/CS-psa53/J63/J63_cluster_000_particles.cs 0 \
  --input-rotation /mnt/csparc_storage/CS-psa53/J63/J63_cluster_001_particles.cs 30
```

The default rotation mode is `global`, meaning:

```text
new_pose = z_rotation * original_pose
```

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

## CryoSPARC I2 Fivefold Local Reconstruction Note

On 2026-08-02, `ISECC_csparc_symmetryexpand.py` was created as the forward
strategy-table implementation for I1/I2 subparticle expansion. The transitional
scripts used during debugging were moved to `LEGACY/`:

```text
LEGACY/ISECC_csparc_symmetryexpand_unique.py
LEGACY/ISECC_csparc_symmetryexpand_i2_sensible.py
```

The original I2 fivefold bug was in subparticle creation with
`--subparticle-distance`. The broken behavior generated 12 rows per source
particle, but several rows used duplicated 2D shifted centers. In CryoSPARC this
looked like fewer than 12 circles on the source particle and repeated lowpass
tiles.

For I2 fivefold local/C1 reconstructions, use the default I2 fivefold strategy
in `ISECC_csparc_symmetryexpand.py`. It uses native I2 operators mapped into the
validated I2/I1-frame convention. This was visually validated on J24: the wrong
native-I2/per-vertex attempts produced non-fivefold-looking density, while the
strategy-table I2 fivefold output restored the expected fivefold-axis feature.

Unique subparticle expansion now checks a deterministic 1% sample of input
particles before writing output. For the selected symmetry and vertex, the
script applies the chosen operators with the real offset convention and fails if
any duplicate subparticle centers are produced in the sampled particles.

Default unique expansion uses the duplicate-free inverse operator selection:

```text
sym.inverse.rotate(ideal_vertex)
```

For I1 fivefold, this gives the old hard-coded ISECC list:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 31]
```

For native I2 inverse fivefold, this gives:

```text
[0, 1, 6, 8, 9, 11, 24, 29, 30, 32, 37, 55]
```

For the production I2 fivefold strategy, native I2 operators are ordered to
match the validated I2/I1-frame convention:

```text
[0, 55, 1, 24, 32, 11, 30, 37, 9, 8, 6, 29]
```

The archived J45 forward-operator convention
`[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 24]` can duplicate subparticle centers
under the actual offset convention, so it is no longer the default unique path.

Offsets should continue to come from:

```text
(symmetry * pose).inverse.rotate(subparticle_vector)
```

Recommended J24 I2 strategy-table output:

```text
/mnt/csparc_storage/CS-psa53/J24/J24_006_particles_I2_sensibleNative_symmetryexpand_unique_fivefold_toZ_subparticle_shiftMinus.cs
```

Generation command:

```bash
/home/daniel/coding/ISECC/isecc/.venv/bin/python \
  /home/daniel/coding/ISECC/isecc/ISECC_csparc_symmetryexpand.py \
  /mnt/csparc_storage/CS-psa53/J24/J24_006_particles.cs \
  --symmetry I2 \
  --vertex fivefold \
  --align-to-z \
  --align-to-z-mode csparc \
  --subparticle-distance 374.4 \
  --passthrough /mnt/csparc_storage/CS-psa53/J24/J24_passthrough_particles.cs \
  --output /mnt/csparc_storage/CS-psa53/J24/J24_006_particles_I2_sensibleNative_symmetryexpand_unique_fivefold_toZ_subparticle_shiftMinus.cs
```

Verification from the corrected file:

```text
Input particles: 4392
Output particles: 52704
Unique 2D shift count histogram at 0.01 px: {12: 4392}
SHA256: af6a1547ac9ed96db130d558b7828a3472f9b64125cf92c608b61e4ffa6b32c2
```

Exhaustive duplicate audits run on 2026-08-02:

```text
J45 I1 input:
  /mnt/csparc_storage/CS-ken10-blade-polyfab/J45/J45_007_particles.cs
  particles checked: 23203 / 23203
  I1 fivefold operators: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 31]
  unique 3D local vector histogram at 0.001: {12: 23203}

J24 I2 input:
  /mnt/csparc_storage/CS-psa53/J24/J24_006_particles.cs
  particles checked: 4392 / 4392
  native I2 inverse fivefold operators: [0, 1, 6, 8, 9, 11, 24, 29, 30, 32, 37, 55]
  unique 3D local vector histogram at 0.001: {12: 4392}

J24 generated outputs:
  sensible native I2 output, unique 2D shift histogram at 0.01 px: {12: 4392}
  legacy I2->I1-frame output, unique 2D shift histogram at 0.01 px: {12: 4392}
```

Import command template:

```bash
/home/daniel/coding/ISECC/isecc/.venv/bin/python \
  /home/daniel/coding/ISECC/isecc/import_symexpanded_particles_to_cryosparc.py \
  /mnt/csparc_storage/CS-psa53/J24/J24_006_particles_I2_sensibleNative_symmetryexpand_unique_fivefold_toZ_subparticle_shiftMinus.cs \
  --project P? \
  --workspace W? \
  --host krypton \
  --base-port 61000 \
  --title "J24 I2 fivefold subparticles sensible native I2"
```

The native-I2 merge candidate has now been folded into:

```text
ISECC_csparc_symmetryexpand.py
```

This script uses native I2 fivefold operators, mapped into the validated
legacy-frame order:

```text
[0, 55, 1, 24, 32, 11, 30, 37, 9, 8, 6, 29]
```

Those native I2 operators were chosen by matching the local fivefold vectors
from the weird-but-correct legacy path. The script computes a per-operator
alignment correction so the final poses match the legacy-frame output while the
expansion itself uses native I2 operators.

Equivalence test command:

```bash
/home/daniel/coding/ISECC/isecc/.venv/bin/python \
  /home/daniel/coding/ISECC/isecc/ISECC_csparc_symmetryexpand.py \
  /mnt/csparc_storage/CS-psa53/J24/J24_006_particles.cs \
  --symmetry I2 \
  --vertex fivefold \
  --align-to-z \
  --align-to-z-mode csparc \
  --subparticle-distance 374.4 \
  --passthrough /mnt/csparc_storage/CS-psa53/J24/J24_passthrough_particles.cs \
  --output /tmp/J24_new_strategy_I2.cs
```

Comparison to the weird-but-correct J24 reference, excluding randomized `uid`:

```text
alignments3D/pose: exact match
alignments3D/shift: max abs diff 0.0165606 px
ctf/df1_A: max abs diff 0.0253906 A
ctf/df2_A: max abs diff 0.0253906 A
```

For the first J24 particle specifically:

```text
alignments3D/pose: exact match
alignments3D/shift: max abs diff 0.0147324 px
ctf/df1_A and ctf/df2_A: max abs diff 0.0214844 A
```

Paired I2/I1 same-capsid reconstruction test:

```text
Validated visually on 2026-08-02.

I2 version:
  CryoSPARC: P4 W2 J44
  title: J24 I2 fivefold subparticles sensible native I2
  generated file:
    /mnt/csparc_storage/CS-psa53/J24/J24_006_particles_I2_sensibleNative_symmetryexpand_unique_fivefold_toZ_subparticle_shiftMinus.cs
  imported particles:
    /mnt/csparc_storage/CS-psa53/J44/particles_0000.cs
  SHA256:
    af6a1547ac9ed96db130d558b7828a3472f9b64125cf92c608b61e4ffa6b32c2
  command copies:
    /mnt/csparc_storage/CS-psa53/J44/J44_I2_generation_command.txt
    /mnt/csparc_storage/CS-psa53/J44/J44_I2_import_command.txt

I1 version:
  CryoSPARC: P4 W2 J45
  title: J43 I1 fivefold subparticles
  generated file:
    /mnt/csparc_storage/CS-psa53/J43/J43_004_particles_I1_symmetryexpand_unique_fivefold_toZ_subparticle_shiftMinus.cs
  imported particles:
    /mnt/csparc_storage/CS-psa53/J45/particles_0000.cs
  SHA256:
    15639b3c5e9a286f98447e43c8e494fa0705f5c082c9637b91ee7930e698343b
  command copies:
    /mnt/csparc_storage/CS-psa53/J45/J45_I1_generation_command.txt
    /mnt/csparc_storage/CS-psa53/J45/J45_I1_import_command.txt
```
