# TO-DO

## CryoSPARC Tail-Tip Candidate Extraction Utility

Add an ISECC utility for extracting candidate phage tail-tip particles from a
CryoSPARC `particles.cs` file containing overlapping tail-segment picks after
helical refinement.

Within each micrograph, particles from a single tail should form an
approximately linear or gently curved chain. Short gaps may occur because
individual segments were missed, and refined segment orientations should
maintain consistent tail polarity.

### Core Workflow

- Group particles by source micrograph.
- Cluster particles into individual tail chains using coordinates and refined
  orientations.
- Bridge short gaps while avoiding connections between unrelated, crossing, or
  adjacent tails.
- Order particles along each reconstructed chain.
- Identify both ends and optionally use pose polarity to label the distal and
  capsid-proximal ends.
- Retain the final 2-3 observed particles at each selected end.
- Generate one extrapolated pick by default, or optionally two, beyond the last
  observed particle at each selected end.
- Export all selected observed and extrapolated particles to a
  CryoSPARC-compatible `.cs` file.

### Extrapolation Requirements

- Estimate the typical segment spacing robustly from consecutive particles
  within the chain.
- Estimate the endpoint direction from a local fit to the final several
  particles rather than only the last pair.
- Place extrapolated picks at successive segment-spacing intervals beyond the
  observed endpoint.
- Extrapolate orientations from the terminal refined poses while preserving
  polarity.
- Reject predictions outside the micrograph or conflicting with another chain.
- Create valid new particle identifiers and preserve required source-micrograph
  and image metadata.

### Endpoint Modes

- Default to selecting and extrapolating from both ends. The capsid-proximal and
  true tail-tip candidates can then be separated by 2D classification.
- After validating CryoSPARC's pose-direction convention, add an optional mode
  that outputs only the inferred distal end.

### Adjustable Parameters

- Expected segment spacing.
- Neighbor-search distance and maximum bridged gap.
- Positional and orientation tolerances.
- Minimum chain length.
- Number of observed particles retained at each end.
- Number of extrapolated particles: default 1, optional 0-2.
- Number of terminal particles used for trajectory fitting.
- Endpoint mode: both or distal.

### Diagnostics

Produce a diagnostic table containing:

- Micrograph.
- Chain ID.
- Ordered chain position.
- Endpoint assignment.
- Polarity score.
- Original-versus-extrapolated status.
- Coordinates.
- Orientation.
- Confidence.
- Selection status.

Ideally, also generate annotated micrographs showing reconstructed chains,
bridged gaps, fitted endpoint trajectories, and extrapolated picks.

Ambiguous branches, crossings, inconsistent orientations, strongly curved
endpoints, and low-confidence extrapolations should be flagged or excluded
rather than silently processed.
