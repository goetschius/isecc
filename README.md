# isecc
Icosahedral Subparticle Extraction and Correlated Classification

Project transfer from GitLab underway

Versions of ISECC_recombine are currently out-of-date and should not be used.

BUG NOTICE:
ISECC_star_subparticle_subtract will fail if the input star already has rlnOriginalImageName. 
-- i.e., from a relion subtract job
-- current work-around is to remove that column using awk
-- long-term fix is to avoid repurposing that metadata item with ISECC
