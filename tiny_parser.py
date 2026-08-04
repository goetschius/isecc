import re
import numpy as np

def print_np_array(R, name=None, precision=5):
    fmt = f"{{:.{precision}f}}"

    if name:
        print(f"{name} = np.array([")
    else:
        print("np.array([")

    for i, row in enumerate(R):
        row_str = ",".join(fmt.format(x).rstrip('0').rstrip('.') if '.' in fmt.format(x) else fmt.format(x) for x in row)
        if i < 2:
            print(f"[{row_str}],")
        else:
            print(f"[{row_str}]")

    print("])\n")

def parse_relion_symops(path):
    """
    Parse RELION --print_symmetry_ops output into a list of 3x3 numpy arrays.
    """
    with open(path, 'r') as f:
        text = f.read()

    # Match blocks like:
    # R(1)=
    #   1.000  0.000  0.000
    #   0.000  1.000  0.000
    #   0.000  0.000  1.000
    pattern = re.compile(
        r'R\(\d+\)=\s*\n'
        r'\s*([-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+)\s*\n'
        r'\s*([-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+)\s*\n'
        r'\s*([-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+)',
        re.MULTILINE
    )

    ops = []
    for match in pattern.findall(text):
        rows = []
        for line in match:
            rows.append([float(x) for x in line.split()])
        ops.append(np.array(rows, dtype=float))

    return ops

ops = parse_relion_symops("relion_I1_ops.txt")

for i, R in enumerate(ops, 1):
    print_np_array(R, name=f"I1Matrix_{i}")
