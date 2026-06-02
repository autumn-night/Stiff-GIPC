#!/usr/bin/env python3
"""
stanford2ipc.py — Convert Stanford 3D Repository meshes to IPC-compatible OBJ.

Supports:
  .ply  (binary LE/BE, ascii)  — Stanford's main format
  .ply.gz                      — gzip-compressed PLY (e.g. bun_zipper.ply.gz)
  .obj  .off  .stl             — other common formats

Output: plain v + f OBJ (no normals, no UV, no mtl) — ready for:
  • IPC / Stiff-GIPC  kinematic collision objects  (use as-is)
  • fTetWild → .msh  for deformable bodies         (pipe the OBJ in)

Usage examples:
  # Single file
  python3 stanford2ipc.py bunny.ply
  python3 stanford2ipc.py bunny.ply -o bunny_ipc.obj

  # Gzipped PLY (Stanford bunny reconstruction)
  python3 stanford2ipc.py bun_zipper.ply.gz -o bunny.obj

  # Batch: all PLY in a folder
  python3 stanford2ipc.py models/ -o output/

  # Normalise to unit size (recommended for IPC scenes)
  python3 stanford2ipc.py bunny.ply --normalize
"""

import sys, os, gzip, shutil, argparse, tempfile, struct
import trimesh
import numpy as np


# ──────────────────────────── loading ────────────────────────────────────────

def load_mesh(path: str) -> trimesh.Trimesh:
    path = os.path.abspath(path)

    if path.lower().endswith('.ply.gz'):
        tmp = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        tmp.close()
        try:
            with gzip.open(path, 'rb') as gi, open(tmp.name, 'wb') as fo:
                shutil.copyfileobj(gi, fo)
            mesh = _load_with_fallback(tmp.name)
        finally:
            os.unlink(tmp.name)
    else:
        mesh = _load_with_fallback(path)

    # Merge multi-body scene into one mesh
    if isinstance(mesh, trimesh.Scene):
        parts = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(parts)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not extract a triangle mesh from {path!r}")
    return mesh


def _load_with_fallback(path: str) -> trimesh.Trimesh:
    try:
        mesh = _load(path)
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) and len(mesh.faces):
            return mesh
    except Exception as exc:
        if not _is_range_grid_ply(path):
            raise exc

    if _is_range_grid_ply(path):
        return _load_range_grid_ply(path)

    return _load(path)


def _load(path: str) -> trimesh.Trimesh:
    return trimesh.load(path, process=False, force='mesh')


def _is_range_grid_ply(path: str) -> bool:
    if not path.lower().endswith('.ply'):
        return False

    with open(path, 'rb') as fh:
        head = fh.read(4096).decode('latin1', errors='ignore').lower()
    return 'element range_grid' in head


def _load_range_grid_ply(path: str) -> trimesh.Trimesh:
    meta, body = _read_ply_header_and_body(path)
    vertices = _read_range_grid_vertices(meta, body)
    grid = _read_range_grid_indices(meta, body)
    faces = _range_grid_to_faces(grid)

    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Could not reconstruct mesh faces from range_grid PLY {path!r}")

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    valid = mesh.area_faces > 1e-15
    if not np.all(valid):
        mesh.update_faces(valid)
        mesh.remove_unreferenced_vertices()
    return mesh


def _read_ply_header_and_body(path: str) -> tuple[dict, bytes]:
    meta = {
        'format': None,
        'vertex_count': 0,
        'range_grid_count': 0,
        'num_rows': None,
        'num_cols': None,
    }

    with open(path, 'rb') as fh:
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header in {path!r}")

            text = line.decode('latin1').strip()
            lower = text.lower()

            if lower.startswith('format '):
                meta['format'] = text.split()[1]
            elif lower.startswith('element vertex '):
                meta['vertex_count'] = int(text.split()[2])
            elif lower.startswith('element range_grid '):
                meta['range_grid_count'] = int(text.split()[2])
            elif lower.startswith('obj_info num_rows '):
                meta['num_rows'] = int(text.split()[2])
            elif lower.startswith('obj_info num_cols '):
                meta['num_cols'] = int(text.split()[2])
            elif lower == 'end_header':
                body = fh.read()
                break

    if meta['format'] not in {'ascii', 'binary_big_endian', 'binary_little_endian'}:
        raise ValueError(f"Unsupported PLY format {meta['format']!r} in {path!r}")
    if not meta['vertex_count'] or not meta['range_grid_count']:
        raise ValueError(f"Missing vertex/range_grid elements in {path!r}")
    if not meta['num_rows'] or not meta['num_cols']:
        raise ValueError(f"Missing num_rows/num_cols metadata in {path!r}")
    if meta['num_rows'] * meta['num_cols'] != meta['range_grid_count']:
        raise ValueError(f"range_grid size mismatch in {path!r}")

    return meta, body


def _read_range_grid_vertices(meta: dict, body: bytes) -> np.ndarray:
    vertex_count = meta['vertex_count']
    if meta['format'] == 'ascii':
        text = body.decode('latin1')
        lines = text.splitlines()
        vertices = np.array([
            [float(value) for value in lines[i].split()[:3]]
            for i in range(vertex_count)
        ], dtype=np.float64)
        return vertices

    endian = '>' if meta['format'] == 'binary_big_endian' else '<'
    values = np.frombuffer(body, dtype=np.dtype(endian + 'f4'), count=vertex_count * 3)
    return values.astype(np.float64, copy=False).reshape(vertex_count, 3)


def _read_range_grid_indices(meta: dict, body: bytes) -> np.ndarray:
    rows = meta['num_rows']
    cols = meta['num_cols']
    vertex_count = meta['vertex_count']
    grid = np.full(rows * cols, -1, dtype=np.int64)

    if meta['format'] == 'ascii':
        text = body.decode('latin1')
        lines = text.splitlines()
        range_lines = lines[vertex_count: vertex_count + meta['range_grid_count']]
        for i, line in enumerate(range_lines):
            parts = line.split()
            if not parts:
                continue
            count = int(parts[0])
            if count:
                grid[i] = int(parts[1])
        return grid.reshape(rows, cols)

    endian = '>' if meta['format'] == 'binary_big_endian' else '<'
    pos = vertex_count * 12
    for i in range(meta['range_grid_count']):
        count = body[pos]
        pos += 1
        if count:
            grid[i] = struct.unpack_from(endian + 'i', body, pos)[0]
            pos += 4 * count
    if pos != len(body):
        raise ValueError('PLY is unexpected length!')
    if np.any(grid >= vertex_count):
        raise ValueError('range_grid references invalid vertex index')
    return grid.reshape(rows, cols)


def _range_grid_to_faces(grid: np.ndarray) -> np.ndarray:
    faces = []
    rows, cols = grid.shape
    for r in range(rows - 1):
        for c in range(cols - 1):
            a = grid[r, c]
            b = grid[r, c + 1]
            d = grid[r + 1, c]
            e = grid[r + 1, c + 1]

            if a >= 0 and b >= 0 and d >= 0:
                faces.append((a, b, d))
            if b >= 0 and e >= 0 and d >= 0:
                faces.append((b, e, d))

    if not faces:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)


# ──────────────────────────── analysis ───────────────────────────────────────

def analyse(mesh: trimesh.Trimesh, name: str) -> None:
    print(f"\n[{name}]")
    print(f"  Vertices   : {len(mesh.vertices):,}")
    print(f"  Faces      : {len(mesh.faces):,}")
    print(f"  Watertight : {mesh.is_watertight}")
    if mesh.is_watertight:
        print(f"  Volume     : {mesh.volume:.6g}")
    bb = mesh.bounds
    print(f"  Bounds     : X[{bb[0,0]:.4g}, {bb[1,0]:.4g}]  "
          f"Y[{bb[0,1]:.4g}, {bb[1,1]:.4g}]  Z[{bb[0,2]:.4g}, {bb[1,2]:.4g}]")

    degen = np.sum(mesh.area_faces < 1e-15)
    if degen:
        print(f"  ⚠ WARNING : {degen} degenerate (zero-area) faces detected")
    if not mesh.is_watertight:
        print("  ⚠ NOTE    : not watertight — run fTetWild before using as deformable body")


# ──────────────────────────── export ─────────────────────────────────────────

def export_ipc_obj(mesh: trimesh.Trimesh, out_path: str) -> None:
    """Write IPC-compatible OBJ: plain v / f lines only, 1-indexed faces."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("# IPC-compatible surface mesh — generated by stanford2ipc.py\n")
        f.write(f"# vertices {len(mesh.vertices)}  faces {len(mesh.faces)}\n\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.10g} {v[1]:.10g} {v[2]:.10g}\n")
        f.write('\n')
        for tri in mesh.faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ──────────────────────────── normalise ──────────────────────────────────────

def normalize_mesh(mesh: trimesh.Trimesh, size: float = 1.0) -> trimesh.Trimesh:
    """Centre at origin, scale longest bbox edge → size."""
    mesh = mesh.copy()
    mesh.vertices -= mesh.centroid
    mesh.vertices *= size / mesh.bounding_box.extents.max()
    return mesh


# ──────────────────────────── CLI ────────────────────────────────────────────

def collect_inputs(paths: list[str]) -> list[str]:
    exts = {'.ply', '.ply.gz', '.obj', '.off', '.stl'}
    files = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                low = name.lower()
                if any(low.endswith(e) for e in exts):
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: {p!r} not found, skipping.", file=sys.stderr)
    return files


def derive_output(inp: str, out_arg: str | None, batch_dir: str | None) -> str:
    base = os.path.basename(inp)
    for ext in ('.ply.gz', '.ply', '.obj', '.off', '.stl'):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    obj_name = base + '.obj'
    if out_arg and not os.path.isdir(out_arg):
        return out_arg                        # explicit single-file output
    if batch_dir:
        return os.path.join(batch_dir, obj_name)
    return os.path.join(os.path.dirname(inp) or '.', obj_name)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', metavar='INPUT',
                   help='Input file(s) or directory')
    p.add_argument('-o', '--output', default=None, metavar='OUTPUT',
                   help='Output .obj file (single) or output directory (batch)')
    p.add_argument('--normalize', action='store_true',
                   help='Centre mesh and scale longest edge to --size')
    p.add_argument('--size', type=float, default=1.0,
                   help='Target size for --normalize (default: 1.0)')
    args = p.parse_args()

    input_files = collect_inputs(args.inputs)
    if not input_files:
        print("No mesh files found.", file=sys.stderr); sys.exit(1)

    batch_dir = args.output if (args.output and os.path.isdir(args.output)) else None
    if batch_dir is None and len(input_files) > 1:
        batch_dir = os.path.dirname(input_files[0]) or '.'
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            batch_dir = args.output

    ok, fail = 0, []
    for inp in input_files:
        out = derive_output(inp, args.output, batch_dir)
        try:
            mesh = load_mesh(inp)
            analyse(mesh, os.path.basename(inp))
            if args.normalize:
                mesh = normalize_mesh(mesh, args.size)
                print(f"  Normalised → size {args.size}")
            export_ipc_obj(mesh, out)
            print(f"  ✓ → {out}")
            ok += 1
        except Exception as e:
            print(f"  ✗ FAILED {inp}: {e}", file=sys.stderr)
            fail.append(inp)

    print(f"\nDone: {ok} converted, {len(fail)} failed.")
    if fail:
        for f in fail: print(f"  ✗ {f}")
        sys.exit(1)


if __name__ == '__main__':
    main()
