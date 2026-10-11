"""Linear Catmull-Clark surface stencils for manifold, uncreased polygon meshes.

The renderer subdivides after skinning. Applying these stencils after the same
skin deformation retains blended joints and their posed surface curvature.
"""
from collections import defaultdict

import numpy as np
from scipy import sparse


def catmull_clark(vertices, faces, levels=2, limit=False):
    if not isinstance(levels, int) or levels < 0:
        raise ValueError('Subdivision levels must be a nonnegative integer')
    count = int(vertices)
    operator = sparse.eye(count, format='csr')
    faces = [list(map(int, face)) for face in faces]
    for _level in range(levels):
        edge_faces, neighbors, vertex_faces = {}, defaultdict(set), defaultdict(list)
        for fi, face in enumerate(faces):
            if len(face) < 3 or min(face) < 0 or max(face) >= count or len(set(face)) != len(face):
                raise ValueError('Subdivision requires valid polygon indices')
            for vi, first in enumerate(face):
                second = face[(vi + 1) % len(face)]
                key = tuple(sorted((first, second)))
                edge_faces.setdefault(key, []).append(fi)
                neighbors[first].add(second)
                neighbors[second].add(first)
                vertex_faces[first].append(fi)
        edges = list(edge_faces)
        edge_ids = {edge: count + i for i, edge in enumerate(edges)}
        boundaries = defaultdict(list)
        for edge, adjacent in edge_faces.items():
            if len(adjacent) > 2:
                raise ValueError('Nonmanifold subdivision edges need a different stencil')
            if len(adjacent) == 1:
                a, b = edge
                boundaries[a].append(b)
                boundaries[b].append(a)
        rows, columns, weights = [], [], []

        def add(row, column, weight):
            rows.append(row); columns.append(column); weights.append(weight)

        def face_average(row, fi, weight):
            for vertex in faces[fi]:
                add(row, vertex, weight / len(faces[fi]))

        for vertex in range(count):
            boundary = boundaries[vertex]
            if boundary:
                if len(boundary) != 2:
                    raise ValueError('Nonmanifold boundary vertex cannot share this stencil')
                add(vertex, vertex, .75)
                for neighbor in boundary:
                    add(vertex, neighbor, .125)
            else:
                n = len(vertex_faces[vertex])
                if n == 0 or n != len(neighbors[vertex]):
                    raise ValueError('Isolated or nonmanifold vertex')
                add(vertex, vertex, (n - 2.) / n)
                for neighbor in neighbors[vertex]:
                    add(vertex, neighbor, 1. / n ** 2)
                for fi in vertex_faces[vertex]:
                    face_average(vertex, fi, 1. / n ** 2)
        for edge, row in edge_ids.items():
            adjacent = edge_faces[edge]
            if len(adjacent) == 1:
                add(row, edge[0], .5); add(row, edge[1], .5)
            else:
                add(row, edge[0], .25); add(row, edge[1], .25)
                for fi in adjacent:
                    face_average(row, fi, .25)
        face_start = count + len(edges)
        for fi in range(len(faces)):
            face_average(face_start + fi, fi, 1.)
        next_count = face_start + len(faces)
        step = sparse.coo_matrix((weights, (rows, columns)), shape=(next_count, count)).tocsr()
        operator = step @ operator
        refined = []
        for fi, face in enumerate(faces):
            for vi, vertex in enumerate(face):
                after = edge_ids[tuple(sorted((vertex, face[(vi + 1) % len(face)])))]
                before = edge_ids[tuple(sorted((face[(vi - 1) % len(face)], vertex)))]
                refined.append([vertex, after, face_start + fi, before])
        count, faces = next_count, refined
    if limit:
        # OpenSubdiv/Blender can place refined vertices on the limit surface.
        # All refined faces are quads; the limit stencil is exact at vertices,
        # including extraordinary valences and smooth open boundaries.
        if any(len(face) != 4 for face in faces):
            raise ValueError('Limit projection requires a refined quad mesh')
        edges, neighbors, diagonals = defaultdict(int), defaultdict(set), defaultdict(list)
        for face in faces:
            for i, vertex in enumerate(face):
                neighbor = face[(i + 1) % 4]
                edges[tuple(sorted((vertex, neighbor)))] += 1
                neighbors[vertex].add(neighbor)
                neighbors[neighbor].add(vertex)
                diagonals[vertex].append(face[(i + 2) % 4])
        boundary = defaultdict(list)
        for (a, b), adjacent in edges.items():
            if adjacent == 1:
                boundary[a].append(b); boundary[b].append(a)
        rr, cc, ww = [], [], []
        for vertex in range(count):
            if boundary[vertex]:
                if len(boundary[vertex]) != 2:
                    raise ValueError('Nonmanifold limit boundary')
                rr.append(vertex); cc.append(vertex); ww.append(2. / 3.)
                for neighbor in boundary[vertex]:
                    rr.append(vertex); cc.append(neighbor); ww.append(1. / 6.)
            else:
                n = len(neighbors[vertex])
                denominator = n * (n + 5.)
                rr.append(vertex); cc.append(vertex); ww.append(n * n / denominator)
                for neighbor in neighbors[vertex]:
                    rr.append(vertex); cc.append(neighbor); ww.append(4. / denominator)
                for diagonal in diagonals[vertex]:
                    rr.append(vertex); cc.append(diagonal); ww.append(1. / denominator)
        operator = sparse.coo_matrix((ww, (rr, cc)), shape=(count, count)).tocsr() @ operator
    return operator, np.array(faces, dtype=np.int32)
