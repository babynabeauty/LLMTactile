"""Analytic rigid-object geometry for the multi-object material-transfer study.

This module has no Genesis import and can be checked on CPU before any scene is
built. Dimensions are physical collider dimensions, never visual-only scaling.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class ObjectSpec:
    shape: str = 'cube'
    side_m: float = .045
    radius_m: float = .0225
    height_m: float = .045
    size_m: tuple = (.045,.060,.045)
    across_flats_m: float = .045

    def __post_init__(self):
        if self.shape not in ('cube', 'cuboid', 'hex_prism', 'cylinder', 'sphere'):
            raise ValueError('Object shape must be cube, cuboid, hex_prism, cylinder, or sphere')
        for name in ('side_m', 'radius_m', 'height_m','across_flats_m'):
            value = getattr(self, name)
            if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be one finite positive number')
        raw_size=np.asarray(self.size_m)
        if raw_size.shape!=(3,) or raw_size.dtype.kind not in 'fiu' or not np.isfinite(raw_size).all() or (raw_size<=0).any():
            raise ValueError('size_m must contain three finite positive numeric dimensions')
        object.__setattr__(self,'size_m',tuple(float(v) for v in raw_size))

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(shape=value)
        if isinstance(value, Mapping):
            return cls(**value)
        raise TypeError('object_spec must be an ObjectSpec, shape name, or specification dict')

    @property
    def volume_m3(self):
        if self.shape == 'cube':
            return self.side_m**3
        if self.shape == 'cuboid':
            return float(np.prod(self.size_m))
        if self.shape == 'hex_prism':
            return 3*np.sqrt(3)/2*self.hex_circumradius_m**2*self.height_m
        if self.shape == 'cylinder':
            return np.pi*self.radius_m**2*self.height_m
        return 4*np.pi*self.radius_m**3/3

    @property
    def bounds_size_m(self):
        if self.shape == 'cube':
            return np.full(3, self.side_m)
        if self.shape == 'cuboid':
            return np.asarray(self.size_m)
        if self.shape == 'hex_prism':
            return np.ptp(self.vertices_local_m,axis=0)
        return np.array([2*self.radius_m, 2*self.radius_m,
                         self.height_m if self.shape == 'cylinder' else 2*self.radius_m])

    @property
    def inertia_per_kg(self):
        if self.shape == 'cube':
            return np.eye(3)*self.side_m**2/6
        if self.shape == 'cuboid':
            x,y,z=self.size_m
            return np.diag([y*y+z*z,x*x+z*z,x*x+y*y])/12
        if self.shape == 'hex_prism':
            r,h=self.hex_circumradius_m,self.height_m
            return np.diag([5*r*r/24+h*h/12]*2+[5*r*r/12])
        if self.shape == 'sphere':
            return np.eye(3)*2*self.radius_m**2/5
        radial = (3*self.radius_m**2+self.height_m**2)/12
        return np.diag([radial, radial, self.radius_m**2/2])

    @property
    def hex_circumradius_m(self):
        return self.across_flats_m/(2*np.cos(np.pi/6))

    @property
    def vertices_local_m(self):
        if self.shape!='hex_prism':
            raise ValueError('Explicit vertices are provided for hex_prism only')
        angles=np.deg2rad(30+60*np.arange(6))
        xy=self.hex_circumradius_m*np.column_stack([np.cos(angles),np.sin(angles)])
        return np.concatenate([np.column_stack([xy,np.full(6,z)])
                               for z in (-self.height_m/2,self.height_m/2)])

    @property
    def mesh_obj_bytes(self):
        vertices=self.vertices_local_m
        faces=[]
        for i in range(1,5):
            faces.extend([(0,i+1,i),(6,6+i,6+i+1)])
        for i in range(6):
            j=(i+1)%6
            faces.extend([(i,j,j+6),(i,j+6,i+6)])
        lines=['# Deterministic regular hexagonal prism; meters; vertices 30+60k degrees.']
        lines.extend('v '+' '.join(format(float(v),'.17g') for v in row) for row in vertices)
        lines.extend('f '+' '.join(str(v+1) for v in row) for row in faces)
        return ('\n'.join(lines)+'\n').encode('ascii')

    def mesh_asset(self,directory=None,create=False):
        """Content-addressed OBJ; requesting metadata alone never writes files."""
        content=self.mesh_obj_bytes
        digest=hashlib.sha256(content).hexdigest()
        directory=Path(directory or Path(__file__).parents[1]/'generated/rigid_shapes')
        path=directory/f'hex_prism_{digest[:16]}.obj'
        if create:
            directory.mkdir(parents=True,exist_ok=True)
            # Exclusive creation avoids overwriting a previously audited asset.
            try:
                with path.open('xb') as stream:stream.write(content)
            except FileExistsError:
                if path.read_bytes()!=content:
                    raise ValueError('Existing rigid-object mesh differs from its expected content hash')
        return {'path':str(path.resolve()),'sha256':digest,'format':'OBJ meters',
                'vertices':12,'triangles':20,'convexify':True,'decimate':False}

    def initial_position(self, xy=(-.60, -.1333), table_top_z=.2):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(table_top_z):
            raise ValueError('Initial XY and tabletop height must be finite')
        return np.r_[xy, table_top_z+self.bounds_size_m[2]/2]

    def vertical_extent(self, quaternion_wxyz):
        """Exact support radius along world Z, including tilted cylinders."""
        q = quaternion_wxyz
        if isinstance(q, torch.Tensor):
            q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
            w, x, y, z = q.unbind(-1)
            abs_, sqrt = torch.abs, torch.sqrt
        else:
            q = np.asarray(q, dtype=float)
            q = q/np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            w, x, y, z = np.moveaxis(q, -1, 0)
            abs_, sqrt = np.abs, np.sqrt
        row_x, row_y, row_z = 2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)
        if self.shape in ('cube','cuboid'):
            half=self.bounds_size_m/2
            return half[0]*abs_(row_x)+half[1]*abs_(row_y)+half[2]*abs_(row_z)
        if self.shape == 'hex_prism':
            if isinstance(q,torch.Tensor):
                vertices=torch.as_tensor(self.vertices_local_m,dtype=q.dtype,device=q.device)
                row=torch.stack([row_x,row_y,row_z],dim=-1)
                return (row@vertices.T).amax(dim=-1)
            row=np.stack([row_x,row_y,row_z],axis=-1)
            return (row@self.vertices_local_m.T).max(axis=-1)
        if self.shape == 'cylinder':
            return self.height_m/2*abs_(row_z)+self.radius_m*sqrt(row_x**2+row_y**2)
        return row_z*0+self.radius_m

    def protocol(self):
        result={**asdict(self), 'volume_m3':float(self.volume_m3),
                'bounds_size_m':self.bounds_size_m.tolist(),
                'inertia_per_kg_m2':self.inertia_per_kg.tolist(),
                'lowest_surface':'center_world_z - exact analytic support radius for current orientation'}
        if self.shape=='hex_prism':
            result.update(hex_circumradius_m=float(self.hex_circumradius_m),
                vertices_local_m=self.vertices_local_m.tolist(),mesh_asset=self.mesh_asset(),
                lowest_surface='center_world_z + minimum rotated local vertex z over all 12 actual mesh vertices')
        return result
