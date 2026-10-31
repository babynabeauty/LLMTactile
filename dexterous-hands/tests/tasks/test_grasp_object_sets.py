import json

import pytest

from entities.objects import load_manifest_object_set


def write_manifest(tmp_path, *, count=2):
    objects = []
    for index in range(count):
        asset = tmp_path / f"object-{index}.urdf"
        asset.write_text("<robot name='object'><link name='object'/></robot>")
        objects.append(
            {
                "asset": asset.name,
                "density_kg_m3": 600.0,
            }
        )
    (tmp_path / "objaverse-train-16.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "object_count": count,
                "objects": objects,
            }
        )
    )


def test_manifest_object_set_builds_mesh_options(monkeypatch, tmp_path):
    write_manifest(tmp_path)
    monkeypatch.setenv("DEXTEROUS_OBJECT_CACHE", str(tmp_path))

    objects = load_manifest_object_set("objaverse_16")

    assert len(objects) == 2
    assert all(object_options.material.rho == 600.0 for object_options in objects)
    assert all(object_options.recompute_inertia for object_options in objects)
    assert all(not object_options.convexify for object_options in objects)


def test_manifest_object_set_requires_declared_multiple_objects(monkeypatch, tmp_path):
    write_manifest(tmp_path, count=1)
    monkeypatch.setenv("DEXTEROUS_OBJECT_CACHE", str(tmp_path))

    with pytest.raises(ValueError, match="declared multiple-object set"):
        load_manifest_object_set("objaverse_16")


import pytest


@pytest.mark.parametrize("set_name,factor", [("OBJECTS_8CM_16", 2.0), ("OBJECTS_10CM_16", 2.5)])
def test_scaled_primitive_sets_are_the_4cm_set_scaled(set_name, factor):
    from eden.options.entities import BoxOptions

    from entities import objects
    from entities.objects import OBJECTS_4CM_16

    scaled = getattr(objects, set_name)
    assert len(scaled) == len(OBJECTS_4CM_16)
    for small, large in zip(OBJECTS_4CM_16, scaled):
        if isinstance(small, BoxOptions):
            assert tuple(large.size) == pytest.approx(tuple(factor * v for v in small.size))
        else:
            assert (large.height, large.radius) == pytest.approx((factor * small.height, factor * small.radius))
