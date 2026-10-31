from types import SimpleNamespace

import genesis as gs
import torch
from eden.managers import EventTerm

from shared_terms import SetSampledBottomAlignedPos


class FakeEntity:
    """Two variants: a centred box and a mesh whose origin sits 2 cm above its base."""

    def __init__(self):
        self.pos = torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]])
        self.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
        self.origin_above_base = torch.tensor([0.03, 0.02])
        self.height = torch.tensor([0.06, 0.06])

    def get_default_root_pose(self):
        return self.pos.clone(), self.quat.clone()

    def set_pos(self, pos, envs_idx=None):
        self.pos = pos.clone() if envs_idx is None else self.pos.index_copy(0, envs_idx.long(), pos)

    def set_quat(self, quat, envs_idx=None):
        self.quat = quat.clone() if envs_idx is None else self.quat.index_copy(0, envs_idx.long(), quat)

    def get_AABB(self):
        low = self.pos[:, 2] - self.origin_above_base
        high = low + self.height
        ones = torch.ones(2)
        return torch.stack([torch.stack([-ones, -ones, low], -1), torch.stack([ones, ones, high], -1)], 1)


def test_bottom_lands_on_the_table_for_centred_and_offset_origins(monkeypatch):
    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    monkeypatch.setattr(EventTerm, "build", lambda self: None)
    entity = FakeEntity()
    term = object.__new__(SetSampledBottomAlignedPos)
    term.entity_name = "obj"
    term.pos_z = 0.5
    term.range_x = (0.0, 0.0)
    term.range_y = (0.0, 0.0)
    term.range_yaw = (0.0, 0.0)
    term._env = SimpleNamespace(entities={"obj": entity}, num_envs=2, device=torch.device("cpu"))
    term.build()
    torch.testing.assert_close(term.offset_z, torch.tensor([0.53, 0.52]))
    term.compute(torch.tensor([0, 1]))
    torch.testing.assert_close(entity.get_AABB()[:, 0, 2], torch.full((2,), 0.5))
