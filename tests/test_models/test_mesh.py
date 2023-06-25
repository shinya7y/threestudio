import torch

from threestudio.models.mesh import Mesh


def test_mesh():
    v_pos = torch.zeros(6, 3)
    t_pos_idx = torch.zeros(8, 3)
    mesh = Mesh(v_pos, t_pos_idx)

    # check vertex color
    v_rgb_input = torch.rand(6, 3)
    mesh.set_vertex_color(v_rgb_input)
    v_rgb_output = mesh.v_rgb
    assert torch.equal(v_rgb_input, v_rgb_output)
