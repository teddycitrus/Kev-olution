import torch


def attention_reference(q, k, v, sm_scale):
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)
