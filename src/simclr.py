import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature: float = 0.5):
    B = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    pos = torch.sum(z1 * z2, dim=1) / temperature
    pos = torch.cat([pos, pos], dim=0)

    loss = -pos + torch.logsumexp(sim, dim=1)
    return loss.mean()
