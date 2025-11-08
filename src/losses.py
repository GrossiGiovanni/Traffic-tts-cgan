
import torch
import torch.nn.functional as F


def d_hinge_loss(real_scores, fake_scores):
    loss_real = torch.relu(1.0 - real_scores).mean()
    loss_fake = torch.relu(1.0 + fake_scores).mean()
    return loss_real + loss_fake




def g_hinge_loss(fake_scores):
    return (-fake_scores).mean()




def r1_regularizer(real_scores, real_x, gamma=1.0):
    # real_x: [B,L,C]
    grads = torch.autograd.grad(outputs=real_scores.sum(), inputs=real_x,
    create_graph=True, retain_graph=True, only_inputs=True)[0]
    penalty = grads.reshape(grads.size(0), -1).pow(2).sum(dim=1).mean()
    return 0.5 * gamma * penalty