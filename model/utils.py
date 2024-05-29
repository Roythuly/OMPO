import math
import torch

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def orthogonal_regularization(model, reg_coef=1e-4):
    """Orthogonal regularization v2.
  
    See equation (3) in https://arxiv.org/abs/1809.11096.
  
    Args:
      model: A PyTorch model to apply regularization for.
      reg_coef: Orthogonal regularization coefficient.
  
    Returns:
      A regularization loss term.
    """
  
    reg = 0.0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight
            prod = torch.matmul(weight.t(), weight)
            eye_matrix = torch.eye(prod.shape[0], device=weight.device)
            reg += torch.sum(torch.square(prod * (1 - eye_matrix)))
  
    return reg * reg_coef
