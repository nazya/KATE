import torch
from codes import DictEnum, auto


class Optimizer(DictEnum):
    ADAM = auto()
    KATE = auto()


class ADAM(torch.optim.Optimizer):
    def __init__(self, params, cfg, device):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(device=device, lr=cfg.lr,
                        betas=(cfg.beta1_, cfg.beta2_), eps=cfg.eps)  # , weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:

            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Momentum (Exponential MA of gradients)
                    state['m'] = torch.zeros_like(p.data, device=group['device'])

                    # RMS Prop componenet. (Exponential MA of squared gradients). Denominator.
                    state['v'] = torch.zeros_like(p.data, device=group['device'])

                m, v = state['m'], state['v']

                beta1, beta2 = group['betas']
                state['step'] += 1

                # # Add weight decay if any
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Momentum
                m = torch.mul(m, beta1) + (1 - beta1)*grad

                # RMS
                v = torch.mul(v, beta2) + (1-beta2)*(grad*grad)

                mhat = m / (1 - beta1 ** state['step'])
                vhat = v / (1 - beta2 ** state['step'])

                denom = torch.sqrt(vhat) + group['eps']

                p.data = p.data - group['lr'] * mhat / denom

                # Save state
                state['m'], state['v'] = m, v

        return loss




class KATE(torch.optim.Optimizer): # delta 0 or 1e-8
    def __init__(self, params, cfg, device):  # lr=1e-3, eta=0.9, eps=1e-8, delta=0, weight_decay=0):
        defaults = dict(device=device, lr=cfg.lr, eta=cfg.eta_, eps=cfg.eps)  # , eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        

    def step(self):
        loss = None
        for group in self.param_groups:

            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data, device=group['device'])
                    state['b'] = torch.zeros_like(p.data, device=group['device'])

                m, b = state['m'], state['b']
                eta = group['eta']
                
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                g = grad*grad

                b = b + g
                denom = b + group['eps']
                m = m + torch.mul(eta, g) + g / denom

                p.data = p.data - group['lr'] * torch.sqrt(m) * grad / denom

                # Save state
                state['m'], state['b'] = m, b
                state['step'] += 1

        return loss


# class KATEADAM(torch.optim.Optimizer): #eta 0, 1e-3, 1e-1, lr=1e-3
#     def __init__(self, params, cfg, device):  # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
#         defaults = dict(device=device, lr=cfg.lr,
#                         betas=(cfg.beta1_, cfg.beta2_), eta=cfg.eta_, eps=cfg.eps)  # , weight_decay=weight_decay)
#         super().__init__(params, defaults)

#     def step(self):
#         loss = None
#         for group in self.param_groups:

#             for p in group['params']:
#                 grad = p.grad.data
#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['b'] = torch.zeros_like(p.data, device=group['device']) + group['eps']
#                     state['m'] = torch.zeros_like(p.data, device=group['device'])
#                     state['v'] = torch.zeros_like(p.data, device=group['device'])

#                 b, m, v = state['b'], state['m'], state['v']

#                 beta1, beta2 = group['betas']
#                 eta = group['eta']
#                 state['step'] += 1

#                 # # Add weight decay if any
#                 # if group['weight_decay'] != 0:
#                 #     grad = grad.add(group['weight_decay'], p.data)

#                 g = grad*grad

#                 b = torch.mul(b, beta2) + (1-beta2) * g
#                 m = torch.mul(m, beta2) + torch.mul(1-beta2, torch.mul(g, eta) + g / b)
#                 v = torch.mul(v, beta1) + (1 - beta1) * grad

#                 bhat = b / (1 - beta2 ** state['step'])
#                 mhat = m / (1 - beta2 ** state['step'])
#                 vhat = v / (1 - beta1 ** state['step'])

#                 denom = bhat# + group['eps']
                
#                 p.data = p.data - group['lr'] * torch.sqrt(mhat) * vhat / denom
#                 # p.data = p.data - group['lr'] * m * vhat / b

#                 # Save state
#                 state['b'], state['m'], state['v'] = b, m, v

#         return loss


# class AdaGrad(torch.optim.Optimizer):
#     def __init__(self, params, cfg, device):  # lr=1e-3, beta=0.999, eps=1e-8, weight_decay=0):
#         defaults = dict(device=device, lr=cfg.lr, eps=cfg.eps)
#         super().__init__(params, defaults)

#     def step(self):
#         loss = None
#         for group in self.param_groups:

#             for p in group['params']:
#                 grad = p.grad.data
#                 state = self.state[p]

#                 # State initialization
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['v'] = torch.zeros_like(p.data, device=group['device'])

#                 v = state['v']

#                 state['step'] += 1

#                 # # Add weight decay if any
#                 # if group['weight_decay'] != 0:
#                 #     grad = grad.add(group['weight_decay'], p.data)

#                 v = v + grad*grad

#                 denom = torch.sqrt(v) + group['eps']

#                 p.data = p.data - group['lr'] * grad / denom / state['step']

#                 # Save state
#                 state['v'] = v

#         return loss