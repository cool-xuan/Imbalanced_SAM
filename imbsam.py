import torch
from collections import defaultdict


class SAM():
    
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)
        
    @torch.no_grad()
    def first_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
        
    @torch.no_grad()
    def second_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()
        
class ImbSAM:
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grad_normal = self.state[p].get("grad_normal")
            if grad_normal is None:
                grad_normal = torch.clone(p).detach()
                self.state[p]["grad_normal"] = grad_normal
            grad_normal[...] = p.grad[...]
        self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def third_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
            p.grad.add_(self.state[p]["grad_normal"])
        self.optimizer.step()
        self.optimizer.zero_grad()
