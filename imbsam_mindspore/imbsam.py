import mindspore as ms
import mindspore.ops as ops
from collections import defaultdict

class SAM():
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.parameters = optimizer.parameters
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)

    def first_step(self, grads):
        grads_ = []
        for grad in grads:
            grads_.append(ops.norm(grad))
        grad_norm = ops.norm(ops.stack(grads_)) + 1.e-16
        
        for p, grad in zip(self.model.trainable_params(), grads):
            eps = self.state[p].get('eps')
            if eps is None:
                eps = p.copy()
                self.state[p]["eps"] = eps
            eps[...] = grad[...]
            eps[...] = eps.mul(self.rho / grad_norm)[...]
            p[...] = p.add(eps)[...]

    def second_step(self, grads):
        for p in self.model.trainable_params():
            p[...] = p.sub(self.state[p]["eps"])[...]
        self.optimizer(grads)

class ImbSAM():
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.parameters = optimizer.parameters
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)
    
    def first_step(self, grads):
        for p, grad in zip(self.model.trainable_params(), grads):
            grad_normal = self.state[p].get("grad_normal")
            if grad_normal is None:
                grad_normal = p.copy()
                self.state[p]['grad_normal'] = grad_normal
            grad_normal[...] = grad[...]
        
    def second_step(self, grads):
        grads_ = []
        for grad in grads:
            grads_.append(ops.norm(grad))
        grad_norm = ops.norm(ops.stack(grads_)) + 1.e-16
        for p, grad in zip(self.model.trainable_params(), grads):
            eps = self.state[p].get('eps')
            if eps is None:
                eps = p.copy()
                self.state[p]['eps'] = eps
            eps[...] = grad[...]
            eps[...] = eps.mul(self.rho / grad_norm)[...]
            p[...] = p.add(eps)[...]
    
    def third_step(self, grads):
        for p, grad in zip(self.model.trainable_params(), grads):
            p[...] = p.sub(self.state[p]['eps'])[...]
            grad[...] = grad.add(self.state[p]['grad_normal'])[...]
        self.optimizer(grads)
