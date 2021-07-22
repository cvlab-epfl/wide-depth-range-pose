import torch

def anneal_linear(start, end, proportion, params=None):
    return start + proportion * (end - start)

def anneal_multi_steps(start, end, proportion, params):
    steps = params['steps']
    gamma = params['gamma']
    lr = start
    for s in steps:
        if proportion >= s:
            lr *= gamma
    return lr

class Phase:
    def __init__(self, start, end, n_iter, anneal_fn, params = None):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.params = params
        self.n = 0

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter, self.params)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter

class WarmupScheduler:
    def __init__(
        self, 
        optimizer, 
        lr_base, 
        max_iter,
        strategy, 
        params,
        warmup_iter = 1000
    ):
        self.optimizer = optimizer
        
        warmup_lr_start = 0
        ph1 = Phase(warmup_lr_start, lr_base, warmup_iter, anneal_linear)

        if strategy == 'multi_steps':
            steps = [(s - warmup_iter)/(max_iter - warmup_iter) for s in params['steps']]
            gamma = params['gamma']
            tmp_params = {'steps': steps, 'gamma': gamma}
            ph2 = Phase(lr_base, None, max_iter - warmup_iter, anneal_multi_steps, tmp_params)
        else:
            print('Not supported scheduler strategy "%s"' % strategy)
            assert(0)

        self.lr_phase = [ph1, ph2]
        self.phase = 0

    def step(self):
        lr = self.lr_phase[self.phase].step()
        if self.lr_phase[self.phase].is_done:
            self.phase += 1
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def step_multiple(self, num_step):
        lr = 0
        for i in range(num_step):
            lr = self.lr_phase[self.phase].step()
            if self.lr_phase[self.phase].is_done:
                self.phase += 1
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr
