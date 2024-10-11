import math

class Cyclic_Annealer():
  def __init__(self, total_iters, n_cycles, beta = 1.0, ratio=0.5, function=lambda x: x*2, z_capacity_base=32, z_capacity_step=1, increase = True):
    self.current = 0
    self.n_cycles = n_cycles
    self.ratio = ratio
    self.tau = lambda x: (self.current%math.ceil(total_iters/n_cycles)) / (total_iters/n_cycles)
    self.function = function
    self.cycle_ended = False
    self.z_capacity = (n_cycles-1)*z_capacity_step + z_capacity_base
    self.cycle = 0
    self.z_capacity_base = z_capacity_base
    self.z_capacity_step = z_capacity_step
    self.beta = beta
    self.increase = increase

  def step(self):
    assert self.function(self.ratio) == 1, 'Function must be 1 at ratio'
    tau = self.tau(self.current)
    self.current += 1
    if self.increase:
      self.z_capacity = self.cycle*self.z_capacity_step - (self.n_cycles-1)*self.z_capacity_step + self.z_capacity_base
    else:
      self.z_capacity = (self.n_cycles-1)*self.z_capacity_step - self.cycle*self.z_capacity_step + self.z_capacity_base
    if tau <= self.ratio:
      if self.cycle_ended:
        self.cycle += 1
        self.cycle_ended = False
      return self.function(tau)*self.beta, self.z_capacity
    else:
      self.cycle_ended = True 
      return self.beta, self.z_capacity


  def reset(self):
    self.current = 0
  