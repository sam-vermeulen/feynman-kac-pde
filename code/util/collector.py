import torch

class BatchCollector():
    def __init__(self, domain, process, batch_size, max_exits):
        self.domain = domain
        self.process = process
        self.max_exits = max_exits
        self.batch_size = batch_size
        self.points = domain.sample_points(batch_size)
        self.remaining_walkers = self.max_exits
        
    def __iter__(self):
        while self.remaining_walkers > 0:

            old_points = self.points.clone()
            self.points = self.process.apply_noise(self.points)
            
            exited = ~self.domain.points_inside(self.points)

            if torch.any(exited):
                clipped_points = torch.where(exited[:, None], self.domain.exit_point(old_points, self.points), self.points)
                num_exited = torch.count_nonzero(exited).item()
                self.remaining_walkers -= num_exited
                self.points = self.domain.resample_points(self.points, where=exited)
            else:
                clipped_points = self.points

            yield old_points, self.points, exited, clipped_points
