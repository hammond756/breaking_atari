import argparse

class BaseParser(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParser, self).__init__()

        # defaults come from Mnih et al. (2015)
        self.add_argument('--batch_size', type=int, required=False, default=32)
        self.add_argument('--gamma', type=float, required=False, default=0.99)
        self.add_argument('--eps_start', type=float, required=False, default=1.0)
        self.add_argument('--eps_stop', type=float, required=False, default=0.1)
        self.add_argument('--eps_steps', type=int, required=False, default=1000000)
        self.add_argument('--target_update', type=int, required=False, default=10000)
        self.add_argument('--num_frames', type=int, required=False, default=10000000)
        self.add_argument('--num_eval', type=int, required=False, default=10000)
        self.add_argument('--eval_every', type=int, required=False, default=100000)
        self.add_argument('--lr', type=float, required=False, default=0.00025)
        self.add_argument('--memory', type=int, required=False, default=1000000)
        self.add_argument('--exploration_phase', type=int, required=False, default=50000)
        self.add_argument('--frame_stack', type=int, required=False, default=4)
        self.add_argument('--environment', type=str, required=True)
        self.add_argument('--output_dir', type=str, required=True)
        self.add_argument('--optimize_every', type=int, required=False, default=4)
        self.add_argument('--device', type=str, required=True)
        self.add_argument('--n_validation_states', type=int, required=False, default=512)