import torch

class Dummy:
    def __init__(self):
        pass

    def preprocess_inputs(self, inputs):
        return inputs
    
    def loss(self, inputs, gaze_outputs, task_outputs):
        """
        inputs:
            image: B, C, H, W
            gt_gazing_pos: B, N
        gaze_outputs:
            log_action_probs: B, N
            num_gazing_each_frame: B
            if_padded_gazing: B, N
        task_outputs:
            outputs: dict of various outputs of the task
            loss: G*B
            reward: G*B, num_reward_each_traj  (There can be multiple rewards taken at different step index for each trajectory)
            traj_len_each_reward: list with length of num_reward_each_traj  (The length of the trajectory before each reward is taken)
            metrics: dict of metrics of the task
            task_losses: B, N (optional, used for task loss prediction)
            task_losses_mask: B, N (optional, used for task loss prediction)
        """
        return torch.zeros(gaze_outputs['gazing_pos'].shape[0], device=gaze_outputs['gazing_pos'].device)

    def __call__(self, inputs, gaze_outputs, task_outputs):
        loss = self.loss(inputs, gaze_outputs, task_outputs)
        metrics = {"dummy_loss": loss.mean()}

        to_return = {
            'loss': loss,
            'metrics': metrics,
        }
        return to_return