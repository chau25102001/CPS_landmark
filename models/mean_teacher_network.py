import torch
import torch.nn as nn
from models.network import SingleNetwork


class MeanTeacher_CPS(nn.Module):
    def __init__(self, num_classes, model_size, model):
        super(MeanTeacher_CPS, self).__init__()
        self.branch1 = SingleNetwork(num_classes, model_size, model)
        self.branch2 = SingleNetwork(num_classes, model_size, model)

        self.teacher_branch1 = SingleNetwork(num_classes, model_size, model)
        self.teacher_branch2 = SingleNetwork(num_classes, model_size, model)

        for param in self.teacher_branch1.parameters():
            param.requires_grad = False
            param.detach_()

        for param in self.teacher_branch2.parameters():
            param.requires_grad = False
            param.detach_()

        for t_param, s_param in zip(self.teacher_branch1.parameters(), self.branch1.parameters()):
            t_param.data.copy_(s_param.data)  # copy weight from student to teacher

        for t_param, s_param in zip(self.teacher_branch2.parameters(), self.branch2.parameters()):
            t_param.data.copy_(s_param.data)  # copy weight from student to teacher

    def forward(self, data, step=1):
        # if current_iter == 0:
        #     for t_param, s_param in zip(self.teacher_branch1.parameters(), self.branch1.parameters()):
        #         t_param.data.copy_(s_param.data)  # copy weight from student to teacher
        #
        #     for t_param, s_param in zip(self.teacher_branch2.parameters(), self.branch2.parameters()):
        #         t_param.data.copy_(s_param.data)  # copy weight from student to teacher
        if step == 1:  # forward using student 1 and teacher 1
            s_pred = self.branch1(data)

            with torch.no_grad():
                t_pred = self.teacher_branch1(data)

        elif step == 2:  # forward using student 2 and teacher 2
            s_pred = self.branch2(data)

            with torch.no_grad():
                t_pred = self.teacher_branch2(data)
        # else:  # ema update
        #     self._update_ema_variables(config.ema_decay)

        return s_pred, t_pred

    def _update_ema_variables(self, ema_decay):  # ema update for 2 teachers
        for t_param, s_param in zip(self.teacher_branch1.parameters(), self.branch1.parameters()):
            t_param.data.mul_(ema_decay).add_(other=s_param.data, alpha=1 - ema_decay)
        for t_param, s_param in zip(self.teacher_branch2.parameters(), self.branch2.parameters()):
            t_param.data.mul_(ema_decay).add_(other=s_param.data, alpha=1 - ema_decay)
