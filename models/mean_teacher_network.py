import torch
import torch.nn as nn
from models.network import SingleNetwork
from models.hgnet import HGNet


class MeanTeacher_CPS(nn.Module):
    def __init__(self, num_classes, model_size, model):
        super(MeanTeacher_CPS, self).__init__()
        if model == 'hgnet':
            module = HGNet
        else:
            module = SingleNetwork
        self.branch1 = module(num_classes=num_classes, model_size=model_size, model=model)
        self.branch2 = module(num_classes=num_classes, model_size=model_size, model=model)

        self.teacher_branch1 = module(num_classes=num_classes, model_size=model_size, model=model)
        self.teacher_branch2 = module(num_classes=num_classes, model_size=model_size, model=model)

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
            s_pred, s_class = self.branch1(data)

            with torch.no_grad():
                t_pred, t_class = self.teacher_branch1(data)

        elif step == 2:  # forward using student 2 and teacher 2
            s_pred, s_class = self.branch2(data)

            with torch.no_grad():
                t_pred, t_class = self.teacher_branch2(data)
        # else:  # ema update
        #     self._update_ema_variables(config.ema_decay)

        return s_pred, t_pred, s_class, t_class

    def _update_ema_variables(self, ema_decay):  # ema update for 2 teachers
        for t_param, s_param in zip(self.teacher_branch1.parameters(), self.branch1.parameters()):
            t_param.data.mul_(ema_decay).add_(other=s_param.data, alpha=1 - ema_decay)
        for t_param, s_param in zip(self.teacher_branch2.parameters(), self.branch2.parameters()):
            t_param.data.mul_(ema_decay).add_(other=s_param.data, alpha=1 - ema_decay)


if __name__ == "__main__":
    model = MeanTeacher_CPS(20, '18', 'hgnet')
    m1 = model.branch1
    m2 = model.branch2
    t1 = model.teacher_branch1
    t1.train()
    for pt in t1.parameters():
        if pt.requires_grad:
            print("here")
    count1 = 0
    count2 = 0
    for ((n1, p1), (n2, p2)) in zip(m1.named_parameters(), m2.named_parameters()):
        count1 += p1.numel()
        if ((p1 - p2) == 0).all():
            print(n1)
            count2 += p2.numel()
    print(count1, count2)
