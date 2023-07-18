import torch
import torch.nn as nn
import sys
from MAE.decoder_head.UNETR import get_model_from_pretrained_path


class MeanTeacher_CPS_ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.branch1 = get_model_from_pretrained_path(config)
        self.branch2 = get_model_from_pretrained_path(config)

        self.teacher_branch1 = get_model_from_pretrained_path(config)
        self.teacher_branch2 = get_model_from_pretrained_path(config)

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

        if step == 1:
            s_pred = self.branch1(data)
            with torch.no_grad():
                t_pred = self.teacher_branch1(data)

        elif step == 2:
            s_pred = self.branch2(data)
            with torch.no_grad():
                t_pred = self.teacher_branch2(data)

        return s_pred, t_pred

    def _update_ema_variables(self, ema_decay):  # ema update for 2 teachers
        for t_param, s_param in zip(self.teacher_branch1.parameters(), self.branch1.parameters()):
            t_param.data.mul_(ema_decay).add_(other=s_param.data, alpha=1 - ema_decay)
        for t_param, s_param in zip(self.teacher_branch2.parameters(), self.branch2.parameters()):
            t_param.data.mul_(ema_decay).add_(other=s_param.data, alpha=1 - ema_decay)


if __name__ == "__main__":
    from config.config_finetune_vit import get_config

    cfg = get_config(train=False)

    model = MeanTeacher_CPS_ViT(cfg)
    count = 0
    for p in model.parameters():
        count += p.numel()

    print(count)

    a = torch.randn((1, 3, 128, 128))
    outputs = model(a)
    print(outputs[0].shape, outputs[1].shape)
