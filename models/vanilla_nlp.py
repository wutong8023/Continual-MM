import torch
from utils.per_buffer_NLP import PERBufferNLP
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        'Experience Replay for NLP.')
    # add related arguments
    add_management_args(parser)
    add_experiment_args(parser)

    return parser


class VanillaNLP(ContinualModel):
    """
    feature: with logits regularization; start from instance
    """
    NAME = 'vanillanlp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    
    def __init__(self, backbone, loss, args, transform):
        super(VanillaNLP, self).__init__(backbone, loss, args, transform)

    def observe(self, inputs, inputs_mask, labels, labels_name=None, labels_mask=None, task_labels=None):
        # begin: Loss 1
        outputs = self.net(inputs, inputs_mask)
        # features = self.net.features(inputs)
        loss = self.loss(outputs, labels)
        # end: Loss 1
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        
        return loss.item()

