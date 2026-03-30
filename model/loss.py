import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        # Smooth value prevents division by zero
        smooth = 1.0
        
        # Flatten the tensors
        predicted = torch.softmax(predicted, dim=1)
        
        dice_score = 0
        num_classes = predicted.shape[1]
        
        for cls in range(num_classes):
            pred_cls = predicted[:, cls, :, :]
            target_cls = (target == cls).float()
            
            intersection = (pred_cls * target_cls).sum()
            dice_score += (2.0 * intersection + smooth) / (
                pred_cls.sum() + target_cls.sum() + smooth
            )
        
        # Return loss (1 - dice because lower loss = better)
        return 1 - (dice_score / num_classes)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, predicted, target):
        # Combine both losses for better training
        return self.ce(predicted, target) + self.dice(predicted, target)