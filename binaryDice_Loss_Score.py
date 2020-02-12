import torch

def binaryDiceLoss(pred, target, eps=1e-5):
    if torch.max(pred) > 1:
        pred = pred.contiguous() / 255
    else:
        pred = pred.contiguous()

    if torch.max(target) > 1:
        target = target.contiguous() / 255
    else:
        target = target.contiguous()

    """
    # This is incorrect. (1-(ab/a+b) + 1-(cd/c+d)) is not same with 1*2(ab+cd/a+b+c+d) 
    inter = torch.dot(pred.view(-1), target.view(-1))
    union = torch.sum(pred) + torch.sum(target)

    loss = 1*batch_num - (2 * inter + smooth) / (union + smooth) # 1*2(ab+cd/a+b+c+d) 
    """
    if len(pred.size()) == 4 and len(target.size()) == 4:  # case of batch (Batchsize, C==1, H, W)
        intersection = (pred * target).sum(dim=2).sum(dim=2)  # sum of H,W axis
        loss = (1 - ((2. * intersection + eps) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + eps)))
        # loss shape : (batch_size, 1)
    elif len(pred.size()) == 3 and len(target.size()) == 3:  # case of image shape (C==1,H,W)
        intersection = (pred * target).sum(dim=1).sum(dim=1)
        coeff = (1 - (2. * intersection) / (pred.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + eps))
    return loss.mean()  # (1-(ab/a+b) + 1-(cd/c+d)) / batch_size


def binaryDiceCoeff(pred, target, eps=1e-5):
    if torch.max(pred) > 1:
        pred = pred.contiguous() / 255
    else:
        pred = pred.contiguous()

    if torch.max(target) > 1:
        target = target.contiguous() / 255
    else:
        target = target.contiguous()

    if len(pred.size()) == 4 and len(target.size()) == 4:
        intersection = (pred * target).sum(dim=2).sum(dim=2)  # sum of H,W axis
        coeff = (2. * intersection + eps) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + eps)
    elif len(pred.size()) == 3 and len(target.size()) == 3:
        intersection = (pred * target).sum(dim=1).sum(dim=1)  # H, W f
        coeff = (2. * intersection) / (pred.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + eps)

    return coeff.mean()