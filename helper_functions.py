import torch


def gaussian_normalization(inputs, sigmas, num_channels, spatial_size, device):
    """

    :param inputs:
    :param sigmas:
    :param num_channels:
    :param spatial_size:
    :param device:
    :return:
    """

    sigmas_2d = sigmas.unsqueeze(dim=1).repeat_interleave(num_channels, dim=1).to(device)
    sigmas_3d = sigmas_2d.unsqueeze(dim=2).repeat_interleave(spatial_size, dim=2).to(device)
    sigmas_image_size = sigmas_3d.unsqueeze(dim=3).repeat_interleave(spatial_size, dim=3).to(device)

    noisy_inputs = inputs + torch.randn_like(inputs, device=device) * sigmas_image_size
    return noisy_inputs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def toy_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        pred = output.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()
