import torch
import argparse
import time
import datetime
import os

from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR

from datasets import get_dataset, DATASETS
from models.architectures import ARCHITECTURES, get_architecture
from models.random_smooth import RandSmoothedClassifier
from models.input_dependent_rs import InputDependentRSClassifier
from models.biased_idrs import BiasedInputDependentRSClassifier
from knn import KNNDistComp
from helper_functions import gaussian_normalization, AverageMeter, accuracy, init_logfile, log


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str,
                    help='folder to save model and training log)')
parser.add_argument('--num_workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--base_sigma', default=0.12, type=float,
                    help="Base sigma used for gaussian augmentation")
parser.add_argument('--rate', default=0.01, type=float,
                    help="The rate used for gaussian augmentation")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print_freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--num_nearest', default=20, type=int, help='How many nearest neighbors to use')
parser.add_argument('--input_dependent', default=False, type=bool,
                    help="Indicator whether to use input-dependent computation of the variance")
parser.add_argument('--alt_sigma_aug', default=-1, type=float,
                    help="Alternative sigma to use when doing input dependent smoothing")
parser.add_argument('--id_augmentation', default=False, type=bool,
                    help="Indicator whether to use input-dependent gaussian data augmentation")
parser.add_argument('--biased', default=False, type=bool,
                    help="Indicator whether to use a biased")
parser.add_argument('--bias_weight', default=0, type=float,
                    help="Weight of bias")
args = parser.parse_args()


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # --- prepare data ---
    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    # if "toy" in args.dataset:
    #     train_dataset.visualize_itself()
    #     test_dataset.visualize_itself()

    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    if args.dataset == 'cifar10':
        spatial_size = 32
        num_classes = 3
        norm_const = 5
    elif args.dataset == 'mnist':
        spatial_size = 28
        num_classes = 1
        norm_const = 1.5
    elif "toy" in args.dataset:
        spatial_size = 2
        num_classes = 2
        norm_const = 0
    else:
        print("shit happens...")
        spatial_size = 0
        num_classes = 0
        norm_const = 0
    # --------------------

    dist_computer = KNNDistComp(train_dataset, 0, device)

    base_model = get_architecture(args.arch, args.dataset, device)

    if args.input_dependent:
        model = InputDependentRSClassifier(base_classifier=base_model, num_classes=num_classes, sigma=args.base_sigma,
                                           distances=None, rate=args.rate, m=norm_const, device=device).to(device)
        add_model_name = "_id"
    elif args.biased:
        model = BiasedInputDependentRSClassifier(base_classifier=base_model, num_classes=num_classes,
                                                 sigma=args.base_sigma, bias_weight=args.bias_weight, oracles=None,
                                                 rate=args.rate, m=norm_const, device=device).to(device)
        add_model_name = "_biased_id"
    else:
        model = RandSmoothedClassifier(base_classifier=base_model, num_classes=num_classes, sigma=args.base_sigma,
                                       device=device).to(device)
        add_model_name = ""

    logfile_name = os.path.join(args.outdir, f'log{add_model_name}.txt')
    init_logfile(logfile_name, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttest loss\ttest acc")

    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    if args.input_dependent and args.alt_sigma_aug > -1:
        used_sigma = args.alt_sigma_aug
    else:
        used_sigma = args.base_sigma

    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch}')
        before = time.time()
        train_loss, train_acc = train(train_loader,
                                      model,
                                      criterion,
                                      optimizer,
                                      used_sigma,
                                      args.rate,
                                      dist_computer,
                                      spatial_size,
                                      num_classes,
                                      args.biased,
                                      args.bias_weight,
                                      args.id_augmentation,
                                      norm_const,
                                      device,
                                      epoch,)
        test_loss, test_acc = test(test_loader,
                                   model,
                                   criterion,
                                   used_sigma,
                                   args.rate,
                                   dist_computer,
                                   spatial_size,
                                   num_classes,
                                   args.biased,
                                   args.bias_weight,
                                   args.id_augmentation,
                                   norm_const,
                                   device,)
        scheduler.step()
        after = time.time()

        log(logfile_name, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_last_lr()[0], train_loss, train_acc, test_loss, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, f'checkpoint{add_model_name}.pth.tar'))

    if "toy" in args.dataset:
        train_dataset.visualize_with_classifier(model, file_path=args.outdir, add_file_name=add_model_name)
        test_dataset.visualize_with_classifier(model, file_path=args.outdir, add_file_name=add_model_name)


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, base_sigma: float,
          rate: float, dist_computer: KNNDistComp, spatial_size: int, num_classes: int, biased: bool,
          bias_weight: float, input_dependent_aug: bool, norm_const: float, device: torch.device, epoch: int):
    """
    Run one epoch of training.

    :param loader: loader containing the training data
    :param model: representation of the model that is trained
    :param criterion: loss function
    :param optimizer: optimizer for the model
    :param base_sigma: value of the base sigma
    :param rate: semi-elasticity constant (for chosen sigma function)
    :param dist_computer: module to compute the distance to each sample in the training data
    :param spatial_size: dimension/size of the image (height and width)
    :param num_classes: number of possible classes in the data
    :param input_dependent_aug:
    :param norm_const: normalization constant for the data set
    :param device: device used for the computations
    :param epoch:
    :return: average loss and accuracy
    """
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    for i, (inputs, labels) in enumerate(loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor).to(device)

        # start_random = time.time()
        if input_dependent_aug:
            dists = dist_computer.compute_dist(inputs, k=args.num_nearest)
            sigmas = base_sigma * torch.exp(rate * (dists - norm_const))
            sigmas = gaussian_normalization(inputs, sigmas, num_classes, spatial_size, device)
        else:
            sigmas = base_sigma

        if biased:
            bias = labels.clone()
            bias[bias == 0] = -1
            bias = bias.unsqueeze(dim=1).repeat_interleave(2, dim=1)

            orthogonal_vector = [-1, 1]
            bias[:, 0] = bias[:, 0] * orthogonal_vector[0]
            bias[:, 1] = bias[:, 1] * orthogonal_vector[1]

            inputs = inputs + bias_weight * bias + torch.randn_like(inputs, device=device) * sigmas
        else:
            inputs = inputs + torch.randn_like(inputs, device=device) * sigmas

        # end_random = time.time()
        # print(f"Finished randomizing after {end_random-start_random} seconds.")

        # forward + loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), inputs.size(0))

        # compute accuracy
        acc1 = accuracy(outputs, labels)
        acc.update(acc1[0].cpu().numpy()[0], inputs.size(0))

        # zero the parameter gradients and run the optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        acc.val = acc.val.item()
        
        # if i % 10 == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
        #         epoch, i, len(loader), batch_time=batch_time,
        #         data_time=data_time, loss=losses, top1=acc))

    return losses.avg, acc.avg


def test(loader: DataLoader, model: torch.nn.Module, criterion, base_sigma: float, rate: float,
         dist_computer: KNNDistComp, spatial_size: int, num_classes: int, biased: bool, bias_weight: float,
         input_dependent_aug: bool, norm_const: float, device: torch.device):
    """
    Run one epoch of testing.

    :param loader: loader containing the training data
    :param model: representation of the model that is trained
    :param criterion: loss function
    :param base_sigma: value of the base sigma
    :param rate: semi-elasticity constant (for chosen sigma function)
    :param dist_computer: module to compute the distance to each sample in the training data
    :param spatial_size: dimension/size of the image (height and width)
    :param num_classes: number of possible classes in the data
    :param input_dependent_aug:
    :param norm_const: normalization constant for the data set
    :param device: device used for the computations
    :return: average loss and accuracy
    """

    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader, 0):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            if input_dependent_aug:
                dists = dist_computer.compute_dist(inputs, k=args.num_nearest)
                sigmas = base_sigma * torch.exp(rate * (dists - norm_const))
                sigmas = gaussian_normalization(inputs, sigmas, num_classes, spatial_size, device)
            else:
                sigmas = base_sigma

            if biased:
                bias = labels.clone()
                bias[bias == 0] = -1
                bias = bias.unsqueeze(dim=1).repeat_interleave(2, dim=1)

                orthogonal_vector = [-1, 1]
                bias[:, 0] = bias[:, 0] * orthogonal_vector[0]
                bias[:, 1] = bias[:, 1] * orthogonal_vector[1]

                inputs = inputs + bias_weight * bias + torch.randn_like(inputs, device=device) * sigmas
            else:
                inputs = inputs + torch.randn_like(inputs, device=device) * sigmas

            # forward + compute accuracy
            outputs = model(inputs)
            acc1 = accuracy(outputs, labels)
            acc.update(acc1[0].cpu().numpy()[0], inputs.size(0))

            # compute loss + backward
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

    return losses.avg, acc.avg


if __name__ == "__main__":
    main()
