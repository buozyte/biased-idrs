import argparse
import time
import datetime
import os

from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.nn import CrossEntropyLoss, Sequential
from torch.optim.lr_scheduler import StepLR

import datasets as ds
from models.base_models.architectures import ARCHITECTURES, get_architecture
from models import rs
from input_dependent_functions.bias_functions import *
from input_dependent_functions.variance_functions import VARIANCE_FUNCTIONS
from knn import KNNDistComp
import utils
from constants import DATASETS

from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger_file = logging.getLogger(__name__+"_file")
logger = logging.getLogger(__name__)

NUM_WORKERS = 10

# TODO: add simple way to include other bias function in training
#       (currently unsure which option is the best)
# NOTE: normal rs and idrs can be defined via biased idrs
#       (either leaving both functions or only bias function as None)
def main_train(dataset, arch, out_dir,
               epochs=90, batch=256, lr=0.2, lr_step_size=20,
               gamma=0.1, momentum=0.9, weight_decay=1e-4,
               base_sigma=0.12, print_freq=10, alt_sigma_aug=-1, gaussian_augmentation=False,
               id_var=False, num_nearest=20, var_func=None, rate=0.01, biased=False,
               bias_weight=1, bias_func=None, lipschitz_const=0.0,
               add_bias_layer=False, load_from_checkpoint=False, optimize_bias_only=False,
               rel_change_stopping=0.01, patience_stopping=5):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_dir = Path(out_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- prepare data ---
    train_dataset = ds.get_dataset(dataset, 'train')
    test_dataset = ds.get_dataset(dataset, 'test')
    num_classes = ds.get_num_classes(dataset)

    pin_memory = (dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)

    spatial_size, num_channels, norm_const = \
        ds.get_dataset_additional_parameters(dataset)
    # --------------------

    dist_computer = KNNDistComp(train_dataset, device)

    base_model = get_architecture(arch, dataset, device)

    # load from a checkpoint if specified
    if load_from_checkpoint:
        checkpoint = torch.load(load_from_checkpoint)
        base_model.load_state_dict(
            {k_[16:]:v_ for k_, v_ in checkpoint["state_dict"].items()})

    # add bias layer if necessary
    if add_bias_layer:
        bias_layer = utils.AddSkipConnection(
            utils.ConstantBiasLayer((num_channels, spatial_size, spatial_size)))
        base_model = Sequential(bias_layer, base_model)

    # Construct the smoothed model
    if biased:
        model = rs.BiasedIDRSClassifier(base_classifier=base_model, num_classes=num_classes, sigma=base_sigma,
                                     device=device, bias_func=bias_func, variance_func=var_func,
                                     bias_weight=bias_weight, lipschitz=lipschitz_const,
                                     rate=rate, m=norm_const).to(device)
    elif id_var:
        model = rs.IDRSClassifier(base_classifier=base_model, num_classes=num_classes, sigma=base_sigma,
                               distances=None, rate=rate, m=norm_const, device=device).to(device)
    else:
        model = rs.RSClassifier(base_classifier=base_model, num_classes=num_classes, sigma=base_sigma,
                             device=device).to(device)


    logfile_name = out_dir / 'log.txt'
    logger_file.addHandler(logging.FileHandler(logfile_name))
    logger_file.info(f"epoch | lr    | tr.loss | tr.acc  | ts.loss | ts.acc | time.epoch | time.overall")

    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(
        model.parameters() if not optimize_bias_only else bias_layer.parameters(),
        lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    if alt_sigma_aug > -1:
        used_sigma = alt_sigma_aug
    else:
        used_sigma = base_sigma

    # init lists with training metrics
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "timer_epoch": [],
        "timer_overall": [],
        "lr": []
    }

    start_overall = time.time()
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch}")
        start_epoch = time.time()
        train_loss, train_acc = train(train_loader,
                                      model,
                                      criterion,
                                      optimizer,
                                      used_sigma,
                                      rate,
                                      dist_computer,
                                      num_nearest,
                                      spatial_size,
                                      num_channels,
                                      biased,
                                      bias_weight,
                                      bias_func,
                                      gaussian_augmentation,
                                      norm_const,
                                      device,
                                      epoch,)
        test_loss, test_acc = test(test_loader,
                                   model,
                                   criterion,
                                   used_sigma,
                                   rate,
                                   dist_computer,
                                   num_nearest,
                                   spatial_size,
                                   num_channels,
                                   biased,
                                   bias_weight,
                                   bias_func,
                                   gaussian_augmentation,
                                   norm_const,
                                   device,)
        scheduler.step()
        timer_epoch = str(datetime.timedelta(seconds=round(time.time() - start_epoch)))
        timer_overall= str(datetime.timedelta(seconds=round(time.time() - start_overall)))

        # save and print the stats
        results = utils.dict_append(results, {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "timer_epoch": timer_epoch,
            "timer_overall": timer_overall,
            "lr": scheduler.get_last_lr()
            })

        logger_file.info(
            f"{epoch:<6}| " + \
            f"{scheduler.get_last_lr()[0]:<6.3}| " + \
            f"{train_loss:<8.3}| " + \
            f"{train_acc:<8.3}| " + \
            f"{test_loss:<8.3}| " + \
            f"{test_acc:<7.3}| " + \
            f"{timer_epoch:<11}| " + \
            f"{timer_overall:<11}")

        torch.save({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'results': results,
        }, out_dir / 'checkpoint.pth.tar')

        # checking stopping criterion
        if epoch > patience_stopping:
            test_acc_list = results["test_acc"]
            max_rel_change = 0.0
            for i in range(1, patience_stopping+1):
                max_rel_change = max(
                    max_rel_change,
                    np.abs(test_acc_list[-i] - test_acc_list[-i-1]) / test_acc_list[-i-1]
                    )

            if max_rel_change < rel_change_stopping:
                logger.info(f"Stopping criterion satisfied: {max_rel_change:.4f} < {rel_change_stopping}")
                break


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, base_sigma: float,
          rate: float, dist_computer: KNNDistComp, num_nearest: int, spatial_size: int, num_channels: int, biased: bool,
          bias_weight: float, bias_func: str, gaussian_aug: bool, norm_const: float, device: torch.device, epoch: int):
    """
    Run one epoch of training.

    :param loader: loader containing the training data
    :param model: representation of the model that is trained
    :param criterion: loss function
    :param optimizer: optimizer for the model
    :param base_sigma: value of the base sigma
    :param rate: semi-elasticity constant (for chosen sigma function)
    :param dist_computer: module to compute the distance to each sample in the training data
    :param num_nearest:
    :param spatial_size: dimension/size of the image (height and width)
    :param num_channels: number of channels in the data
    :param biased: indicator whether a bias should be used
    :param bias_weight: "weight" of the bias
    :param bias_func: chosen bias function
    :param gaussian_aug: indicator whether a gaussian augmentation w.r.t. input-dependency should be performed
    :param norm_const: normalization constant for the data set
    :param device: device used for the computations
    :param epoch: current epoch
    :return: average loss and accuracy
    """
    
    batch_time = utils.AverageMeter()
    data_loading_time = utils.AverageMeter()
    end = time.time()

    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    model.train()

    for i, (inputs, labels) in enumerate(loader, 0):
        # measure data loading time
        data_loading_time.update(time.time() - end)

        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor).to(device)
        batch_size = len(labels)

        augmentation = utils.get_augmentation(
            inputs, dist_computer, num_nearest,
            gaussian_aug, base_sigma, rate, norm_const, num_channels, spatial_size,
            biased, bias_func, bias_weight)

        inputs += augmentation

        # end_random = time.time()
        # print(f"Finished randomizing after {end_random-start_random} seconds.")

        # forward + loss
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.update(loss.item(), batch_size)

        # compute accuracy
        acc1 = utils.accuracy(outputs, labels)
        acc.update(acc1[0].cpu().numpy()[0], batch_size)

        # zero the parameter gradients and run the optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        acc.val = acc.val.item()
        
        if i % 10 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(epoch, i, len(loader), batch_time=batch_time,
                                                                   data_time=data_loading_time, loss=losses, top1=acc))

    return losses.avg, acc.avg


def test(loader: DataLoader, model: torch.nn.Module, criterion, base_sigma: float, rate: float,
         dist_computer: KNNDistComp, num_nearest: int, spatial_size: int, num_channels: int, biased: bool,
         bias_weight: float, bias_func: str, gaussian_aug: bool, norm_const: float, device: torch.device):
    """
    Run one epoch of testing.

    :param loader: loader containing the training data
    :param model: representation of the model that is trained
    :param criterion: loss function
    :param base_sigma: value of the base sigma
    :param rate: semi-elasticity constant (for chosen sigma function)
    :param dist_computer: module to compute the distance to each sample in the training data
    :param num_nearest:
    :param spatial_size: dimension/size of the image (height and width)
    :param num_channels: number of channels in the data
    :param biased: indicator whether a bias should be used
    :param bias_weight: "weight" of the bias
    :param bias_func: chosen bias function
    :param gaussian_aug: indicator whether a gaussian augmentation w.r.t. input-dependency should be performed
    :param norm_const: normalization constant for the data set
    :param device: device used for the computations
    :return: average loss and accuracy
    """

    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader, 0):
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor).to(device)

            augmentation = utils.get_augmentation(
                inputs, dist_computer, num_nearest,
                gaussian_aug, base_sigma, rate, norm_const, num_channels, spatial_size,
                biased, bias_func, bias_weight)

            inputs += augmentation

            # forward + compute accuracy
            outputs = model(inputs)
            acc1 = utils.accuracy(outputs, labels)
            acc.update(acc1[0].cpu().numpy()[0], inputs.size(0))

            # compute loss + backward
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

    return losses.avg, acc.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('dataset', type=str, choices=DATASETS)
    parser.add_argument('arch', type=str, choices=ARCHITECTURES)
    parser.add_argument('out_dir', type=str,
                        help='folder to save model and training log)')
    parser.add_argument('--add_bias_layer', type=bool, default=False)
    parser.add_argument('--load_from_checkpoint', type=str, default=False)
    parser.add_argument('--optimize_bias_only', type=bool, default=False)
    parser.add_argument('--rel_change_stopping', type=float, default=0.001,
                        help='stopping threshold for the relative change in evaluation accuracy')
    parser.add_argument('--patience_stopping', default=8, type=int,
                        help='patience of stopping criterion checking')
    #parser.add_argument('--num_workers', default=2, type=int, metavar='N',
    #                    help='number of data loading workers (default: 4)')
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
    #parser.add_argument('--gpu', default=None, type=str,
    #                    help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--print_freq', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--alt_sigma_aug', default=-1, type=float,
                        help="Alternative sigma to use when doing input dependent smoothing")
    parser.add_argument('--gaussian_augmentation', default=False, type=bool,
                        help="Indicator whether to use input-dependent gaussian data augmentation")
    # variance
    parser.add_argument('--id_var', default=False, type=bool,
                        help="Indicator whether to use input-dependent computation of the variance")
    parser.add_argument('--num_nearest', default=20, type=int,
                        help='How many nearest neighbors to use')
    parser.add_argument('--var_func', default=None, type=str, choices=VARIANCE_FUNCTIONS,
                        help='Choice for the variance function to be used')
    parser.add_argument('--rate', default=0.01, type=float,
                        help="The rate used for gaussian augmentation")
    # bias
    parser.add_argument('--biased', default=False, type=bool,
                        help="Indicator whether to use a biased")
    parser.add_argument('--bias_weight', default=1, type=float,
                        help="Weight of bias")
    parser.add_argument('--bias_func', default=None, type=str, choices=BIAS_FUNCTIONS,
                        help='Choice for the bias function to be used')
    parser.add_argument('--lipschitz_const', default=0.0, type=float,
                        help="Lipschitz constant of the bias functions")
    args = parser.parse_args()

    args_dict = vars(args)

    main_train(**args_dict)
