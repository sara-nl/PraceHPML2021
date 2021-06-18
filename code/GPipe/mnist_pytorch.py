import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch.optim as optim
from torchvision import datasets, transforms

from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time, balance_by_size

from typing import cast
import time
from collections import OrderedDict

from gpipemodels.resnet import *
from gpipemodels.vgg import *
from gpipemodels.mobilenetv2 import *

model_names = {
    'resnet18'   : mnist_resnet18(),
    'resnet50'   : mnist_resnet50(),
    'resnet152'  : mnist_resnet152(),
    'vgg11'      : mnist_vgg11(),
    'vgg16'      : mnist_vgg16(),
    'mobilenetv2': mnist_mobilenetv2(),
}

## Things you'll need to adapt to make a code use GPipe
## - Make sure the model is of the nn.Sequential(...) type (already the case for this example)
## - Model no longer needs to be moved to device
## - Determine how to split the model
## - Wrap sequential model as torchgpipe.GPipe model
## - Send input to first device, output to last device
## Hints as to what needs to happen where are indicated as '##HINT'

## HINT: evaluate(...) contains the .to(device=...) for both input and target.
##       you'll need to change it's dev arguments to two arguments:
##       one for the input, and one for the output device.
##       Then, use those arguments to send the input & target to the right device

## HINT: In general: search this code for 'dev', and consider if you need to replace 
##       it by the input device, or output device

# Validation
def evaluate(test_loader, dev, model):
    tick = time.time()
    steps = len(test_loader)
    data_tested = 0
    loss_sum = torch.zeros(1, device=dev)
    accuracy_sum = torch.zeros(1, device=dev)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            current_batch = input.size(0)
            data_tested += current_batch
            input = input.to(device=dev)
            target = target.to(device=dev)

            output = model(input)

            loss = F.nll_loss(output, target)
            loss_sum += loss * current_batch

            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum()
            accuracy_sum += correct

            if i % log_inter == 0:
                percent = i / steps * 100
                throughput = data_tested / (time.time() - tick)
                print('valid | %d%% | %.3f samples/sec (estimated)'
                    '' % (percent, throughput))

    loss = loss_sum / data_tested
    accuracy = accuracy_sum / data_tested

    return loss.item(), accuracy.item()

## HINT: you can have the torch.cuda.synchronize(...) function act on the input device
##       in production code, you wouldn't do this explicit synchronization: it is only
##       here for accurate timing

# Single training epoch
def run_epoch(args, model, dev, train_loader, test_loader, epoch, optimizer):
    torch.cuda.synchronize(dev)
    tick = time.time()

    steps = len(train_loader)
    data_trained = 0
    loss_sum = torch.zeros(1, device=dev)
    model.train()
    for i, (input, target) in enumerate(train_loader):
        data_trained += batch_size
        input = input.to(device=dev, non_blocking=True)
        target = target.to(device=dev, non_blocking=True)

        output = model(input)
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach() * batch_size

        if i % log_inter == 0:
            percent = i / steps * 100
            throughput = data_trained / (time.time()-tick)
            print(f'train | %d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                '' % (epoch+1, epochs, percent, throughput))

            mem_used = ''
            mem_reserved = ''
            for device in range(0, torch.cuda.device_count()):
                torch.cuda.synchronize(device)
                print('{0} GPU {1}: allocated {2:.3f} GB, reserved {3:.3f} GB'.format(mem_used, device, torch.cuda.memory_allocated(device)/(1024*1024*1024), torch.cuda.memory_reserved(device)/(1024*1024*1024)))

    torch.cuda.synchronize(dev)
    tock = time.time()

    train_loss = loss_sum.item() / data_trained
    valid_loss, valid_accuracy = evaluate(test_loader, dev, model)
    torch.cuda.synchronize(dev)

    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    print('%d/%d epoch | train loss:%.3f %.3f samples/sec | '
        'valid loss:%.3f accuracy:%.3f'
        '' % (epoch+1, epochs, train_loss, throughput,
                valid_loss, valid_accuracy))

    return throughput, elapsed_time


if __name__ == '__main__':
    # Get default datadir
    jhl_teacher_dir = os.getenv('TEACHER_DIR')
    if jhl_teacher_dir is not None:
        default_datadir = os.path.join(jhl_teacher_dir, 'JHL_data')
    else:
        default_datadir = os.getenv('TMPDIR')

    # Training settings
    parser = argparse.ArgumentParser(description='D-DNN mnist benchmark')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--datadir', default=default_datadir, help='Directory where "benchmark/mnist/MNIST.tar.gz" was extracted')
    parser.add_argument('--batchsize', type=int, default=256, help='The batch size. When using with GPipe, make sure that the batch size is divisible by the number of microbatches/chunks.')
    parser.add_argument('--num_microbatches', type=int, default=24, help='The number of microbatches, also known as "chunks", used by GPipe.')
    parser.add_argument('--epochs', type=int, default=1, help='The amount of epochs to train.')
    parser.add_argument('--num_workers_dataloader', type=int, default=6, help='The amount of workers to be used by the dataloader. Typically, one would set this equal to amount of available CPU cores.')
    parser.add_argument('--log_interval', type=int, default=1, help='Progress, speed and memory consumption are reported every "log_interval" iterations.')
    parser.add_argument('--balance_by', default='time', choices=['time', 'memory'], help='Let GPipe balance the model by time, or memory.')
    args = parser.parse_args()

    batch_size = args.batchsize
    epochs = args.epochs
    log_inter = args.log_interval

    dataloader_kwargs = {'pin_memory': True}

    # Fixed seeds for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Initialize DataLoaders
    print("== Creating DataLoader ==")
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.datadir, train=True, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, num_workers=args.num_workers_dataloader,
        **dataloader_kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.datadir, train=False, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True, num_workers=args.num_workers_dataloader,
        **dataloader_kwargs)

    #---------------------------------------------------------------------------------
    # Move model to GPU.
    print("== Creating model '{}' ==".format(args.arch))

## HINT: does the model still need to be moved explicitely when using GPipe? 
    dev = torch.device("cuda")
    model = model_names[args.arch].to(dev)

## HINT: This is where you'll want to determine how to partition the model (the 'balance')
##       using balance_by_time or balance_by_memory.
##       You may find the torch.cuda.device_count() function useful to determine the amount of partitions
##       The GPipe balance functions require a sample input that you can make with
##       torch.empty(batch_size, 1, 28, 28) - this has the same dimensions as MNIST

## HINT: After balancing, you'll want to wrap the model as a GPipe model before running the training loops

## HINT: You'll want to specify the input and output devices.
##       The input device is the first device returned by list(model.devices), the output device is the last one.
##       Store those in a variable so you can pass these to the run_epoch(...) and evaluate(...) functions


    throughputs = []
    elapsed_times = []
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print("== Start training ==")
    for epoch in range(epochs):
        ## HINT: remember to change the device argument into an input and output device argument
        throughput, elapsed_time = run_epoch(args, model, dev, train_loader, test_loader, epoch, optimizer)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    ## HINT: remember to change the device argument into an input and output device argument
    _, valid_accuracy = evaluate(test_loader, dev, model)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))
