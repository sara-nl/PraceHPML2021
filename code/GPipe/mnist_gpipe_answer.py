'''
Mnist example in pytorch
https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py
'''

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

# Validation
def evaluate(test_loader, in_device, out_device, model):
    tick = time.time()
    steps = len(test_loader)
    data_tested = 0
    loss_sum = torch.zeros(1, device=out_device)
    accuracy_sum = torch.zeros(1, device=out_device)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            current_batch = input.size(0)
            data_tested += current_batch
            input = input.to(device=in_device)
            target = target.to(device=out_device)

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

# Single training epoch
def run_epoch(args, model, in_device, out_device, train_loader, test_loader, epoch, optimizer):
    torch.cuda.synchronize(in_device)
    tick = time.time()

    steps = len(train_loader)
    data_trained = 0
    loss_sum = torch.zeros(1, device=out_device)
    model.train()
    for i, (input, target) in enumerate(train_loader):
        data_trained += batch_size
        input = input.to(device=in_device, non_blocking=True)
        target = target.to(device=out_device, non_blocking=True)

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

    torch.cuda.synchronize(in_device)
    tock = time.time()

    train_loss = loss_sum.item() / data_trained
    valid_loss, valid_accuracy = evaluate(test_loader, in_device, out_device, model)
    torch.cuda.synchronize(in_device)

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
    parser.add_argument('--batchsize', type=int, default=512, help='The batch size. When using with GPipe, make sure that the batch size is divisible by the number of microbatches/chunks.')
    parser.add_argument('--num_microbatches', type=int, default=6, help='The number of microbatches, also known as "chunks", used by GPipe.')
    parser.add_argument('--epochs', type=int, default=1, help='The amount of epochs to train.')
    parser.add_argument('--num_workers_dataloader', type=int, default=6, help='The amount of workers to be used by the dataloader. Typically, one would set this equal to amount of available CPU cores.')
    parser.add_argument('--log_interval', type=int, default=1, help='Progress, speed and memory consumption are reported every "log_interval" iterations.')
    parser.add_argument('--balance_by', default='time', choices=['time', 'memory'], help='Let GPipe balance the model by time, or memory.')
    args = parser.parse_args()

    batch_size = args.batchsize
    epochs = args.epochs
    log_inter = args.log_interval

    device = torch.device("cuda")
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
    # model = model_names[args.arch].cuda()
    model = model_names[args.arch]

    print("== Autobalancing partitions ==")
    partitions = torch.cuda.device_count()
    sample = torch.empty(batch_size, 1, 28, 28)
    if args.balance_by == 'time':
        balance = balance_by_time(partitions, model, sample)
    elif args.balance_by == 'memory':
        balance = balance_by_size(partitions, model, sample)
    else:
        raise NotImplementedError("Unsupport value specified for 'balance_by' argument")

    print("== Wrapping model as GPipe model ==")
    model = GPipe(model, balance, chunks=args.num_microbatches)
    
    #---------------------------------------------------------------------------------
    # Specify input and output to the correct device
    devices = list(model.devices)
    in_device  = devices[0]
    out_device = devices[-1]

    throughputs = []
    elapsed_times = []
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    print("== Start training ==")
    for epoch in range(epochs):
        throughput, elapsed_time = run_epoch(args, model, in_device, out_device, train_loader, test_loader, epoch, optimizer)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

    _, valid_accuracy = evaluate(test_loader, in_device, out_device, model)

    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))
