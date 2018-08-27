from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from models import MNISTNet, MNISTFIG2Net


def train(args, model, criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if args.vis:
            output, _ = model(x=data, target=target)
        else:
            output = model(x=data, target=target)
        loss = criterion(input=output, target=target)
        optimizer.zero_grad()
        # clip_grad_norm_(model.parameters(), max_norm=10)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.vis:
                output, _ = model(x=data)
            else:
                output = model(x=data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def plot_2d_features(args, model, device, test_loader):
    net_logits = np.zeros((10000, 2), dtype=np.float32)
    net_labels = np.zeros((10000,), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for b_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            _, output2d = model(x=data)
            output2d = output2d.cpu().data.numpy()
            target = target.cpu().data.numpy()
            net_logits[b_idx * args.test_batch_size: (b_idx + 1) * args.test_batch_size, :] = output2d
            net_labels[b_idx * args.test_batch_size: (b_idx + 1) * args.test_batch_size] = target
        for label in range(10):
            idx = net_labels == label
            plt.scatter(net_logits[idx, 0], net_logits[idx, 1])
        plt.legend(np.arange(10, dtype=np.int32))
        plt.show()


def adjust_learning_rate(args, optimizer, epoch):
    # Decreasing the learning rate to the factor of 0.1 at epochs 51 and 65
    # with a batch size of 256 this would comply with changing the lr at iterations 12k and 15k
    if 50 < epoch < 65:
        lr = args.lr * 0.1
    elif epoch >= 65:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser(description='PyTorch L-Softmax MNIST Example')
    parser.add_argument('--margin', type=int, default=4, metavar='M',
                        help='the margin for the l-softmax formula (m=1, 2, 3, 4)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='initial learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                        help='SGD weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--vis', default=False, metavar='V',
                        help='enables visualizing 2d features (default: False).')
    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    criterion = nn.CrossEntropyLoss().to(device)
    if args.vis:
        model = MNISTFIG2Net(margin=args.margin, device=device).to(device)
    else:
        model = MNISTNet(margin=args.margin, device=device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, criterion, device, train_loader, optimizer, epoch)
        test(args, model, criterion, device, test_loader)

    if args.vis:
        plot_2d_features(args, model, device, test_loader)


if __name__ == '__main__':
    main()
