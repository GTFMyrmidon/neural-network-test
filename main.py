#! /usr/bin/env python3
import sys
import argparse
import torch
import cnn
import utility

parser = argparse.ArgumentParser(description='CNN Car Classifier')

# Learning Parameters
parser.add_argument('-lr', type=float, default=1e-3, help='Set the learning rate [Default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='Number of epochs to train for [Default: 256]')
parser.add_argument('-batch_size', type=int, default=64, help='Batch size for training [Default: 64]')
parser.add_argument('-num_filters', type=int, default=16, help='Number of filters to be used [Default: 16]')
parser.add_argument('-filter_size', type=int, default=3, help='Size of filters [Default: 3]')

# Data Parameters

# Model Parameters
parser.add_argument('-')

# Device Parameters
parser.add_argument('-GPU', action='store_true', default=False, help='Enable GPU acceleration [Default: False]')

# Other options
parser.add_argument('-save', type=str, default='model.pkl', help='Save the model to the following path [Default: model.pkl]')
parser.add_argument('-load', type=str, default=None, help='Filename of saved model to load [Default: None]')
parser.add_argument('-classify', type=str, default=None, help='Classify image at given location')

def main():
    
    args = parser.parse_args()

    # Create new model, or load one
    if args.load is None:
        cnn = cnn.CNN()
    else:
        print("Loading model from [%s]" % args.load)
        try:
            cnn.load_state_dict(torch.load(args.load))
        except:
            print("The selected model does not exist.")
            sys.exit()

    if args.GPU:
        # Check if multiple GPUs are present
        if torch.cuda.device_count() > 1:
            print (torch.cuda.device_count(), "GPUs detected")
            cnn = nn.DataParallel(cnn)

        # Check if CUDA is available
        if torch.cuda.is_available():
            cnn.cuda()

    train_dataset = dsets.CIFAR10(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.CIFAR10(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Train, test, or predict
    if args.predict is not None:
        label = utility.classify(cnn, image)
    elif args.test:
        utility.eval(model, test_loader, args)
    else:
        utility.train(model, train_loader, args)

if __name__ == '__main__':
    main()
