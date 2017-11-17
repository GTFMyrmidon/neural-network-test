import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable

def train(model, train_loader, args):
    num_epochs = args.epochs
    learning_rate = args.learning_rate

    if args.GPU:
        # Check if multiple GPUs are present
        if torch.cuda.device_count() > 1:
            print (torch.cuda.device_count(), "GPUs detected")
            model = nn.DataParallel(model)

        # Check if CUDA is available
        if torch.cuda.is_available():
            model.cuda()

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            # Zero-out gradients
            optimizer.zero_grad()

            # Compute forward pass
            outputs = model(images)

            # Compute loss
            loss = loss_fn(outputs, labels)

            # Compute backward pass
            loss.backward()

            # Update weights/parameters
            optimizer.step()

            # Print training progress
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Save the Trained Model
    torch.save(model.state_dict(), 'model.pkl')

def test(model, test_loader, args):
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = Variable(images).cuda()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

def classify(model, image):
    model.eval()
    x = Variable(image).cuda()
    output = model(x)
    _, predicted = torch.max(output, 1)
    return
