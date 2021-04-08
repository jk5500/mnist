import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorboard as tb


# to download mnist data
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# for embedding
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


class CNN(nn.Module):

    def __init__(self, input_channel=1, num_classes=10):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16*7*7, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)

        return x


# helper functions

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8','9')


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from
    a trained network and a list of images
    """

    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(64, 64))

    for idx in np.arange(16):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]), fontsize=100,
            color=("green" if preds[idx] == labels[idx].item() else "red"))

    return fig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
num_epochs = 100
learning_rate = 0.001
model = CNN(input_channel=1, num_classes=10)
model = model.to(device=device)

writer = SummaryWriter(f"runs/MNIST/BatchSize {batch_size} LR {learning_rate}")
train_data = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(),
                            download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# to let network know model is training
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)

losses = []
accuracies = []
steps_in_tensorboard = 0
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        _, prediction = scores.max(1)
        num_correct = (prediction == targets).sum()
        running_training_acc = float(num_correct)/float(data.shape[0])
        image_grid = torchvision.utils.make_grid(data)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=steps_in_tensorboard)
        writer.add_scalar("Training accuracy", running_training_acc, global_step=steps_in_tensorboard)
        writer.add_image("mnist images", image_grid, global_step=steps_in_tensorboard)
        writer.add_histogram("cnn1", model.conv1.weight, global_step=steps_in_tensorboard)
        writer.add_histogram("cnn2", model.conv2.weight, global_step=steps_in_tensorboard)
        writer.add_histogram("fc1", model.fc1.weight, global_step=steps_in_tensorboard)
        writer.add_figure('predictions vs. actuals',
                          plot_classes_preds(model, data, targets),
                          global_step=steps_in_tensorboard)

        if batch_idx == 230:
            features = data.reshape(data.shape[0], -1)
            class_labels = [classes[label] for label in prediction]
            writer.add_embedding(
                features,
                metadata=class_labels,
                label_img=data,
                global_step=batch_idx,
            )

        steps_in_tensorboard += 1










