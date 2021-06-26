import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
from random import randint, gauss


epochs = 2000
batch_size = 1024
verbose = 10


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        #encoder
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size = (3, 3),
                                             padding = 1, stride = 2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(4, 8, kernel_size = (3, 3),
                                             padding = 1, stride = 2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(8, 16, kernel_size = (3, 3),
                                             padding = 1, stride = 2)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, kernel_size = (5, 5),
                                             padding = 2, stride = 2))
        self.fc6 = nn.Linear(64, 20)
        self.mu = nn.Linear(20, 2)
        self.log_sq = nn.Linear(20, 2)
        #decoder
        self.fc7 = nn.Linear(2, 50)
        self.fc8 = nn.Linear(50, 64)
        
        self.deconv9 = nn.ConvTranspose2d(1, 16, kernel_size=(4, 4),
                                          padding = 1, stride = 2)
        self.deconv10 = nn.ConvTranspose2d(16, 8, kernel_size=(4, 4),
                                           padding = 1)
        self.deconv11 = nn.ConvTranspose2d(8, 4, kernel_size=(4, 4),
                                           padding = 2)
        
        self.deconv12 = nn.ConvTranspose2d(4, 1, kernel_size=(4, 4),
                                           padding = 1, stride=2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.01)

    def encode(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(-1, 64)
        x = self.relu(self.fc6(x))
        return self.mu(x), self.log_sq(x)

    def decode(self, x):
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = x.view(-1, 1, 8, 8)

        x = self.relu(self.deconv9(x))
        x = self.relu(self.deconv10(x))
        x = self.relu(self.deconv11(x))
        x = self.sigmoid(self.deconv12(x))
        return x

    def forward(self, x, deviation = True):
        mu, log_sq = self.encode(x)
        hidden = (mu, log_sq)
        if deviation:
            x = self.point_from_q(mu, log_sq)
        else:
            x = mu
        x = self.decode(x)
        return x, hidden[0], hidden[1]

    def point_from_q(self, mu, log_sq):
		    epsilon = Variable(torch.randn(mu.size()),
                           requires_grad=False).type(torch.FloatTensor)
		    sigma = torch.exp(log_sq / 2)
		    return mu + sigma * epsilon * 0

    def show_parameters(self):
        tensor_list = list(self.state_dict().items())
        for layer_tensor_name, tensor in tensor_list:
            print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, kernel_size = (3, 3),
                                             padding = 1, stride = 2)
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(4, 8, kernel_size = (3, 3),
                                             padding = 1, stride = 2)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(8, 16, kernel_size = (3, 3),
                                             padding = 1, stride = 2)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, kernel_size = (5, 5),
                                             padding = 2, stride = 2))
        self.fc6 = nn.Linear(64, 20)
        self.fc7 = nn.Linear(20, 10)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(-1, 64)
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return self.sigmoid(x)


def L1_reg(model, k = 0.00000001):
    L1_reg = torch.tensor(0., requires_grad=True)
    length = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            L1_reg = L1_reg + torch.norm(param, 1) ** 2
            length += 1
    loss = k * L1_reg
    #print(loss)
    return loss / length

def lossk1(mu, log_sq):
    return torch.mean(.5 * torch.sum((mu**2) + torch.exp(log_sq) - 1 - log_sq, 1))

def same_class_loss(real_img, fake_img):
    m1 = classifier(real_img)
    m2 = classifier(fake_img)
    l = torch.mean((m1 - m2) ** 2)
    return l

def show_tensors(tensors):
    fig, axes = plt.subplots(1, len(tensors))
    for ind, tensor in enumerate(tensors):
        img = tensor.detach().permute(1, 2, 0).numpy()
        img = cv2.resize(img, (256, 256),interpolation = cv2.INTER_AREA)
        if len(tensors) == 1:
            axes.imshow(img)
            break
        axes[ind].imshow(img)
    plt.show()

def vis(hidden, pred):
    hidden = hidden.detach().numpy()
    pred = pred.detach().numpy()
    colors = {
        0: "Gray",
        1: "Red",
        2: "Green",
        3: "Blue",
        4: "Orange",
        5: "Black",
        6: "Yellow",
        7: "Pink",
        8: "Brown",
        9: "Purple"
        }
    for i, (x, y) in enumerate(hidden):
        col = colors[pred[i]]
        plt.plot(x, y,'o', color = col)

def vec2image(vec):
    vec = torch.tensor(vec, dtype = torch.float)
    vec.reshape(1, -1)
    show_tensors(autoencoder.decode(vec))

def slide(vec1, vec2, n = 20):
    vec1 = torch.tensor(vec1, dtype = torch.float)
    vec2 = torch.tensor(vec2, dtype = torch.float)
    for i in range(n):
        vec = (vec1 * (n - i) + vec2 * i) / n
        print(vec)
        vec2image(vec)


def train_classifier():
    loss = nn.NLLLoss()
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=0.01)
    for epoch in range(5):
        for step, (x, y) in enumerate(dataset_loader):
            optimizer_C.zero_grad()
            y_pred = classifier.forward(x)
            c_loss = loss(y_pred, y)
            c_loss.backward()
            optimizer_C.step()
            if step % 100 == 0:
                print(c_loss)
    torch.save(classifier.state_dict(), (r'c.pth'))



if __name__ == '__main__':
    dataset_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/files/', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
        batch_size=batch_size, shuffle=True)
    
    autoencoder = Autoencoder()
    loss = torch.nn.MSELoss()
    
    try:
        autoencoder.load_state_dict(torch.load(r'a.pth'))
        print('loaded model')
        
    except:
        print('new model')
        autoencoder.show_parameters()
        print()
    optimizer_A = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    Train = True
    
    classifier = Classifier()
    try:
        classifier.load_state_dict(torch.load(r'c.pth'))
    except:
        train_classifier()
        
    for epoch in range(epochs):
        for step, (real_img, y_train) in enumerate(dataset_loader):
            optimizer_A.zero_grad()
            fake_img, mu, log_sq = autoencoder.forward(real_img)
            
            visual_dif = loss(fake_img, real_img)
            class_dif = same_class_loss(real_img, fake_img) * 1
            k1 = lossk1(mu, log_sq) * 0.03
            reg = L1_reg(autoencoder)
            a_loss = visual_dif + class_dif + k1 + reg
            
            a_loss.backward()
            optimizer_A.step()
              
            if step % verbose == 0:
                #print(mu[0])
                print()
                print('epoch {} / {},\nvis_loss   = {:f}'.format(epoch, epochs, visual_dif))
                print('class_loss = {:f},\nreg_loss   = {:f}'.format(class_dif, reg))
                print('k1_loss    = {:f}'.format(k1))
                print()
                show_tensors([real_img[0], fake_img[0]])
                vis(mu, y_train)
                plt.show()
                torch.save(autoencoder.state_dict(),
                          (r'a.pth'))
    

