import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models

import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trainset = datasets.ImageFolder(root='~/data/Fruit-Images-Dataset/Training', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=150,
                                          shuffle=True, num_workers=4)
testset = datasets.ImageFolder(root='~/data/Fruit-Images-Dataset/Test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=150,
                                         shuffle=False, num_workers=4)

model_ft = models.vgg16(pretrained=True)
num_ftrs = model_ft.classifier[0].in_features
model_ft.classifier = nn.Linear(num_ftrs, 118)

optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
loss = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

trainer = create_supervised_trainer(model_ft, optimizer, loss, device=device)
evaluator = create_supervised_evaluator(model_ft,
                                        metrics={
                                            'accuracy': Accuracy(),
                                            'CrossEntropy': Loss(loss)},
                                        device=device)

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    message = f"Epoch[{trainer.state.epoch}] " \
              f"Iteration[{trainer.state.iteration}] " \
              f"Loss: {trainer.state.output:.10f}"
    print(message)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(trainloader)
    metrics = evaluator.state.metrics
    message = f"Training Results - Epoch: {trainer.state.epoch}  " \
              f"Avg accuracy: {metrics['accuracy']:.10f} " \
              f"Avg loss: {metrics['CrossEntropy']:.10f}"
    print(message)

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(testloader)
    metrics = evaluator.state.metrics
    message = f"Validation Results - Epoch: {trainer.state.epoch}  " \
              f"Avg accuracy: {metrics['accuracy']:.10f} " \
              f"Avg loss: {metrics['CrossEntropy']:.10f}"
    print(message)

epochs = 1
trainer.run(trainloader, max_epochs=epochs)
