import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os


def show_img(tensor):
    img = transforms.ToPILImage()(tensor)
    plt.imshow(img)
    plt.show()
    return img


def save_model(model, dirname, modelfilename):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(model,os.path.join(dirname,modelfilename))


def show_loss(train_loss_history, test_loss_history):
    # history_dict = history.history
    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']
    loss_values = train_loss_history
    val_loss_values = test_loss_history
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Train and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss.jpg")
    plt.show()


def show_accuracy(train_acc_history, test_acc_history):
    # history_dict = history.history
    # acc = history_dict['acc']
    # val_acc = history_dict['val_acc']
    acc = train_acc_history
    val_acc = test_acc_history
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Train and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("acc.jpg")
    plt.show()
