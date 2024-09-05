from model_quick_draw import quick_draw_CNN
from datasets_quick_draw import Quickdrawdatasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor, Compose, Resize
import warnings
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
from tqdm.autonotebook import tqdm 
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the Quick Draw model proposed by Google""")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--checkpoint_dir", type=str, default="trained_models_quick_draw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--early_stopping_duration", type=int, default=3)
    parser.add_argument("--log_dir", type=str, default="tensorboard_quick_draw", help = "Place to save logging infor")

    args = parser.parse_args()
    return args
def train(args):
    Train_dataset = Quickdrawdatasets(root = "data_quick_draw", total_img_per_class=500, percent=0.8, mode="train")
    print(len(Train_dataset))
    Train_dataloader = DataLoader(
        dataset = Train_dataset,
        batch_size = 16,
        num_workers = 4,
        drop_last = True,
        shuffle=True
    )
    Val_dataset = Quickdrawdatasets(root = "data_quick_draw", total_img_per_class=500, percent=0.8, mode="test")
    print(len(Val_dataset))
    Val_dataloader = DataLoader(
        dataset = Val_dataset,
        batch_size = 16,
        num_workers = 4,
        drop_last = True,
        shuffle=True
    )
    model = quick_draw_CNN()
    optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    if os.path.isdir(args.checkpoint_dir):
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "last.pt"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_accuracy = checkpoint['best_accuracy']
        #shutil.rmtree(args.checkpoint_dir)
    else:
        os.makedirs(args.checkpoint_dir)
        epoch = 0
        best_epoch = 0
        best_accuracy = -1

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)
    num_iters_per_epoch = len(Train_dataloader)
    
    for epoch in range(epoch,args.num_epochs):
        model.train()
        progress_bar = tqdm(Train_dataloader, colour="blue")
        for iter, (image, label) in enumerate(progress_bar):
            predictions  = model(image)
            loss_value = criterion(predictions,label)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch+1, args.num_epochs, loss_value.item()))
            writer.add_scalar("Train/Loss", loss_value, global_step=epoch*num_iters_per_epoch+iter)
        model.eval()
        all_loss = []
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for iter, (image, label) in enumerate(Val_dataloader):
                outputs  = model(image)
                loss_value = criterion(outputs,label)
                predictions = torch.argmax(outputs, dim=1)
                all_labels.extend(label.tolist())
                all_predictions.extend(predictions.tolist())
                all_loss.append(loss_value)
            accuracy = accuracy_score(all_labels,all_predictions)
            avg_loss = np.mean(all_loss)
            print("Epoch {}/{}. Loss {:0.4f}. Acc {:0.4f}".format(epoch+1, args.num_epochs, avg_loss, accuracy))
            writer.add_scalar("Val/Loss", avg_loss, global_step=epoch)
            writer.add_scalar("Val/Accuracy", accuracy, global_step=epoch)
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch+1,
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch
            }

            if accuracy > best_accuracy:
                best_epoch = epoch
                best_accuracy = accuracy
                torch.save(checkpoint,os.path.join(args.checkpoint_dir, "best.pt"))
            torch.save(checkpoint,os.path.join(args.checkpoint_dir, "last.pt"))
        if epoch - best_epoch > args.early_stopping_duration:
            print("Stop the training process at epoch {}".format(epoch+1))
            exit(0)
if __name__ == '__main__':
    args = get_args()
    train(args)