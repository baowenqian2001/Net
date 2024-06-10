import os 
import argparse 
import math 
import shutil 
import random 
import numpy as np 
import torch 
import torch.optim as optim 
from torchvision import transforms 
import torch.optim.lr_scheduler as lr_scheduler

from network import * 
from utils.lr_methods import warmup 
from dataload.dataload_five_flower import Five_Flowers_Loda 
from utils.train_engin import train_one_epoch, evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=5, help='the number of classes')
parser.add_argument('--epochs', type=int, default=50, help='the number of training epoch')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size for training')
parser.add_argument('--lr', type=float, default=0.0002, help='star learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='end learning rate') 
parser.add_argument('--seed', default=False, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--use_amp', default=False, action='store_true', help=' training with mixed precision') 
parser.add_argument('--data_path', type=str, default="/mnt/d/Datasets/flower")
parser.add_argument('--model', type=str, default="vgg", help=' select a model for training') 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed) # # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed) # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('random seed has been fixed')
    seed_torch() 

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print('args')

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter 
        # 存放你要使用的tensorboard显示的数据的绝对路径
        log_path = os.path.join('/home/vipuser/workspace/result/tensorboard', args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path)) 

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path) #当log文件存在时删除文件夹。记得在代码最开始import shutil 

        # 实例化一个tensorboard
        tb_writer = SummaryWriter(log_path)

    data_transform = {
    "train": transforms.Compose(
        [transforms.RandomResizedCrop(224),  # 随机裁剪
            transforms.RandomHorizontalFlip(),  # 随机旋转
            transforms.ToTensor(),
            # 把shape=(H x W x C) 的像素值为 [0, 255] 的 PIL.Image 和 numpy.ndarray转换成shape=(C,H,WW)的像素值范围为[0.0, 1.0]的 torch.FloatTensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
    ),
    "test": transforms.Compose(
        [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
    )}
    # # 对标pytorch封装好的ImageFlolder，我们自己实现了一个数据加载类 Five_Flowers_Load，并使用指定的预处理操作来处理图像，结果会同时返回图像和对应的标签。
    
    data_path = '/home/vipuser/workspace/data/cnn/data'
    net.to(device)
    loss_function = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=0.0002)


    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    vaild_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["test"])
    train_num = len(train_dataset)
    test_num = len(vaild_dataset)

    n_works = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 9])
    print('Using {} dataloader workers every process'.format(n_works))

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_works)
    valid_loader = DataLoader(vaild_dataset, batch_size=batch_size, shuffle=False, num_workers=n_works, drop_last=True)

    print('using {} images for traing and using {} images for testing'.format(train_num, test_num))

    epochs = 40
    save_path = os.path.join(os.getcwd(), '../../checkpoints/alex')
    if os.path.isdir(save_path):
        print("checkpoints save in " + save_path)
    else:
        os.makedirs(save_path)
        print("new a dir to save checkpoints: " + save_path)

    best_acc = 0.0 
    train_steps = len(train_loader)

    # training 
    for epoch in range(epochs):
        time_start = time.time() 
        net.train() 
        running_loss = 0.0 
        for step, data in enumerate(train_loader):
            images, labels = data 
            optimizer.zero_grad() 
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward() 
            optimizer.step()

            # print statistics 
            running_loss += loss.item() 
            print("Epoch" + str(epoch) + ": processing:" + str(step) + "/" + str(train_steps))
        
        # validate
        time_end = time.time() 
        net.eval() 
        acc = 0.0 
        with torch.no_grad():
            for val_data in valid_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / test_num 
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f time_one_epoch: %.3f ' %
              (epoch + 1, running_loss / train_steps, val_accurate, (-time_start + time_end)))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(save_path, 'alex_flower.pth'))

    print('Finished Training')
 
if __name__ == '__main__':
    main()