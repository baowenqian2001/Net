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

import network
from network import * 
from utils.lr_methods import warmup 
from dataload.dataload_five_flower import Five_Flowers_Load
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
parser.add_argument('--data_path', type=str, default="/home/vipuser/workspace/data/data")
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
    
    print('opt')

    if opt.tensorboard:
        from torch.utils.tensorboard import SummaryWriter 
        # 存放你要使用的tensorboard显示的数据的绝对路径
        log_path = os.path.join('/home/vipuser/workspace/result/tensorboard', opt.model)
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
    "val": transforms.Compose(
        [transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
    )}

    # # 对标pytorch封装好的ImageFlolder，我们自己实现了一个数据加载类 Five_Flowers_Load，并使用指定的预处理操作来处理图像，结果会同时返回图像和对应的标签。
    train_dataset = Five_Flowers_Load(os.path.join(opt.data_path, 'train'), transform=data_transform['train'])
    val_dataset = Five_Flowers_Load(os.path.join(opt.data_path, 'val'),transform=data_transform['val'])

    if opt.num_classes != train_dataset.num_class:
        raise ValueError('dataset have {} classes, but input {}'.format(train_dataset, num_class, opt.num_classes))
    
    nw = min([os.cpu_count(), opt.batch_size if opt.batch_size > 1 else 0, 8]) # number os workers
    print('Using {} dataloader workers every process'.format(nw))

    # 使用DataLoader将加载的数据集处理成批量（batch）加载模式
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    # create model 
    model = network.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=opt.lr)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x:((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc = 0.

    # save parameters path 
    save_path = os.path.join(os.getcwd(), '/home/vipuser/workspace/result/weights', opt.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    for epoch in range(opt.epochs):
        # train
        mean_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, use_amp=opt.use_amp, lr_method=warmup)
        scheduler.step() 
        # validate
        val_acc = evaluate(model=model, data_loader=val_loader, device=device)
        print('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc))   
        with open(os.path.join(save_path, "AlexNet_log.txt"), 'a') as f: 
                f.writelines('[epoch %d] train_loss: %.3f  train_acc: %.3f  val_accuracy: %.3f' %  (epoch + 1, mean_loss, train_acc, val_acc) + '\n')

        if opt.tensorboard:
            tags = ["train_loss", "train_acc", "val_accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_acc, epoch)
            tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"], epoch)

        # 判断当前验证集的准确率是否是最大的，如果是，则更新之前保存的权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, "AlexNet.pth")) 



if __name__ == '__main__':         
    main()

  