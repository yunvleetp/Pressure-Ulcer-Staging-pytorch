# from torchvision import datasets
# import torchvision.transforms as transforms
import os
import monai, torch, time, tqdm
from pygments import highlight
from torch.autograd import Variable
import wandb
import argparse
import os
import torch
import pandas as pd
from pathlib import Path
import numpy as np
import monai.transforms as T
from monai.data import Dataset, DataLoader, PILReader
import pdb

def getDataDict(dataPath, fold, lbl=1, test=False, aug=False, low=False, high=False, class7=False):
    dPath = Path(dataPath)
    if lbl:
        labTable = pd.read_excel("/send/fdgClass/pp/sevJoint_0903_class.xlsx",engine='openpyxl')

#     elif lbl==0: # lbl=0 if using label based on the clinical diagnosis
#         labTable = pd.read_csv()
    # kmeans label 말고 AD/CN/MCI label로 교체66

    dictList = list()
    if test:
        if low: testSbj = list(labTable[(labTable.l_fold == fold) & (labTable["4class"] == 0)].imageID)
        elif high: testSbj = list(labTable[(labTable.h_fold == fold) & (labTable["4class"] ==1)].imageID)
        elif class7: testSbj = list(labTable[labTable["7fold"] == fold].imageID)
        else: testSbj = list(labTable[labTable.fold == fold].imageID)
        for sbj in testSbj:
            sPath = dPath / str(sbj + ".jpg")
            _dict = dict()
            _dict["img"] = sPath
            if low: _dict['lbl']=int(list(labTable[labTable.imageID == sbj]["Grade"])[0])-1
            elif high: _dict['lbl']=int(list(labTable[labTable.imageID == sbj]["Grade"])[0])-3
            elif class7: _dict["lbl"]=int(list(labTable[labTable.imageID == sbj]["7class"])[0])
            else: _dict["lbl"]=int(list(labTable[labTable.imageID == sbj]["4class"])[0])
            dictList.append(_dict)
        
    else:
        if low: trainSbj = list(labTable[(labTable.l_fold != fold) & (labTable["4class"] == 0)].imageID)
        elif high: trainSbj = list(labTable[(labTable.h_fold != fold) & (labTable["4class"] ==1)].imageID)
        elif class7: trainSbj = list(labTable[labTable["7fold"] != fold].imageID)
        else: trainSbj = list(labTable[labTable.fold != fold].imageID)
        for sbj in trainSbj:
            sPath = dPath / str(sbj + ".jpg")
            _dict = dict()
            _dict["img"] = sPath
            if low: _dict['lbl']=int(list(labTable[labTable.imageID == sbj]["Grade"])[0])-1
            elif high: _dict['lbl']=int(list(labTable[labTable.imageID == sbj]["Grade"])[0])-3
            elif class7: _dict["lbl"]=int(list(labTable[labTable.imageID == sbj]["7class"])[0])
            else: _dict["lbl"]=int(list(labTable[labTable.imageID == sbj]["4class"])[0])
            dictList.append(_dict)
    
    return dictList
    
def loadData(dataroot, batch_size, fold, size, aug=False, test=False, low=False, high=False, class7=False):
    dataDict = getDataDict(dataroot, fold, aug= aug, test=test, low=low, high=high, class7=class7)
    if not test:
        if aug:
            transform = T.Compose(
                [
                    T.LoadImaged(keys=['img']),
                    T.EnsureChannelFirstd(keys=['img']),
                    T.Resized(['img'],[size, size]),
                    T.ScaleIntensityd(keys=['img']),
                    # T.AddChanneld(keys=['img']),
                    # # T.Resized(['img'],256),
                    # T.ScaleIntensityd(keys=['img']),
                    # T.Orientationd(keys=["fdgx", "syn"], axcodes="RAS"),
                    T.RandAffined(keys=['img'], prob=0.5, translate_range=10), 
                    T.RandRotated(keys=['img'], prob=0.5, range_x=30.0),
                    T.RandGaussianNoised(keys=['img'], prob=0.5),
                    T.EnsureTyped(keys=['img']),
                ]
            )
        else: 
            transform = T.Compose(
                [
                    T.LoadImaged(keys=['img']),
                    T.EnsureChannelFirstd(keys=['img']),
                    T.Resized(['img'],[size, size]),
                    T.ScaleIntensityd(keys=['img']),
                    T.EnsureTyped(keys=['img']),
                ]
            )
        
    else: 
        transform = T.Compose(
            [
                T.LoadImaged(keys=['img']),
                T.EnsureChannelFirstd(keys=['img']),
                T.Resized(['img'],[size, size]),
                T.ScaleIntensityd(keys=['img']),
                T.EnsureTyped(keys=['img']),
            ]
        )

    
    dataset= Dataset(  # use imageDataSet or DataSet need to look at difference
        data=dataDict,
        transform=transform,
    )

    dataloader= DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=40,
        drop_last = False
        
    )

    return dataloader
class classOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--dataroot', required=True, help='path to images')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--depthSize', type=int, default=128, help='depth for 3d images')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output classes')
        self.parser.add_argument('--aug', action='store_true', help='data augmentation')
        self.parser.add_argument('--class7', action='store_true', help='low grade binary classifcation')
        self.parser.add_argument('--low', action='store_true', help='low grade binary classifcation')
        self.parser.add_argument('--high', action='store_true', help='high grade binary classification')
        self.parser.add_argument('--model', type=str, help='what network to use')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--no_wandb', action='store_true', help='no wandb')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--fold', type=int, default=0, help='K-fold validation fold number')
        
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        self.parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        self.parser.add_argument('--val_interval', type=int, default=1, help='validation loop frequency')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')


        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt


# path2data = '/send/fdgClass/pp/SevJoint_0514'

# # if not exists the path, make the path
# if not os.path.exists(path2data):
#     os.mkdir(path2data)

opt = classOptions().parse()
config_exclude_key = [
        'gpu_ids','checkpoint_dir','no_wandb'
    ]
if not opt.no_wandb:
    wandb.init(
        project = "amulda",
        entity = "yunvleetp",
        name = opt.name,
        config = opt,
        config_exclude_keys = config_exclude_key,
        reinit=True,
        dir = opt.checkpoints_dir
    )

if opt.model =="dense121":
        model = monai.netwrks.nets.DenseNet121(spatial_dims=2, in_channels=opt.input_nc, out_channels=opt.output_nc, pretrained=True).to(opt.gpu_ids[0])
if opt.model =="SEResNext101":
    model = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=opt.input_nc, num_classes=opt.output_nc, pretrained=True).to(opt.gpu_ids[0])
if opt.model =="AHNet":
    model = monai.networks.nets.AHnet(in_channels=opt.input_nc, out_channels=opt.output_nc,pretrained=True).to(opt.gpu_ids[0])
if opt.model =="viT":
    model = monai.networks.nets.ViT(in_channels=opt.input_nc, img_size=(128,128,128), patch_size=(16,16,16), pos_embed='conv', classification=True, num_classes=3).to(opt.gpu_ids[0])

model = torch.nn.DataParallel(model, opt.gpu_ids)

Tensor = torch.cuda.FloatTensor
inputTensor = Tensor(opt.batchSize, opt.input_nc, opt.depthSize, opt.depthSize)
labels = torch.cuda.LongTensor(opt.batchSize)

# pdb.set_trace()
train_data = loadData(opt.dataroot, opt.batchSize, opt.fold, opt.depthSize, aug = opt.aug, low = opt.low, high = opt.high, class7 = opt.class7)
val_data = loadData(opt.dataroot, opt.batchSize, opt.fold, opt.depthSize, aug = opt.aug, test=True, low=opt.low, high=opt.high, class7=opt.class7)

# start a typical PyTorch training
optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))
best_metric1 = -1
best_metric2 = 1000
best_metric_epoch1 = -1
best_metric_epoch2 = -1
max_epochs = 5
criterionCE = torch.nn.CrossEntropyLoss()

for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay+1 ):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    print("-" * 10)
    print(f"epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay+1}")
    model.train()
    train_epoch_loss = 0
    metric_count = 0
    num_correct=0
    step = 0
    
    # pdb.set_trace()
    for i, batch_data in tqdm.tqdm(enumerate(train_data)):
        iter_start_time = time.time()
        step += 1
        inputTensor = batch_data['img']
        # if inputTensor.shape != (opt.batchSize,3, opt.depthSize,opt.depthSize): 
        #     print('fuck')
        #     continue
        labels = Variable(batch_data['lbl'].type(torch.cuda.LongTensor))
        optimizer.zero_grad()
        outputs = model(inputTensor) #squeeze?
        # pdb.set_trace()
        loss = criterionCE(outputs, labels)
        loss.backward()
        optimizer.step()
        train_epoch_loss += loss.detach()
        # pdb.set_trace()

        value = torch.eq(outputs.argmax(dim=1), labels)
        metric_count += len(value)
        num_correct += value.sum().detach()
    
    train_epoch_loss /= step
    train_epoch_acc = num_correct / metric_count
    print(f"epoch {epoch} average loss: {train_epoch_loss:.4f} current accuracy: {train_epoch_acc:.4f}")
   
    # pdb.set_trace()
    if epoch % opt.val_interval == 0:
        model.eval()

        num_correct = 0.0
        metric_count = 0
        epoch_loss = 0 
        step = 0 
        for i, batch_data in tqdm.tqdm(enumerate(val_data)):
            # pdb.set_trace()
            step += 1
            inputTensor = batch_data['img']
            # if inputTensor.shape != (opt.batchSize,3, opt.depthSize,opt.depthSize): 
            #     print('fuck')
            #     continue
            labels = Variable(batch_data['lbl'].type(torch.cuda.LongTensor))
            with torch.no_grad():
                val_outputs = model(inputTensor)
                loss = criterionCE(val_outputs, labels)
                epoch_loss += loss.detach()
                value = torch.eq(val_outputs.argmax(dim=1), labels)
                metric_count += len(value)
                num_correct += value.sum().detach()
        epoch_loss /= step
        metric1 = num_correct / metric_count
        metric2 = epoch_loss

        if metric1 > best_metric1:
            best_metric1 = metric1
            best_metric_epoch = epoch + 1
            save_dir = os.path.join(opt.checkpoints_dir, opt.name)
            torch.save(model.state_dict(), os.path.join(save_dir,f"{opt.name}_acc.pth"))
            print("saved new best metric model")
        if metric2 < best_metric2:
                best_metric2 = metric2
                best_metric_epoch2 = epoch
                save_dir = os.path.join(opt.checkpoints_dir, opt.name)
                torch.save(model.state_dict(), os.path.join(save_dir,f"{opt.name}_loss.pth"))
                print("saved new best metric model")
        print(f"Current epoch: {epoch}  average loss: {epoch_loss:.4f} current accuracy: {metric1:.4f} ")
        print(f"Best accuracy: {best_metric1:.4f} at epoch {best_metric_epoch1}")
        print(f"Best loss: {best_metric2:.4f} at epoch {best_metric_epoch2}")
        if not opt.no_wandb:
            wandb.log(
                {
                "training_loss": train_epoch_loss,
                "training_acc": train_epoch_acc, 
                "validation_loss": epoch_loss,
                "validation_acc": metric1,
                "best_acc": best_metric1,
                "best_loss": best_metric2
                }
            )
    print(f"Training completed, best_acc: {best_metric1:.4f} at epoch: {best_metric_epoch1}, best_acc: {best_metric2:.4f} at epoch: {best_metric_epoch2}")