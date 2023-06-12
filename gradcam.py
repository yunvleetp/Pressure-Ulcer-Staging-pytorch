from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50
import os
import monai, torch
import monai.transforms as T
import argparse
from monai.data import Dataset, DataLoader, PILReader
from PIL import Image
from pathlib import Path
import pandas as pd
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import tqdm
import numpy as np
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.environ["CUDA_VISIBLE_DEVICES"] ="0"

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
            _dict["id"] = sbj
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
            _dict["id"] = sbj
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

def load_network(network, log_dir, network_label, epoch_label):
        save_filename = '%s_%s.pth' % (network_label,epoch_label)
        save_path = os.path.join(log_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
for fold in range(10):
    opt = argparse.Namespace(gpu_ids=[0], dataroot = "/send/fdgClass/pp/sevJoint", checkpoints_dir= \
                                        "/send/fdgClass/pp/log", input_nc=3 , output_nc = 7, name =f"7class_{fold}_balanced" ,fold = fold , batchSize=24, \
                                        model = 'SEResNext101', class7= True, depthSize=512, low= False, high=False)


    model = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=opt.input_nc, num_classes=opt.output_nc, pretrained=True).to(opt.gpu_ids[0])
    # print(model)
    model = torch.nn.DataParallel(model, opt.gpu_ids)
    print(model.module)
    log_dir = os.path.join(opt.checkpoints_dir, opt.name)
    load_network(model, log_dir, opt.name, 'acc')
    # print(model.module.layer4[-1])
    target_layers = [model.module.layer4[-1]]

    val_data = loadData(opt.dataroot, opt.batchSize, opt.fold, opt.depthSize, test=True, low=opt.low, high=opt.high, class7=opt.class7)
    for i, batch in tqdm.tqdm(enumerate(val_data)):
        input_tensor= batch["img"]
        targets= [ClassifierOutputTarget(x) for x in batch["lbl"]]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        # print(input_tensor.shape)
        # input_tensor = torch.permute(input_tensor, (0,2,3,1))
        # print(input_tensor.shape)
        grayscale_cam = cam(input_tensor=input_tensor, targets = targets)
        sbj = batch["id"]
        preds = model(input_tensor)
        preds = torch.argmax(preds.detach().cpu(), axis=1)
        # print(preds.shape, batch["lbl"].shape)
        mask = (preds == batch["lbl"])
        lbl = batch["lbl"]
        # print(mask)


        

        for i in range(grayscale_cam.shape[0]):
            gray = grayscale_cam[i,:]
            vis = show_cam_on_image(input_tensor[i], gray, use_rgb=True)
            if mask[i] == True:
                im = Image.fromarray(vis)
                im.save(f"/send/fdgClass/pp/gradcam/right/{lbl[i]}/{sbj[i]}_gradCAM.jpg")
                # print(input_tensor[i].shape, vis.shape)
                origin = np.transpose(np.array(input_tensor[i]), (1,2,0))
                origin = Image.fromarray(np.uint8(255 * origin))
                origin.save(f"/send/fdgClass/pp/gradcam/right/{lbl[i]}/{sbj[i]}.jpg")
            if mask[i] == False:
                im = Image.fromarray(vis)
                im.save(f"/send/fdgClass/pp/gradcam/wrong/{lbl[i]}/{sbj[i]}_gradCAM.jpg")
                # print(input_tensor[i].shape, vis.shape)
                origin = np.transpose(np.array(input_tensor[i]), (1,2,0))
                origin = Image.fromarray(np.uint8(255 * origin))
                origin.save(f"/send/fdgClass/pp/gradcam/wrong/{lbl[i]}/{sbj[i]}.jpg")
