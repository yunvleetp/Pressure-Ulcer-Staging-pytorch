import pandas as pd
from pathlib import Path
import numpy as np
import monai.transforms as T
from monai.data import Dataset, DataLoader


def getDataDict(dataPath, fold, lbl=1, test=False, aug=False):
    dPath = Path(dataPath)
    if lbl:
        labTable = pd.read_excel("/send/fdgClass/pp/sevJoint_0514_class.xslx",engine='openpyxl')

#     elif lbl==0: # lbl=0 if using label based on the clinical diagnosis
#         labTable = pd.read_csv()
    # kmeans label 말고 AD/CN/MCI label로 교체66

    dictList = list()
    if test:
        testSbj = list(labTable[labTable.fold == fold].iamgeID)
        for sbj in testSbj:
            sPath = dPath / sbj
            _dict = dict()
            _dict["img"] = sPath
            _dict["lbl"]=int(np.array(labTable[labTable.imageID == sbj]["4class"])[0])
            dictList.append(_dict)
        
    else:
        trainSbj = list(labTable[labTable.fold != fold].subject)
        for sbj in trainSbj:
            sPath = dPath / sbj
            _dict = dict()
            _dict["img"] = sPath
            _dict["lbl"]=int(np.array(labTable[labTable.imageID == sbj]["4class"])[0])
            dictList.append(_dict)
    
def loadData(dataroot, batch_size, fold, size, aug=False, test=False):
    dataDict = getDataDict(dataroot, fold, aug= aug, test=test)
    if not test:
        if aug:
            transform = T.Compose(
                [
                    T.LoadImaged(keys=['img']),
                    T.AddChanneld(keys=['img']),
                    T.Resized([size, size]),
                    T.ScaleIntensityd(keys=['img']),
                    # T.Orientationd(keys=["fdgx", "syn"], axcodes="RAS"),
                    T.RandAffined(keys=['img'], prob=0.5, translate_range=10), 
                    T.RandRotated(keys=['img'], prob=0.5, range_x=10.0),
                    T.RandGaussianNoised(keys=['img'], prob=0.5),
                    T.EnsureTyped(keys=['img']),
                ]
            )
        else: 
            transform = T.Compose(
                [
                    T.LoadImaged(keys=['img']),
                    T.Resized([size, size]),
                    T.ScaleIntensityd(keys=['img']),
                    T.AddChanneld(keys=['img']),
                    T.EnsureTyped(keys=['img']),
                ]
            )
        
    else: 
        transform = T.Compose(
            [
                T.LoadImaged(keys=['img']),
                T.Resized([size, size]),
                T.ScaleIntensityd(keys=['img']),
                T.AddChanneld(keys=['img']),
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
        drop_last = True
        
    )

    return dataloader