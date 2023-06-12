import pandas as pd
import os
import shutil

_q2 = pd.read_excel("/send/fdgClass/pp/sevJoint_0514_class.xlsx", engine ='openpyxl')
path = '/send/fdgClass/pp/SevJoint_0514'
for  index, row in _q2.iterrows():
    try: 
        fname = os.path.join(path, row['imageID']+".jpg")
        newPath = os.path.join(path, "class", str(row["4class"]), row['imageID']+".jpg")
        # print(fname, index)
        shutil.copyfile(fname,newPath)
    except Exception as ex: # Type of the error
        print('An error has occured', ex)