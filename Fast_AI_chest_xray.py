

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from fastai import *
from fastai.vision import *


import os
import glob
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

path = Path('/kaggle/input/sample-images/sample-images/Scoring')
print(path)
path.ls()[:5]

path = Path('/kaggle/input/data')
print(path)
path.ls()

image_path = glob.glob('../input/data/images*/*/*.png') 
#image_path[:5]
print("No. of images: ",len(image_path))

df_data_info_orig = pd.read_csv(path/'Data_Entry_2017.csv')
df_data_info = pd.DataFrame()
df_data_info = df_data_info_orig[["Image Index", "Finding Labels"]]
print("#Image Labels: ",len(df_data_info))
df_data_info.head(5)

# creating "Image_Paths" dictionary with "Image Index" as Key and "image path" as values
Image_Paths = {os.path.basename(x): x for x in glob.glob(os.path.join('..', 'input','data', 'images*', '*', '*.png'))}
print("#Image_Paths: ",len(Image_Paths))
df_data_info['image_address'] = df_data_info['Image Index'].map(Image_Paths.get)
df_data_info.head(5)

df_test_list = pd.read_table(path/'test_list.txt')
df_train_val_list = pd.read_table(path/'train_val_list.txt')

image_data = df_data_info[['image_address', 'Finding Labels']]
#image_data = image_data[['image_address', 'processed_labels']].rename(columns = {"processed_labels" : "Labels"})
image_data["Labels"] = image_data['Finding Labels'].str.replace("|",",",regex=True).replace(";",",",regex=True).replace(":",",",regex=True)
image_data = image_data[['image_address', 'Labels']]
image_data.sample(5)

image_data_v2 = image_data.copy()
image_data_v2["Atelectasis"]=np.where(image_data_v2.Labels.str.contains("Atelectasis", flags=re.IGNORECASE),1,0)
image_data_v2["Consolidation"]=np.where(image_data_v2.Labels.str.contains("Consolidation", flags=re.IGNORECASE),1,0)
image_data_v2["Infiltration"]=np.where(image_data_v2.Labels.str.contains("Infiltration", flags=re.IGNORECASE),1,0)
image_data_v2["Pneumothorax"]=np.where(image_data_v2.Labels.str.contains("Pneumothorax", flags=re.IGNORECASE),1,0)
image_data_v2["Edema"]=np.where(image_data_v2.Labels.str.contains("Edema", flags=re.IGNORECASE),1,0)
image_data_v2["Emphysema"]=np.where(image_data_v2.Labels.str.contains("Emphysema", flags=re.IGNORECASE),1,0)
image_data_v2["Fibrosis"]=np.where(image_data_v2.Labels.str.contains("Fibrosis", flags=re.IGNORECASE),1,0)
image_data_v2["Effusion"]=np.where(image_data_v2.Labels.str.contains("Effusion", flags=re.IGNORECASE),1,0)
image_data_v2["Pneumonia"]=np.where(image_data_v2.Labels.str.contains("Pneumonia", flags=re.IGNORECASE),1,0)
image_data_v2["Pleural_Thickening"]=np.where(image_data_v2.Labels.str.contains("thickening", flags=re.IGNORECASE),1,0)
image_data_v2["Cardiomegaly"]=np.where(image_data_v2.Labels.str.contains("Cardiomegaly", flags=re.IGNORECASE),1,0)
image_data_v2["Nodule"]=np.where(image_data_v2.Labels.str.contains("Nodule", flags=re.IGNORECASE),1,0)
image_data_v2["Mass"]=np.where(image_data_v2.Labels.str.contains("Mass", flags=re.IGNORECASE),1,0)
image_data_v2["Hern"]=np.where(image_data_v2.Labels.str.contains("Hern", flags=re.IGNORECASE),1,0)
image_data_v2["No_Findings"]=np.where(image_data_v2.Labels.str.contains("finding", flags=re.IGNORECASE),1,0)
image_data_v2.head(5)

image_data_subset_1 = image_data_v2[(image_data_v2.Hern == 1) | (image_data_v2.Pneumonia == 1) | (image_data_v2.Fibrosis == 1) | (image_data_v2.Edema == 1) | (image_data_v2.Emphysema == 1) | (image_data_v2.Cardiomegaly == 1) | (image_data_v2.Pleural_Thickening == 1)].reset_index(drop = True)
print(len(image_data_subset_1))
image_data_subset_1 = image_data_subset_1.append(image_data_v2[(image_data_v2.Consolidation == 1) & 
                                                               (image_data_v2.image_address.isin(image_data_subset_1.image_address) == False)].sample(2200, random_state = 22),
                                                 ignore_index=True)
print(len(image_data_subset_1))
image_data_subset_1 = image_data_subset_1.append(image_data_v2[(image_data_v2.Pneumothorax == 1) & 
                                                               (image_data_v2.image_address.isin(image_data_subset_1.image_address) == False)].sample(1700, random_state = 17),
                                                 ignore_index=True)
print(len(image_data_subset_1))
image_data_subset_1 = image_data_subset_1.append(image_data_v2[(image_data_v2.Mass == 1) & 
                                                               (image_data_v2.image_address.isin(image_data_subset_1.image_address) == False)].sample(1500, random_state = 151),
                                                 ignore_index=True)
print(len(image_data_subset_1))
image_data_subset_1 = image_data_subset_1.append(image_data_v2[(image_data_v2.Nodule == 1) & 
                                                               (image_data_v2.image_address.isin(image_data_subset_1.image_address) == False)].sample(1500, random_state = 152),
                                                 ignore_index=True)
print(len(image_data_subset_1))
image_data_subset_1 = image_data_subset_1.append(image_data_v2[(image_data_v2.No_Findings == 1) & 
                                                               (image_data_v2.image_address.isin(image_data_subset_1.image_address) == False)].sample(3000, random_state = 30),
                                                 ignore_index=True)
print(len(image_data_subset_1))
image_data_subset_1.drop_duplicates(inplace=True)
image_data_subset_1.reset_index(drop = True)
print(len(image_data_subset_1))
image_data_subset_1.head(5)

image_data_subset_final = image_data_subset_1[['image_address','Labels']]
image_data_subset_final.head(5)

path = "."
images_list = ImageList.from_df(image_data_subset_final, path).split_by_rand_pct(0.3).label_from_df(label_delim=',')

images_list.classes

xtra_tfms = zoom_crop(scale=(1.05,1.19), do_rand=True, p = 1.0)
tfms = get_transforms(flip_vert=False, max_lighting=0.7, max_zoom=1.1, max_warp=0., xtra_tfms = xtra_tfms)

np.random.seed(42) # set random seed so we always get the same validation set
data = (images_list.transform(tfms, size=256)
        # Apply transforms and scale images to 128x128
        .databunch(bs=96).normalize(imagenet_stats)
        # Create databunch with batchsize=64 and normalize the images
)

data.classes

data.show_batch(rows=3, figsize=(12, 9))

# create metrics
acc_02 = partial(accuracy_thresh, thresh=0.2)
acc_03 = partial(accuracy_thresh, thresh=0.3)
acc_04 = partial(accuracy_thresh, thresh=0.4)
acc_05 = partial(accuracy_thresh, thresh=0.5)
f_score = partial(fbeta, thresh=0.2)
# create cnn with the resnet50 architecture
learn = cnn_learner(data, models.resnet50, metrics=[acc_02, acc_03, acc_04, acc_05, f_score])

learn.lr_find() # find learning rate

learn.recorder.plot() # plot learning rate

lr = 0.01 # chosen learning rate
learn.fit_one_cycle(5, lr) # train model for 3 epochs

learn.save('xray_g12_3_RN50_BS64_256_lr01_FT_zm12_stage1') # save model

learn.unfreeze()
learn.fit_one_cycle(8, max_lr = slice(1e-5, lr/5))

learn.save('xray_g12_3_RN50_BS64_256_lr01_FT_zm12_stage1') # save model

learn.export('/kaggle/working/export.pkl')

Path('/kaggle/input/hack-data-new/scoring2/Scoring2').ls()

test_path = Path('/kaggle/input/hack-data-new/scoring2/Scoring2/')
# print(test_path.ls())

test_data = ImageList.from_folder(test_path)
print('len(test_data): ', len(test_data))

learn = load_learner(Path('/kaggle/input/ara-hackathon-arch4/'), test=test_data)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)

# help(learn.get_preds)
print(learn.data.classes)
print(preds.shape)

names = [os.path.basename(test_data.items[i].name)  for i in range(len(test_data))]
# print('names: ', names)

pred_cls_temp = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hern','Infiltration','Mass','No_findings','Nodule','Pleural_thickening','Pneumonia','Pneumothorax']

res = pd.DataFrame()
res['ImageID'] = names
for i  in range(len(learn.data.classes)):
#     print(i)
#     res[learn.data.classes[i]] = preds[:, i]
    res[pred_cls_temp[i]] = preds[:, i]
print(res.head())
print(res.shape)

pred_cls =                      ['ImageID', 'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hern','Infiltration','Mass','No_findings','Nodule','Pleural_thickening','Pneumonia','Pneumothorax']
submit = pd.DataFrame(columns = ['ImageID', 'Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion','Pneumonia','Pleural_thickening','Cardiomegaly','Nodule','Mass','Hern','No_findings']) 

pred2sub = {}

for col in list(submit):
#     print(col)
    submit[col] = res[col]
    
# print(submit)
print(submit.shape)
submit.to_csv('submit3_152.csv', index = False)

from IPython.display import HTML
import base64


def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


create_download_link(submit)
