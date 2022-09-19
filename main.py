#!/usr/bin/env python
# coding: utf-8

# In[1]:

import jsonify
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import os
from random import randint
import uuid
import torchvision
import torch
from PIL import Image
# In[ ]:
class_name_ = ['pizza','steak','suchi']


effnet_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
eff_transforms = effnet_weights.transforms()
path = r'C:\Users\bleym\Desktop\Dart Practice\ModelFastAPI\effnetb2_model.pth'
eff = torch.load(path)

def predict(img,model,class_name,transforms):
    img = transforms(img).unsqueeze(0)
    pred = torch.softmax(model(img), dim =1)
    predmax = torch.argmax(pred,dim =1)
    return class_name[predmax]

# In[4]:

imgdir = r'C:\Users\bleym\Desktop\Dart Practice\calorifythis\uploadedimages'

app = FastAPI()

db = []

@app.post("/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!
    #filepath = imgdir+'\'+ file.filename
    filepath = f'{imgdir}\\{file.filename}'

    with open(filepath, 'wb' ) as f:
        f.write(contents)
    img = Image.open(filepath)

    prediction = predict(
                    img = img,
                    model = eff,
                    transforms = eff_transforms,
                    class_name = class_name_)

    return prediction


@app.get("/")
async def index():
    return {'hello': 'world'}

# In[ ]:





# In[ ]:
