from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from PIL import Image
import torch.nn as nn




my_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

num_classes=2
learning_rate=0.0001
batch_size=4
num_epochs=5


#device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Cuda Availability:',torch.cuda.is_available())

class Wood_Plastic_Dataset(Dataset):
        
    def __init__ (self, csv_file,root_dir,transform=None):
    
        self.DataFrame = pd.read_csv(csv_file)
        
        self.root=root_dir
        
        self.transform= transform
    
    
    def __len__(self):
        
        return(len(self.DataFrame ))
    
    def __getitem__(self,index):
        
        image_path= os.path.join(self.root, self.DataFrame.iloc[index,0 ])
        
        img= Image.open(image_path)
        
        img=np.asarray(img)
        
        image=Image.fromarray(img)
        
    
        img_label=torch.tensor(self.DataFrame.iloc[index,1])
        
        
        
        
        # print('The image shape is ',image.shape)
        
        # print('The image array type is ,',type(image))
        
        if self.transform:
            image=self.transform(image)
        
        img_sample=(image,img_label)
            
            # print('The image new array type is ,',type(img_sample[0]))
            
        return img_sample  
    
# class ToTensor:
         
#     def __call__(self,img_sample):
            
#         img,Label= img_sample 
        
#         return torch.from_numpy(img), torch.tensor(Label)

    
    

            
dataset=Wood_Plastic_Dataset(os.path.join(os.path.dirname(__file__), 'wood&plasticNew.csv'),os.path.join(os.path.dirname(__file__), 'Wood_Plastic'),transform= my_transform)



sample= dataset[1]

print(sample)
print('my', sample[0].shape)

image=sample[0]
print(image[0].mean())


plt.imshow(sample[0].reshape(227,227,3))



train_dataset,test_dataset=torch.utils.data.random_split(dataset=dataset, lengths=[800,200])

train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle = True )  

test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle = True )    

example=iter(train_dataloader)

print(len(train_dataloader))

print(len(test_dataloader))


for i in example.next():
    
    print(i.shape)
    
classes=('wood','plasticBags')



class Alexnet(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.convolutional =  nn.Sequential(
        
        nn.Conv2d(in_channels=3,out_channels=96, kernel_size=11, stride=4 ),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3,stride=2),
        
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,padding=2),
        nn.ReLU(inplace= True),
        nn.MaxPool2d(kernel_size=3,stride=2),
    
        nn.Conv2d(256,384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        nn.MaxPool2d(kernel_size=3, stride=2),
        
        )
        
        self.averagepooling=nn.AdaptiveAvgPool2d((6,6))
        
        self.FullyConnected=nn.Sequential(
            
        nn.Dropout(),
        nn.Linear(256*6*6,4096),
        nn.ReLU(),
            
        nn.Dropout(),
        nn.Linear(4096,4096),
        nn.ReLU(),
        
        nn.Linear(4096,num_classes),
        )
        
    def forward(self, x ):
            
        x=self.convolutional(x)
            
        x=self.averagepooling(x)
            
        x=torch.flatten(x,1)

        x=self.FullyConnected(x)
            
        return x
        
model=Alexnet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer= torch.optim.Adam(model.parameters(),lr=learning_rate)

num_total_steps=len(train_dataloader)



# ----------Training Loop(uncommenet to train)---------------

for epoch in range(num_epochs):
    
     for i,(images,labels)in enumerate(train_dataloader):
        
         images=images.to(device)
    
         labels=labels.to(device)
        
         output=model(images)
        
         loss=criterion(output,labels)
       
         optimizer.zero_grad ()
        
         loss.backward()
         
         optimizer.step()
        
        
         if (i+1)%2==0:
             print(f'epoch:{epoch+1}/{num_epochs},step:{i+1}/{num_total_steps},loss:{loss.item()}')
            
        
#----------Testing---------


correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' %(100 * correct / total))