#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from itertools import product
import time
import pandas as pd
from collections import OrderedDict
from collections import namedtuple
from IPython.display import clear_output
import json
from skimage.transform import resize

class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = [] # keep track of parameter values and the result of each epoch
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
    def begin_run(self, run,network,image_size):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.tb = SummaryWriter(comment=f'-{run}')
        
        #self.loader = loader
        #self.batch_size = batch_size
        self.image_size = image_size
        
        #images,angle,torque,speed = self.loader
        #if len(images.shape) == 5:
            #image = []
            #image1 = []
            #for t in range(images.shape[0]):
                #for j in range(images.shape[1]):
                    #image.append(images[t,j])
                #image_stack = torch.stack(image,dim=0)
                #image.clear()
                #gridimage = torchvision.utils.make_grid(image_stack)
                #image1.append(gridimage)
            #imageall = torch.stack(image1,dim=0)
            #grid = torchvision.utils.make_grid(imageall).cuda()
            #images = images.permute(0,2,1,3,4).cuda()
        #self.tb.add_image('Images',grid)
        #self.tb.add_graph(self.network,images)
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time()-self.run_start_time
        
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        
        self.tb.add_scalar('Loss',loss,self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        
        for name,param in self.network.named_parameters():
            self.tb.add_histogram( name,param, self.epoch_count)
            self.tb.add_histogram( f'{name}.grad',param.grad, self.epoch_count)
            
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration
        
        for k,v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data,orient='columns')
        clear_output(wait=True)
        display(df)
        
    def track_loss(self,loss):
        self.epoch_loss +=loss.item() * self.loader.batch_size
        
    def track_num_correct(self,preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds,labels)
        
    @torch.no_grad()
    def _get_num_correct(self,preds,labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient = 'columns'
            
        ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json','w',encoading = 'utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii = False, indent =4)