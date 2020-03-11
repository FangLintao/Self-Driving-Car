#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import Sampler
import random

class ConsecutiveBatchSampler(Sampler):
    
    def __init__(self, data_source, batch_size, seq_len, drop_last=False, shuffle=True, use_all_frames=False):
        r""" Sampler to generate consecutive Batches
        
        Args:
            data_source: Source of data
            batch_size: Size of batch
            seq_len: Number of frames in each sequence (used for context for prediction)
            drop: Wether to drop the last incomplete batch
            shuffle: Wether to shuffle the data
        Return:
            List of iterators, size: [batch_size x seq_len x n_channels x height x width]
        """
        super(ConsecutiveBatchSampler, self).__init__(data_source)
        
        self.data_source = data_source
        
        assert seq_len >= 1, "Invalid batch size: {}".format(seq_len)
        self.seq_len = seq_len
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.use_all_frames_ = use_all_frames
    
    def __iter__(self):
        
        data_size = len(self.data_source)
        
        if self.use_all_frames_:
            start_indices = list(range(data_size))
        else:
            start_indices = list(range(1, data_size, self.seq_len))
            
        if self.shuffle:
            random.shuffle(start_indices)
        
        batch = []
        for idx, ind in enumerate(start_indices):
            if data_size - idx < self.batch_size and self.drop_last: # if last batch
                break
                
            seq = []
            if ind + 1 < self.seq_len:
                seq.extend([0]*(self.seq_len - ind - 1) + list(range(0, ind+1)))
            else:
                seq.extend(list(range(ind-self.seq_len+1, ind+1)))
            
            batch.append(seq)
            
            if len(batch) == self.batch_size or idx == data_size - 1:
                yield batch
                batch = []

    
    def __len__(self):
        length = len(self.data_source)
        batch_size = self.batch_size
        
        if length % batch_size == 0 or self.drop_last:
            return length // batch_size
        
        return length // batch_size + 1

