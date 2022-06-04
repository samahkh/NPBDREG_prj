# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:22:39 2020

@author: ssamahkh
This callback saves model weights and loss after each iteration 
"""
import keras


class History_LAW(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch = []
        self.weights = []
        self.history = {}
        

    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
    
        '''modelWeights = []
        for layer in self.model.layers:
            layerWeights = []
            for weight in layer.get_weights():
                layerWeights.append(weight)
            modelWeights.append(layerWeights)
        #self.weights.append(modelWeights)'''
        Weights = self.model.get_weights()
        self.weights.append(Weights)