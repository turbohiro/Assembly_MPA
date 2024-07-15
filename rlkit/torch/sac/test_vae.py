import torch
import unittest
from vanilla_vae import VanillaVAE
from torchsummary import summary

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from experiment import VAEXperiment



class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        # self.model2 = VAE(3, 10)
        with open('vae.yaml', 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        #config = yaml.safe_load('vae.yaml')
        model = VanillaVAE(**config['model_params'])
        #self.model.build_layers()    
        PATH = '/homeL/wchen/Assembly_RPL/rlkit/rlkit/torch/sac/VanillaVAE/channel1/checkpoints/last.ckpt'
        checkpoint = torch.load(PATH)
       
        experiment = VAEXperiment(model,config['exp_params'])
        self.vae = experiment.load_from_checkpoint(PATH,vae_model = model, params=config['exp_params'])
        self.vae.eval()

        
        
    def test_summary(self):
        print(summary(self.model, (1, 64, 64), device='cpu'))
        # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 1, 64, 64)

        result,input,mu,std = self.vae(x)
        feature = torch.cat([mu, std],axis=-1)
        cuda = torch.cuda.is_available()
        f = feature.cpu() if cuda else feature
        f = f.data.view(x.shape[0], -1).numpy().tolist()

        #features.extend(f)
        import pdb
        pdb.set_trace()
       
       

        #y = self.experiment(x)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())


if __name__ == '__main__':
    unittest.main()