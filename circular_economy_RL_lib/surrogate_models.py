import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

apap_path = "best_model_apap.h5"
pap_path = "best_model_pap.h5"
hyd_path = "best_model_hyd.h5"

class PyTorchMLPSurrogate(nn.Module):
    """
    Recreates Keras Dense/BatchNormalization MLP architectures as native PyTorch modules
    to execute forward passes without TensorFlow graph overhead.
    """
    def __init__(self, keras_model):
        super(PyTorchMLPSurrogate, self).__init__()
        self.layers = nn.ModuleList()
        for layer in keras_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                w, b = layer.get_weights()
                in_features, out_features = w.shape
                linear = nn.Linear(in_features, out_features)
                linear.weight.data = torch.tensor(w.T, dtype=torch.float32)
                linear.bias.data = torch.tensor(b, dtype=torch.float32)
                self.layers.append(linear)
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                gamma, beta, mean, variance = layer.get_weights()
                bn = nn.BatchNorm1d(len(gamma), eps=1e-5)
                bn.weight.data = torch.tensor(gamma, dtype=torch.float32)
                bn.bias.data = torch.tensor(beta, dtype=torch.float32)
                bn.running_mean.data = torch.tensor(mean, dtype=torch.float32)
                bn.running_var.data = torch.tensor(variance, dtype=torch.float32)
                self.layers.append(bn)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                if layer != self.layers[-1]:
                    x = F.relu(layer(x))
                else:
                    x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                if x.shape[0] == 1:
                    layer.eval()
                x = layer(x)
        return x

class SurrogateModel:
    def __init__(self):
        # Load once using Keras
        apap_keras = tf.keras.models.load_model(apap_path, compile=False)
        pap_keras = tf.keras.models.load_model(pap_path, compile=False)
        hyd_keras = tf.keras.models.load_model(hyd_path, compile=False)

        # Convert to native PyTorch
        self.apap_model = PyTorchMLPSurrogate(apap_keras)
        self.pap_model = PyTorchMLPSurrogate(pap_keras)
        self.hyd_model = PyTorchMLPSurrogate(hyd_keras)

        # Flush Keras graph and clear TensorFlow memory
        tf.keras.backend.clear_session()
        print("--- All Surrogate Models Successfully Ported to PyTorch ---")

    @torch.no_grad()
    def get_apap_model_outputs(self, apap_input):
        return self.apap_model(apap_input).numpy()

    @torch.no_grad()
    def get_pap_model_outputs(self, pap_input):
        return self.pap_model(pap_input).numpy()

    @torch.no_grad()
    def get_hyd_model_outputs(self, hyd_input):
        return self.hyd_model(hyd_input).numpy()
