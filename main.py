import setGPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import inference.train_autoencoder as train

# train model
model = train.train()

# apply model






