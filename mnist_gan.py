from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Reshape, Conv2DTranspose
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand

(train_data, _), (_,_) = tf.keras.datasets.mnist.load_data()
train_data.reshape(len(train_data), 28, 28, 1)
train_data = train_data/255 # Formats it in [0,1]
sample_dimensions=100

batch=256
buffer=70000
def format(sample, n): return sample.reshape(n,28,28,1)

def noise(n):
  noise=np.random.randn(n*sample_dimensions).reshape(n, sample_dimensions) # create noise in [-1,1] shape into n chunks of 100 values
  return noise

def gen_real_samples(n):
  X = []
  for i in range(n):
    X.append(np.array(train_data[random.randint(0, train_data.shape[0]-1)])) # fetch real data and append to our collection
  X = np.expand_dims(X, axis=1)
  y = np.ones((n, 1)) # mimick labelling from discriminator -- "positive match"
  return X,y

def gen_fake_samples(n,g):
  X=g.predict(noise(n)) # create feedback loop -- generator trains based on discriminator, and vice versa
  y = np.zeros((n, 1)) # not real data, label it as such
  return X,y

def make_discriminator(in_shape=(28,28,1)):
  model = Sequential()
  model.add(Conv2D(64, 3, strides=2, padding='same', input_shape=in_shape)) # 64 filters, kernel 3, padding -- fills out empty space with noise to meet shape reqs
  model.add(LeakyReLU(alpha=0.2)) # Leaky ReLU -- Best practice, alpha -- trial + error tuning
  model.add(Dropout(0.3))
  model.add(Conv2D(64, 3, strides=2, padding='same')) # breaking down the image
  model.add(LeakyReLU(alpha=0.3)) # Leaky ReLU again, more tuning
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid')) # data must be between [0,1]
  opt = Adam(lr=0.00021, beta_1=0.5) # adam optimizer -- research this more
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) # binary crossentropy -- to classify between 2 choices based on accuracy of comparison to samples
  return model

def make_generator():
  model = Sequential()
  n_nodes = 128 * 7 * 7 # define format -- 128 renditions of 7x7 pixel "art"
  model.add(Dense(n_nodes, input_dim=sample_dimensions)) # Dense layer with formatted shape, accepts input noise
  model.add(LeakyReLU(alpha=0.2)) # Welcome back, Leaky ReLU!
  model.add(Reshape((7, 7, 128))) # Reformat data -- prepares for merging of many images
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # Increase resolution to 14x14 ("Upsample") -- 7x7 * 2x2(strides) = 14x14 pixel image
  model.add(LeakyReLU(alpha=0.2)) # More Leaky ReLU yayyyy
  model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # stride to 28x28 -- kernel dictates height and width of convo window -- research more
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same')) # output layer -- in [0,1], kicks up kernel size and completes
  # no compilation -- model is not directly updating itself
  return model

def make_gan(d, g):
  d.trainable = False # we do not want to update the discriminator on potentially false values -- it is already trained
  model=Sequential()
  model.add(g)
  model.add(d) # add both models to the GAN
  opt = Adam(lr=0.0002, beta_1=0.5) # trial + error + best practices settings
  model.compile(loss='binary_crossentropy', optimizer=opt) # identifying accuracy -- proportional loss function
  return model

discriminator=make_discriminator()
generator=make_generator()
gan=make_gan(discriminator, generator)

def train(g, d, gan, e=100, batches=256): # combo model
  bat_per_epo = int(train_data.shape[0] / batches)
  half_batch = int(batches / 2)
  for i in range(e):
    for j in range(bat_per_epo):
      X_real, y_real = gen_real_samples(half_batch) # generate large samples of real and fake data
      X_fake, y_fake = gen_fake_samples(half_batch, g) # with corresponding tags
      X_fake = format(X_fake, half_batch)
      X_real = format(X_real, half_batch) # format them
      X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake)) # vertically stack data for training
      d_loss, _ = d.train_on_batch(X, y) # improve discriminator model
      X_gan = noise(batches) # generator input
      y_gan = np.ones((batches, 1)) # fake tag to trick discriminator
      g_loss = gan.train_on_batch(X_gan, y_gan) # train the GAN, since discriminator is locked only generator will update
      print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss)) # updates

def generate_img(g):  
  plt.imshow((g.predict(noise(1))*255).reshape(28,28))

generate_img(generator)