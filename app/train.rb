require 'bundler'
Bundler.require

include PyCall::Import

pyimport 'numpy', as: :np
pyimport 'matplotlib', as: :mp
pyimport 'matplotlib.pyplot', as: :plt

require './lib/loader'
require './lib/util'
require './lib/network'

limit = 100
x_train, t_train, x_test, t_test = Loader.load_mnist(true, limit)

iters_num = 500
train_size = x_train.shape[0]
batch_size = 10 
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = [train_size / batch_size, 1].max

network = Network.new(784, 50, 10)

iters_num.times do |i|
  batch_mask = np.random.choice.(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  # grads = network.numerical_gradient(x_batch, t_batch)
  grads = network.gradient(x_batch, t_batch)

  network.params.keys.each do |key|
    network.params[key] -= learning_rate * grads[key]
  end

  loss = network.loss(x_batch, t_batch)
  train_loss_list << loss

  if i % iter_per_epoch == 0
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list << train_acc
    test_acc_list << test_acc
    p "train acc, test acc | #{train_acc}, #{test_acc}"
  end
end

# x = np.arange.(train_loss_list.size)
# plt.plot.(x, np.array.(train_loss_list))

x = np.arange.(train_acc_list.size)
plt.plot.(x, train_acc_list)
plt.plot.(x, test_acc_list, '--')
plt.xlabel.('epochs')
plt.ylabel.('accuracy')
plt.ylim.(0, 1.0)
plt.legend.('lower right')
plt.show.()
