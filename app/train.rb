require './lib'

require 'benchmark'

pyimport 'numpy', as: :np
pyimport 'matplotlib.pyplot', as: :plt

limit = ARGV[0].nil? ? 100 : ARGV[0].to_i
iters_num = ARGV[1].to_i.nonzero? || 500
batch_size = ARGV[2].to_i.nonzero? || 10
path = ARGV[3] || 'tmp/params.json'

train_acc_list = []
test_acc_list = []
result = Benchmark.measure {
  _, train_acc_list, test_acc_list, params = Trainer.train_mnist(limit, iters_num, batch_size)
  Trainer.save_params(path, params)
}
puts Benchmark::CAPTION
puts result

x = np.arange.(train_acc_list.size)
plt.plot.(x, train_acc_list)
plt.plot.(x, test_acc_list, '--')
plt.xlabel.('epochs')
plt.ylabel.('accuracy')
plt.ylim.(0, 1.0)
plt.legend.('lower right')
plt.show.()
