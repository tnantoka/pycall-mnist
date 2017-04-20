require './lib'

require 'benchmark'

pyimport 'matplotlib.pyplot', as: :plt

limit = ARGV[0].nil? ? 100 : ARGV[0].to_i
iters_num = ARGV[1].to_i.nonzero? || 500
batch_size = ARGV[2].to_i.nonzero? || 10
path = ARGV[3] || 'tmp/params.json'

train_acc_list = []
test_acc_list = []

memprof = false
plot = true

network = if File.exist?(path)
            Predictor.new(path).network
          else
            Network.new
          end

Memprof2.start if memprof

result = Benchmark.measure do
  trainer = Trainer.new(network)
  _, train_acc_list, test_acc_list = trainer.train_mnist(limit, iters_num, batch_size, 0.1, plot)
  Trainer.save_params(path, network.params)
end
puts Benchmark::CAPTION
puts result

puts "train acc, test acc | #{train_acc_list.last}%, #{test_acc_list.last}%"

if memprof
  Memprof2.report(out: 'tmp/memprof2_report')
  Memprof2.stop
end

if plot
  x = NP.arange(train_acc_list.size)
  plt.plot.(x, train_acc_list)
  plt.plot.(x, test_acc_list, '--')
  plt.xlabel.('epochs')
  plt.ylabel.('accuracy')
  plt.ylim.(0, 1.0)
  plt.legend.('lower right')
  plt.show.()
end
