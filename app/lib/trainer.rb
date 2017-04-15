class Trainer
  class << self
    def save_params(path, params)
      json = params.map do |key, value|
        param_to_json(key, value)
      end.to_h
      File.write(path, json.to_json)
    end

    def load_params(path)
      JSON.parse(File.read(path)).map { |k, v| [k.to_sym, NP.array(v)] }.to_h
    end

    private

    def param_to_json(key, value)
      if value.ndim == 1
        [key, Util.np_array_to_a(value)]
      else
        [key, Array.new(value.shape[0]) { |i| Util.np_array_to_a(value[i]) }]
      end
    end
  end

  def initialize(network)
    self.network = network
    self.optimizer = SGD.new
  end

  def train_mnist(limit, iters_num, batch_size, learning_rate = 0.1, listing = true)
    mnist = Loader.load_mnist(true, limit, listing)
    x_train = mnist[0]

    train_size = x_train.shape[0]

    list = { train_loss: [], train_acc: [], test_acc: [] }

    iter_per_epoch = [train_size / batch_size, 1].max

    train_batch(iters_num, learning_rate, batch_size, train_size, mnist) do |i, x_batch, t_batch|
      loss_to_list(list, x_batch, t_batch) if listing
      acc_to_list(list, mnist) if listing && (i % iter_per_epoch).zero?
    end

    list.values
  end

  private

  attr_accessor :network, :optimizer

  def batch(mnist, batch_size, train_size)
    x_train, t_train = mnist
    batch_mask = NP.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    [x_batch, t_batch]
  end

  def train_batch(iters_num, learning_rate, batch_size, train_size, mnist)
    progressbar = ProgressBar.create(total: iters_num, format: '%c / %C')
    iters_num.times do |i|
      x_batch, t_batch = batch(mnist, batch_size, train_size)

      # grads = network.numerical_gradient(x_batch, t_batch)
      grads = network.gradient(x_batch, t_batch)

      optimizer.update(network.params, grads, learning_rate)

      yield i, x_batch, t_batch
      progressbar.increment
    end
  end

  def loss_to_list(list, x_batch, t_batch)
    loss = network.loss(x_batch, t_batch)
    list[:train_loss] << loss
  end

  def acc_to_list(list, mnist)
    x_train, t_train, x_test, t_test = mnist

    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    list[:train_acc] << train_acc
    list[:test_acc] << test_acc
  end
end
