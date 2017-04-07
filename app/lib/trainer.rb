class Trainer
  class << self
    def train_mnist(limit, iters_num, batch_size, learning_rate = 0.1)
      pyimport 'numpy', as: :np

      x_train, t_train, x_test, t_test = Loader.load_mnist(true, limit)

      train_size = x_train.shape[0]

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
        end
      end

      [train_loss_list, train_acc_list, test_acc_list, network.params]
    end

    def save_params(path, params)
      json = params.map do |key, value|
        if value.ndim == 1
          [key, Util.np_array_to_a(params[key])]
        else
          [key, params[key].shape[0].times.map { |i| Util.np_array_to_a(params[key][i]) }]
        end
      end.to_h
      File.write(path, json.to_json)
    end

    def load_params(path)
      pyimport 'numpy', as: :np
      json = JSON.parse(File.read(path)).map { |k, v| [k, np.array.(v)] }.to_h
    end
  end
end
