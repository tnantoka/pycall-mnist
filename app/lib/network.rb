class Network
  attr_accessor :params, :input_size, :hidden_size, :output_size

  def initialize(input_size = 784, hidden_size = 50, output_size = 10, weight_init_std = 0.01)
    pyimport 'numpy', as: :np

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.params = {
      W1: weight_init_std * np.random.randn.(input_size, hidden_size),
      b1: np.zeros.(hidden_size),
      W2: weight_init_std * np.random.randn.(hidden_size, output_size),
      b2: np.zeros.(output_size)
    }
  end

  def predict(x, skip_activate_output = false)
    w1, w2 = params[:W1], params[:W2]
    b1, b2 = params[:b1], params[:b2]

    a1 = np.dot.(x, w1) + b1
    z1 = Util.sigmoid(a1)
    a2 = np.dot.(z1, w2) + b2
    y = skip_activate_output ? a2 : Util.softmax(a2)

    { w2: w2, z1: z1, y: y }
  end

  def loss(x, t)
    y = predict(x)[:y]
    Util.cross_entropy_error(y, t)
  end

  def accuracy(x, t)
    y = predict(x, true)[:y]
    y = np.argmax.(y, 1)
    t = np.argmax.(t, 1)

    np.sum.(y == t) / x.shape[0].to_f
  end

  def numerical_gradient(x, t)
    {
      W1: Util.numerical_gradient(loss_w(:W1, x, t), params[:W1]),
      b1: Util.numerical_gradient(loss_w(:b1, x, t), params[:b1]),
      W2: Util.numerical_gradient(loss_w(:W2, x, t), params[:W2]),
      b2: Util.numerical_gradient(loss_w(:b2, x, t), params[:b2])
    }
  end

  def loss_w(key, x, t)
    lambda do |w|
      tmp_w = params[key]
      params[key] = w
      l = loss(x, t)
      params[key] = tmp_w
      l
    end
  end

  def gradient(x, t)
    grads = {}

    batch_num = x.shape[0]

    forward = predict(x)
    w2, z1, y = forward[:w2], forward[:z1], forward[:y]

    dy = (y - t) / batch_num
    grads[:W2] = np.dot.(z1.T, dy)
    grads[:b2] = np.sum.(dy, 0)

    da1 = np.dot.(dy, w2.T)
    dz1 = Util.sigmoid_grad(z1) * da1
    grads[:W1] = np.dot.(x.T, dz1)
    grads[:b1] = np.sum.(dz1, 0)

    grads
  end
end
