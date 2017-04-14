class Network
  attr_accessor :params, :input_size, :hidden_size, :output_size

  def initialize(input_size = 784, hidden_size = 50, output_size = 10, weight_init_std = 0.01)
    pyimport 'numpy', as: :np

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    init_params(weight_init_std)
  end

  def predict(x, skip_activate_output = false)
    a1 = NP.dot(x, w1) + b1
    z1 = Util.sigmoid(a1)
    a2 = NP.dot(z1, w2) + b2
    y = skip_activate_output ? a2 : Util.softmax(a2)

    { w2: w2, z1: z1, y: y }
  end

  def loss(x, t)
    y = predict(x)[:y]
    Util.cross_entropy_error(y, t)
  end

  def accuracy(x, t)
    y = predict(x, true)[:y]
    y = NP.argmax(y, 1)
    t = NP.argmax(t, 1)

    np.sum.(y == t) / x.shape[0].to_f
  end

  def numerical_gradient(x, t)
    {
      W1: Util.numerical_gradient(loss_w(:W1, x, t), w1),
      b1: Util.numerical_gradient(loss_w(:b1, x, t), b1),
      W2: Util.numerical_gradient(loss_w(:W2, x, t), w2),
      b2: Util.numerical_gradient(loss_w(:b2, x, t), b2)
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
    forward = predict(x)
    z1 = forward[:z1]
    dy, dz1 = gradient_delta(x, t, forward)
    {
      W1: NP.dot(x.T, dz1),
      b1: NP.sum(dz1, 0),
      W2: NP.dot(z1.T, dy),
      b2: NP.sum(dy, 0)
    }
  end

  private

  def gradient_delta(x, t, forward)
    batch_num = x.shape[0]
    w2, z1, y = forward[:w2], forward[:z1], forward[:y]

    dy = (y - t) / batch_num

    da1 = NP.dot(dy, w2.T)
    dz1 = Util.sigmoid_grad(z1) * da1

    [dy, dz1]
  end

  def w1
    params[:W1]
  end

  def w2
    params[:W2]
  end

  def b1
    params[:b1]
  end

  def b2
    params[:b2]
  end

  def init_params(weight_init_std)
    self.params = {
      W1: weight_init_std * NP.randn(input_size, hidden_size),
      b1: NP.zeros(hidden_size),
      W2: weight_init_std * NP.randn(hidden_size, output_size),
      b2: NP.zeros(output_size)
    }
  end
end
