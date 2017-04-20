class Network
  attr_accessor :params, :input_size, :hidden_size, :output_size, :layers, :lastLayer

  def initialize(input_size = 784, hidden_size = 50, output_size = 10, weight_init_std = 0.01)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    init_params(weight_init_std)
  end

  def predict(x)
    y = x
    layers.values.each do |layer|
      y = layer.forward(y)
    end
    y
  end

  def loss(x, t)
    y = predict(x)
    lastLayer.forward(y, t)
  end

  def accuracy(x, t)
    y = predict(x)
    y = NP.argmax(y, 1)
    t = NP.argmax(t, 1)

    NP.sum(y == t) / x.shape[0].to_f
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
      tmp_params = params
      self.params = params.merge(key => w)
      l = loss(x, t)
      self.params = tmp_params
      l
    end
  end

  def gradient(x, t)
    gradient_delta(x, t)
    {
      W1: layers[:affine1].dw,
      b1: layers[:affine1].db,
      W2: layers[:affine2].dw,
      b2: layers[:affine2].db
    }
  end

  def params=(params)
    @params = params
    init_layers
  end

  private

  def gradient_delta(x, t)
    loss(x, t)

    dout = 1
    ([lastLayer] + layers.values.reverse).each do |layer|
      dout = layer.backward(dout)
    end
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

  def init_layers
    self.layers = {
      affine1: Affine.new(w1, b1),
      relu1: Relu.new,
      # sigmoid1: Sigmoid.new,
      affine2: Affine.new(w2, b2)
    }
    self.lastLayer = SoftmaxWithLoss.new
  end
end
