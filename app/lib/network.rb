class Network
  attr_accessor :params

  def initialize(input_size, hidden_size, output_size, weight_init_std = 0.01)
    pyimport 'numpy', as: :np

    self.params = {
      W1: weight_init_std * np.random.randn.(input_size, hidden_size),
      b1: np.zeros.(hidden_size),
      W2: weight_init_std * np.random.randn.(hidden_size, output_size),
      b2: np.zeros.(output_size),
    }
  end

  def predict(x)
    w1, w2 = params[:W1], params[:W2]
    b1, b2 = params[:b1], params[:b2]

    a1 = np.dot.(x, w1) + b1
    z1 = Util.sigmoid(a1)
    a2 = np.dot.(z1, w2) + b2

    Util.softmax(a2)
  end

  def loss(x, t)
    y = predict(x)
    Util.cross_entropy_error(y, t)
  end

  def accuracy(x, t)
    y = predict(x)
    y = np.argmax.(y, 1)
    t = np.argmax.(t, 1)

    np.sum.(y == t) / x.shape[0].to_f
  end

  def numerical_gradient(x, t)
    loss_W = -> w { loss(x, t) }
    {
      W1: Util.numerical_gradient(loss_W, params[:W1]),
      b1: Util.numerical_gradient(loss_W, params[:b1]),
      W2: Util.numerical_gradient(loss_W, params[:W2]),
      b2: Util.numerical_gradient(loss_W, params[:b2]),
    }
  end  

  def gradient(x, t)
    w1, w2 = params[:W1], params[:W2]
    b1, b2 = params[:b1], params[:b2]

    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = np.dot.(x, w1) + b1
    z1 = Util.sigmoid(a1)
    a2 = np.dot.(z1, w2) + b2
    y = Util.softmax(a2)

    # backward
    dy = (y - t) / batch_num
    grads[:W2] = np.dot.(z1.T, dy)
    grads[:b2] = np.sum.(dy, 0)

    da1 = np.dot.(dy, w2.T)
    dz1 = Util.sigmoid_grad(a1) * da1
    grads[:W1] = np.dot.(x.T, dz1)
    grads[:b1] = np.sum.(dz1, 0)

    return grads
  end
end
