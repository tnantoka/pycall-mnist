require 'bundler'
Bundler.require

include PyCall::Import

class Util
  pyimport 'numpy', as: :np

  class << self
    def identity(x)
      x
    end

    def sigmoid(x)
      1 / (1 + np.exp.(-1 * x))
    end

    def sigmoid_grad(out)
      (1.0 - out) * out
    end

    def softmax(x)
      xt = x.T
      c = np.max.(xt, 0)
      exp_x = np.exp.(xt - c)
      sum_exp_x = np.sum.(exp_x, 0)
      y = exp_x / sum_exp_x
      y.T 
    end

    def cross_entropy_error(y, t)
      not_batch = y.ndim == 1
      if not_batch
        t = t.reshape.(1, t.size)
        y = y.reshape.(1, y.size)
      end

      one_hot_vector = t.size == y.size
      t = t.argmax.(1) if one_hot_vector

      batch_size = y.shape[0]
      output_for_answer = y[np.arange.(batch_size), t]
      sum = -np.sum.(np.log.(output_for_answer))
      return sum / batch_size
    end

    def numerical_gradient(f, x)
      x = np.copy.(x)
      h = 1e-4
      grad = np.zeros_like.(x)

      it = np.nditer.(x, ['multi_index'], ['readwrite'])
      loop do
        i = it.multi_index
        tmp_val = x[i]

        x[i] = tmp_val + h
        forward = f.call(x)    

        x[i] = tmp_val - h
        backward = f.call(x)

        central = (forward - backward) / (2 * h)
        grad[i] = central

        x[i] = tmp_val

        break unless it.iternext.()
      end

      grad
    end
  end
end
