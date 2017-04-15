class Util
  class << self
    def identity(x)
      x
    end

    def sigmoid(x)
      1 / (1 + NP.exp(-1 * x))
    end

    def sigmoid_grad(out)
      (1.0 - out) * out
    end

    def softmax(x)
      xt = x.T
      c = NP.max(xt, 0)
      exp_x = NP.exp(xt - c)
      sum_exp_x = NP.sum(exp_x, 0)
      y = exp_x / sum_exp_x
      y.T
    end

    def cross_entropy_error(y, t)
      y, t = to_batch(y, t)
      t = revert_one_hot_t(y, t)

      batch_size = y.shape[0]
      output_for_answer = y[NP.arange(batch_size), t]
      sum = -NP.sum(NP.log(output_for_answer))
      sum / batch_size
    end

    def numerical_gradient(f, x)
      x = NP.copy(x)
      grad = NP.zeros_like(x)

      it = NP.nditer(x, ['multi_index'], ['readwrite'])
      loop do
        i = it.multi_index
        grad[i] = central_diff(f, x, i)
        break unless it.iternext.()
      end

      grad
    end

    def np_array_to_a(array)
      (0...array.size).map { |i| array[i] }
    end

    private

    def central_diff(f, x, i)
      h = 1e-4
      tmp_val = x[i]

      x[i] = tmp_val + h
      forward = f.call(x)

      x[i] = tmp_val - h
      backward = f.call(x)

      x[i] = tmp_val

      (forward - backward) / (2 * h)
    end

    def to_batch(y, t)
      not_batch = y.ndim == 1
      if not_batch
        y = y.reshape.(1, y.size)
        t = t.reshape.(1, t.size)
      end
      [y, t]
    end

    def revert_one_hot_t(y, t)
      one_hot_vector = t.size == y.size
      t = NP.argmax(t, 1) if one_hot_vector
      t
    end
  end
end
