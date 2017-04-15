class NP
  pyimport 'numpy', as: :np

  class << self
    def array(object)
      np.array.(object)
    end

    def array_equal(a1, a2)
      np.array_equal.(a1, a2)
    end

    def dot(a, b)
      np.dot.(a, b)
    end

    def copy(a)
      np.copy.(a)
    end

    def zeros_like(a)
      np.zeros_like.(a)
    end

    def sum(a, axis = nil)
      axis.nil? ? np.sum.(a) : np.sum.(a, axis)
    end

    def exp(x)
      np.exp.(x)
    end

    def argmax(a, axis)
      np.argmax.(a, axis)
    end

    def max(a, axis)
      np.max.(a, axis)
    end

    def randn(d0, d1)
      np.random.randn.(d0, d1)
    end

    def choice(a, size)
      np.random.choice.(a, size)
    end

    def zeros(shape)
      np.zeros.(shape)
    end

    def log(x)
      np.log.(x)
    end

    def arange(stop)
      np.arange.(stop)
    end

    def nditer(op, flags, op_flags)
      np.nditer.(op, flags, op_flags)
    end
  end
end
