class NP
  pyimport 'numpy', as: :np

  class << self
    def dot(a, b)
      np.dot.(a, b)
    end

    def sum(a, axis = nil)
      axis.nil? ? np.sum.(a) : np.sum.(a, axis)
    end

    def argmax(a, axis)
      np.argmax.(a, axis)
    end

    def randn(d0, d1)
      np.random.randn.(d0, d1)
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
  end
end
