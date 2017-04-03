class Loader
  class << self
    def load_mnist(one_hot_label = false, limit = nil)
      pyimport 'numpy', as: :np

      files = %w(
        train-images-idx3
        train-labels-idx1
        t10k-images-idx3
        t10k-labels-idx1
      )
      x_train = load_images('train')
      t_train = load_labels('train')
      x_test = load_images('t10k')
      t_test = load_labels('t10k')

      if one_hot_label
        t_train = one_hot_labels(t_train)
        t_test = one_hot_labels(t_test)
      end

      [x_train, t_train, x_test, t_test].map { |a| limit.to_i.zero? ? a : a[0...limit] }.map { |a| np.array.(a) }
    end

    private

    def load_images(type)
      Mnist.load_images(path("#{type}-images-idx3"))[2].map do |image|
        image.unpack('C*').map { |p| p / 255.0 }
      end
    end

    def load_labels(type)
      Mnist.load_labels(path("#{type}-labels-idx1")) 
    end

    def path(file)
      File.expand_path("../../data/#{file}-ubyte.gz", __FILE__)
    end

    def one_hot_labels(x)
      x.map do |label|
        Array.new(10, 0).tap { |a| a[label] = 1 }
      end
    end
  end
end
