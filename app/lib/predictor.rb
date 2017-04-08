class Predictor
  attr_accessor :network

  class << self
    def parse(data)
      pyimport 'numpy', as: :np

      length = 28
      canvas = ChunkyPNG::Canvas.from_data_url(data).resize(length, length)
      pixels = np.array.(length.times.map do |i|
        length.times.map do |j|
          canvas[j, i] / 255.0
        end
      end.flatten)
      [canvas, pixels]
    end
  end

  def initialize(path)
    params = Trainer.load_params(path)
    self.network = Network.new
    network.params = params
  end

  def predict(x, skip_activate_output = true)
    network.predict(x, skip_activate_output)
  end
end
