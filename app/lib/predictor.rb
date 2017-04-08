class Predictor
  def initialize(path)
    params = Trainer.load_params(path)
    @network = Network.new(784, 50, 10)
    @network.params = params
  end

  def predict(x)
    @network.predict(x, true)
  end
end
