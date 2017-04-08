class Predictor
  def initialize(path)
    params = Trainer.load_params(path)
    @network = Network.new
    @network.params = params
  end

  def predict(x)
    @network.predict(x, true)
  end
end
