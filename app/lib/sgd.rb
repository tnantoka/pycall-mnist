class SGD
  def update(network, grads, learning_rate)
    network.params = network.params.map do |key, value|
      [key, value - learning_rate * grads[key]]
    end.to_h
  end
end
