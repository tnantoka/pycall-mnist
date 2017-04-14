class SGD
  def update(params, grads, learning_rate)
    params.keys.each do |key|
      params[key] -= learning_rate * grads[key]
    end
  end
end
