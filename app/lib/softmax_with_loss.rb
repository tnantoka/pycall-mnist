class SoftmaxWithLoss
  def forward(x, t)
    self.t = t
    self.y = Util.softmax(x)
    self.loss = Util.cross_entropy_error(y, t)
  end

  def backward(_)
    batch_size = t.shape[0]
    (y - t) / batch_size
  end

  private

  attr_accessor :loss, :y, :t
end
