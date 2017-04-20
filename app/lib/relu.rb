class Relu
  def forward(x)
    self.mask = x <= 0

    out = x.copy.()
    out[mask] = 0
    out
  end

  def backward(dout)
    dout[mask] = 0
    dout
  end

  private

  attr_accessor :mask
end
