class Affine
  attr_accessor :dw, :db

  def initialize(w, b)
    self.w = w
    self.b = b
  end

  def forward(x)
    self.x = x
    NP.dot(x, w) + b
  end

  def backward(dout)
    dx = NP.dot(dout, w.T)
    self.dw = NP.dot(x.T, dout)
    self.db = NP.sum(dout, 0)
    dx
  end

  private

  attr_accessor :w, :b, :x
end
