module PyCallHelper
  def array_to_a(array, digits = nil)
    (0...array.size).map { |i| digits.nil? ? array[i] : array[i].round(digits) }
  end
end
