require 'simplecov'
SimpleCov.start

require './lib'

require 'fileutils'

require './spec/support/pycall_helper'

RSpec.configure do |config|
  config.before(:each) do
    pyimport 'numpy', as: :np
  end
  config.include PyCallHelper
end
