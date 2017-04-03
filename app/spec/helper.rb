require 'simplecov'
SimpleCov.start

require 'bundler'
Bundler.require

require './spec/support/pycall_helper'

include PyCall::Import

require './lib/util'
require './lib/loader'
require './lib/network'

RSpec.configure do |config|
  config.before(:each) do
    pyimport 'numpy', as: :np
  end
  config.include PyCallHelper
end
