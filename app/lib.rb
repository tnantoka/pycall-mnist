require 'bundler'
Bundler.require

require 'json'

include PyCall::Import

require './lib/loader'
require './lib/util'
require './lib/network'
require './lib/trainer'
