$(function() {
  var $canvas = $('canvas')
  var canvas = $canvas.get(0)
  var context = canvas.getContext('2d')

  context.lineWidth = 14
  context.lineJoin = 'round'

  // https://software.intel.com/en-us/html5/articles/touch-drawing-app-using-html5-canvas
  $.fn.drawTouch = function() {
    var $this = $(this)
    var start = function(e) {
      e.preventDefault()
      var touchEvent = e.originalEvent.changedTouches[0]
      context.beginPath()
      context.moveTo(touchEvent.pageX - $this.offset().left, touchEvent.pageY - $this.offset().top)
    }
    var move = function(e) {
      e.preventDefault()
      var touchEvent = e.originalEvent.changedTouches[0]
      context.lineTo(touchEvent.pageX - $this.offset().left, touchEvent.pageY - $this.offset().top)
      context.stroke()
    }
    $this.on('touchstart', start)
    $this.on('touchmove', move)
  }
  $.fn.drawMouse = function() {
    var $this = $(this)
    var clicked = 0
    var start = function(e) {
      clicked = 1
      context.beginPath()
      context.moveTo(e.pageX - $this.offset().left, e.pageY - $this.offset().top)
    }
    var move = function(e) {
      if (clicked){
        context.lineTo(e.pageX - $this.offset().left, e.pageY - $this.offset().top)
        context.stroke()
      }
    }
    var stop = function(e) {
      clicked = 0
    }
    $this.mousedown(start)
    $this.mousemove(move)
    $this.mouseup(stop)
  }
  $canvas.drawTouch()
  $canvas.drawMouse()

  var $table = $('table')
  var clearCanvas = function() {
    context.clearRect(0, 0, canvas.width, canvas.height)
  }

  $submit = $('.js-submit')
  $submit.click(function(e) {
    var $this = $(this)
    var dataURL = canvas.toDataURL()
    $this.prop('disabled', true)
    $.post('/predict', { data_url: dataURL }, function(json) {
      $this.prop('disabled', false)
      var $img = $('<img>').prop('src', json.image)
      var $tr = $('<tr>')
      $('<td>').html($img).appendTo($tr)
      $('<td>').html(json.label).appendTo($tr)
      $('<td>').html(json.percent + '%').appendTo($tr)
      $tr.prependTo($table).hide().fadeIn()
      clearCanvas()
    }, 'json')
  }) 

  $('.js-clear').click(function(e) {
    clearCanvas()
    $submit.prop('disabled', false)
    $table.html('')
  })
})
