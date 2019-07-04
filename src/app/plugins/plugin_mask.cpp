#include "plugin_mask.h"
#include <QTabWidget>
#include <QStackedWidget>

#include <iostream>

PluginMask::PluginMask(FrameBuffer *buffer, ConvexHullImageMask &mask) :
  VisionPlugin(buffer), _widget(nullptr), _settings(nullptr),
  _v_enable(nullptr), _mask(mask) {

  _v_enable = new VarBool("enable", false);

  _settings = new VarList("Mask");
  _settings->addChild(_v_enable);
}

PluginMask::~PluginMask() {
  delete _settings;
  delete _v_enable;
}

VarList *PluginMask::getSettings() {
  return _settings;
}

std::string PluginMask::getName() {
  return "Mask";
}

QWidget * PluginMask::getControlWidget() {
  if (_widget==0)
    _widget = new MaskWidget();

  return (QWidget *) _widget;
}

ProcessResult PluginMask::process(FrameData* data, RenderOptions* options) {
  // We can only allocate _mask once we get the first frame.
  // Until then we do not know the size of the image.
  if (_mask.getNumPixels() != data->video.getNumPixels())
    _mask.setSize(data->video.getWidth(), data->video.getHeight());
  
  if (!_v_enable->getBool())
    _mask.reset();

  return ProcessingOk;
}

void PluginMask::_addPoint(const int x, const int y) {
  if (!_v_enable->getBool())
    return;

  _mask.addPoint(x, y);
}

void PluginMask::_removePoint(const int x, const int y) {
  if (!_v_enable->getBool())
    return;

  _mask.removePoint(x, y, 5);
}

void PluginMask::_mouseEvent(QMouseEvent *event, const pixelloc loc) {
  QTabWidget *tabw = (QTabWidget*) _widget->parentWidget()->parentWidget();  
  if (tabw->currentWidget() != _widget) {
    event->ignore();
    return;
  }

  FrameBuffer *fb = getFrameBuffer();
  if (!fb)
    return;

  if (event->buttons() == Qt::LeftButton) {
    event->accept();
    fb->lockRead();
    {
      int fb_idx = fb->curRead();
      FrameData *frame = fb->getPointer(fb_idx);

      const int video_width = frame->video.getWidth();
      const int video_height = frame->video.getHeight();

      int x = loc.x;
      int y = loc.y;

      // clean the click location
      {
	if (x < 0)
          x = 0;
	else if (x >= video_width)
	  x = video_width - 1;
	if (y < 0)
          y = 0;
	else if (y >= video_height)
	  y = video_height - 1;
      }
      // if (// frame->video.getNumPixels() > 1 &&  // when can this happen?
      //     loc.x >= 0 && loc.x < video_width &&
      //     loc.y >= 0 && loc.y < video_height) {
      if (event->modifiers() == Qt::ShiftModifier)
        _removePoint(x, y);
      else
        _addPoint(x, y);
    }
    fb->unlockRead();
  }
}

void PluginMask::mousePressEvent(QMouseEvent *event, pixelloc loc) {
  _mouseEvent(event, loc);
}
