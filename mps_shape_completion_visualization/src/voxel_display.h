#ifndef VOXEL_DISPLAY_H
#define VOXEL_DISPLAY_H

#include <mps_shape_completion_msgs/OccupancyStamped.h>
#include <rviz/message_filter_display.h>

namespace Ogre
{
class SceneNode;
}

namespace rviz
{
class ColorProperty;
class FloatProperty;
class IntProperty;
}

namespace mps_shape_completion_visualization
{

class VoxelGridVisual;

// TODO: inheriting from MessageFilterDisplay gives us topic handling (yay!), but also the undesirable "unreliabel" checkbox. 
class VoxelGridDisplay: public rviz::MessageFilterDisplay<mps_shape_completion_msgs::OccupancyStamped>
{
Q_OBJECT
public:
  // Constructor.  pluginlib::ClassLoader creates instances by calling
  // the default constructor, so make sure you have one.
  VoxelGridDisplay();
  virtual ~VoxelGridDisplay();

  // Overrides of protected virtual functions from Display.  As much
  // as possible, when Displays are not enabled, they should not be
  // subscribed to incoming data and should not show anything in the
  // 3D view.  These functions are where these connections are made
  // and broken.
protected:
  virtual void onInitialize();

  // A helper to clear this display back to the initial state.
  virtual void reset();

  // These Qt slots get connected to signals indicating changes in the user-editable properties.
private Q_SLOTS:
  void updateColorAndAlpha();


private:
  // Function to handle an incoming ROS message.
  void processMessage( const mps_shape_completion_msgs::OccupancyStamped::ConstPtr& msg);

  std::unique_ptr<VoxelGridVisual > visual_;

  // User-editable property variables.
  std::unique_ptr<rviz::ColorProperty> color_property_;
  std::unique_ptr<rviz::FloatProperty> alpha_property_;
  std::unique_ptr<rviz::BoolProperty> binary_display_property_;
  std::unique_ptr<rviz::FloatProperty> cutoff_property_;
};

} // end namespace mps_shape_completion_visualization

#endif // VOXEL_DISPLAY_H
