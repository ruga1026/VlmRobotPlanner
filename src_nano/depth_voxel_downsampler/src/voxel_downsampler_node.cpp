#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/filter.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace std::chrono_literals;

class VoxelDownsamplerNode : public rclcpp::Node
{
public:
  explicit VoxelDownsamplerNode(const rclcpp::NodeOptions & options)
  : Node("voxel_downsampler_node", options)
  {
    // ===== Parameters =====
    input_topic_   = declare_parameter<std::string>("input_topic", "/camera/depth/color/points");
    output_topic_  = declare_parameter<std::string>("output_topic", "/depth/points_downsampled");              // Best-Effort
    output_topic_reliable_ = declare_parameter<std::string>("output_topic_reliable", "/depth/points_downsampled_reliable"); // Reliable 복제
    publish_reliable_duplicate_ = declare_parameter<bool>("publish_reliable_duplicate", true);

    leaf_size_     = declare_parameter<double>("leaf_size", 0.05); // 5 cm
    z_min_         = declare_parameter<double>("z_min", 0.10);
    z_max_         = declare_parameter<double>("z_max", 4.50);
    drop_rgb_      = declare_parameter<bool>("drop_rgb", true);    // true: XYZ만
    remove_nan_    = declare_parameter<bool>("remove_nan", true);
    rate_limit_hz_ = declare_parameter<double>("rate_limit_hz", 15.0); // 0 => 제한 없음
    qos_depth_     = declare_parameter<int>("qos_depth", 5);

    // ===== QoS =====
    // Subscriber: Realsense 등의 센서 토픽 구독 → SensorDataQoS(베스트에포트)
    auto sub_qos = rclcpp::SensorDataQoS();

    // Publisher #1: Best-Effort (센서 데이터 파이프: SLAM/Nav2용)
    rclcpp::QoS pub_best_effort_qos(qos_depth_);
    pub_best_effort_qos.best_effort();
    pub_best_effort_qos.durability_volatile();

    // Publisher #2: Reliable (RViz 등 신뢰성 강한 구독자용, 선택)
    rclcpp::QoS pub_reliable_qos(qos_depth_);
    pub_reliable_qos.reliable();
    pub_reliable_qos.durability_volatile();

    pub_best_effort_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, pub_best_effort_qos);
    if (publish_reliable_duplicate_) {
      pub_reliable_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_reliable_, pub_reliable_qos);
    }

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, sub_qos,
      std::bind(&VoxelDownsamplerNode::cloudCallback, this, std::placeholders::_1));

    // 동적 파라미터 콜백 (주의: 퍼블리셔 QoS는 런타임 변경 불가)
    param_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&VoxelDownsamplerNode::onParamChange, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "VoxelDownsampler started.\n  input: %s\n  output(best-effort): %s\n  output(reliable): %s (%s)\n  leaf: %.3f m  z:[%.2f, %.2f] m  rate: %.1f Hz  drop_rgb:%s remove_nan:%s",
      input_topic_.c_str(),
      output_topic_.c_str(),
      output_topic_reliable_.c_str(),
      publish_reliable_duplicate_ ? "enabled" : "disabled",
      leaf_size_, z_min_, z_max_, rate_limit_hz_,
      drop_rgb_ ? "true" : "false", remove_nan_ ? "true" : "false");
  }

private:
  rcl_interfaces::msg::SetParametersResult onParamChange(const std::vector<rclcpp::Parameter> & params)
  {
    for (const auto & p : params) {
      if (p.get_name() == "leaf_size")             leaf_size_ = p.as_double();
      else if (p.get_name() == "z_min")            z_min_ = p.as_double();
      else if (p.get_name() == "z_max")            z_max_ = p.as_double();
      else if (p.get_name() == "drop_rgb")         drop_rgb_ = p.as_bool();
      else if (p.get_name() == "remove_nan")       remove_nan_ = p.as_bool();
      else if (p.get_name() == "rate_limit_hz")    rate_limit_hz_ = p.as_double();
      // QoS / 토픽 이름은 런타임 변경해도 퍼블리셔 재생성이 필요하므로 여기서 적용하지 않음
    }
    rcl_interfaces::msg::SetParametersResult r; r.successful = true; return r;
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // ===== Rate limiting =====
    if (rate_limit_hz_ > 0.0) {
      const auto now = this->now();
      const double period = 1.0 / rate_limit_hz_;
      if (last_pub_time_.nanoseconds() != 0) {
        if ((now - last_pub_time_).seconds() < period) {
          return; // Skip this frame
        }
      }
      last_pub_time_ = now;
    }

    // ===== Convert ROS -> PCL, optional NaN removal, Z-crop, VoxelGrid =====
    sensor_msgs::msg::PointCloud2 out;
    if (drop_rgb_) {
      // XYZ only
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
      pcl::fromROSMsg(*msg, *cloud);

      if (remove_nan_) {
        std::vector<int> idx;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);
      }

      // Z crop
      if (z_min_ > 0.0 || z_max_ > 0.0) {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_min_, z_max_);
        pass.filter(*cloud);
      }

      // Voxel grid (5cm default)
      pcl::VoxelGrid<pcl::PointXYZ> vg;
      vg.setInputCloud(cloud);
      vg.setLeafSize(static_cast<float>(leaf_size_), static_cast<float>(leaf_size_), static_cast<float>(leaf_size_));
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZ>());
      vg.filter(*cloud_ds);

      pcl::toROSMsg(*cloud_ds, out);
    } else {
      // XYZRGB
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl::fromROSMsg(*msg, *cloud);

      if (remove_nan_) {
        std::vector<int> idx;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);
      }

      if (z_min_ > 0.0 || z_max_ > 0.0) {
        pcl::PassThrough<pcl::PointXYZRGB> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_min_, z_max_);
        pass.filter(*cloud);
      }

      pcl::VoxelGrid<pcl::PointXYZRGB> vg;
      vg.setInputCloud(cloud);
      vg.setLeafSize(static_cast<float>(leaf_size_), static_cast<float>(leaf_size_), static_cast<float>(leaf_size_));
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ds(new pcl::PointCloud<pcl::PointXYZRGB>());
      vg.filter(*cloud_ds);

      pcl::toROSMsg(*cloud_ds, out);
    }

    // Preserve header (frame_id, stamp)
    out.header = msg->header;

    // ===== Publish to both QoS flavors =====
    if (pub_best_effort_) {
      pub_best_effort_->publish(out);
    }
    if (publish_reliable_duplicate_ && pub_reliable_) {
      pub_reliable_->publish(out);
    }
  }

  // ===== Members =====
  std::string input_topic_;
  std::string output_topic_;
  std::string output_topic_reliable_;
  bool publish_reliable_duplicate_ = true;

  double leaf_size_;
  double z_min_;
  double z_max_;
  bool drop_rgb_;
  bool remove_nan_;
  double rate_limit_hz_;
  int qos_depth_;

  rclcpp::Time last_pub_time_; // init to 0

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_best_effort_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_reliable_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true); // 복사 최소화
  rclcpp::spin(std::make_shared<VoxelDownsamplerNode>(options));
  rclcpp::shutdown();
  return 0;
}
