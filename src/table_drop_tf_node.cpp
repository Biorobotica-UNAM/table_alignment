#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using namespace std::chrono_literals;

class TableDropTfNode : public rclcpp::Node {
public:
  TableDropTfNode() : Node("table_drop_tf_node") {
    declare_parameter<std::string>("pointcloud_topic", "/hardware/camera/points");
    declare_parameter<std::string>("fixed_frame_id", "map");
    declare_parameter<std::string>("validation_frame_id", "base_link");
    declare_parameter<std::string>("table_frame_id", "table_frame");
    declare_parameter<std::string>("drop_start_frame_id", "table_drop_start");
    declare_parameter<std::string>("drop_points_file",
                                   "src/Vision/table_alignment/src/"
                                   "table_drop_points.txt");
    declare_parameter<double>("start_height_m", 0.10);
    declare_parameter<double>("front_margin_m", 0.15);
    declare_parameter<double>("side_margin_m", 0.10);
    declare_parameter<double>("back_margin_m", 0.10);
    declare_parameter<double>("obstacle_margin_m", 0.15);
    declare_parameter<double>("obstacle_min_height_m", 0.03);
    declare_parameter<double>("obstacle_max_height_m", 0.35);
    declare_parameter<double>("preferred_x_m", 0.0);
    declare_parameter<int>("min_front_edge_points", 30);
    declare_parameter<int>("min_front_edge_line_inliers", 20);
    declare_parameter<double>("safe_x_min_m", 0.15);
    declare_parameter<double>("safe_x_max_m", 1.20);
    declare_parameter<double>("safe_abs_y_max_m", 0.80);
    declare_parameter<double>("safe_z_min_m", 0.35);
    declare_parameter<double>("safe_z_max_m", 1.40);
    declare_parameter<double>("safe_dist_xy_max_m", 1.40);
    declare_parameter<double>("roi_x_min_m", -1.5);
    declare_parameter<double>("roi_x_max_m", 1.5);
    declare_parameter<double>("roi_y_min_m", -0.8);
    declare_parameter<double>("roi_y_max_m", 0.8);
    declare_parameter<double>("roi_z_min_m", 0.0);
    declare_parameter<double>("roi_z_max_m", 1.5);
    declare_parameter<bool>("active_on_start", false);

    pointcloud_topic_ = get_parameter("pointcloud_topic").as_string();
    fixed_frame_id_ = get_parameter("fixed_frame_id").as_string();
    validation_frame_id_ = get_parameter("validation_frame_id").as_string();
    table_frame_id_ = get_parameter("table_frame_id").as_string();
    drop_start_frame_id_ = get_parameter("drop_start_frame_id").as_string();
    drop_points_file_ = get_parameter("drop_points_file").as_string();
    active_ = get_parameter("active_on_start").as_bool();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    static_tf_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&TableDropTfNode::cloudCallback, this, std::placeholders::_1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "camera/table_drop_markers", 10);

    resetDropPointsFile();

    enable_srv_ = create_service<std_srvs::srv::SetBool>(
        "table_drop_tf/enable",
        std::bind(&TableDropTfNode::enableCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(),
                "[table_drop_tf] Nodo iniciado. pointcloud_topic=%s "
                "fixed_frame=%s active=%s",
                pointcloud_topic_.c_str(), fixed_frame_id_.c_str(),
                active_ ? "true" : "false");
  }

private:
  struct Candidate {
    float x{0.0f};
    float y{0.0f};
    float score{-std::numeric_limits<float>::max()};
  };

  struct SavedDropPoint {
    int id{0};
    Eigen::Vector3f point{0.0f, 0.0f, 0.0f};
  };

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!active_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[table_drop_tf] inactivo; ignorando nubes hasta recibir enable=true");
      return;
    }

    if (drop_point_saved_this_activation_) {
      RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[table_drop_tf] esfera segura ya guardada en esta activacion");
      return;
    }

    std::string error;
    if (!processCloud(*msg, error)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "[table_drop_tf] No se pudo calcular TF: %s",
                           error.c_str());
    }
  }

  void enableCallback(
      const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
      std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    active_ = request->data;
    if (active_) {
      drop_point_saved_this_activation_ = false;
      publishMarkers();
    }

    response->success = true;
    response->message = active_ ? "table_drop_tf_enabled"
                                : "table_drop_tf_disabled";

    RCLCPP_INFO(get_logger(), "[table_drop_tf] %s", response->message.c_str());
  }

  bool processCloud(const sensor_msgs::msg::PointCloud2 &input_msg,
                    std::string &error_out) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input_msg, *cloud);

    RCLCPP_DEBUG(get_logger(),
                 "[table_drop_tf] nube recibida frame=%s width=%u height=%u "
                 "puntos=%zu",
                 input_msg.header.frame_id.c_str(), input_msg.width,
                 input_msg.height, cloud->size());

    if (cloud->empty()) {
      error_out = "empty_cloud";
      return false;
    }

    // 1) Recortamos la nube a la zona donde esperamos ver la mesa.
    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    *roi_cloud = *cloud;

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(roi_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(get_parameter("roi_z_min_m").as_double(),
                         get_parameter("roi_z_max_m").as_double());
    pass.filter(*roi_cloud);
    const auto after_z = roi_cloud->size();

    pass.setInputCloud(roi_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(get_parameter("roi_y_min_m").as_double(),
                         get_parameter("roi_y_max_m").as_double());
    pass.filter(*roi_cloud);
    const auto after_y = roi_cloud->size();

    pass.setInputCloud(roi_cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(get_parameter("roi_x_min_m").as_double(),
                         get_parameter("roi_x_max_m").as_double());
    pass.filter(*roi_cloud);
    const auto after_x = roi_cloud->size();

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[table_drop_tf] ROI puntos: entrada=%zu z=%zu y=%zu x=%zu "
        "limites_x=[%.2f, %.2f] limites_y=[%.2f, %.2f] limites_z=[%.2f, %.2f]",
        cloud->size(), after_z, after_y, after_x,
        get_parameter("roi_x_min_m").as_double(),
        get_parameter("roi_x_max_m").as_double(),
        get_parameter("roi_y_min_m").as_double(),
        get_parameter("roi_y_max_m").as_double(),
        get_parameter("roi_z_min_m").as_double(),
        get_parameter("roi_z_max_m").as_double());

    if (roi_cloud->size() < 500) {
      error_out = "insufficient_points_after_roi count=" +
                  std::to_string(roi_cloud->size());
      return false;
    }

    // 2) Buscamos el plano dominante de la mesa con RANSAC.
    pcl::ModelCoefficients::Ptr plane_coefficients(
        new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> plane_seg;
    plane_seg.setOptimizeCoefficients(true);
    plane_seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    plane_seg.setMethodType(pcl::SAC_RANSAC);
    plane_seg.setAxis(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    plane_seg.setEpsAngle(45.0 * (M_PI / 180.0));
    plane_seg.setDistanceThreshold(0.02);
    plane_seg.setMaxIterations(200);
    plane_seg.setInputCloud(roi_cloud);
    plane_seg.segment(*plane_inliers, *plane_coefficients);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[table_drop_tf] RANSAC plano: inliers=%zu coef=[%.4f %.4f %.4f %.4f]",
        plane_inliers->indices.size(), plane_coefficients->values[0],
        plane_coefficients->values[1], plane_coefficients->values[2],
        plane_coefficients->values[3]);

    if (plane_inliers->indices.empty()) {
      error_out = "plane_not_found";
      return false;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr table_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_table_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(roi_cloud);
    extract.setIndices(plane_inliers);
    extract.setNegative(false);
    extract.filter(*table_cloud);
    extract.setNegative(true);
    extract.filter(*non_table_cloud);

    RCLCPP_DEBUG(get_logger(),
                 "[table_drop_tf] separacion mesa/no_mesa: mesa=%zu no_mesa=%zu",
                 table_cloud->size(), non_table_cloud->size());

    if (table_cloud->size() < 300) {
      error_out = "insufficient_table_points count=" +
                  std::to_string(table_cloud->size());
      return false;
    }

    // 3) La normal del plano define "arriba" de la mesa. En el frame optico
    // del Kinect normalmente arriba apunta a -Y; por eso corregimos el signo.
    Eigen::Vector3f table_z(plane_coefficients->values[0],
                            plane_coefficients->values[1],
                            plane_coefficients->values[2]);
    table_z.normalize();
    if (table_z.dot(Eigen::Vector3f(0.0f, -1.0f, 0.0f)) < 0.0f) {
      table_z = -table_z;
    }

    RCLCPP_DEBUG(get_logger(),
                 "[table_drop_tf] normal mesa corregida=(%.4f, %.4f, %.4f)",
                 table_z.x(), table_z.y(), table_z.z());

    // 4) Encontramos el borde frontal de la mesa tomando, para cada franja en
    // X, el punto de mesa mas cercano al robot/camara (menor Z).
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (!extractFrontEdge(table_cloud, edge_cloud, error_out)) {
      return false;
    }

    pcl::ModelCoefficients::Ptr line_coefficients(
        new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr line_inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> line_seg;
    line_seg.setOptimizeCoefficients(true);
    line_seg.setModelType(pcl::SACMODEL_LINE);
    line_seg.setMethodType(pcl::SAC_RANSAC);
    line_seg.setDistanceThreshold(0.02);
    line_seg.setMaxIterations(200);
    line_seg.setInputCloud(edge_cloud);
    line_seg.segment(*line_inliers, *line_coefficients);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[table_drop_tf] borde frontal: puntos=%zu inliers_linea=%zu",
        edge_cloud->size(), line_inliers->indices.size());

    const auto min_edge_points =
        static_cast<std::size_t>(get_parameter("min_front_edge_points").as_int());
    const auto min_line_inliers = static_cast<std::size_t>(
        get_parameter("min_front_edge_line_inliers").as_int());
    if (edge_cloud->size() < min_edge_points ||
        line_inliers->indices.size() < min_line_inliers) {
      error_out = "front_edge_too_weak points=" +
                  std::to_string(edge_cloud->size()) +
                  " min_points=" + std::to_string(min_edge_points) +
                  " line_inliers=" +
                  std::to_string(line_inliers->indices.size()) +
                  " min_line_inliers=" + std::to_string(min_line_inliers);
      return false;
    }

    if (line_inliers->indices.empty()) {
      error_out = "front_edge_line_not_found";
      return false;
    }

    Eigen::Vector3f table_x(line_coefficients->values[3],
                            line_coefficients->values[4],
                            line_coefficients->values[5]);
    table_x.normalize();
    if (table_x.x() < 0.0f) {
      table_x = -table_x;
    }

    Eigen::Vector3f table_y = table_z.cross(table_x).normalized();
    table_z = table_x.cross(table_y).normalized();

    Eigen::Vector3f edge_origin = centroidFromInliers(edge_cloud, line_inliers);
    Eigen::Vector3f table_centroid = centroid(table_cloud);

    // La direccion correcta hacia dentro de la mesa es la que va del borde
    // frontal hacia el centro de la nube de puntos de la superficie.
    if ((table_centroid - edge_origin).dot(table_y) < 0.0f) {
      table_y = -table_y;
      table_z = table_x.cross(table_y).normalized();
    }

    float x_min, x_max, y_min, y_max;
    computeLocalBounds(table_cloud, edge_origin, table_x, table_y, x_min, x_max,
                       y_min, y_max);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[table_drop_tf] frame mesa: origen=(%.3f %.3f %.3f) "
        "x=(%.3f %.3f %.3f) y=(%.3f %.3f %.3f) z=(%.3f %.3f %.3f)",
        edge_origin.x(), edge_origin.y(), edge_origin.z(), table_x.x(),
        table_x.y(), table_x.z(), table_y.x(), table_y.y(), table_y.z(),
        table_z.x(), table_z.y(), table_z.z());

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[table_drop_tf] limites locales mesa: x=[%.3f, %.3f] y=[%.3f, %.3f]",
        x_min, x_max, y_min, y_max);

    // 5) Elegimos un punto libre dentro de la mesa, lejos de bordes y objetos.
    std::vector<Eigen::Vector2f> obstacles =
        obstacleProjections(non_table_cloud, edge_origin, table_x, table_y,
                            table_z, x_min, x_max, y_min, y_max);

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1500,
        "[table_drop_tf] obstaculos proyectados sobre mesa=%zu altura=[%.2f, %.2f]",
        obstacles.size(), get_parameter("obstacle_min_height_m").as_double(),
        get_parameter("obstacle_max_height_m").as_double());

    Candidate best;
    if (!selectCandidate(x_min, x_max, y_min, y_max, obstacles, best,
                         error_out)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1500,
          "[table_drop_tf] seleccion de candidato fallo: %s", error_out.c_str());
      return false;
    }

    Eigen::Vector3f surface_point =
        edge_origin + table_x * best.x + table_y * best.y;
    Eigen::Vector3f start_point =
        surface_point +
        table_z * get_parameter("start_height_m").as_double();

    if (!validateDropStartPose(input_msg.header, start_point, error_out)) {
      return false;
    }

    Eigen::Vector3f fixed_edge_origin;
    Eigen::Vector3f fixed_surface_point;
    Eigen::Vector3f fixed_start_point;
    Eigen::Vector3f fixed_table_x;
    Eigen::Vector3f fixed_table_y;
    Eigen::Vector3f fixed_table_z;
    if (!transformGeometryToFixedFrame(input_msg.header, edge_origin,
                                       surface_point, start_point, table_x,
                                       table_y, table_z, fixed_edge_origin,
                                       fixed_surface_point, fixed_start_point,
                                       fixed_table_x, fixed_table_y,
                                       fixed_table_z, error_out)) {
      return false;
    }

    publishTransforms(fixed_edge_origin, fixed_table_x, fixed_table_y,
                      fixed_table_z, best);
    saveDropPoint(fixed_start_point);
    publishMarkers();
    drop_point_saved_this_activation_ = true;

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[table_drop_tf] TF %s publicada: local=(%.3f, %.3f, %.3f) "
        "superficie=(%.3f %.3f %.3f) score=%.3f",
        drop_start_frame_id_.c_str(), best.x, best.y,
        get_parameter("start_height_m").as_double(), surface_point.x(),
        surface_point.y(), surface_point.z(), best.score);

    error_out.clear();
    return true;
  }

  bool extractFrontEdge(const pcl::PointCloud<pcl::PointXYZ>::Ptr &table_cloud,
                        pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud,
                        std::string &error_out) const {
    float theta_min = std::numeric_limits<float>::max();
    float theta_max = -std::numeric_limits<float>::max();
    float phi_max = -std::numeric_limits<float>::max();
    float min_x = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();

    for (const auto &pt : table_cloud->points) {
      theta_min = std::min(theta_min, std::atan2(pt.x, pt.z));
      theta_max = std::max(theta_max, std::atan2(pt.x, pt.z));
      phi_max = std::max(phi_max, std::atan2(pt.y, pt.z));
      min_x = std::min(min_x, pt.x);
      max_x = std::max(max_x, pt.x);
    }

    const int num_bins = 100;
    const float bin_width = (max_x - min_x) / static_cast<float>(num_bins);
    if (bin_width <= 0.0f) {
      error_out = "invalid_table_width min_x=" + std::to_string(min_x) +
                  " max_x=" + std::to_string(max_x);
      return false;
    }

    std::vector<float> bin_min_z(num_bins,
                                 std::numeric_limits<float>::max());
    std::vector<pcl::PointXYZ> bin_min_pt(num_bins);
    pcl::PointCloud<pcl::PointXYZ>::VectorType raw_edge_points;
    std::size_t rejected_left_fov = 0;
    std::size_t rejected_right_fov = 0;
    std::size_t rejected_top_fov = 0;
    std::size_t rejected_far = 0;

    for (const auto &pt : table_cloud->points) {
      int bin = static_cast<int>((pt.x - min_x) / bin_width);
      bin = std::clamp(bin, 0, num_bins - 1);
      if (pt.z < bin_min_z[bin]) {
        bin_min_z[bin] = pt.z;
        bin_min_pt[bin] = pt;
      }
    }

    for (int i = 0; i < num_bins; ++i) {
      if (bin_min_z[i] == std::numeric_limits<float>::max()) {
        continue;
      }

      const auto &pt = bin_min_pt[i];
      raw_edge_points.push_back(pt);
      const float theta = std::atan2(pt.x, pt.z);
      const float phi = std::atan2(pt.y, pt.z);

      // Quitamos puntos que probablemente vienen del corte artificial del FOV.
      const bool near_left_fov = theta < theta_min + 0.05f;
      const bool near_right_fov = theta > theta_max - 0.05f;
      const bool near_top_fov = phi > phi_max - 0.05f;
      const bool too_far = pt.z > 1.45f;
      const bool artificial =
          near_left_fov || near_right_fov || near_top_fov || too_far;
      if (!artificial) {
        edge_cloud->points.push_back(pt);
      } else {
        rejected_left_fov += near_left_fov ? 1 : 0;
        rejected_right_fov += near_right_fov ? 1 : 0;
        rejected_top_fov += near_top_fov ? 1 : 0;
        rejected_far += too_far ? 1 : 0;
      }
    }

    if (edge_cloud->size() < 10) {
      RCLCPP_WARN(
          get_logger(),
          "[table_drop_tf] filtro anti-FOV dejo pocos puntos de borde: "
          "filtrados=%zu crudos=%zu rechazados(left=%zu right=%zu top=%zu "
          "far=%zu) theta=[%.3f, %.3f] phi_max=%.3f. "
          "Usando candidatos crudos como fallback.",
          edge_cloud->size(), raw_edge_points.size(), rejected_left_fov,
          rejected_right_fov, rejected_top_fov, rejected_far, theta_min,
          theta_max, phi_max);

      edge_cloud->points = raw_edge_points;
      edge_cloud->width = static_cast<std::uint32_t>(edge_cloud->points.size());
      edge_cloud->height = 1;
      edge_cloud->is_dense = false;

      if (edge_cloud->size() < 10) {
        error_out = "insufficient_front_edge_points count=" +
                    std::to_string(edge_cloud->size());
        return false;
      }
    }

    edge_cloud->width = static_cast<std::uint32_t>(edge_cloud->points.size());
    edge_cloud->height = 1;
    edge_cloud->is_dense = false;

    return true;
  }

  Eigen::Vector3f centroid(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) const {
    Eigen::Vector4f c4;
    pcl::compute3DCentroid(*cloud, c4);
    return Eigen::Vector3f(c4.x(), c4.y(), c4.z());
  }

  Eigen::Vector3f centroidFromInliers(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      const pcl::PointIndices::Ptr &inliers) const {
    Eigen::Vector3f c(0.0f, 0.0f, 0.0f);
    for (const auto idx : inliers->indices) {
      const auto &pt = cloud->points[idx];
      c += Eigen::Vector3f(pt.x, pt.y, pt.z);
    }
    return c / static_cast<float>(inliers->indices.size());
  }

  void computeLocalBounds(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                          const Eigen::Vector3f &origin,
                          const Eigen::Vector3f &axis_x,
                          const Eigen::Vector3f &axis_y, float &x_min,
                          float &x_max, float &y_min, float &y_max) const {
    x_min = y_min = std::numeric_limits<float>::max();
    x_max = y_max = -std::numeric_limits<float>::max();

    for (const auto &pt : cloud->points) {
      Eigen::Vector3f p(pt.x, pt.y, pt.z);
      Eigen::Vector3f rel = p - origin;
      const float lx = rel.dot(axis_x);
      const float ly = rel.dot(axis_y);
      x_min = std::min(x_min, lx);
      x_max = std::max(x_max, lx);
      y_min = std::min(y_min, ly);
      y_max = std::max(y_max, ly);
    }
  }

  std::vector<Eigen::Vector2f> obstacleProjections(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
      const Eigen::Vector3f &origin, const Eigen::Vector3f &axis_x,
      const Eigen::Vector3f &axis_y, const Eigen::Vector3f &axis_z,
      float x_min, float x_max, float y_min, float y_max) const {
    std::vector<Eigen::Vector2f> obstacles;

    const float min_h =
        static_cast<float>(get_parameter("obstacle_min_height_m").as_double());
    const float max_h =
        static_cast<float>(get_parameter("obstacle_max_height_m").as_double());

    for (const auto &pt : cloud->points) {
      Eigen::Vector3f p(pt.x, pt.y, pt.z);
      Eigen::Vector3f rel = p - origin;
      const float h = rel.dot(axis_z);
      const float lx = rel.dot(axis_x);
      const float ly = rel.dot(axis_y);

      if (h > min_h && h < max_h && lx >= x_min && lx <= x_max &&
          ly >= y_min && ly <= y_max) {
        obstacles.emplace_back(lx, ly);
      }
    }

    return obstacles;
  }

  bool selectCandidate(float x_min, float x_max, float y_min, float y_max,
                       const std::vector<Eigen::Vector2f> &obstacles,
                       Candidate &best, std::string &error_out) const {
    const float side_margin =
        static_cast<float>(get_parameter("side_margin_m").as_double());
    const float front_margin =
        static_cast<float>(get_parameter("front_margin_m").as_double());
    const float back_margin =
        static_cast<float>(get_parameter("back_margin_m").as_double());
    const float obstacle_margin =
        static_cast<float>(get_parameter("obstacle_margin_m").as_double());
    const float preferred_x =
        static_cast<float>(get_parameter("preferred_x_m").as_double());

    const float ux_min = x_min + side_margin;
    const float ux_max = x_max - side_margin;
    const float uy_min = std::max(y_min + front_margin, front_margin);
    const float uy_max = y_max - back_margin;

    RCLCPP_DEBUG(get_logger(),
                 "[table_drop_tf] area util: ux=[%.3f, %.3f] uy=[%.3f, %.3f] "
                 "margenes side=%.3f front=%.3f back=%.3f obstacle=%.3f",
                 ux_min, ux_max, uy_min, uy_max, side_margin, front_margin,
                 back_margin, obstacle_margin);

    if (ux_min >= ux_max || uy_min >= uy_max) {
      error_out = "usable_table_area_too_small ux=[" + std::to_string(ux_min) +
                  "," + std::to_string(ux_max) + "] uy=[" +
                  std::to_string(uy_min) + "," + std::to_string(uy_max) + "]";
      return false;
    }

    const float preferred_y = (uy_min + uy_max) * 0.5f;
    const int grid_x = 15;
    const int grid_y = 15;

    bool found = false;
    for (int ix = 0; ix < grid_x; ++ix) {
      const float x = ux_min + (ux_max - ux_min) *
                                   (static_cast<float>(ix) /
                                    static_cast<float>(grid_x - 1));
      for (int iy = 0; iy < grid_y; ++iy) {
        const float y = uy_min + (uy_max - uy_min) *
                                     (static_cast<float>(iy) /
                                      static_cast<float>(grid_y - 1));

        float min_obstacle_distance = std::numeric_limits<float>::max();
        for (const auto &obs : obstacles) {
          const float d = std::hypot(x - obs.x(), y - obs.y());
          min_obstacle_distance = std::min(min_obstacle_distance, d);
        }

        if (min_obstacle_distance < obstacle_margin) {
          continue;
        }

        const float border_clearance =
            std::min({x - ux_min, ux_max - x, y - uy_min, uy_max - y});
        const float center_cost = std::hypot(x - preferred_x, y - preferred_y);
        const float obstacle_score =
            std::min(min_obstacle_distance, obstacle_margin * 2.0f);

        Candidate candidate;
        candidate.x = x;
        candidate.y = y;
        candidate.score = border_clearance + obstacle_score - 0.25f * center_cost;

        if (!found || candidate.score > best.score) {
          best = candidate;
          found = true;
        }
      }
    }

    if (!found) {
      error_out = "no_free_drop_candidate obstacles=" +
                  std::to_string(obstacles.size()) +
                  " obstacle_margin=" + std::to_string(obstacle_margin);
      return false;
    }

    RCLCPP_DEBUG(get_logger(),
                 "[table_drop_tf] mejor candidato: local=(%.3f, %.3f) "
                 "score=%.3f obstaculos=%zu",
                 best.x, best.y, best.score, obstacles.size());

    return true;
  }

  bool validateDropStartPose(const std_msgs::msg::Header &source_header,
                             const Eigen::Vector3f &start_point,
                             std::string &error_out) {
    if (source_header.frame_id.empty()) {
      error_out = "input_cloud_frame_empty";
      return false;
    }

    if (validation_frame_id_.empty()) {
      error_out = "validation_frame_id_empty";
      return false;
    }

    Eigen::Vector3f validation_point = start_point;
    if (validation_frame_id_ != source_header.frame_id) {
      try {
        const auto tf = tf_buffer_->lookupTransform(
            validation_frame_id_, source_header.frame_id, tf2::TimePointZero,
            200ms);

        Eigen::Quaternionf q(tf.transform.rotation.w, tf.transform.rotation.x,
                             tf.transform.rotation.y, tf.transform.rotation.z);
        q.normalize();
        const Eigen::Matrix3f rotation = q.toRotationMatrix();
        const Eigen::Vector3f translation(tf.transform.translation.x,
                                          tf.transform.translation.y,
                                          tf.transform.translation.z);
        validation_point = rotation * start_point + translation;
      } catch (const tf2::TransformException &ex) {
        error_out = "validation_transform_failed target=" +
                    validation_frame_id_ + " source=" + source_header.frame_id +
                    " error=" + ex.what();
        return false;
      }
    }

    const double x = validation_point.x();
    const double y = validation_point.y();
    const double z = validation_point.z();
    const double distance_xy = std::hypot(x, y);
    const double x_min = get_parameter("safe_x_min_m").as_double();
    const double x_max = get_parameter("safe_x_max_m").as_double();
    const double abs_y_max = get_parameter("safe_abs_y_max_m").as_double();
    const double z_min = get_parameter("safe_z_min_m").as_double();
    const double z_max = get_parameter("safe_z_max_m").as_double();
    const double dist_max = get_parameter("safe_dist_xy_max_m").as_double();

    if (x < x_min || x > x_max || std::fabs(y) > abs_y_max || z < z_min ||
        z > z_max || distance_xy > dist_max) {
      error_out = "unsafe_drop_start_pose frame=" + validation_frame_id_ +
                  " x=" + std::to_string(x) + " y=" + std::to_string(y) +
                  " z=" + std::to_string(z) +
                  " dist_xy=" + std::to_string(distance_xy);
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "[table_drop_tf] candidato inseguro en %s: x=%.3f y=%.3f z=%.3f "
          "dist_xy=%.3f limites x=[%.2f, %.2f] abs_y<=%.2f z=[%.2f, %.2f] "
          "dist<=%.2f",
          validation_frame_id_.c_str(), x, y, z, distance_xy, x_min, x_max,
          abs_y_max, z_min, z_max, dist_max);
      return false;
    }

    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[table_drop_tf] candidato seguro en %s: x=%.3f y=%.3f z=%.3f "
        "dist_xy=%.3f",
        validation_frame_id_.c_str(), x, y, z, distance_xy);
    return true;
  }

  bool transformGeometryToFixedFrame(
      const std_msgs::msg::Header &source_header,
      const Eigen::Vector3f &edge_origin,
      const Eigen::Vector3f &surface_point,
      const Eigen::Vector3f &start_point,
      const Eigen::Vector3f &axis_x,
      const Eigen::Vector3f &axis_y,
      const Eigen::Vector3f &axis_z,
      Eigen::Vector3f &fixed_edge_origin,
      Eigen::Vector3f &fixed_surface_point,
      Eigen::Vector3f &fixed_start_point,
      Eigen::Vector3f &fixed_axis_x,
      Eigen::Vector3f &fixed_axis_y,
      Eigen::Vector3f &fixed_axis_z,
      std::string &error_out) {
    if (source_header.frame_id.empty()) {
      error_out = "input_cloud_frame_empty";
      return false;
    }

    if (fixed_frame_id_.empty()) {
      error_out = "fixed_frame_id_empty";
      return false;
    }

    if (fixed_frame_id_ == source_header.frame_id) {
      fixed_edge_origin = edge_origin;
      fixed_surface_point = surface_point;
      fixed_start_point = start_point;
      fixed_axis_x = axis_x;
      fixed_axis_y = axis_y;
      fixed_axis_z = axis_z;
      return true;
    }

    try {
      const auto tf = tf_buffer_->lookupTransform(
          fixed_frame_id_, source_header.frame_id, tf2::TimePointZero, 200ms);

      Eigen::Quaternionf q(
          tf.transform.rotation.w, tf.transform.rotation.x,
          tf.transform.rotation.y, tf.transform.rotation.z);
      q.normalize();
      const Eigen::Matrix3f rotation = q.toRotationMatrix();
      const Eigen::Vector3f translation(
          tf.transform.translation.x, tf.transform.translation.y,
          tf.transform.translation.z);

      fixed_edge_origin = rotation * edge_origin + translation;
      fixed_surface_point = rotation * surface_point + translation;
      fixed_start_point = rotation * start_point + translation;
      fixed_axis_x = (rotation * axis_x).normalized();
      fixed_axis_y = (rotation * axis_y).normalized();
      fixed_axis_z = (rotation * axis_z).normalized();

      RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "[table_drop_tf] geometria transformada a %s desde %s",
          fixed_frame_id_.c_str(), source_header.frame_id.c_str());
      return true;
    } catch (const tf2::TransformException &ex) {
      error_out = "fixed_frame_transform_failed target=" + fixed_frame_id_ +
                  " source=" + source_header.frame_id + " error=" + ex.what();
      return false;
    }
  }

  void publishTransforms(const Eigen::Vector3f &edge_origin,
                         const Eigen::Vector3f &axis_x,
                         const Eigen::Vector3f &axis_y,
                         const Eigen::Vector3f &axis_z,
                         const Candidate &candidate) {
    Eigen::Matrix3f rotation;
    rotation.col(0) = axis_x;
    rotation.col(1) = axis_y;
    rotation.col(2) = axis_z;
    Eigen::Quaternionf q(rotation);
    q.normalize();

    geometry_msgs::msg::TransformStamped table_tf;
    table_tf.header.frame_id = fixed_frame_id_;
    table_tf.header.stamp = now();
    table_tf.child_frame_id = table_frame_id_;
    table_tf.transform.translation.x = edge_origin.x();
    table_tf.transform.translation.y = edge_origin.y();
    table_tf.transform.translation.z = edge_origin.z();
    table_tf.transform.rotation.x = q.x();
    table_tf.transform.rotation.y = q.y();
    table_tf.transform.rotation.z = q.z();
    table_tf.transform.rotation.w = q.w();

    geometry_msgs::msg::TransformStamped drop_tf;
    drop_tf.header.stamp = table_tf.header.stamp;
    drop_tf.header.frame_id = table_frame_id_;
    drop_tf.child_frame_id = drop_start_frame_id_;
    drop_tf.transform.translation.x = candidate.x;
    drop_tf.transform.translation.y = candidate.y;
    drop_tf.transform.translation.z = get_parameter("start_height_m").as_double();
    drop_tf.transform.rotation.w = 1.0;

    static_tf_broadcaster_->sendTransform(table_tf);
    static_tf_broadcaster_->sendTransform(drop_tf);

    RCLCPP_DEBUG(get_logger(),
                 "[table_drop_tf] TF emitidas parent=%s table=%s drop=%s "
                 "drop_local=(%.3f %.3f %.3f)",
                 fixed_frame_id_.c_str(), table_frame_id_.c_str(),
                 drop_start_frame_id_.c_str(), drop_tf.transform.translation.x,
                 drop_tf.transform.translation.y, drop_tf.transform.translation.z);
  }

  void clearMarkers() {
    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear;
    clear.header.frame_id =
        fixed_frame_id_.empty() ? validation_frame_id_ : fixed_frame_id_;
    clear.header.stamp = rclcpp::Time(0);
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);
    marker_pub_->publish(markers);
  }

  void resetDropPointsFile() {
    std::ofstream file(drop_points_file_, std::ios::trunc);
    if (!file.is_open()) {
      RCLCPP_WARN(get_logger(),
                  "[table_drop_tf] no se pudo reiniciar archivo de puntos: %s",
                  drop_points_file_.c_str());
      return;
    }

    file << "id,frame,x,y,z\n";
    RCLCPP_INFO(get_logger(),
                "[table_drop_tf] archivo de puntos reiniciado: %s",
                drop_points_file_.c_str());
  }

  void saveDropPoint(const Eigen::Vector3f &start_point) {
    SavedDropPoint saved;
    saved.id = next_drop_point_id_++;
    saved.point = start_point;
    saved_drop_points_.push_back(saved);

    std::ofstream file(drop_points_file_, std::ios::app);
    if (!file.is_open()) {
      RCLCPP_WARN(get_logger(),
                  "[table_drop_tf] no se pudo escribir archivo de puntos: %s",
                  drop_points_file_.c_str());
      return;
    }

    file << saved.id << "," << fixed_frame_id_ << "," << std::fixed
         << std::setprecision(6) << saved.point.x() << "," << saved.point.y()
         << "," << saved.point.z() << "\n";
    RCLCPP_INFO(get_logger(),
                "[table_drop_tf] punto drop_%d guardado en %s: frame=%s "
                "pos=(%.3f %.3f %.3f)",
                saved.id, drop_points_file_.c_str(), fixed_frame_id_.c_str(),
                saved.point.x(), saved.point.y(), saved.point.z());
  }

  void publishMarkers() {
    visualization_msgs::msg::MarkerArray markers;
    std_msgs::msg::Header marker_header;
    marker_header.frame_id = fixed_frame_id_;
    marker_header.stamp = rclcpp::Time(0);

    visualization_msgs::msg::Marker clear;
    clear.header = marker_header;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);

    for (const auto &saved : saved_drop_points_) {
      visualization_msgs::msg::Marker marker;
      marker.header = marker_header;
      marker.ns = "table_drop";
      marker.id = saved.id + 1;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position.x = saved.point.x();
      marker.pose.position.y = saved.point.y();
      marker.pose.position.z = saved.point.z();
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 0.055;
      marker.scale.y = 0.055;
      marker.scale.z = 0.055;
      marker.color.r = 0.0f;
      marker.color.g = 0.7f;
      marker.color.b = 1.0f;
      marker.color.a = 0.9f;
      marker.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(marker);
    }

    marker_pub_->publish(markers);
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "[table_drop_tf] esferas seguras publicadas: count=%zu frame=%s",
        saved_drop_points_.size(), fixed_frame_id_.c_str());
  }

  std::string pointcloud_topic_;
  std::string fixed_frame_id_;
  std::string validation_frame_id_;
  std::string table_frame_id_;
  std::string drop_start_frame_id_;
  std::string drop_points_file_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
  std::vector<SavedDropPoint> saved_drop_points_;
  int next_drop_point_id_{0};
  bool drop_point_saved_this_activation_{false};
  bool active_{false};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TableDropTfNode>());
  rclcpp::shutdown();
  return 0;
}
