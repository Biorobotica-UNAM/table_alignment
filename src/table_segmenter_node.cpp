#include <cmath>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/centroid.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>

class TableSegmenterNode : public rclcpp::Node {
public:
  TableSegmenterNode() : Node("table_segmenter_node") {
    RCLCPP_INFO(this->get_logger(), "[table_segmenter] Nodo iniciado");

    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/hardware/camera/points", 10,
        std::bind(&TableSegmenterNode::point_cloud_callback, this,
                  std::placeholders::_1));

    // Publicador de nube de puntos
    publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "camera/table_points", 10);

    // Publicador de ángulo
    angle_publisher_ = this->create_publisher<std_msgs::msg::Float64>(
        "camera/table_alignment", 10);

    // Publicadores para debug visual
    marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "camera/table_normal_marker", 10);
    
    marker_array_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "camera/table_axes_markers", 10);

    plane_marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "camera/table_plane_marker", 10);

    // Publicador del vector de desalineación
    misalignment_vector_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "camera/table_misalignment_vector", 10);

    // Publicador de distancia de desalineación
    misalignment_distance_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>(
        "camera/table_misalignment_distance", 10);

    // TF broadcaster para visualizar ejes
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
  }

private:
  const double tolerance_deg = 1.0; // tolerancia

  void
  point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Recibida nube: frame='%s' size_msg=%u width=%u height=%u",
                 msg->header.frame_id.c_str(), msg->width * msg->height,
                 msg->width, msg->height);

    // msg a PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Nube convertida a PCL: puntos=%zu",
                 cloud->size());

    if (cloud->empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "[table_segmenter] Nube vacía, se omite procesamiento");
      return;
    }

    // ROI
    pcl::PassThrough<pcl::PointXYZ> pass;

    // Eje Z
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.5);
    pass.filter(*cloud);
    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Filtro Z aplicado: puntos=%zu",
                 cloud->size());

    // Eje Y
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-0.8, 0.0);
    pass.filter(*cloud);
    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Filtro Y aplicado: puntos=%zu",
                 cloud->size());

    // Eje X
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-0.5, 0.5);
    pass.filter(*cloud);
    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Filtro X aplicado: puntos=%zu",
                 cloud->size());

    if (cloud->size() < 500) {
      RCLCPP_WARN(this->get_logger(),
                  "[table_segmenter] Muy pocos puntos tras ROI (%zu), no se segmenta",
                  cloud->size());
      return;
    }

    // RANSAC
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01); // 1 cm
    seg.setMaxIterations(200);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] RANSAC terminado: inliers=%zu coeffs=%zu",
                 inliers->indices.size(), coefficients->values.size());

    if (inliers->indices.empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "[table_segmenter] No se encontró un plano");
      return;
    }

    // coeficientes del plano: ax + by + cz + d = 0
    // vector normal (a, b, c).
    float a = coefficients->values[0];
    float b = coefficients->values[1];
    float c = coefficients->values[2];
    // float d = coefficients->values[3];

    // Calcular el eje Y del plano proyectando el eje Y del mundo sobre el plano
    float norm_tmp = std::sqrt(a * a + b * b + c * c);
    float nx_tmp = a / norm_tmp;
    float ny_tmp = b / norm_tmp;
    float nz_tmp = c / norm_tmp;

    Eigen::Vector3f normal_world(nx_tmp, ny_tmp, nz_tmp);
    Eigen::Vector3f ref_y_world(0.0f, 1.0f, 0.0f); // +Y hacia el usuario en el frame de la camara
    Eigen::Vector3f plane_y = (ref_y_world - normal_world * ref_y_world.dot(normal_world)).normalized();

    // Ángulo (radianes) de desalineación alrededor de Z: 0 cuando plane_y == (0,1,0)
    double angle_rad = std::atan2(plane_y[0], plane_y[1]);
    double angle_deg = angle_rad * (180.0 / M_PI);

    // Normalizar a [-90,90]
    if (angle_deg > 90)
      angle_deg -= 180;
    if (angle_deg < -90)
      angle_deg += 180;

    // tolerancia
    if (std::abs(angle_deg) < tolerance_deg) {
      angle_deg = 0.0;
      angle_rad = 0.0;
    }

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Plano: a=%.6f b=%.6f angle_rad=%.6f angle_deg=%.3f tol=%.3f %s",
                 a, b, angle_rad, angle_deg, tolerance_deg,
                 (angle_deg == 0.0 ? "ALINEADA" : "DESALINEADA"));

    // logs
    auto angle_msg = std_msgs::msg::Float64();
    angle_msg.data = angle_deg;
    angle_publisher_->publish(angle_msg);

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Publicado camera/table_alignment = %.6f rad",
                 angle_msg.data);

    RCLCPP_INFO(this->get_logger(),
                "[table_segmenter] Giro necesario: %.2f grados [%s]",
                angle_deg, (angle_deg == 0.0 ? "ALINEADA" : "DESALINEADA"));

    // publicar puntos
    pcl::PointCloud<pcl::PointXYZ>::Ptr table_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*table_cloud);

    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*table_cloud, output_msg);
    output_msg.header = msg->header;
    publisher_->publish(output_msg);

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Publicado camera/table_points: puntos=%zu",
                 table_cloud->size());

    // ============ DEBUG VISUAL ============
    // Calcular centroide del plano
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*table_cloud, centroid);

    // Vector normal normalizado
    float norm = std::sqrt(a * a + b * b + c * c);
    float nx = a / norm;
    float ny = b / norm;
    float nz = c / norm;

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Normal: nx=%.3f ny=%.3f nz=%.3f | Centroide: x=%.3f y=%.3f z=%.3f",
                 nx, ny, nz, centroid[0], centroid[1], centroid[2]);

    // 1. Publicar la normal como una flecha (Marker)
    publish_normal_marker(msg->header, centroid, nx, ny, nz);

    // 2. Publicar los ejes del plano como Markers (3 flechas: X, Y, Z)
    publish_axes_markers(msg->header, centroid, nx, ny, nz);

    // 3. Publicar TF del plano (para visualizar frame)
    publish_plane_tf(msg->header, centroid, nx, ny, nz);

    // 4. Publicar el plano como un polígono visualizado
    publish_plane_polygon_marker(msg->header, table_cloud);

    // 5. Publicar vector de desalineación
    publish_misalignment_vector(msg->header, centroid, angle_rad, nx, ny, nz);

    // 6. Publicar distancia de desalineación
    publish_misalignment_distance(msg->header, centroid, angle_deg);
  }

  void publish_normal_marker(const std_msgs::msg::Header& header, 
                            const Eigen::Vector4f& centroid,
                            float nx, float ny, float nz) {
    auto marker = visualization_msgs::msg::Marker();
    marker.header = header;
    marker.ns = "table_normal";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Punto inicial (centroide)
    marker.points.resize(2);
    marker.points[0].x = centroid[0];
    marker.points[0].y = centroid[1];
    marker.points[0].z = centroid[2];

    // Punto final (centroide + normal * 0.1)
    marker.points[1].x = centroid[0] + nx * 0.1;
    marker.points[1].y = centroid[1] + ny * 0.1;
    marker.points[1].z = centroid[2] + nz * 0.1;

    marker.scale.x = 0.01;  // shaft diameter
    marker.scale.y = 0.02;  // head diameter
    marker.scale.z = 0.03;  // head length

    marker.color.r = 1.0f;  // Rojo para la normal
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.8f;

    marker.lifetime = rclcpp::Duration(0, 500000000);  // 0.5 segundos

    marker_publisher_->publish(marker);
  }

  void publish_axes_markers(const std_msgs::msg::Header& header,
                           const Eigen::Vector4f& centroid,
                           float nx, float ny, float nz) {
    // Crear dos vectores ortogonales al normal para formar los ejes
    Eigen::Vector3f normal(nx, ny, nz);
    
    // Encontrar un vector ortogonal
    Eigen::Vector3f axis1;
    if (std::abs(nx) < 0.9f) {
      axis1 = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    } else {
      axis1 = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    }

    // Producto cruzado para obtener eje ortogonal
    Eigen::Vector3f axis2 = normal.cross(axis1).normalized();
    axis1 = axis2.cross(normal).normalized();

    auto marker_array = visualization_msgs::msg::MarkerArray();

    // Eje 1 (Verde - X del plano)
    {
      auto marker = visualization_msgs::msg::Marker();
      marker.header = header;
      marker.ns = "table_axes";
      marker.id = 1;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.points.resize(2);
      marker.points[0].x = centroid[0];
      marker.points[0].y = centroid[1];
      marker.points[0].z = centroid[2];

      marker.points[1].x = centroid[0] + axis1[0] * 0.15;
      marker.points[1].y = centroid[1] + axis1[1] * 0.15;
      marker.points[1].z = centroid[2] + axis1[2] * 0.15;

      marker.scale.x = 0.008;
      marker.scale.y = 0.015;
      marker.scale.z = 0.025;

      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 0.8f;

      marker.lifetime = rclcpp::Duration(0, 500000000);

      marker_array.markers.push_back(marker);
    }

    // Eje 2 (Azul - Y del plano)
    {
      auto marker = visualization_msgs::msg::Marker();
      marker.header = header;
      marker.ns = "table_axes";
      marker.id = 2;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.points.resize(2);
      marker.points[0].x = centroid[0];
      marker.points[0].y = centroid[1];
      marker.points[0].z = centroid[2];

      marker.points[1].x = centroid[0] + axis2[0] * 0.15;
      marker.points[1].y = centroid[1] + axis2[1] * 0.15;
      marker.points[1].z = centroid[2] + axis2[2] * 0.15;

      marker.scale.x = 0.008;
      marker.scale.y = 0.015;
      marker.scale.z = 0.025;

      marker.color.r = 0.0f;
      marker.color.g = 0.0f;
      marker.color.b = 1.0f;
      marker.color.a = 0.8f;

      marker.lifetime = rclcpp::Duration(0, 500000000);

      marker_array.markers.push_back(marker);
    }

    // Eje 3 (Rojo - Normal Z)
    {
      auto marker = visualization_msgs::msg::Marker();
      marker.header = header;
      marker.ns = "table_axes";
      marker.id = 3;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.points.resize(2);
      marker.points[0].x = centroid[0];
      marker.points[0].y = centroid[1];
      marker.points[0].z = centroid[2];

      marker.points[1].x = centroid[0] + nx * 0.15;
      marker.points[1].y = centroid[1] + ny * 0.15;
      marker.points[1].z = centroid[2] + nz * 0.15;

      marker.scale.x = 0.008;
      marker.scale.y = 0.015;
      marker.scale.z = 0.025;

      marker.color.r = 1.0f;
      marker.color.g = 0.0f;
      marker.color.b = 0.0f;
      marker.color.a = 0.8f;

      marker.lifetime = rclcpp::Duration(0, 500000000);

      marker_array.markers.push_back(marker);
    }

    marker_array_publisher_->publish(marker_array);
  }

  void publish_plane_tf(const std_msgs::msg::Header& header,
                       const Eigen::Vector4f& centroid,
                       float nx, float ny, float nz) {
    // Crear vectores ortogonales
    Eigen::Vector3f normal(nx, ny, nz);
    Eigen::Vector3f axis1;

    if (std::abs(nx) < 0.9f) {
      axis1 = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    } else {
      axis1 = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
    }

    Eigen::Vector3f axis2 = normal.cross(axis1).normalized();
    axis1 = axis2.cross(normal).normalized();

    // Crear quaternion para representar la rotación
    Eigen::Matrix3f rotation;
    rotation.col(0) = axis1;
    rotation.col(1) = axis2;
    rotation.col(2) = normal;

    Eigen::Quaternionf quat(rotation);

    geometry_msgs::msg::TransformStamped tf;
    tf.header = header;
    tf.child_frame_id = "table_plane_frame";

    tf.transform.translation.x = centroid[0];
    tf.transform.translation.y = centroid[1];
    tf.transform.translation.z = centroid[2];

    tf.transform.rotation.x = quat.x();
    tf.transform.rotation.y = quat.y();
    tf.transform.rotation.z = quat.z();
    tf.transform.rotation.w = quat.w();

    tf_broadcaster_->sendTransform(tf);
  }

  void publish_plane_polygon_marker(const std_msgs::msg::Header& header,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    auto marker = visualization_msgs::msg::Marker();
    marker.header = header;
    marker.ns = "table_plane";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Agregar todos los puntos dela nube como mesh
    for (const auto& pt : cloud->points) {
      geometry_msgs::msg::Point p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      marker.points.push_back(p);
    }

    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;

    marker.color.r = 0.5f;
    marker.color.g = 0.5f;
    marker.color.b = 1.0f;
    marker.color.a = 0.3f;

    marker.lifetime = rclcpp::Duration(0, 500000000);

    plane_marker_publisher_->publish(marker);
  }

  void publish_misalignment_vector(const std_msgs::msg::Header& header,
                                   const Eigen::Vector4f& centroid,
                                   float angle_rad,
                                   float nx, float ny, float nz) {
    // Crear vector que representa la dirección de rotación necesaria
    // El vector perpendicular al normal en el plano
    Eigen::Vector3f normal(nx, ny, nz);
    
    // Vector perpendicular al normal (en el plano XY del mundo, aproximadamente)
    Eigen::Vector3f perp;
    if (std::abs(nz) < 0.9f) {
      perp = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    } else {
      perp = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
    }
    
    perp = (perp - normal * normal.dot(perp)).normalized();

    auto marker = visualization_msgs::msg::Marker();
    marker.header = header;
    marker.ns = "table_misalignment";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Punto inicial
    marker.points.resize(2);
    marker.points[0].x = centroid[0];
    marker.points[0].y = centroid[1];
    marker.points[0].z = centroid[2];

    // Punto final - escalar según magnitud del ángulo
    float scale = std::abs(angle_rad) * 0.3f; // 0.3 metros por radian
    marker.points[1].x = centroid[0] + perp[0] * scale;
    marker.points[1].y = centroid[1] + perp[1] * scale;
    marker.points[1].z = centroid[2] + perp[2] * scale;

    marker.scale.x = 0.01;  // shaft diameter
    marker.scale.y = 0.025; // head diameter
    marker.scale.z = 0.035; // head length

    marker.color.r = 1.0f;  // Amarillo para desalineación
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.9f;

    marker.lifetime = rclcpp::Duration(0, 500000000);

    misalignment_vector_publisher_->publish(marker);
  }

  void publish_misalignment_distance(const std_msgs::msg::Header& header,
                                     const Eigen::Vector4f& centroid,
                                     float angle_deg) {
    auto marker = visualization_msgs::msg::Marker();
    marker.header = header;
    marker.ns = "table_distance";
    marker.id = 0;
    marker.type  = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = centroid[0];
    marker.pose.position.y = centroid[1];
    marker.pose.position.z = centroid[2] + 0.15; // Mostrar encima del centroide

    marker.scale.z = 0.05; // Tamaño del texto

    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0f;

    // Formatear texto con la distancia de desalineación
    char buffer[100];
    snprintf(buffer, sizeof(buffer), "Misalignment: %.2f°", angle_deg);
    marker.text = buffer;

    marker.lifetime = rclcpp::Duration(0, 500000000);

    misalignment_distance_publisher_->publish(marker);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr angle_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_marker_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr misalignment_vector_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr misalignment_distance_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TableSegmenterNode>());
  rclcpp::shutdown();
}