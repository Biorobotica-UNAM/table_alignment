#include <Eigen/Dense>
#include <cmath>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <mutex>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64.hpp>
#include <vision_interfaces/srv/get_table_alignment.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class TableSegmenterNode : public rclcpp::Node {
public:
  TableSegmenterNode() : Node("table_segmenter_node") {
    RCLCPP_INFO(this->get_logger(), "[table_segmenter] Nodo iniciado");

    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/hardware/camera/points", 10,
        std::bind(&TableSegmenterNode::point_cloud_callback, this,
                  std::placeholders::_1));

    get_alignment_srv_ =
        this->create_service<vision_interfaces::srv::GetTableAlignment>(
            "/vision/get_table_alignment",
            std::bind(&TableSegmenterNode::get_alignment_callback, this,
                      std::placeholders::_1, std::placeholders::_2));

    // Publicador de nube de puntos
    publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "camera/table_points", 10);

    // Publicador de ángulo
    angle_publisher_ = this->create_publisher<std_msgs::msg::Float64>(
        "camera/table_alignment", 10);

    // Publicadores para debug visual
    marker_array_publisher_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "camera/table_axes_markers", 10);
  }

private:
  const double tolerance_deg = 0.5; // tolerancia
  double last_angle_ =
      0.0; // guarda el último ángulo para no saltar entre bordes

  void
  point_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
    latest_cloud_msg_ = msg;
  }

  void get_alignment_callback(
      const std::shared_ptr<
          vision_interfaces::srv::GetTableAlignment::Request> request,
      std::shared_ptr<vision_interfaces::srv::GetTableAlignment::Response>
          response) {
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
    {
      std::lock_guard<std::mutex> lock(latest_cloud_mutex_);
      cloud_msg = latest_cloud_msg_;
    }

    if (!cloud_msg) {
      response->success = false;
      response->message = "no_pointcloud_available";
      response->angle = 0.0;
      RCLCPP_WARN(this->get_logger(),
                  "[table_segmenter] Solicitud de align sin nube disponible");
      return;
    }

    double angle_deg = 0.0;
    std::string error;
    const bool ok = process_cloud(*cloud_msg, request->is_round, angle_deg, error);

    response->success = ok;
    response->angle = ok ? angle_deg : 0.0;
    response->message = ok ? "ok" : error;
  }

  bool process_cloud(const sensor_msgs::msg::PointCloud2 &input_msg,
                     bool is_round,
                     double &angle_deg_out, std::string &error_out) {
    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Recibida nube: frame='%s' size_msg=%u "
                 "width=%u height=%u",
                 input_msg.header.frame_id.c_str(),
                 input_msg.width * input_msg.height, input_msg.width,
                 input_msg.height);

    // msg a PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input_msg, *cloud);

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Nube convertida a PCL: puntos=%zu",
                 cloud->size());

    if (cloud->empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "[table_segmenter] Nube vacía, se omite procesamiento");
      error_out = "empty_cloud";
      return false;
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
    pass.setFilterLimits(-0.8, 0.8);
    pass.filter(*cloud);
    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Filtro Y aplicado: puntos=%zu",
                 cloud->size());

    // Eje X
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-1.5, 1.5);
    pass.filter(*cloud);
    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Filtro X aplicado: puntos=%zu",
                 cloud->size());

    if (cloud->size() < 500) {
      RCLCPP_WARN(
          this->get_logger(),
          "[table_segmenter] Muy pocos puntos tras ROI (%zu), no se segmenta",
          cloud->size());
      error_out = "insufficient_points_after_roi";
      return false;
    }

    // 1. RANSAC para el plano de la mesa
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(
        pcl::SACMODEL_PERPENDICULAR_PLANE); // Forzar plano horizontal
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setAxis(Eigen::Vector3f(0.0, 1.0, 0.0)); // Eje Y de la cámara
    seg.setEpsAngle(45.0 * (M_PI / 180.0)); // Tolerancia de 45 grados (permite
                                            // tilt de cámara, rechaza paredes)
    seg.setDistanceThreshold(0.02);         // 2 cm
    seg.setMaxIterations(200);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] RANSAC plano terminado: inliers=%zu",
                 inliers->indices.size());

    if (inliers->indices.empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "[table_segmenter] No se encontró la superficie de la mesa");
      error_out = "plane_not_found";
      return false;
    }

    // 2. Extraer puntos de la mesa
    pcl::PointCloud<pcl::PointXYZ>::Ptr table_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*table_cloud);

    if (is_round) {
      if (table_cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "[table_segmenter] Nube de la mesa vacia");
        error_out = "empty_table_cloud";
        return false;
      }

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*table_cloud, centroid);

      // En el frame óptico de la cámara: X es la desviación horizontal y Z es el eje adelante (profundidad).
      double angle_rad = std::atan2(centroid[0], centroid[2]);
      double angle_deg = angle_rad * (180.0 / M_PI);

      if (std::abs(angle_deg) < tolerance_deg) {
        angle_deg = 0.0;
      }

      angle_deg_out = angle_deg;

      RCLCPP_INFO(this->get_logger(),
                  "[table_segmenter] [Mesa Redonda] Centroide=(%.3f, %.3f, %.3f), Giro necesario: %.2f grados",
                  centroid[0], centroid[1], centroid[2], angle_deg);

      // Publicar el ángulo
      auto angle_msg = std_msgs::msg::Float64();
      angle_msg.data = angle_deg;
      angle_publisher_->publish(angle_msg);

      // Publicar nube de puntos de la mesa
      sensor_msgs::msg::PointCloud2 output_msg;
      pcl::toROSMsg(*table_cloud, output_msg);
      output_msg.header = input_msg.header;
      publisher_->publish(output_msg);

      // Publicar marcadores visuales para RViz
      Eigen::Vector3f normal(0.0, -1.0, 0.0); // Aproximacion de la normal del plano (hacia arriba)
      Eigen::Vector3f edge_dir(1.0, 0.0, 0.0); // Eje horizontal ficticio
      publish_axes_markers(input_msg.header, centroid, normal, edge_dir);

      error_out = "";
      return true;
    }

    // 3. Encontrar el borde frontal de la mesa
    // Primero, calcular los límites del campo de visión (FOV) para ignorar los
    // cortes artificiales
    float theta_min = std::numeric_limits<float>::max();
    float theta_max = -std::numeric_limits<float>::max();
    float phi_max =
        -std::numeric_limits<float>::max(); // phi máximo es el borde inferior
                                            // de la imagen

    for (const auto &pt : table_cloud->points) {
      float theta = std::atan2(pt.x, pt.z);
      float phi = std::atan2(pt.y, pt.z);
      if (theta < theta_min)
        theta_min = theta;
      if (theta > theta_max)
        theta_max = theta;
      if (phi > phi_max)
        phi_max = phi;
    }

    // Dividir en bins a lo largo del eje X para encontrar el frente a lo ancho
    // de la mesa
    float min_x = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    for (const auto &pt : table_cloud->points) {
      if (pt.x < min_x)
        min_x = pt.x;
      if (pt.x > max_x)
        max_x = pt.x;
    }

    int num_bins = 100; // Mayor resolución para los bins
    float bin_width = (max_x - min_x) / num_bins;
    std::vector<float> bin_min_z(num_bins, std::numeric_limits<float>::max());
    std::vector<pcl::PointXYZ> bin_min_pt(num_bins);

    for (const auto &pt : table_cloud->points) {
      if (bin_width <= 0)
        break;
      int bin = (pt.x - min_x) / bin_width;
      if (bin < 0)
        bin = 0;
      if (bin >= num_bins)
        bin = num_bins - 1;

      if (pt.z < bin_min_z[bin]) {
        bin_min_z[bin] = pt.z;
        bin_min_pt[bin] = pt;
      }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < num_bins; ++i) {
      if (bin_min_z[i] < std::numeric_limits<float>::max()) {
        pcl::PointXYZ pt = bin_min_pt[i];
        float theta = std::atan2(pt.x, pt.z);
        float phi = std::atan2(pt.y, pt.z);

        // Rechazar puntos que pertenezcan a cortes artificiales del FOV o del
        // filtro Z
        bool is_artificial = (theta < theta_min + 0.05f) ||
                             (theta > theta_max - 0.05f) ||
                             (phi > phi_max - 0.05f) || (pt.z > 1.45f);

        if (!is_artificial) {
          edge_cloud->points.push_back(pt);
        }
      }
    }

    if (edge_cloud->size() < 10) {
      RCLCPP_WARN(this->get_logger(), "[table_segmenter] No se encontraron "
                                      "suficientes puntos en el borde frontal");
      error_out = "edge_not_found";
      return false;
    }

    // 4. RANSAC para encontrar la línea 3D sobre el borde frontal
    pcl::ModelCoefficients::Ptr line_coefficients1(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr line_inliers1(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> line_seg;

    line_seg.setOptimizeCoefficients(true);
    line_seg.setModelType(pcl::SACMODEL_LINE);
    line_seg.setMethodType(pcl::SAC_RANSAC);
    line_seg.setDistanceThreshold(0.02); // 2 cm
    line_seg.setMaxIterations(200);
    line_seg.setInputCloud(edge_cloud);
    line_seg.segment(*line_inliers1, *line_coefficients1);

    if (line_inliers1->indices.empty()) {
      RCLCPP_WARN(
          this->get_logger(),
          "[table_segmenter] No se pudo ajustar una línea al borde frontal");
      error_out = "line_not_found";
      return false;
    }

    // Calcular el ángulo del primer borde detectado
    float lx1 = line_coefficients1->values[3];
    float ly1 = line_coefficients1->values[4];
    float lz1 = line_coefficients1->values[5];
    if (lx1 < 0) {
      lx1 = -lx1;
      ly1 = -ly1;
      lz1 = -lz1;
    }
    double angle1 = std::atan2(lz1, lx1) * (180.0 / M_PI);

    // Intentar buscar una SEGUNDA línea (por si hay mesas en forma de L)
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_remaining(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract_edge;
    extract_edge.setInputCloud(edge_cloud);
    extract_edge.setIndices(line_inliers1);
    extract_edge.setNegative(true);
    extract_edge.filter(*edge_cloud_remaining);

    double chosen_angle = angle1;
    pcl::ModelCoefficients::Ptr final_line_coefficients = line_coefficients1;
    pcl::PointIndices::Ptr final_line_inliers = line_inliers1;

    if (edge_cloud_remaining->size() >= 10) {
      pcl::ModelCoefficients::Ptr line_coefficients2(
          new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr line_inliers2(new pcl::PointIndices);
      line_seg.setInputCloud(edge_cloud_remaining);
      line_seg.segment(*line_inliers2, *line_coefficients2);

      if (!line_inliers2->indices.empty() &&
          line_inliers2->indices.size() > 5) {
        float lx2 = line_coefficients2->values[3];
        float ly2 = line_coefficients2->values[4];
        float lz2 = line_coefficients2->values[5];
        if (lx2 < 0) {
          lx2 = -lx2;
          ly2 = -ly2;
          lz2 = -lz2;
        }
        double angle2 = std::atan2(lz2, lx2) * (180.0 / M_PI);

        // Elegir la línea cuyo ángulo se parezca más al último ángulo conocido
        // Esto evita saltos bruscos cuando hay dos bordes formando una esquina
        // (mesa en L)
        if (std::abs(angle2 - last_angle_) < std::abs(angle1 - last_angle_)) {
          chosen_angle = angle2;
          final_line_coefficients = line_coefficients2;
          final_line_inliers = line_inliers2;
        }
      }
    }

    // Actualizar nuestro last_angle_
    last_angle_ = chosen_angle;

    // Usar la línea final elegida
    float lx = final_line_coefficients->values[3];
    float ly = final_line_coefficients->values[4];
    float lz = final_line_coefficients->values[5];
    if (lx < 0) {
      lx = -lx;
      ly = -ly;
      lz = -lz;
    }

    // Calcular ángulo de la línea respecto al eje X (visto desde arriba en el
    // plano X-Z) Cuando la mesa está perfectamente de frente, el borde es
    // horizontal (Lz = 0, Lx > 0) -> angle = 0
    double angle_rad = std::atan2(lz, lx);
    double angle_deg = angle_rad * (180.0 / M_PI);

    // Normalizar si es necesario (ya debería estar entre -180 y 180 por atan2)
    // Aplicar tolerancia de desalineación
    if (std::abs(angle_deg) < tolerance_deg) {
      angle_deg = 0.0;
      angle_rad = 0.0;
    }

    angle_deg_out = angle_deg;

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Borde: vector=(%.3f, %.3f, %.3f) "
                 "angle_rad=%.6f angle_deg=%.3f tol=%.3f %s",
                 lx, ly, lz, angle_rad, angle_deg, tolerance_deg,
                 (angle_deg == 0.0 ? "ALINEADA" : "DESALINEADA"));

    // Publicar ángulo
    auto angle_msg = std_msgs::msg::Float64();
    angle_msg.data = angle_deg;
    angle_publisher_->publish(angle_msg);

    RCLCPP_DEBUG(
        this->get_logger(),
        "[table_segmenter] Publicado camera/table_alignment = %.6f grados",
        angle_msg.data);

    RCLCPP_INFO(this->get_logger(),
                "[table_segmenter] Giro necesario: %.2f grados [%s]", angle_deg,
                (angle_deg == 0.0 ? "ALINEADA" : "DESALINEADA"));

    // Publicar nube de puntos de la mesa
    sensor_msgs::msg::PointCloud2 output_msg;
    pcl::toROSMsg(*table_cloud, output_msg);
    output_msg.header = input_msg.header;
    publisher_->publish(output_msg);

    RCLCPP_DEBUG(this->get_logger(),
                 "[table_segmenter] Publicado camera/table_points: puntos=%zu",
                 table_cloud->size());

    // ============ DEBUG VISUAL ============
    Eigen::Vector4f centroid;
    centroid[0] = final_line_coefficients->values[0];
    centroid[1] = final_line_coefficients->values[1];
    centroid[2] = final_line_coefficients->values[2];
    centroid[3] = 1.0f;

    Eigen::Vector3f normal(coefficients->values[0], coefficients->values[1],
                           coefficients->values[2]);
    Eigen::Vector3f edge_dir(lx, ly, lz);

    // Publicar los ejes del borde frontal
    publish_axes_markers(input_msg.header, centroid, normal, edge_dir);

    error_out = "";
    return true;
  }

  void publish_axes_markers(const std_msgs::msg::Header &header,
                            const Eigen::Vector4f &centroid,
                            const Eigen::Vector3f &normal,
                            const Eigen::Vector3f &edge_dir) {
    // Eje X (Verde): Dirección del borde de la mesa
    Eigen::Vector3f axis_x = edge_dir.normalized();

    // Eje Z (Rojo): Normal de la superficie de la mesa
    Eigen::Vector3f axis_z = normal.normalized();

    // Eje Y (Azul): Ortogonal (hacia el interior de la mesa o hacia afuera)
    Eigen::Vector3f axis_y = axis_z.cross(axis_x).normalized();

    // Asegurar ortogonalidad perfecta ajustando el Z
    axis_z = axis_x.cross(axis_y).normalized();

    auto marker_array = visualization_msgs::msg::MarkerArray();

    // Función lambda para no repetir código de markers
    auto add_arrow_marker = [&](int id, const Eigen::Vector3f &dir, float r,
                                float g, float b) {
      auto marker = visualization_msgs::msg::Marker();
      marker.header = header;
      marker.ns = "table_axes";
      marker.id = id;
      marker.type = visualization_msgs::msg::Marker::ARROW;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.points.resize(2);
      marker.points[0].x = centroid[0];
      marker.points[0].y = centroid[1];
      marker.points[0].z = centroid[2];

      marker.points[1].x = centroid[0] + dir[0] * 0.15;
      marker.points[1].y = centroid[1] + dir[1] * 0.15;
      marker.points[1].z = centroid[2] + dir[2] * 0.15;

      marker.scale.x = 0.008; // shaft
      marker.scale.y = 0.015; // head width
      marker.scale.z = 0.025; // head length

      marker.color.r = r;
      marker.color.g = g;
      marker.color.b = b;
      marker.color.a = 0.8f;
      marker.lifetime = rclcpp::Duration(0, 500000000); // 0.5 sec

      marker_array.markers.push_back(marker);
    };

    // Eje X (Borde, Verde)
    add_arrow_marker(1, axis_x, 0.0f, 1.0f, 0.0f);
    // Eje Y (Adentro, Azul)
    add_arrow_marker(2, axis_y, 0.0f, 0.0f, 1.0f);
    // Eje Z (Arriba/Normal, Rojo)
    add_arrow_marker(3, axis_z, 1.0f, 0.0f, 0.0f);

    marker_array_publisher_->publish(marker_array);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Service<vision_interfaces::srv::GetTableAlignment>::SharedPtr
      get_alignment_srv_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr angle_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_array_publisher_;
  sensor_msgs::msg::PointCloud2::SharedPtr latest_cloud_msg_;
  std::mutex latest_cloud_mutex_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TableSegmenterNode>());
  rclcpp::shutdown();
}