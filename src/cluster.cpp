#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/iterators/PolygonIterator.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point;
typedef CGAL::Alpha_shape_vertex_base_2<K> Vb;
typedef CGAL::Alpha_shape_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
typedef CGAL::Alpha_shape_2<Delaunay, CGAL::Tag_false> AlphaShape;

using namespace grid_map;

static double euclideanDist(const Point& a, const Point& b) {
  double dx = a.x() - b.x();
  double dy = a.y() - b.y();
  return std::sqrt(dx * dx + dy * dy);
}

// DBSCAN clustering algorithm 
// TODO: Use a more efficient clustering algorithm 
static std::vector<std::vector<Point>> clusterPoints(const std::vector<Point>& points, double eps, int min_pts) {
  std::vector<std::vector<Point>> clusters;
  std::vector<bool> visited(points.size(), false);
  std::vector<bool> clustered(points.size(), false);

  for (size_t i = 0; i < points.size(); ++i) {
    if (visited[i]) continue;
    visited[i] = true;

    std::vector<size_t> neighbors;
    for (size_t j = 0; j < points.size(); ++j) {
      if (euclideanDist(points[i], points[j]) <= eps)
        neighbors.push_back(j);
    }

    if (neighbors.size() < static_cast<size_t>(min_pts))
      continue;

    std::vector<Point> cluster;
    std::vector<size_t> seed = neighbors;
    while (!seed.empty()) {
      size_t idx = seed.back();
      seed.pop_back();

      if (!visited[idx]) {
        visited[idx] = true;

        std::vector<size_t> new_neighbors;
        for (size_t k = 0; k < points.size(); ++k) {
          if (euclideanDist(points[idx], points[k]) <= eps)
            new_neighbors.push_back(k);
        }

        if (new_neighbors.size() >= static_cast<size_t>(min_pts))
          seed.insert(seed.end(), new_neighbors.begin(), new_neighbors.end());
      }

      if (!clustered[idx]) {
        cluster.push_back(points[idx]);
        clustered[idx] = true;
      }
    }

    if (!cluster.empty())
      clusters.push_back(cluster);
  }

  return clusters;
}

float packRGB(uint8_t r, uint8_t g, uint8_t b) {
  uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                  static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
  float rgb_float;
  std::memcpy(&rgb_float, &rgb, sizeof(float));
  return rgb_float;
}

void markAlphaShapeObstacleClusters(GridMap& map, const std::string& layer, double alpha_value, rclcpp::Logger logger, double clustering_eps = 0.5, int min_pts = 3)
{
  std::vector<Point> input_points;

  grid_map::Matrix& layer_data = map[layer];

  // Collect all obstacle cells
  for (GridMapIterator it(map); !it.isPastEnd(); ++it) {

    const std::size_t i = it.getLinearIndex();


    if (std::isnan(layer_data(i))) continue;
    if (layer_data(i) < 1.0) continue;

    RCLCPP_INFO(logger, "Processing point");

    Position pos;
    map.getPosition(*it, pos);
    input_points.emplace_back(pos.x(), pos.y());
  }

  if (input_points.size() < 4) return;

  auto clusters = clusterPoints(input_points, clustering_eps, min_pts);

  for (const auto& cluster_pts : clusters) {
    if (cluster_pts.size() < 2) continue;

    AlphaShape alpha_shape(cluster_pts.begin(), cluster_pts.end(), alpha_value, AlphaShape::GENERAL);
    alpha_shape.set_mode(AlphaShape::REGULARIZED);

    // Collect boundary points (no need for pairs)
    std::vector<grid_map::Position> boundary_pts;
    for (auto it = alpha_shape.alpha_shape_edges_begin(); it != alpha_shape.alpha_shape_edges_end(); ++it) {
      auto seg = alpha_shape.segment(*it);
      boundary_pts.emplace_back(
        CGAL::to_double(seg.source().x()), CGAL::to_double(seg.source().y()));
      boundary_pts.emplace_back(
        CGAL::to_double(seg.target().x()), CGAL::to_double(seg.target().y()));
    }

    // Remove duplicates
    std::sort(boundary_pts.begin(), boundary_pts.end(), [](const Position& a, const Position& b) {
      return a.x() == b.x() ? a.y() < b.y() : a.x() < b.x();
    });
    boundary_pts.erase(std::unique(boundary_pts.begin(), boundary_pts.end()), boundary_pts.end());

    if (boundary_pts.size() < 3) continue;

    // Build and fill polygon
    Polygon poly;
    for (const auto& pt : boundary_pts)
      poly.addVertex(Position(pt.x(), pt.y()));

    for (PolygonIterator it(map, poly); !it.isPastEnd(); ++it) {
      map.at(layer, *it) = 1000;
    }
  }
}
