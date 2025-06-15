#include <grid_map_core/GridMap.hpp>
#include <grid_map_core/iterators/GridMapIterator.hpp>
#include <algorithm>       // for std::max / std::min
#include <limits>          // for std::numeric_limits<float>::max()

/**
 * @brief Applies a 3×3 morphological closing (dilation then erosion)
 *        on the specified layer of `map`. 
 *
 * @param map    Reference to a grid_map::GridMap that already contains `layer`.
 *               After this call, `map[layer]` is overwritten by the closed result.
 * @param layer  Name of the layer to process. That layer is assumed to have
 *               values 0.0f (free) or 1000.0f (occupied).
 */
void morphologicalClose3x3(grid_map::GridMap& map, const std::string& layer)
{
  // 1) Extract a read‐only reference to the input layer. Internally, GridMap
  //    stores each layer as an Eigen::MatrixXf, so operator[] gives us that.
  const grid_map::Matrix& input = map[layer];

  // 2) We'll build two temporary matrices of the same size: one for dilation, one for erosion.
  const int rows = input.rows();
  const int cols = input.cols();

  // Temporary storage for dilation result:
  grid_map::Matrix dilated(rows, cols);

  // // --- DILATION (3×3 max filter) ---
  // // For each cell (i,j), look at all neighbors (i+di, j+dj) with di,dj ∈ {-1,0,1}.
  // // dilated(i,j) = max{ input(n_i, n_j) : valid neighbors }.
  // for (int i = 0; i < rows; ++i) {
  //   for (int j = 0; j < cols; ++j) {
  //     float maxVal = 0.0f; // because occupied=1000, free=0 → max starts at 0
  //     for (int di = -1; di <= 1; ++di) {
  //       for (int dj = -1; dj <= 1; ++dj) {
  //         int ni = i + di;
  //         int nj = j + dj;
  //         if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
  //           // compare with each neighbor's value
  //           maxVal = std::max(maxVal, input(ni, nj));
  //         }
  //       }
  //     }
  //     dilated(i, j) = maxVal;
  //   }
  // }

  // --- DILATION (4×4 max filter) ---
  // For each cell (i,j), look at all neighbors (i+di, j+dj) with di,dj ∈ {-2,-1,0,1}.
  // dilated(i,j) = max{ input(n_i, n_j) : valid neighbors }.
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float maxVal = 0.0f; // because occupied=1000, free=0 → max starts at 0
      for (int di = -2; di <= 1; ++di) {
        for (int dj = -2; dj <= 1; ++dj) {
          int ni = i + di;
          int nj = j + dj;
          if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
            // compare with each neighbor's value
            maxVal = std::max(maxVal, input(ni, nj));
          }
        }
      }
      dilated(i, j) = maxVal;
    }
  }

  // 3) Prepare storage for the erosion step:
  grid_map::Matrix eroded(rows, cols);

  // --- EROSION (3×3 min filter on the dilated result) ---
  // For each cell (i,j), look at all neighbors in dilated, take the minimum.
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      // We know that dilated values are either 0.0f or 1000.0f. Still,
      // initialize with a very large float so the first min is correct.
      float minVal = std::numeric_limits<float>::max();
      for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
          int ni = i + di;
          int nj = j + dj;
          if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
            minVal = std::min(minVal, dilated(ni, nj));
          }
        }
      }
      eroded(i, j) = minVal;
    }
  }

  // 4) Overwrite the original layer with the closed result:
  map[layer] = eroded;
}
