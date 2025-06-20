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
void morphologicalClose(grid_map::GridMap& map,
                        const std::string& layer,
                        int dilation_radius,
                        int erosion_radius)
{
  const grid_map::Matrix& input = map[layer];
  const int rows = input.rows();
  const int cols = input.cols();

  grid_map::Matrix dilated(rows, cols);

  // --- DILATION ---
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float maxVal = 0.0f;
      for (int di = -dilation_radius; di <= dilation_radius; ++di) {
        for (int dj = -dilation_radius; dj <= dilation_radius; ++dj) {
          int ni = i + di;
          int nj = j + dj;
          if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
            maxVal = std::max(maxVal, input(ni, nj));
          }
        }
      }
      dilated(i, j) = maxVal;
    }
  }

  grid_map::Matrix eroded(rows, cols);

  // --- EROSION ---
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      float minVal = std::numeric_limits<float>::max();
      for (int di = -erosion_radius; di <= erosion_radius; ++di) {
        for (int dj = -erosion_radius; dj <= erosion_radius; ++dj) {
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

  // Store result back into map
  map[layer] = eroded;
}
