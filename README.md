# Isolation-Forest
c++20 header only implementation of anomaly detection via isolation forest.

example:
```cpp
// data
std::vector<double> data = { 1.2, 1.8, 0.99, 10.4, 2.0, 1.86, 0.899, 1.3, 0.901, 1.345,
                             1.25, 1.9, 0.96, 1.48, 1.97, 1.867, 1.9, 1.48, 0.001, 1.45, };

// forest
IsolationForest::Forest<double> forest{ 25, 100 };
forest.build(data.begin(), data.end());

// calculate outlier score
std::vector<double> outlier_score;
for (const auto& val : data) {
  outlier_score.emplace_back(forest.score(val, data.size()));
}

// assumed outlier is
const auto max_element_iter = std::max_element(outlier_score.begin(), outlier_score.end());
const auto max_element_index = std::distance(outlier_score.begin(), max_element_iter);
std::cout << "suspected outlier is " << data[max_element_index] << '\n'; // <- should be 10.4
```
