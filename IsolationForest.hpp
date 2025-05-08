#pragma once
#include <type_traits>
#include <concepts>
#include <vector>
#include <span>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <assert.h>

/**
* implementation of isolation forest anomaly detector algorithm
**/
namespace IsolationForest {

	/**
* interfaces, concepts and traits for isolation forest
**/
	namespace Interface {
		/**
		* \brief concept of a node in isolation tree
		**/
		template<class NODE>
		concept INode = std::is_integral_v<typename NODE::size_type> &&
			            std::is_floating_point_v<typename NODE::value_type> &&
			            requires (NODE node) {
			std::is_same_v<decltype(node.split_value), typename NODE::value_type>; // node split value
			std::is_same_v<decltype(node.left), typename NODE::size_type>;         // how many "left" partitions the node underwent
			std::is_same_v<decltype(node.right), typename NODE::size_type>;        // how many "right" partitions the node underwent
		};

		/**
		* \brief concept of an isolation tree
		**/
		template<class TREE, class ITER>
		concept ITree = INode<typename TREE::node_type> &&
			            std::is_same_v<typename TREE::size_type, typename TREE::node_type::size_type>&&
			            std::is_same_v<typename TREE::value_type, typename TREE::node_type::value_type>&&
			            std::is_same_v<typename TREE::tree_type, std::vector<typename TREE::node_type>>&&
			            requires (TREE tree, ITER it, TREE::value_type value, TREE::size_type size) {

			/**
			* \brief return root node id
			* @param {size, out} tree root node id
			**/
			{ tree.root_id() } -> std::same_as<typename TREE::size_type>;

			/**
			* \brief build tree from data given by range iterators to a given collection
			* @param {forward_iterator, in} iterator for first element in collection
			* @param {forward_iterator, in} iterator for last element in collection
			**/
			{ tree.build(it, it) } -> std::same_as<void>;

			/**
			* \brief return the path length of a given value
			* @param {value_type,   in}  value
			* @param {std::size_t,  in}  node
			* @param {std::size_t,  in}  depth
			* @param {value_type,   out} path length
			**/
			{ tree.path_length(value, size, size) } -> std::same_as<typename TREE::value_type>;
		};

		/**
		* \brief concept of an isolation forest
		**/
		template<class FOREST, class ITER>
		concept IForest = ITree<typename FOREST::tree_type, ITER>&&
			              std::is_same_v<typename FOREST::size_type, typename FOREST::tree_type::size_type>&&
			              std::is_same_v<typename FOREST::value_type, typename FOREST::tree_type::value_type>&&
			              requires (FOREST forest, ITER it, FOREST::value_type value, std::size_t size) {

			/**
			* \brief build forest from data given by range iterators to a given collection
			* @param {forward_iterator, in} iterator for first element in collection
			* @param {forward_iterator, in} iterator for last element in collection
			**/
			{ forest.build(it, it) } -> std::same_as<void>;

			/**
			* \brief calculate given value "outlier" score
			* @param {value_type,   in}  value
			* @param {std::size_t,  in}  data size
			* @param {value_type,   out} outlier score
			**/
			{ forest.score(value, size) } -> std::same_as<typename FOREST::value_type>;
		};
	};

	/**
	* implementation of isolation forest interface
	**/
	namespace Implementation {

		/**
		* \brief Interface::INode implementation
		**/
		template<typename T, typename I = std::int64_t>
			requires(std::is_floating_point_v<T>)
		struct INode {
			using size_type = I;
			using value_type = T;

			value_type split_value{};
			size_type left{ -1 };
			size_type right{ -1 };
		};
		static_assert(Interface::INode<INode<double>>);

		/**
		* \brief Interface::ITree implementation
		**/
		template<Interface::INode Node>
		struct ITree {
			using node_type = Node;
			using size_type = typename Node::size_type;
			using value_type = typename Node::value_type;
			using tree_type = std::vector<node_type>;

			/**
			* \brief construct ITree with predefined maximal depth
			* @param {size_type, in} maximal depth
			**/
			explicit ITree(size_type _max_depth) : max_depth(_max_depth) {
				this->tree.reserve(static_cast<std::size_t>((_max_depth > 100) ? _max_depth * (_max_depth / 100) : _max_depth));
			}

			// ITree is regular
			ITree(const ITree&) = default;
			ITree(ITree&&) = default;
			ITree& operator =(const ITree&) = delete;
			ITree& operator =(ITree&&) = delete;
			~ITree() = default;

			/**
			* \brief return root node id
			* @param {size_t, out} tree root node id
			**/
			constexpr size_type root_id() const {
				return static_cast<size_type>(this->tree.size() - 1);
			};

			/**
			* \brief build tree from data given by range iterators to a given collection
			*        notice that this function is recursive.
			* @param {forward_iterator, in} iterator for first element in collection
			* @param {forward_iterator, in} iterator for last element in collection
			**/
			template<std::forward_iterator It>
				requires(std::is_same_v<value_type, typename std::decay_t<decltype(*std::declval<It>())>>)
			constexpr void build(It first, It last) {
				std::vector<value_type> data(first, last);
				this->build_recursively(data, size_type{}, static_cast<size_type>(std::distance(first, last)), size_type{});
			};

			/**
			* \brief return the path length of a given value
			*        notice that this function is recursive.
			* @param {value_type, in}  value
			* @param {size_type,  in}  node index
			* @param {size_type,  in}  node depth
			* @param {value_type, out} path length
			**/
			constexpr value_type path_length(const value_type& value, const size_type node_index, const size_type node_depth) const {
				assert(node_index > 0);
				const node_type& node{ this->tree[static_cast<std::size_t>(node_index)] };

				if (node.left >= 0 || node.right >= 0) [[likely]] {
					if (value < node.split_value && node.left >= 0) {
						return (this->path_length(value, node.left, node_depth + 1));
					}
					else if (node.right >= 0) {
						return (this->path_length(value, node.right, node_depth + 1));
					}
				}

				return static_cast<value_type>(node_depth - 1);
			};

			// internals
			private:
				// properties
				tree_type tree;
				const size_type max_depth;

				/**
				* \brief recursively build tree
				**/
				constexpr size_type build_recursively(std::span<value_type> data,
					                                  const size_type left, const size_type right,
					                                  const size_type depth) {
					using iter_t = std::span<value_type>::iterator;

					if (left >= right || depth >= this->max_depth || right == 0) [[unlikely]] {
						this->tree.push_back(node_type{});
					}
					else {
						const std::size_t anchor_index{ static_cast<std::size_t>(left + rand() % (right - left)) };
						const value_type& anchor{ data[anchor_index] };
						const iter_t anchor_iter{ std::partition(data.begin() + left, data.begin() + right,
																[&anchor](const value_type& v) -> bool { return (v < anchor); }) };
						const size_type mid{ static_cast<size_type>(std::distance(data.begin(), anchor_iter)) };
						const node_type node{
							.split_value = anchor,
			                .left = this->build_recursively(data, left, mid, depth + 1),
			                .right = this->build_recursively(data, mid, right, depth + 1)
						};

						this->tree.push_back(node);
					}

					// output
					return this->root_id();
				}
		};
		static_assert(Interface::ITree<ITree<INode<double>>, std::vector<double>::iterator>);

		/**
		* \brief Interface::IForest implementation
		**/
		template<Interface::INode Node>
		struct IForest {
			using tree_type = ITree<Node>;
			using size_type = typename Node::size_type;
			using value_type = typename Node::value_type;

			constexpr explicit IForest(const std::size_t num_trees, const size_type max_depth) : trees(num_trees, tree_type{ max_depth }) {}

			// IForest is regular
			IForest() = delete;
			IForest(const IForest&) = default;
			IForest(IForest&&) = default;
			IForest& operator =(const IForest&) = default;
			IForest& operator =(IForest&&) = default;
			~IForest() = default;

			/**
			* \brief build forest from data given by range iterators to a given collection
			* @param {forward_iterator, in} iterator for first element in collection
			* @param {forward_iterator, in} iterator for last element in collection
			**/
			template<std::forward_iterator It>
				requires(std::is_same_v<value_type, typename std::decay_t<decltype(*std::declval<It>())>>)
			constexpr void build(It first, It last) {
				std::vector<value_type> data(first, last);

				for (auto& tree : this->trees) {
					this->shuffle(data);
					tree.build(data.begin(), data.end());
				}
			}

			/**
			* \brief calculate given value "outlier" score
			* @param {value_type, in}  value
			* @param {size_type,  in}  data size
			* @param {value_type, out} outlier score
			**/
			constexpr value_type score(const value_type value, const size_type size) {
				value_type avg_path_len{};

				for (const auto& tree : this->trees) {
					avg_path_len += tree.path_length(value, tree.root_id(), 0);
				}
				avg_path_len /= static_cast<value_type>(this->trees.size());

				return static_cast<value_type>(std::pow(static_cast<value_type>(2.0), avg_path_len / this->calc_depth(size)));
			}

			// internals
			private:
				// properties
				std::vector<tree_type> trees;

				/**
				* \brief estimated expected path length for given data size
				**/
				constexpr value_type calc_depth(const size_type size) const {
					if (size <= 1) {
						return value_type{};
					}

					return ((static_cast<value_type>(2.0) * (std::log(static_cast<value_type>(size - 1)) + static_cast<value_type>(0.5772156649))) -
						    (static_cast<value_type>(2.0) * (static_cast<value_type>(size - 1)) / static_cast<value_type>(size)));
				}

				constexpr void shuffle(std::vector<value_type>& vec) const {
					for (std::size_t i{ vec.size() - 1 }; i > 0; --i) {
						std::iter_swap(vec.begin() + i, vec.begin() + static_cast<std::size_t>(rand()) % (i + 1));
					}
				}
		};
		static_assert(Interface::IForest<IForest<INode<double>>, std::vector<double>::iterator>);
	};

	// API
	template<typename T>
		requires(std::is_floating_point_v<T>)
	using Forest = Implementation::IForest<Implementation::INode<T>>;
};
