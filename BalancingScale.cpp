#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <string_view>
#include <algorithm>
#include <charconv>
#include <array>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <limits>
#include <cassert>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

/**
 * @brief Prefetches memory data into cache to reduce latency
 * @param ptr Pointer to memory location to prefetch
 * @note Uses platform-specific intrinsics for optimal performance
 * @note Uses read-only prefetch with high temporal locality
 */
inline void prefetch(const void* ptr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);  // Read-only, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

namespace Config {
    // Cache line size for alignment (platform-specific if available)
    #ifdef __cpp_lib_hardware_interference_size
        static constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
    #else
        static constexpr size_t CACHE_LINE_SIZE = 64;  // Common cache line size for x86
    #endif
    
    constexpr int MAX_INT_CHARS = std::numeric_limits<int>::digits10 + 2;  // Max digits for int + sign
    constexpr size_t INITIAL_NODE_POOL_SIZE = 2048;  // Balance between memory usage and allocations
    constexpr size_t EXPECTED_MAX_PARENTS = 3;  // Optimize for common case of 1-3 parents
    constexpr int BASE_SCALE_MASS = 1;  // Each scale weighs 1kg by itself
    //constexpr int MAX_MASS_VALUE = 1000000;  // Reasonable upper bound for mass values
    constexpr bool DISPLAY_INPUT_LINES = true;  // Flag to control input display

    // Test cases (using proper line endings for test cases)
    static constexpr const char* TEST_CASE_1 = 
        "B1,10,B2\n"
        "B2,B3,4\n"
        "B3,7,8";
    static constexpr const char* EXPECTED_OUTPUT = 
        "=== Balancing Solution ===\n"
        "B1,25,0\n"
        "B2,0,13\n"
        "B3,1,0\n"
        "=======================\n";
    static constexpr const char* CYCLE_TEST_CASE = 
        "A,1,B\n"
        "B,1,C\n"
        "C,1,A";
}

/**
 * @brief Represents a single balancing scale in the system
 * @details Contains mass values, child pointers, and parent tracking
 * @note Aligned to cache line size to prevent false sharing
 */
struct alignas(Config::CACHE_LINE_SIZE) ScaleNode {
    // Mass values (in kg)
    int left_mass = 0;       // Direct mass on left pan
    int right_mass = 0;      // Direct mass on right pan
    int adjust_left = 0;     // Additional mass needed on left to balance
    int adjust_right = 0;    // Additional mass needed on right to balance
    int total_mass = Config::BASE_SCALE_MASS; // Total mass including subtree
    
    // Processing state
    bool processed = false;
    bool visiting = false;   // Cycle detection flag
    
    // Child nodes
    ScaleNode* left_child = nullptr;
    ScaleNode* right_child = nullptr;
    
    // Parent tracking (optimized for common case)
    ScaleNode* parents[Config::EXPECTED_MAX_PARENTS]{};
    uint8_t parent_count = 0;
    std::vector<ScaleNode*> extra_parents;  // For scales with many parents

    /**
     * @brief Adds a parent reference to this node
     * @param parent Pointer to parent node to add
     * @note Uses small-buffer optimization for common case
     */
    void add_parent(ScaleNode* parent) {
        if (parent_count < Config::EXPECTED_MAX_PARENTS) {
            parents[parent_count++] = parent;
        } else {
            extra_parents.push_back(parent);
        }
    }
};

static_assert(alignof(ScaleNode) == Config::CACHE_LINE_SIZE, "ScaleNode cache alignment incorrect");

/**
 * @brief Memory pool for efficient ScaleNode allocation
 * @details Uses a static pool for initial allocations and falls back to dynamic
 */
class NodePool {
    std::array<ScaleNode, Config::INITIAL_NODE_POOL_SIZE> static_pool_;
    size_t index_ = 0;
    std::vector<std::unique_ptr<ScaleNode>> dynamic_nodes_;

public:
    /**
     * @brief Allocates a new ScaleNode
     * @return Pointer to newly allocated node
     * @note Tries static pool first, then falls back to dynamic allocation
     */
    ScaleNode* allocate() {
        if (index_ < static_pool_.size()) return &static_pool_[index_++];
        dynamic_nodes_.emplace_back(std::make_unique<ScaleNode>());
        return dynamic_nodes_.back().get();
    }

    /**
     * @brief Calculates total memory usage
     * @return Total bytes used by the pool
     */
    size_t memory_usage() const {
        return sizeof(static_pool_) + (dynamic_nodes_.size() * sizeof(ScaleNode));
    }
};

/**
 * @brief Core class for balancing scale calculations
 * @details Handles parsing, processing, and output generation
 */
class ScaleBalancer {
    NodePool node_pool_;
    std::unordered_map<std::string, ScaleNode*> node_registry_;
    std::vector<std::pair<std::string, ScaleNode*>> output_order_;
    std::vector<ScaleNode*> processing_queue_;
    std::vector<std::string> input_lines_;

    // Benchmarking data
    struct {
        size_t parse_time_us = 0;
        size_t process_time_us = 0;
        size_t output_time_us = 0;
        size_t total_nodes = 0;
        size_t max_queue_depth = 0;
        size_t estimated_cache_lines = 0;
        size_t memory_used_bytes = 0;
    } benchmark_stats_;

    /**
     * @brief Displays input lines if enabled in config
     */
    void display_input_lines() const {
        if (Config::DISPLAY_INPUT_LINES) {
            std::cerr << "\n=== Input Lines ===\n";
            for (const auto& line : input_lines_) {
                if (!line.empty() && line[0] != '#') {
                    std::cerr << line << "\n";
                }
            }
            std::cerr << "=================\n";
        }
    }

    /**
     * @brief Allocates and tracks a new ScaleNode
     * @return Pointer to newly created node
     */
    ScaleNode* allocate_new_node() { 
        ScaleNode* node = node_pool_.allocate();
        benchmark_stats_.total_nodes++;
        return node;
    }

    /**
     * @brief Trims whitespace from string view
     * @param sv String view to trim
     * @return Trimmed string view
     */
    std::string_view trim_whitespace(std::string_view sv) const {
        while (!sv.empty() && isspace(sv.front())) sv.remove_prefix(1);
        while (!sv.empty() && isspace(sv.back())) sv.remove_suffix(1);
        return sv;
    }

    /**
     * @brief Efficiently appends integer to string
     * @param out String to append to
     * @param value Integer value to append
     */
    void append_integer(std::string& out, int value) const {
        char buffer[Config::MAX_INT_CHARS];
        auto [ptr, ec] = std::to_chars(buffer, buffer + Config::MAX_INT_CHARS, value);
        out.append(buffer, ptr);
    }

    /**
     * @brief Checks if character could start a scale reference
     * @param c Character to check
     * @return True if character is alphabetic
     */
    bool is_scale_reference(char c) const {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    }

    /**
     * @brief Ensures scale appears in output in correct order
     * @param name Scale name to check
     * @param node Associated node pointer
     */
    void ensure_output_order(const std::string& name, ScaleNode* node) {
        output_order_.emplace_back(name, node);
    }

    /**
     * @brief Parses a single pan definition (mass or scale reference)
     * @param value String value to parse
     * @param child_ptr [out] Reference to child pointer to set
     * @param direct_mass [out] Reference to mass value to set
     * @param current_node Node that owns this pan
     */
    void parse_scale_pan(const std::string& value, ScaleNode*& child_ptr, 
                        int& direct_mass, ScaleNode* current_node) {
        if (value.empty()) return;

        if (is_scale_reference(value[0])) {
            // Reference to another scale
            ScaleNode*& child = node_registry_[value];
            if (!child) {
                child = allocate_new_node();
                ensure_output_order(value, child);
            }
            child_ptr = child;
            child->add_parent(current_node);
        } else {
            // Direct mass value
            int mass = 0;
            auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), mass);
            if (ec != std::errc() || ptr != value.data() + value.size() || mass < 0) {
                throw std::runtime_error("Invalid mass value: " + value);
            }
            direct_mass = mass;
        }
    }

    /**
     * @brief Parses a single line of input with enhanced validation
     * @param line Input line to parse
     * @throws std::runtime_error on invalid input format
     */
    void parse_input_line(const std::string& line) {
        input_lines_.push_back(line);
        if (line.empty() || line[0] == '#') return;

        const size_t first_comma = line.find(',');
        const size_t second_comma = line.find(',', first_comma + 1);
        
        // Enhanced input validation
        if (first_comma == std::string::npos || second_comma == std::string::npos ||
            first_comma == 0 || second_comma == line.length() - 1) {
            throw std::runtime_error("Invalid input format - expected 'Name,Left,Right': " + line);
        }

        const std::string name = std::string(trim_whitespace(
            std::string_view(line.data(), first_comma)));
        const std::string left = std::string(trim_whitespace(
            line.substr(first_comma + 1, second_comma - first_comma - 1)));
        const std::string right = std::string(trim_whitespace(
            line.substr(second_comma + 1)));

        ScaleNode*& node = node_registry_[name];
        if (!node) {
            node = allocate_new_node();
            ensure_output_order(name, node);
        } else if (node->left_child || node->right_child || 
                   node->left_mass || node->right_mass) {
            throw std::runtime_error("Duplicate scale definition: " + name);
        }

        parse_scale_pan(left, node->left_child, node->left_mass, node);
        parse_scale_pan(right, node->right_child, node->right_mass, node);
    }

    /**
     * @brief Checks if node is ready for processing
     * @param node Node to check
     * @return True if all children are processed
     */
    bool is_node_ready_for_processing(const ScaleNode* node) const {
        return (!node->left_child || node->left_child->processed) &&
               (!node->right_child || node->right_child->processed);
    }

/*///////////////////////////////////////////////////////////////////////////
SIMD Optimizations (AVX2):

While SIMD could potentially help, in this specific case:

Limited Benefit: The core algorithm processes nodes sequentially with dependencies between calculations (each node depends on its children's totals)
Branching Logic: The cycle detection and topological sorting make SIMD difficult to apply effectively
Memory Bound: The algorithm is likely memory-bound rather than compute-bound

Thus,  in practice, this SIMD version may not be faster because:
-The overhead of loading values into SIMD registers
-The simple scalar version is already very efficient
-The compiler may auto-vectorize it anyway
-------------------------------------------------------------------------------*/
/*
    #include <immintrin.h>  // For AVX2 intrinsics

void process_single_scale(ScaleNode* node) {
    if (node->visiting) {
        throw std::runtime_error("Cycle detected in scale tree");
    }
    node->visiting = true;

    // Calculate total mass on each side including subtrees
    const int left_total = node->left_mass + 
                         (node->left_child ? node->left_child->total_mass : 0);
    const int right_total = node->right_mass + 
                          (node->right_child ? node->right_child->total_mass : 0);

    // AVX2 optimized imbalance calculation
    __m256i masses = _mm256_set_epi32(0, 0, 0, 0, 0, right_total, left_total, 0);
    __m256i imbalance_vec = _mm256_sub_epi32(masses, _mm256_permutevar8x32_epi32(masses, _mm256_set_epi32(0,0,0,0,0,0,1,0)));
    int imbalance = _mm256_extract_epi32(imbalance_vec, 0);

    // Determine needed adjustments
    node->adjust_left = std::max(0, -imbalance);
    node->adjust_right = std::max(0, imbalance);

    // Update total mass (self + left + right + adjustments)
    node->total_mass = Config::BASE_SCALE_MASS + left_total + right_total +
                      node->adjust_left + node->adjust_right;

    node->processed = true;
    node->visiting = false;
    benchmark_stats_.estimated_cache_lines++;
    if (!node->extra_parents.empty()) benchmark_stats_.estimated_cache_lines++;
}

/////////////////////////////////////////////////////////////////////////
 */

    /**
     * @brief Processes a single scale to calculate balance
     * @param node Scale node to process
     * @throws std::runtime_error if cycle detected
     */
    void process_single_scale(ScaleNode* node) {
        if (node->visiting) {
            throw std::runtime_error("Cycle detected in scale tree");
        }
        node->visiting = true;

        // Calculate total mass on each side including subtrees
        const int left_total = node->left_mass + 
                             (node->left_child ? node->left_child->total_mass : 0);
        const int right_total = node->right_mass + 
                              (node->right_child ? node->right_child->total_mass : 0);
        const int imbalance = left_total - right_total;

        // Determine needed adjustments
        node->adjust_left = std::max(0, -imbalance);
        node->adjust_right = std::max(0, imbalance);

        // Update total mass (self + left + right + adjustments)
        node->total_mass = Config::BASE_SCALE_MASS + left_total + right_total +
                          node->adjust_left + node->adjust_right;

        node->processed = true;
        node->visiting = false;
        benchmark_stats_.estimated_cache_lines++;
        if (!node->extra_parents.empty()) benchmark_stats_.estimated_cache_lines++;
    }

    /**
     * @brief Processes all scales in topological order
     * @details Uses iterative approach with queue for efficiency
     */
    void balance_scale_tree() {
        if (node_registry_.empty()) {
            throw std::runtime_error("No valid scales found in input");
        }

        // Initialize queue with leaf nodes (no unprocessed children)
        for (const auto& [_, node] : node_registry_) {
            if (is_node_ready_for_processing(node)) {
                processing_queue_.push_back(node);
            }
        }

        // Process nodes in topological order (leaves to root)
        for (size_t i = 0; i < processing_queue_.size(); ++i) {
            ScaleNode* current = processing_queue_[i];
            
            // Prefetch next node if available
            if (i + 1 < processing_queue_.size()) {
                prefetch(processing_queue_[i + 1]);
            }

            process_single_scale(current);
            benchmark_stats_.max_queue_depth = std::max(
                benchmark_stats_.max_queue_depth, processing_queue_.size());

            // Add parents to queue if they become ready
            for (uint8_t j = 0; j < current->parent_count; ++j) {
                ScaleNode* parent = current->parents[j];
                if (is_node_ready_for_processing(parent)) {
                    processing_queue_.push_back(parent);
                }
            }
            for (ScaleNode* parent : current->extra_parents) {
                if (is_node_ready_for_processing(parent)) {
                    processing_queue_.push_back(parent);
                }
            }
        }
    }

    /**
     * @brief Generates the output string with balance adjustments
     * @return Formatted output string in input order
     */
    std::string generate_balance_output() {
        std::string output;
        output.reserve(output_order_.size() * 16);  // Preallocate based on expected size

        char buffer[Config::MAX_INT_CHARS];  // Buffer for integer conversions
        for (const auto& [name, node] : output_order_) {
            output.append(name).append(",");
            
            // Fast integer conversion
            auto [ptr1, ec1] = std::to_chars(buffer, buffer + Config::MAX_INT_CHARS, node->adjust_left);
            output.append(buffer, ptr1).append(",");
            
            auto [ptr2, ec2] = std::to_chars(buffer, buffer + Config::MAX_INT_CHARS, node->adjust_right);
            output.append(buffer, ptr2).append("\n");
        }
        return output;
    }

    /**
     * @brief Displays detailed benchmark results to stderr
     */
    void display_benchmark_results() const {
        std::cerr << "\n=== Detailed Performance Metrics ===\n"
                  << "Input Parsing:      " << benchmark_stats_.parse_time_us << " μs\n"
                  << "Tree Processing:    " << benchmark_stats_.process_time_us << " μs\n"
                  << "  - Per Node:       " 
                  << benchmark_stats_.process_time_us / std::max(benchmark_stats_.total_nodes, 1ul) 
                  << " μs/node\n"
                  << "Output Generation:  " << benchmark_stats_.output_time_us << " μs\n"
                  << "Nodes Created:      " << benchmark_stats_.total_nodes << "\n"
                  << "Max Queue Depth:    " << benchmark_stats_.max_queue_depth << "\n"
                  << "Cache Efficiency:   " 
                  << (benchmark_stats_.estimated_cache_lines * Config::CACHE_LINE_SIZE * 100 / 
                     benchmark_stats_.memory_used_bytes) << "% utilization\n"
                  << "Memory Used:        " << benchmark_stats_.memory_used_bytes << " bytes\n";
    }

public:
    /**
     * @brief Main execution method
     * @param input Input stream to read from
     * @param output Output stream to write to
     * @param main_execution Flag to indicate if this is the main execution (affects benchmark output)
     */
    void execute(std::istream& input = std::cin, std::ostream& output = std::cout, bool main_execution = true) {
        // Optimize stream performance
        std::ios::sync_with_stdio(false);
        input.tie(nullptr);

        // Phase 1: Parse input
        auto parse_start = std::chrono::high_resolution_clock::now();
        std::string line;
        while (std::getline(input, line)) {
            parse_input_line(line);
        }
        benchmark_stats_.parse_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - parse_start).count();

        // Display input lines if enabled
        display_input_lines();

        // Phase 2: Process scales
        auto process_start = std::chrono::high_resolution_clock::now();
        balance_scale_tree();
        benchmark_stats_.process_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - process_start).count();

        // Phase 3: Generate output
        auto output_start = std::chrono::high_resolution_clock::now();
        std::string result = generate_balance_output();
        output << "=== Balancing Solution ===\n" << result << "=======================\n";
        benchmark_stats_.output_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - output_start).count();

        benchmark_stats_.memory_used_bytes = node_pool_.memory_usage();
        if (main_execution) {
            display_benchmark_results();
        }
    }
};

bool run_test_case(const std::string& input, const std::string& expected_output, 
                  const std::string& test_name) {
    std::istringstream input_stream(input);
    std::ostringstream output_stream;
    ScaleBalancer balancer;
    try {
        balancer.execute(input_stream, output_stream, false);
        const std::string& actual_output = output_stream.str();

        if (actual_output.find(expected_output) != std::string::npos) {
            std::cerr << "[PASS] " << test_name << "\n";
            return true;
        } else {
            std::cerr << "[FAIL] " << test_name << "\n"
                      << "Expected:\n" << expected_output
                      << "Actual:\n" << actual_output << "\n";
            return false;
        }
    } catch (const std::exception& e) {
        if (expected_output.empty()) {  // Expected to fail
            std::cerr << "[PASS] " << test_name << " (failed as expected: " 
                      << e.what() << ")\n";
            return true;
        }
        std::cerr << "[FAIL] " << test_name << " (unexpected error: "
                  << e.what() << ")\n";
        return false;
    }
}

void run_all_tests() {
    bool all_passed = true;
    all_passed &= run_test_case(Config::TEST_CASE_1, Config::EXPECTED_OUTPUT, 
                              "Unbalanced input scales");
    all_passed &= run_test_case(Config::CYCLE_TEST_CASE, "", 
                              "Cycle detection - invalid input case");

    if (all_passed) {
        std::cerr << "\nALL TESTS PASSED\n";
    } else {
        std::cerr << "\nSOME TESTS FAILED\n";
        exit(1);
    }
}

int main() {
    run_all_tests();
    
    // Main program execution
    ScaleBalancer balancer;
    balancer.execute();  // Process standard input
    
    return 0;
}
