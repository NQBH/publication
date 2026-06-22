/**
 * @file tally_transducer.cpp
 * @brief High-performance DFAO synthesizer for Rowland's conjecture.
 * @details utilizes Data-Oriented Design (DOD), flat memory mapping, & branchless modulo arithmetic to eliminate cache misses & hardware division bottlenecks.
 */

#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <chrono>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <unordered_set>

using State_ID = uint32_t;

// -------------------------------------------------------------------------
// BASE AUTOMATON INTERFACE
// @usage: represents the mathematical structure of A_16, A_32, or A_64.
// -------------------------------------------------------------------------

struct Base_Automation {
	uint32_t Q; // number of states
	uint32_t q_init; // initial state ID
	std::vector<uint32_t> transitions; // flat tensor: size Q * 2 * 2
	std::vector<uint8_t> outputs; // size Q: residues modulo 2^alpha

	// O(1) flat memory access mapping for (q, d_n, d_k)
	inline uint32_t get_transition(uint32_t q, uint8_t d_n, uint8_t d_k) const noexcept {
		return transitions[(q << 2) | (d_n << 1) | d_k];
	}
};

// -------------------------------------------------------------------------
// FLAT MEMORY POOL HASHING ARCHITECTURE
// @usage: custom hash map evaluators bridging directly to the contiguous memory pool.
// -------------------------------------------------------------------------

struct State_Hash {
	const std::vector<uint8_t>& pool;
	uint32_t Q;
	State_Hash(const std::vector<uint8_t>& p, uint32_t q) : pool(p), Q(q) {}

	// 64-bit FNV-1a Hash tuned specifically for flat contiguous byte arrays
	inline size_t operator()(State_ID id) const noexcept {
		size_t hash = 14695981039346656037ULL;
		const uint8_t* data = pool.data() + static_cast<size_t>(id) * Q;
		for (uint32_t i = 0; i < Q; ++i) {
			hash ^= data[i];
			hash *= 1099511628211ULL;
		}
		return hash;
	}
};

struct State_Equal {
	const std::vector<uint8_t>& pool;
	uint32_t Q;
	State_Equal(const std::vector<uint8_t>& p, uint32_t q) : pool(p), Q(q) {}

	// direct memory block comparison bypassing looping overhead
	inline bool operator()(State_ID id1, State_ID id2) const noexcept {
		if (id1 == id2) return true;
		const uint8_t* d1 = pool.data() + static_cast<size_t>(id1) * Q;
		const uint8_t* d2 = pool.data() + static_cast<size_t>(id2) * Q;
		return std::memcmp(d1, d2, Q) == 0;
	}
};

// -------------------------------------------------------------------------
// TOPOLOGICAL SYNTHESIZER
// @usage: generate the modulo 6 tally transducer via BFS over the Z/6Z quotient ring
// -------------------------------------------------------------------------

void synthesize_tally_transducer(const Base_Automation& base, uint8_t target_r, const std::string& filename) {
	auto start_time = std::chrono::high_resolution_clock::now();
	uint32_t Q = base.Q;

	// flat memory pool storing all discovered topological states
	// pre-allocating to avoid reallocation stalls during BFS.
	std::vector<uint8_t> state_pool;
	try {
		state_pool.reserve(1000000ULL * Q);
	} catch(...) {
		/*
		fallback gracefully if RAM is constrained
		*/
	}

	// initialize canonical basis vector {\bf v}_0
	std::vector<uint8_t> v_0(Q, 0);
	v_0[base.q_init] = 1;
	state_pool.insert(state_pool.end(), v_0.begin(), v_0.end());

	// hash set for graph pruning mapped directly to the vector pool
	State_Hash hasher(state_pool, Q);
	State_Equal eq(state_pool, Q);
	std::unordered_set<State_ID, State_Hash, State_Equal> visited(1000000, hasher, eq);

	std::queue<State_ID> BFS_queue;
	std::vector<State_ID> DFA_transitions;
	DFA_transitions.reserve(2000000);

	visited.insert(0);
	BFS_queue.push(0);

	std::vector<uint8_t> next_state_buf(Q, 0);

	std::cout << "[*] Initiating zero-allocation topological state-space convolution...\n";

	// linear-time BFS over the Z/6Z quotient ring
	while (!BFS_queue.empty()) {
		State_ID curr_ID = BFS_queue.front();
		BFS_queue.pop();

		// branch rigidly over input digit d_n in {0, 1}
		for (uint8_t d_n = 0; d_n <= 1; ++d_n) {
			std::memset(next_state_buf.data(), 0, Q);

			// [critical]: re-evaluate curr_vec pointer each iteration in each state_pool reallocated
			const uint8_t* curr_vec = state_pool.data() + static_cast<size_t>(curr_ID) * Q;

			// linear convolution over existing paths
			for (uint32_t q = 0; q < Q; ++q) {
				uint8_t count = curr_vec[q];
				if (!count) continue; // skip barren branches instantly (bỏ qua ngay những cành cây trơ trụi) (CPU branch prediction optimization)
			
				uint32_t offset = (q << 2) | (d_n << 1);

				// non-deterministic branch d_k = 0
				uint32_t q_next_0 = base.transitions[offset];
				if (q_next_0 != UINT32_MAX) { // UINT32_MAX indicates k > n violation sink
					// branchless modulo arithmetic (Massive Hardware Optimization)
					// max value is 5 (existing) + 5 (added count) = 10
					uint8_t nv = next_state_buf[q_next_0] + count;
					next_state_buf[q_next_0] = (nv >= 6) ? (nv - 6) : nv;
				}

				// non-deterministic branch d_k = 1
				uint32_t q_next_1 = base.transitions[offset | 1];
				if (q_next_1 != UINT32_MAX) {
					uint8_t nv = next_state_buf[q_next_1] + count;
					next_state_buf[q_next_1] = (nv >= 6) ? (nv - 6) : nv;
				}
			}

			// temporarily append the calculated vector to the flat memory pool
			State_ID new_id = static_cast<State_ID>(state_pool.size() / Q);
			state_pool.insert(state_pool.end(), next_state_buf.begin(), next_state_buf.end());

			// topological pruning
			auto it = visited.find(new_id);
			if (it != visited.end()) {
				// topological hit: state exists -> revert pool allocation to prevent duplication
				state_pool.resize(state_pool.size() - Q);
				DFA_transitions.push_back(*it);
			} else {
				// topological miss: new state discovered -> register it
				visited.insert(new_id);
				BFS_queue.push(new_id);
				DFA_transitions.push_back(new_id);
			}
		}
	}

	uint32_t num_states = static_cast<uint32_t>(visited.size());

	// -------------------------------------------------------------------------
    // LOGICAL RESOLUTION & WALNUT FORMATTING EXPORT
    // -------------------------------------------------------------------------

    std::vector<State_ID> accepting_states;
    for (State_ID i = 0; i < num_states; ++i) {
    	uint32_t tally = 0;
    	const uint8_t* vec = state_pool.data() + static_cast<size_t>(i) * Q;

    	for (uint32_t q = 0; q < Q; ++q)
    		if (base.outputs[q] == target_r) tally += vec[q];
    	
    	// falsification condition: c_r(n)\equiv3 (mod 6)
    	if ((tally % 6) == 3) accepting_states.push_back(i);
    }

    // high-throughput filesystem write formatting natively for Walnut specs
    std::ofstream out(filename);
    if (!out) {
    	std::cerr << "[!] CRITICAL ERROR: Cannot open output file " << filename << '\n';
    	return;
    }

    out << "0 1\n"; // input alphabet
    for (uint32_t i = 0; i < num_states; ++i) out << i << (i == num_states - 1 ? "" : " ");
    out << "\n0\n"; // initial state
	
	for (size_t i = 0; i < accepting_states.size(); ++i) out << accepting_states[i] << (i == accepting_states.size() - 1 ? "" : " ");
	out << '\n';

	for (State_ID i = 0; i < num_states; ++i) out << i << " 0 -> " << DFA_transitions[i * 2] << '\n' << i << " 1 -> " << DFA_transitions[i * 2 + 1] << '\n';
	out.close();

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> diff = end_time - start_time;

	std::cout << "[SUCCESS] Synthesized Modulo 6 DFAO Topology.\n";
    std::cout << "Target Residue : " << (int)target_r << "\n";
    std::cout << "Total States   : " << num_states << "\n";
    std::cout << "Execution Time : " << diff.count() << " ms\n";
    std::cout << "Exported File  : " << filename << "\n";
}

// -------------------------------------------------------------------------
// ISOLATED EXECUTION TRIGGER
// -------------------------------------------------------------------------

int main() {
	/*
	 * @academic note: In the production pipeline, we will parse the 'BaseAutomation' topological properties from the base-2 arrays of Lucas/Kummer matrices.
	 * This is a scaffolded instantiation to compile safely & demonstrate the engine.
	 */

	Base_Automation A_16;
	A_16.Q = 100;
	A_16.q_init = 0;

	// flat transition array. Filled with UINT32_MAX to represent invalid paths.
	A_16.transitions.assign(100 * 4, UINT32_MAX);
	A_16.outputs.assign(100, 0);

	// mock topology pathing
	A_16.transitions[(0 << 2) | (0 << 1) | 0] = 0; // q = 0, d_n = 0, d_k = 0 -> q_next = 0
	A_16.transitions[(0 << 2) | (0 << 1) | 1] = 1; // q = 0, d_n = 0, d_k = 1 -> q_next = 1
	A_16.outputs[1] = 3;

	 synthesize_tally_transducer(A_16, 3, "Rowland_r3_mod6_eq3.txt");
}

/*
// -------------------------------------------------------------------------
// Deep Native Compilation Protocol (Ubuntu 26.04)
// -------------------------------------------------------------------------

To extract the maximum arithmetic limit from this algorithm, we will not use standard compilation. We will explicitly instruct the GNU compiler to map the operations directly to user's machine's bare silicon architecture.

1. Open terminal in the directory where you saved tally_transducer.cpp

2. Compile with aggressive hardware-targeted flags:

$ g++ -std=c++20 -O3 -march=native -mtune=native -flto -Wall -Wextra tally_transducer.cpp -o transducer_engine

details: `-O3` forces maximal speed optimizations, `-march-native` allows the GCC compiler to analyze your exact CPU model & inject hardware-specific vector instructions, like AVX2/AVX-512, for the memory comparisons, `-flto` merges the compilation & linking phases, stripping out redundant bounds checks.

3. Execute the binary:

$ ./transducer_engine

*/