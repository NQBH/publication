/**
 * @file: tally_transducer_ultra.cpp
 * @brief: extreme-scale DFAO synthesizer for higher 2-adic moduli.
 * @details: utilizes 3-bit tensor packing (21 dimensions per 64-bit word) to bypass RAM exhaustion, & native 128-bit hardware registers for massive residues.
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>

// GCC hardware native 128-bit integer handles outputs up to 2^128 instantly
using big_residue = unsigned __int128;
using state_ID = uint32_t;

// helper to print 128-bit integers to terminal
std::string to_string(big_residue n) {
	if (!n)  return "0";
	std::string s;
	while (n > 0) {
		s += (char)('0' + (n % 10));
		n /= 10;
	}
	std::reverse(s.begin(), s.end());
	return s;
}

// -------------------------------------------------------------------------
// BASE AUTOMATON INTERFACE
// -------------------------------------------------------------------------

struct Base_Automaton {
	uint32_t Q, q_init;
	std::vector<uint32_t> transitions;
	std::vector<big_residue> outputs; // supports extremely large residues
};

// -------------------------------------------------------------------------
// ULTRA-DENSE 3-BIT TENSOR ARCHITECTURE
// Math: max value is 5 = (101)_2, requiring exactly 3 bits.
// We pack 21 elements into 1 64-bit chunk (21 * 3 = 63 bits used).
// -------------------------------------------------------------------------

constexpr uint32_t VALS_PER_CHUNK = 21;

inline uint8_t get_packed_val(const std::vector<uint64_t>& vec, uint32_t q) noexcept {
	uint32_t chunk_idx = q / VALS_PER_CHUNK, bit_offset = q % VALS_PER_CHUNK * 3;
	return (vec[chunk_idx] >> bit_offset) & 0b111;
}

inline void add_packed_val_mod6(std::vector<uint64_t>& vec, uint32_t q, uint8_t add_val) noexcept {
	uint32_t chunk_idx = q / VALS_PER_CHUNK, bit_offset = q % VALS_PER_CHUNK * 3;
	uint8_t current = (vec[chunk_idx] >> bit_offset) & 0b111, next_val = current + add_val;

	if (next_val >= 6) next_val -= 6;

	// clear old 3 bits & insert new 3 bits
	vec[chunk_idx] &= ~(0b111ULL << bit_offset);
	vec[chunk_idx] |= (static_cast<uint64_t>(next_val) << bit_offset);
}

// SIMD-optimized hash for packed 64-bit vectors
struct Packed_Vector_Hash {
	size_t operator()(const std::vector<uint64_t>& v) const noexcept {
		size_t hash = 14695981039346656037ULL;
		for (uint64_t chunk : v) {
			hash ^= chunk;
			hash *= 1099511628211ULL;
		}
		return hash;
	}
};

// -------------------------------------------------------------------------
// EXTREME-SCALE SYNTHESIZER
// -------------------------------------------------------------------------

void synthesize_ultra_transducer(const Base_Automaton& base, big_residue target_r, const std::string& filename) {
	auto start_time = std::chrono::high_resolution_clock::now();
	uint32_t Q = base.Q, packed_size = (Q + VALS_PER_CHUNK - 1) / VALS_PER_CHUNK; // ceiling division

	std::vector<uint64_t> v_0(packed_size, 0);
	add_packed_val_mod6(v_0, base.q_init, 1);

	std::unordered_map<std::vector<uint64_t>, state_ID, Packed_Vector_Hash> visited;
	std::queue<std::vector<uint64_t>> BFS_queue;
	std::vector<state_ID> DFA_transitions;

	visited.reserve(2000000);
	DFA_transitions.reserve(4000000);

	visited[v_0] = 0;
	BFS_queue.push(v_0);

	std::cout << "[*] Executing Extreme-Scale 3-Bit Tensor Convolution...\n";

	while (!BFS_queue.empty()) {
		std::vector<uint64_t> curr_vec = std::move(BFS_queue.front());
		BFS_queue.pop();

		for (uint8_t d_n = 0; d_n <= 1; ++d_n) {
			std::vector<uint64_t> next_vec(packed_size, 0);

			for (uint32_t q = 0; q < Q; ++q) {
				uint8_t count = get_packed_val(curr_vec, q);
				if (!count) continue;

				uint32_t offset = (q << 2) | (d_n << 1), q_next_0 = base.transitions[offset];

				if (q_next_0 != UINT32_MAX) add_packed_val_mod6(next_vec, q_next_0, count);
				
				uint32_t q_next_1 = base.transitions[offset | 1];
				if (q_next_1 != UINT32_MAX) add_packed_val_mod6(next_vec, q_next_1, count);
			}

			auto it = visited.find(next_vec);
			if (it != visited.end()) DFA_transitions.push_back(it->second);
			else {
				state_ID new_id = static_cast<state_ID>(visited.size());
				visited[next_vec] = new_id;
				BFS_queue.push(std::move(next_vec));
				DFA_transitions.push_back(new_id);
			}
		}
	}

	uint32_t num_states = static_cast<uint32_t>(visited.size());

	// invert map to write sequential blocks securely for Walnut
	std::vector<const std::vector<uint64_t>*> state_order(num_states);
	for (const auto& pair :visited) state_order[pair.second] = &pair.first;

	std::ofstream out(filename);
	out << "lsd_2\n";
	for (state_ID i = 0; i < num_states; ++i) {
		uint32_t tally = 0;
		const std::vector<uint64_t>& vec = *(state_order[i]);

		for (uint32_t q = 0; q < Q; ++q)
			if (base.outputs[q] == target_r) tally += get_packed_val(vec, q);

		uint8_t out_val = ((tally % 6) == 3) ? 1 : 0;
		out << i << " " << static_cast<int>(out_val) << '\n' << "0 -> " << DFA_transitions[i * 2] << '\n' << "1 -> " << DFA_transitions[i * 2 + 1] << '\n';
	}

	out.close();

	std::chrono::duration<double, std::milli> diff = std::chrono::high_resolution_clock::now() - start_time;

	std::cout << "[SUCCESS] Ultra-Compressed DFAO Compiled.\n";
    std::cout << "Target Residue : " << to_string(target_r) << '\n';
    std::cout << "Total States   : " << num_states << '\n';
    std::cout << "Execution Time : " << diff.count() << " ms\n";
}

int main() {
	Base_Automaton A;
	A.Q = 3;
	A.q_init = 0;
	A.transitions.assign(3 * 4, UINT32_MAX);
	A.outputs.assign(3, 0);

	// mock topology pathing
	A.transitions[(0 << 2) | (0 << 1) | 0] = 0; 
    A.transitions[(0 << 2) | (1 << 1) | 0] = 1; 
    A.transitions[(0 << 2) | (1 << 1) | 1] = 0; 
    A.transitions[(1 << 2) | (0 << 1) | 0] = 1; 
    A.transitions[(1 << 2) | (1 << 1) | 0] = 1; 
    A.transitions[(1 << 2) | (1 << 1) | 1] = 0;

    A.outputs[0] = 1;

    // we can confidently assign a massive 128-bit residue here
    A.outputs[1] = (static_cast<big_residue>(1) << 100) | 3;
    A.outputs[2] = 0;

    synthesize_ultra_transducer(A, A.outputs[1], "Rowland_extreme_cpp.txt");
}

/*
g++ -std=c++26 -O3 -funroll-loops -march=native -mtune=native -flto -Wall -Wextra tally_transducer_ultra.cpp -o transducer_ultra
*/