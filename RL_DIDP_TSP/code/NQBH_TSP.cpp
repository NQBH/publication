#include <bits/stdc++.h>
/*
#include <iostream>
#include <vector>
#include <bitset>
#include <unordered_map>
#include <algorithm>
*/
using namespace std;
using ll = long long;

/* 1st naive solution
source: Geeks for Geeks/Traveling Salesman Problem (TSP) Implementation
url: https://www.geeksforgeeks.org/dsa/traveling-salesman-problem-tsp-implementation/
C++ program to find the shortest possible route that visits every city exactly once & returns to the starting point
time complexity: O(n!) since the algorithm uses the next_permutation function which generates all the possible permutations of the vertex set.
auxiliary space: O(n) since we use a vector to store all the vertices.
*/

int tsp(vector<vector<int>> &cost) {
	int num_nodes = cost.size(); // number of nodes
	vector<int> nodes;
	for (int i = 1; i < num_nodes; ++i) nodes.push_back(i); // initialize the nodes excluding the fixed starting point (node 0)
	int min_cost = INT_MAX;	
	do { // generate all permutations of the remaining nodes
		int curr_cost = 0, curr_node = 0; // start from node 0		
		for (int i = 0; i < (int)nodes.size(); ++i) { // calculate the cost of the current permutation
			curr_cost += cost[curr_node][nodes[i]];
			curr_node = nodes[i];
		}
		curr_cost += cost[curr_node][0]; // add the cost to return to the starting node
		min_cost = min(min_cost, curr_cost); // update the minimum cost if the current cost is lower
	} while (next_permutation(nodes.begin(), nodes.end()));
	return min_cost;
}

/* 2nd naive solution
source: Geeks for Geeks/Traveling Salesman Problem
url: https://www.geeksforgeeks.org/dsa/travelling-salesman-problem-using-dynamic-programming/
Naive approach: use DFS: O(n!) time & O(n) space
*/

int dfs(vector<vector<int>>& cost, vector<bool>& vis, int last, int cnt) {
	int n = cost.size();
	if (cnt == n) return cost[last][0]; // if all visited then return cost back to start 0,
	int min_cost = INT_MAX;
	for (int city = 1; city < n; ++city)
		if (!vis[city]) {
			vis[city] = true; // mark the city as visited & explore all possible paths from there
			min_cost = min(min_cost, cost[last][city] + dfs(cost, vis, city, cnt + 1));
			vis[city] = false; // backtrack
		}
	return min_cost; // return minimum cost among all possible paths
}

int tsp_dfs(vector<vector<int>>& cost) {
	int n = cost.size();
	vector<bool> vis(n, false); // guarantee that each city is visited exactly once
	vis[0] = true;
	return dfs(cost, vis, 0, 1);
}

/* 3rd solution: backtracking O(n!) time & O(n) space
source: Geeks for Geeks/Traveling Salesman Problem implementation using backtracking
url: www.geeksforgeeks.org/dsa/travelling-salesman-problem-implementation-using-backtracking/
C++ program to find the shortest possible route that visits every city exactly once & returns to the starting point using backtracking
*/

void total_cost_backtrack(vector<vector<int>> &cost, vector<bool> &visited, int curr_pos, int n, int cnt, int cost_so_far, int &ans) {
	if (cnt == n && cost[curr_pos][0]) { // if all nodes are visited & there is an edge to starting node
		ans = min(ans, cost_so_far + cost[curr_pos][0]); // update the minimum cost
		return;
	}
	for (int i = 0; i < n; ++i) // try visiting each node from current position
		if (!visited[i] && cost[curr_pos][i]) {
			visited[i] = true; // if node is not visited & has an edge then mark as visited
			total_cost_backtrack(cost, visited, i, n, cnt + 1, cost_so_far + cost[curr_pos][i], ans);
			visited[i] = false;
		}
}

int tsp_backtrack(vector<vector<int>> &cost) {
	int n = cost.size();
	vector<bool> visited(n, false);
	visited[0] = true;
	int ans = INT_MAX;
	total_cost_backtrack(cost, visited, 0, n, 1, 0, ans);
	return ans;
}

/* 4th solution: bitmask representation for tracking visited cities
Use recursion O(n!) time & O(n) space
*/

int total_cost(int mask, int pos, vector<vector<int>>& cost) {
	int n = cost.size();
	if (mask == (1 << n) - 1) return cost[pos][0]; // base case: if all cities are visited, return the cost to return to the starting city 0
	int ans = INT_MAX;
	for (int i = 0; i < n; ++i) // try visiting every city that has not been visited yet
		if (!(mask & (1 << i))) ans = min(ans, cost[pos][i] + total_cost(mask | (1 << i), i, cost)); // if city is not visited, visit it & update the mask
	return ans;
}

int tsp_bitmask(vector<vector<int>>& cost) {
	int mask = 1, pos = 0; // start from city 0, & only city 0 is visited initially (mask = 1)
	return total_cost(mask, pos, cost);
}

/* 5th solution: use top-down dynamic programming (memoization)
O(n^2 * 2^n) time & O(n * 2^n) space
*/

int total_cost_dp_topdown(int mask, int curr, vector<vector<int>>& cost, vector<vector<int>>& dp) {
	int n = cost.size();
	if (mask == (1 << n) - 1) return cost[curr][0]; // base case: if all cities are visited, return the cost to return to the starting city 0
	if (dp[curr][mask] != -1) return dp[curr][mask]; // if the value has already been computed, return it from the DP table
	int ans = INT_MAX;
	for (int i = 0; i < n; ++i) // try visiting every city that has not been visited yet
		if (!(mask & (1 << i))) ans = min(ans, cost[curr][i] + total_cost_dp_topdown(mask | (1 << i), i, cost, dp)); // if city is not visited, visity city i & update the mask
	return dp[curr][mask] = ans;
}

int tsp_dp_topdown(vector<vector<int>>& cost) {
	int n = cost.size();
	vector<vector<int>> dp(n, vector<int>(1 << n, -1));
	int mask = 1, curr = 0; // start from city 0, with only city 0 (visited initially, mask = 1)
	return total_cost_dp_topdown(mask, curr, cost, dp);
}

/* 6th solution: iterative dp, use bottom-up dynamic programming (tabulation)
O(n^2 * 2^n) time & O(n * 2^n) space
*/
int tsp_bottomup(vector<vector<int>>& cost) {
	int n = cost.size();
	if (n <= 1) return n == 1 ? cost[0][0] : 0;
	const int INF = INT_MAX; // maximum cost to visit all cities
	int FULL = 1 << n, full_mask = FULL - 1;
	vector<vector<int>> dp(FULL, vector<int>(n, INF)); // dp[mask][i] represents the minimum cost to visit all cities corresponding to the set bits in mask, ending at city i
	dp[1][0] = 0;
	for (int mask = 1; mask < FULL; ++mask) // iterate over all subsets of cities
		for (int i = 0; i < n; ++i) {
			// skip if city i is not included in mask
			if (!(mask & (1 << i))) continue;
			if (dp[mask][i] == INF) continue;
			for (int j = 0; j < n; ++j) { // try to go to every unvisited city j
				if (mask & ( 1 << j)) continue; // skip if city j is already visited
				// cost to visit new city j from city i s.t. previously visited cities remain visited
				int nxt = mask | (1 << j);
				dp[nxt][j] = min(dp[nxt][j], dp[mask][i] + cost[i][j]);
			}
		}
	int ans = INF;
	for (int i = 0; i < n; ++i)
		if (dp[full_mask][i] != INF) // if last city on path is i & cost of path is not infinity
			ans = min(ans, dp[full_mask][i] + cost[i][0]); // update net cost s.t. city 0 is visited in last
	return ans;
}

/* implementation a Reinforcement-Guided Domain-Independent Dynamic Programming (RG-DIDP) solver in C++, we focus on a structure that allows an external model (the RL agent) to inject heuristic values into a high-performance search engine.
The following architecture uses a Beam Search approach, as it is the standard for memory-efficient DIDP.
Core State Representation: In C++, efficiency is gained by using bitmasks for the set of visited cities and custom hash functions for the DP table.
*/

struct state {
	int curr_city;
	std::bitset<32> visited; // assuming TSP size <= 32 for bitmask
	double g_cost; // path cost from start to here
	bool operator==(const state& other) const {
		return curr_city == other.curr_city && visited == other.visited;
	}
};

// custom hash for DP table
struct state_hash {
	std::size_t operator()(const state& s) const {
        return std::hash<int>()(s.curr_city) ^ (std::hash<std::bitset<32>>()(s.visited) << 1);
    }
};

/*
// Integration with the RL Model: show how the C++ engine communicates with the RL model (typically trained in Python/PyTorch). We use LibTorch (the PyTorch C++ API) for low-latency inference during the search.
#include <torch/script.h> // LibTorch

class RLheuristic {
private:
	torch::jit::script::Module model;
public:
	RLHeuristic(std::string model_path) {
        model = torch::jit::load(model_path);
    }

    double get_h_value(const state& s, const vector<vector<double>>& adj_matrix) {
    	auto tensor_state = state_to_tensor(s); // convert state & adjacency matrix to Torch Tensors
    	auto output = model.forward({tensor_state}).toTensor();
    	return output.item<double>();
    }
};

// RG-DIDP Search Loop: This is where the "Reinforcement-Guided" logic prunes the state space. We use a priority_queue to manage the "Beam" of states.
void solve_RL_DIDP(int n_cities, const vector<vector<double>>& dist) {
	unordered_map<state, double, state_hash> dp_table;
	vector<state> beam;

	// initial state
	state start = {0, bitset<32>(1), 0.0};
	beam.push_back(start);

	for (int step = 1; step < n_cities; ++step) {
		vector<state> next_beam;
		for (const auto& curr : beam)
			for (int next_city = 0; next_city < n_cities; ++next_city)
				if (!curr.visited.test(next_city)) {
					state next_s = {next_city, current.visited, current.g_cost + dist[curr.curr_city][next_city]};
					next_s.visited.set(next_city);
					next_beam.push_back(next_s);
				}
		// Reinforcement Learning guidance step
		// sort by f = g + h_RL
		sort(next_beam.begin(), next_beam.end(), [&](const state& a, const state& b) {
			double f_a = a.g_cost + rl_agent.get_h_value(a, dist);
            double f_b = b.g_cost + rl_agent.get_h_value(b, dist);
            return f_a < f_b;
		});
		// prune: keep only the top 'W' (Beam Width) states
		if (next_beam.size() > BEAM_WIDTH) next_beam.resize(BEAM_WIDTH);
		beam = next_beam;
	}
}
*/

//-----------------------------------------------------------------------------//
// main
//-----------------------------------------------------------------------------//

int main() {
	// small fixed testcase
	vector<vector<int>> cost = {{0, 10, 15, 20, 12}, 
                                {10, 0, 35, 25, 21}, 
                                {15, 35, 0, 30, 7}, 
                                {20, 25, 30, 0, 9},
                            	{23, 37, 57, 19, 0}};
    // input testcase

    // output result
    cout << tsp(cost) << '\n';
    cout << tsp_dfs(cost) << '\n';
    cout << tsp_backtrack(cost) << '\n';
    cout << tsp_bitmask(cost) << '\n';
    cout << tsp_dp_topdown(cost) << '\n';
    cout << tsp_bottomup(cost) << '\n';
	/*
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Random Tensor: " << tensor << std::endl;
    */
}