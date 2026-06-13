#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")

#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace std;

// ---------------------------------------------------------
// I. COMPETITIVE PROGRAMMING STYLE
// ---------------------------------------------------------

const int MAXN = 2048; // maximum universe size
const int MAXN_brute_force = 22; // maximum universe size for brute force algorithm
const double eps = 1e-9; // float epsilon to prevent mapping errors in alpha_val/beta_val

int m, n;
double alpha_val, beta_val;

// bitset for O(1) chunked intersections of geometric subsets
bitset<MAXN> C[MAXN]; // macroscopic covering blocks
bitset<MAXN> N_C[MAXN]; // microscopic minimal neighborhoods {\cal N}_{\cal C}(x)

double H[MAXN]; // discrete harmonic sequence H_n
double nCr[30][30]; // Pascal's triangle for brute-force verification

// ---------------------------------------------------------
// MATHEMATICAL PRECOMPUTATIONS
// ---------------------------------------------------------

void precompute() {
	H[0] = 0.0;
	for (int i = 1; i < MAXN; ++i) H[i] = H[i - 1] + 1.0 / (double)i;

	// Pascal's triangle: only needed if doing O(2^n) verification
		if (n <= 25)
			for (int i = 0; i <= n; ++i) {
				nCr[i][0] = 1.0;
				for (int j = 1; j <= i; ++j) nCr[i][j] = nCr[i - 1][j - 1] + nCr[i - 1][j];
			}
}

// ---------------------------------------------------------
// 2. DISCRETE THRESHOLD BOUNDINGS (Def. 2.4)
// ---------------------------------------------------------


int k_alpha_val(int sz, double a) {
	int k = ceil(a * sz - eps);
	return max(1, min(sz, k)); // geometric bounding shield: k\in[z]
}

int k_beta_val(int sz, double b) {
	int k = floor(b * sz + eps) + 1;
	return max(1, min(sz, k)); // geometric bounding shield: k\in[{\cal N}_{\cal C}(x)]
}

// ---------------------------------------------------------
// CORE ALGORITHM & THEOREM EVALUATION
// ---------------------------------------------------------

void solve() {
	cin >> n >> m >> alpha_val >> beta_val;
	for (int i = 1; i <= m; ++i) {
		int sz;
		cin >> sz;
		C[i].reset();
		for (int j = 0; j < sz; ++j) {
			int u;
			cin >> u;
			C[i].set(u - 1); // 0-index internally
		}
	}

	precompute();

	// phase 1: construct minimal neighborhoods {\cal N}_{\cal C}(x) in O(m * n / 64)
	bitset<MAXN> U;
	for (int i = 0; i < n; ++i) U.set(i);

	double sum_N = 0.0;
	vector<int> N_sz(n, 0);
	for (int x = 0; x < n; ++x) {
		N_C[x] = U; // start with universal set
		bool covered = false;

		for (int i = 1; i <= m; ++i)
			if (C[i].test(x)) {
				N_C[x] &= C[i]; // bitwise geometric intersection
				covered = true;
			}

		// topological vacuum convention
		if (!covered) N_C[x] = U;

		N_sz[x] = N_C[x].count(); // hardware accelerated pop-count
		sum_N += N_sz[x];
	}

	// phase 2: evaluate exact AZ-fractional identities (Sect. 2)
	double theoretical_lower = 0.0, theoretical_upper = 0.0, theoretical_bound = 0.0, theoretical_alpha_val = 0.0, theoretical_beta_val = 0.0;

	for (int x = 0; x < n; ++x) {
		int sz = N_sz[x];
		theoretical_lower += 1.0 / (double)sz; // Thm. 2.2
		theoretical_upper += H[sz]; // Thm. 2.3
		theoretical_bound += H[sz - 1]; // Cor. 2.2
		theoretical_alpha_val += H[sz] - H[k_alpha_val(sz, alpha_val) - 1]; // Thm. 2.5(i)
		theoretical_beta_val += H[sz] - H[k_beta_val(sz, beta_val) - 1]; // Thm. 2.5(ii)
	}

	// phase 3: verify macroscopic capacity bounds: Thm. 2.4 & Lagrange identity
	double mu_C = sum_N / (double)n, Lagrange_dispersion = 0.0;

	for (int x = 0; x < n; ++x)
		for (int y = x + 1; y < n; ++y) { // unique pairs (x < y) implicitly computes 1/2
			double diff = (double)N_sz[x] - N_sz[y];
			Lagrange_dispersion += (diff * diff) / ((double)N_sz[x] * N_sz[y]);
		}

	// divided by n * mu_C because the sum only evaluated strictly over x < y
	double theoretical_Lagrange = (double)n / mu_C + Lagrange_dispersion / ((double)n * mu_C);

	cout << fixed << setprecision(6);
    cout << "========================================================\n";
    cout << " POLYNOMIAL-TIME THEORETICAL EVALUATIONS O(m * n / 64)\n";
    cout << "========================================================\n";
    cout << "[Thm 2.2] Lower Approx : " << theoretical_lower << '\n';
    cout << "[Thm 2.3] Upper Approx : " << theoretical_upper << '\n';
    cout << "[Cor 2.2] Boundary     : " << theoretical_bound << '\n';
    cout << "[Thm 2.4] Lagrange RHS : " << theoretical_Lagrange << " (LHS Diff: " << abs(theoretical_lower - theoretical_Lagrange) << ")\n";
    cout << "[Thm 2.5] Alpha-Lower  : " << theoretical_alpha_val << '\n';
    cout << "[Thm 2.5] Beta-Upper   : " << theoretical_beta_val << "\n\n";

   	// phase 4: exponential brute-force verification: for n <= 22
   	if (n <= MAXN_brute_force) {
   		double brute_force_lower = 0.0, brute_force_upper = 0.0, brute_force_bound = 0, brute_force_alpha_val = 0.0, brute_force_beta_val = 0.0;

   		for (long long mask = 1; mask < (1LL << n); ++mask) {
   			int target_sz = __builtin_popcountll(mask);
   			double weight = 1.0 / ((double)target_sz * nCr[n][target_sz]); // AZ-weight function h_n(|X|)

   			int lower_card = 0, upper_card = 0, alpha_val_card = 0, beta_val_card = 0;

   			for (int x = 0; x < n; ++x) {
   				// extract N_C(x) into 64-bit integer
   				long long nx_mask = 0;
   				for (int i = 0; i < n; ++i) if (N_C[x].test(i)) nx_mask |= (1LL << i);

   				int intersect = __builtin_popcountll(nx_mask & mask);
   				int sz = N_sz[x];

   				if (intersect == sz) ++lower_card; // subset inclusion
   				if (intersect >= 1) ++upper_card;
   				if (intersect >= k_alpha_val(sz, alpha_val)) ++alpha_val_card; // alpha_val threshold
   				if (intersect >= k_beta_val(sz, beta_val)) ++beta_val_card; // beta_val threshold
   			}

   			brute_force_upper += weight * lower_card;
   			brute_force_upper += weight * upper_card;
   			brute_force_bound += weight * (upper_card - lower_card);
   			brute_force_alpha_val += weight * alpha_val_card;
   			brute_force_beta_val += weight * beta_val_card;
   		}

   		cout << "========================================================\n";
        cout << " EXPONENTIAL BRUTE-FORCE VERIFICATION O(n * 2^n)\n";
        cout << "========================================================\n";
        cout << "[Thm 2.2] Lower Approx : " << brute_force_lower << " (Err: " << abs(theoretical_lower - brute_force_lower) << ")\n";
        cout << "[Thm 2.3] Upper Approx : " << brute_force_upper << " (Err: " << abs(theoretical_upper - brute_force_upper) << ")\n";
        cout << "[Cor 2.2] Boundary     : " << brute_force_bound << " (Err: " << abs(theoretical_bound - brute_force_bound) << ")\n";
        cout << "[Thm 2.5] Alpha-Lower  : " << brute_force_alpha_val << " (Err: " << abs(theoretical_alpha_val - brute_force_alpha_val) << ")\n";
        cout << "[Thm 2.5] Beta-Upper   : " << brute_force_beta_val  << " (Err: " << abs(theoretical_beta_val  - brute_force_beta_val)  << ")\n";
   	}
}

// ---------------------------------------------------------
// II. PURE C/C++ PROGRAMMING STYLE
// ---------------------------------------------------------

// ---------------------------------------------------------
// III. OPTIMIZED C/C++ PROGRAMMING STYLE
// ---------------------------------------------------------

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------

int main() {
	ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    solve();
}
/*
compilation on Terminal of Ubuntu:
g++ -O2 -Wall NQBH_nonideal_covering_based_rough_set_identities.cpp -o NQBH_nonideal_covering_based_rough_set_identities
./NQBH_nonideal_covering_based_rough_set_identities
*/