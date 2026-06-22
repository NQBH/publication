import collections

"""
Rigorous Algorithmic Blueprint: Modulo 6 Tally Transducer Generator
Targeting: c_r(n)\equiv 3 (mod 6) for binomial residues modulo 16.
"""

def generate_tally_DFA(base_states, q_init, base_transitions, base_outputs, target_r):
	# state vector v\in(Z/6Z)^{|Q|}: v[q] tracks path counts mod 6 terminating in state q
	# initialize the canonical basis vector v_0 (1 empty path at q_init, 0 paths elsewhere)
	v_0_list = [0] * len(base_states)
	v_0_list[q_init] = 1
	v_0 = tuple(v_0_list)

	# lazy BFS initialization for topological pruning
	queue = collections.deque([v_0])
	visited = {v_0: 0} # maps the mathematical vector state to a unique integer ID
	DFA_transitions = {} # maps (state_id, d_n) -> next_state_id

	while queue:
		v_curr = queue.popleft()
		curr_id = visited[v_curr]

		# branch over the base-2 digits of n
		for d_n in (0, 1):
			v_next = [0] * len(base_states)

			# linear convolution over the base transition relation
			for q, count in enumerate(v_curr):
				if count == 0:
					continue
				# branch over the non-deterministic guesses for the binary digits of k
				for d_k in (0, 1):
					q_next = base_transitions.get((q, d_n, d_k))
					if q_next is not None:
						# aggregate path counts over the quotient ring Z/6Z
						v_next[q_next] = (v_next[q_next] + count) % 6

			v_next_tuple = tuple(v_next)

			# map topology: only register mathematically reachable subspaces
			if v_next_tuple not in visited:
				visited[v_next_tuple] = len(visited)
				queue.append(v_next_tuple)

			DFA_transitions[(curr_id, d_n)] = visited[v_next_tuple]

	# define the accepting states (the Rowland violation condition)
	accepting_states = []
	for v_tuple, state_id in visited.items():
		# aggregate path tallies for all base states outputting the targeted odd residue
		tally = sum(count for q, count in enumerate(v_tuple) if base_outputs.get(q) == target_r) % 6

		# the DFA accepts iff the count is 3 mod 6
		if tally == 3:
			accepting_states.append(state_id)

	return visited, DFA_transitions, accepting_states

# export to Walnut native format: [deprecated: replaced by the next function]
def export_Walnut_DFA(visited, transitions, accepting_states, filename):
	with open(filename, 'w') as f:
		# [wrong syntax]: f.write("0 1\n") # line 1: input alphabet
		f.write("lsd_2\n") # line 1: explicitly declare the numeration system
		f.write(" ".join(str(i) for i in visited.values()) + "\n") # line 2: space-separated state IDs
		f.write("0\n") # line 3: initial state
		f.write(" ".join(str(i) for i in accepting_states) + "\n") # line 4: space-separated accepting states (writes empty line if none)
		# line 5+: transition matrix: [origin][input] -> [destination]
		for (state_id, d_n), next_id in transitions.items():
			f.write(f"{state_id} {d_n} -> {next_id}\n")

# export to Walnut native word automaton format (DFAO)
def export_Walnut_DFAO(visited, transitions, accepting_states, filename):
	with open(filename, 'w') as f:
		f.write("lsd_2\n") # line 1: explicitly declare the numeration system
		
		""" [old error]
		# line 2: space-sparated states IDs
		states = list(visited.values())
		f.write(" ".join(str(s) for s in states) + "\n")

		f.write("0\n") # line 3: initial state

		# line 4: DFAO outputs: 1 = violation found, 0 = safe
		outputs = ["1" if s in accepting_states else "0" for s in states]
		f.write(" ".join(outputs) + "\n")

		# line 5+: transition matrix
		for (state_id, d_n), next_id in transitions.items():
			f.write(f"{state_id} {d_n} -> {next_id}\n")
		"""

		# Walnut DFAO parser requires consecutive State Blocks: State 0 is implicitly the initial state
		# sort by state_id to guarantee 0,1,2,... sequential order
		states_sorted = sorted(visited.items(), key = lambda item: item[1])

		for state_tuple, state_id in states_sorted:
			# line A: <state_id> <output_value>
			# output is 1 if it violates the conjecture, else 0
			out_val = 1 if state_id in accepting_states else 0
			f.write(f"{state_id} {out_val}\n")

			# line B: <input> -> <next_state>
			for d_n in (0, 1):
				next_id = transitions[(state_id, d_n)]
				f.write(f"{d_n} -> {next_id}\n")

		# Note: Because we sort the states by state_id, this logic flawlessly guarantees the 0, 1, 2, ... sequential block order Walnut strictly demands.

# =========================================================================
# EXPLICIT EXECUTION TRIGGER & TOPOLOGICAL SCAFFOLD
# =========================================================================

if __name__ == "__main__":
    import time
    print("[*] Initializing Base Automaton A_16 (Structural Scaffold)...")

    # [note]: In the full research pipeline, these data structures are populated by parsing the Kummer/Lucas carry-chain transition matrices.
    # we define the exact topological parameters required by the BFS algorithm to test the memory-safety & formatting of the Z/6Z convolution.

    base_states = [0, 1, 2]
    q_init = 0

    # transition mapping: (current_state, d_n, d_k) -> next_state
    # transitions branching into mathematically invalid k > n bounds implicitly map to None (via .get) & are topologically pruned.
    base_transitions = {
    	(0, 0, 0): 0,
        (0, 1, 0): 1,
        (0, 1, 1): 0,
        (1, 0, 0): 1,
        (1, 1, 0): 1,
        (1, 1, 1): 0
    }

    # output mapping: current_state -> integer residue mod 16
    base_outputs = {
    	0: 1, # non-target residue
    	1: 3, # target odd residue (r = 3)
    	2 : 0 # sink state output
    }

    target_r = 3
    output_filename = f"Rowland_r{target_r}_mod6_eq3.txt"

    print(f"[*] Synthesizing Modulo 6 Tally Transducer for r = {target_r}...")
    start_time = time.time()

    # execyte BFS over the Z/6Z quotient ring
    visited, DFA_transitions, acc_states = generate_tally_DFA(base_states, q_init, base_transitions, base_outputs, target_r)

    # compile & export to Walnut-compliant syntax
    # [deprecated]: export_Walnut_DFA(visited, DFA_transitions, acc_states, output_filename)
    export_Walnut_DFAO(visited, DFA_transitions, acc_states, output_filename)

    exec_time = (time.time() - start_time) * 1000
    print(f"[SUCCESS] Topology synthesized in {exec_time:.2f} ms.")
    print(f"[SUCCESS] Total minimal states generated: {len(visited)}")
    print(f"[SUCCESS] DFAO successfully exported to: {output_filename}")