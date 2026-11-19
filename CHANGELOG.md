## [unreleased]

### 🚀 Features

- Scaffold workspace with complete core crate and area skeletons
- *(transiogram)* Implement experimental transiograms, 1D and 3D models
- *(hmm)* Implement emissions, forward-backward, Viterbi and Baum-Welch
- *(hsmm)* Implement hidden semi-Markov models and state-count selection
- *(chain)* Implement Markov chain fits, tests, entropy, sojourns and simulation
- *(tpsim)* Implement transition-probability cokriging, SIS and quenching
- *(mcrf)* Implement MCRF simulation and MCP prediction
- *(py)* Bind chains, HMMs, HSMMs, transiograms and grid simulation
- *(py)* Serialise Entropy, Sojourn, Prediction and Realizations

### 🐛 Bug Fixes

- *(core)* Keep expm_rate stochastic at huge lags and round-trip NaN in serde_mat
- *(chain)* Validate order, simulation and bridge sizes before allocating
- *(hmm)* Address review findings in emissions, forward-backward, Viterbi and EM
- *(hsmm)* Reject absurd max_duration and draw counts before allocating
- *(transiogram)* Bound the table search and cost it before allocating
- *(tpsim)* Check cokriging and simulation memory before allocating
- *(mcrf)* Check simulation memory before allocating and test peaks against accepted budgets
- *(hmm)* Keep categorical pseudocounts when restarts reinitialise emissions
- *(hsmm)* Keep pseudocounts across restarts and expose Duration::validate
- *(py)* Freeze the cell that holds a hard point and print test results compactly
- *(plot)* Put the strip-log legend below the strip
- *(core)* Lower the available memory to the tightest cgroup v2 headroom
- *(transiogram)* Raise InvalidInput before allocating an oversized (n, k, k) transition stack
- *(chain)* Reject order tests beyond the data and bound bootstrap memory
- *(core)* Budget memory from process limits and check Grid.coords and emission samples

### 📚 Documentation

- Add charter and roadmap
- Add API contract and per-crate design notes
- *(spec)* Record the shared hmm helpers and NaN handling in serde_mat
- *(examples)* Add narrative examples for every area and the boitata integration
- Add a guide page per capability, API pages and the example pages to the site
- *(hsmm)* Add NumPy docstrings to HiddenSemiMarkovModel methods and getters
- *(examples)* Add an HSMM duration-law page with criteria and a brute-force check
- *(examples)* Add an entrywise transiogram and closeness page
- *(hmm)* Add a counts and mixtures example and document GaussianMixture.log_density
- Record the true status of each milestone in the README roadmap

### 🧪 Testing

- *(py)* Cover every binding with theory checks against brute force and closed forms
- *(hsmm)* Hold brute-force checks to the spec tolerances and test gamma from segment sums
- *(tpsim)* Run M5 at the spec cell for hard data and full-range auto-transiograms
- *(integration)* Segment logs with an HMM and check T-PROGS and MCRF grids for M7

### ⚙️ Miscellaneous Tasks

- Add the compat task that installs the wheel via pip, uv, pixi and conda
