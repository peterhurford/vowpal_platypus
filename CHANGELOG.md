#### v2.1.5

* Added a function `confusion_matrix` to return a dictionary with the confusion matrix data.

#### v2.1.4

* Data from a Cassandra query can be loaded using `load_cassandra_query`.

#### v2.1.3

* Only creates one thread per VP daemon.

#### v2.1.2

* Handle UTF encoding issues in VP input.

#### v2.1.1

* Allow custom port to be specified for a daemon.
* Switch to using Pathos library for multiprocessing.
* Fixed bugs in creating and using daemons for predictions.
* An error that occurs on a particular core is now captured and printed clearly to the user.
* The spanning tree now spins down when an error occurs, preventing further errors occuring from trying to reconnect to a spanning tree.

## v2.1

**Major-ish Changes**

* Evaluation functions added to VW: Log loss, RMSE, percent correct, TPR, TNR, FPR, FNR, precision, recall, F-score, MCC, average accuracy, and AUC.
* Models now import from `vowpal_platypus.models`
* Logistic regression now works from 0 to 1 instead of -1 to 1.
* Utility functions now import from `vowpal_platypus.utils`
* Daemon functions now import from `vowpal_platypus.daemon`
* `run_parallel` supports the pre v2 `run` function for running a manual function for each model on each core.

**Minor/Technical Changes**

* Announcement of shuffling no longer erroneously occurs on single core operations.

-

## v2.0.0

**Major-ish Changes**

* Changed the `run` function to dramatically reduce the amount that needs to be hardcoded.
* Added grid searching.
* Added `termination` parameter for BFGS.
* Change `safe_remove` to take wildcard arguments.
* `split_file` takes an argument to remove headers and now returns a list of the resulting filenames.
* Added error messages to parameter checking.
* Announce multicore shuffling.
* Added quieting to `daemon_predict`.
* `passes` defaults to 1.

**Minor/Technical Changes**

* Ports now spin down silently on process close rather than at the end. Only ports that exist are spun down.
* The VW class now operates on a parameter hash rather than passed in values.
* Spanning ports are no longer defined.
* Only spin down ports that exist.
* Removed a unneeded debugging statement.
* For `load_file`, error if the file is empty.
* Fixed a bug in `load_file` for length-1 keys.
* Fixed a bug in hash parsing for non-string input.

-

#### v1.0.1

* Fix conflict between spanning port and daemon port.
* Allow VP to work on one core.
* Fix bug where `quiet` would not actually quiet model output.

## v1.0.0

* Initial pre-release.
