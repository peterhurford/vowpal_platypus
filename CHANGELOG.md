## v1.0.0

**Major-ish Changes**

* Changed the `run` function to dramatically reduce the amount that needs to be hardcoded.
* Added grid searching.
* The way to run multiple multi-core models (e.g., ensembles) is now different.
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

#### v0.1.1

* Fix conflict between spanning port and daemon port.
* Allow VP to work on one core.
* Fix bug where `quiet` would not actually quiet model output.

## v0.1.0

* Initial pre-release.
