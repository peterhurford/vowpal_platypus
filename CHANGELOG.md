## v1.1

**Major-ish Changes**

* Added grid searching.
* Added `train_on` and `predict_on` as a quick syntax for writing out models.
* The way to run multiple multi-core models (e.g., ensembles) is now different.
* Change `safe_remove` to take wildcard arguments.
* Ports now spin down silently on process close rather than at the end. Only ports that exist are spun down.
* Added quieting to `daemon_predict`.

**Minor/Technical Changes**

* The VW class now operates on a parameter hash rather than passed in values.
* Spanning ports are no longer defined.
* Only spin down ports that exist.
* Removed a unneeded debugging statement.
* For `load_file`, error if the file is empty.

-

#### v1.0.1

* Fix conflict between spanning port and daemon port.
* Allow VP to work on one core.
* Fix bug where `quiet` would not actually quiet model output.

## v1.0

* Initial pre-release.
