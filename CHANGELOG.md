#### v1.0.2

* Reverted spanning port.
* Spin down ports silently on process close rather than at the end.
* Only spin down ports that exist.
* Removed a unneeded debugging statement.
* For `load_file`, error if the file is empty.
* Add quieting to `daemon_predict`.

#### v1.0.1

* Fix conflict between spanning port and daemon port.
* Allow VP to work on one core.
* Fix bug where `quiet` would not actually quiet model output.

## v1.0

* Initial pre-release.
