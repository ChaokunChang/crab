# Third-party notices

Crab includes two runtime archives so the iFlow replay example can run without
installing a live iFlow release or depending on an external package registry at
runtime. These components are not covered by Crab's MIT License; they retain
their respective upstream licenses.

## Node.js 22.18.0 for Linux x86-64

- Repository file:
  `integrations/sandboxes/iflow/cache/node-v22.18.0-linux-x64.tar.xz`
- Upstream project: [Node.js](https://github.com/nodejs/node)
- Upstream archive:
  <https://nodejs.org/dist/v22.18.0/node-v22.18.0-linux-x64.tar.xz>
- SHA-256:
  `c1bfeecf1d7404fa74728f9db72e697decbd8119ccc6f5a294d795756dfcfca7`
- License: the Node.js license and bundled dependency notices are included in
  the archive, beginning with `node-v22.18.0-linux-x64/LICENSE`.

The repository file matches the checksum published in Node.js 22.18.0's
official `SHASUMS256.txt`.

## iFlow CLI replay bundle

- Repository file:
  `integrations/sandboxes/iflow/cache/iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz`
- Upstream project: [iFlow CLI](https://github.com/iflow-ai/iflow-cli)
- Upstream npm package:
  [`@iflow-ai/iflow-cli`](https://www.npmjs.com/package/@iflow-ai/iflow-cli)
- Version reported by the bundled `package/package.json`: `0.4.7`
- SHA-256:
  `e5bec219dc8a17e6b815e715a74d42fd00a5df71d4d5712e96e41fdfdbe10289`
- License: Apache License 2.0, included as `package/LICENSE`; licenses for
  bundled runtime dependencies are included beneath `package/node_modules/`
  and `package/vendors/`.

This is a replay-oriented, self-contained repack of the upstream 0.4.7 npm
package. It includes runtime dependencies that the registry tarball normally
resolves during installation and is therefore not byte-for-byte identical to
the npm registry artifact. The historical repository filename predates the
0.4.7 package metadata; consumers should use the version and checksum above.

The archive is used only to prepare the iFlow sandbox image for the replay and
iFlow SDK examples. Crab does not claim ownership of iFlow CLI, Node.js, or
their bundled dependencies.
