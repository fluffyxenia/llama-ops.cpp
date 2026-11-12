# ggml-rpc: `supports_op` was a stub for 2+ years — fixed, tested, one new bug found

Context for a future PR writeup. Found and fixed while training a small custom RL
policy net (conv+dense, ~289K params) remotely over `ggml-rpc` on an Intel UHD 600
iGPU via Vulkan, as part of an unrelated Sonic-3-&-Knuckles RL project.

## The bug

`ggml_backend_rpc_device_supports_op()` in `ggml/src/ggml-rpc/ggml-rpc.cpp` has
unconditionally returned `true` since the RPC backend was added
(`5e31828` "ggml : add RPC backend (#6829)", 2024-05-14; the stub itself is present
by `172c825` "rpc : fix ggml_backend_rpc_supports_buft() (#7918)", 2024-06-13 — over
two years old as of this writing, 2026-11-09):

```cpp
static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}
```

Because `ggml_backend_sched`'s automatic CPU-fallback-for-unsupported-ops relies
entirely on `supports_op` to decide whether to reroute a node, this stub silently
disables that fallback for ANY RPC-backed device: unsupported ops get dispatched to
the remote device anyway and crash/abort there instead of being caught and rerouted
locally ahead of time. Reproduced directly: a v3 policy-net architecture whose
backward pass needs `GGML_OP_OUT_PROD` (dense-layer weight gradients) for a
mixed f16/f32 input combination that Intel's Mesa ANV Vulkan backend doesn't
implement, crashed the rpc-server with `Missing op: OUT_PROD for f16 and f32 to f32`
the instant `ggml_opt_eval()` tried to execute it remotely - exactly the failure
mode `supports_op` exists to prevent, just never checked.

Likely never noticed because RPC's primary real-world use (splitting one known-good
LLM's inference graph across machines) only ever exercises a small, universally-
supported op set - this gap only bites genuinely novel graphs (e.g. training an
arbitrary small custom architecture) hitting a real per-backend op coverage edge.

## The fix

Implemented a real check instead of guessing: reuses the EXISTING
`serialize_graph()`/`create_node()` infrastructure already used by
`RPC_CMD_GRAPH_COMPUTE` (same wire format: `device | uid | n_nodes | nodes |
n_tensors | tensors`), just for a one-node, non-executing query instead of a real
graph execution.

- New command `RPC_CMD_SUPPORTS_OP` (added at the end of the `rpc_cmd` enum, after
  `RPC_CMD_COMM_FREE`, specifically to not disturb `RPC_CMD_HELLO`'s pinned value of
  14 or any other existing command's wire-protocol number).
- New response struct `rpc_msg_supports_op_rsp { uint8_t result; }` (same shape as
  the existing `rpc_msg_copy_tensor_rsp`).
- Client (`ggml_backend_rpc_device_supports_op`): builds a minimal 1-node
  `ggml_cgraph` wrapping just the `op` tensor, serializes it via the existing
  `serialize_graph()` (uid=0, so the server's graph-reuse cache is never touched by
  these one-shot queries), sends it as `RPC_CMD_SUPPORTS_OP`, returns the bool.
- Server (`rpc_server::supports_op`): parses the same wire format graph_compute
  parses (skipping the uid-based graph-cache logic entirely, since queries are
  always uid=0), reconstructs the op node via the existing `create_node()`, then
  calls the REAL `ggml_backend_dev_supports_op(ggml_backend_get_device(backends[device]), op)`
  against whichever device the server actually serves - no execution, no buffer
  allocation.

Diff is self-contained to `ggml/src/ggml-rpc/ggml-rpc.cpp` (enum entry, cmd-name
case, response struct, client function body, server method + declaration, dispatch
switch case). No protocol version bump attempted here - would need
`RPC_PROTO_MAJOR_VERSION`/`HELLO` handshake consideration for real upstream
submission (old clients talking to a server with this patch, or vice versa, should
degrade safely since unknown commands are already rejected by the dispatch switch's
default case - but this wasn't verified against an actually-mismatched-version pair).

## Verified, not assumed

Built against this tree's Vulkan-enabled ggml (`cmake -DGGML_RPC=ON .` on top of the
existing `-DGGML_VULKAN=ON` configure, then `cmake --build . --target
ggml-rpc-server`). Ran a dedicated test program
(`/home/foxo/sonic-demo-recorder/rpc_train_test.cpp`) against a localhost-only
rpc-server (`-H 127.0.0.1`, never exposed beyond loopback) serving `Vulkan0,CPU`:

- **Before the fix**: crashed on the very first `ggml_opt_eval()` call with the
  `OUT_PROD` error above - the remote server aborted mid-graph-compute.
- **After the fix**: step 0 completed and printed `loss = 9.617403` - the EXACT
  same value the original local (non-RPC) validation of this same architecture
  produced hours earlier (same RNG seed, same init) - a genuine bit-for-bit numeric
  cross-check, not just "didn't crash." Confirms `OUT_PROD` correctly got rerouted
  to the CPU device instead of being sent to Vulkan.

## New issue found (not yet fixed) - worth its own look

Step 1 of the same test hit a different, previously-unreachable assertion:

```
ggml-opt.cpp:783: GGML_ASSERT(opt_ctx->eval_ready) failed
```

This never surfaced before because, prior to this fix, every op was (incorrectly)
assigned to the single RPC/Vulkan backend - there was no real per-node backend
split for `ggml_opt`'s internal state tracking to disagree with. Now that
`supports_op` is real, `ggml_backend_sched` can legitimately split a graph across
two different backends (Vulkan for most ops, CPU for the one op it can't do), and
something about `ggml_opt`'s assumption that a graph's backend assignment/structure
stays stable across repeated `ggml_opt_eval()` calls on the same `opt_ctx` breaks
once that split is real. Root cause not yet investigated - flagging for later
rather than guessing. Reproducer: the same test program/server setup above, just
let it run past step 0.

## Context/attribution

Found and fixed by Foxo (project owner) + Claude (Sonnet 5, orchestrating this
session) while chasing an unrelated question: "can ggml-rpc run training
instructions at all, not just inference?" Yes it can (verified above) - this bug
was actively preventing it from doing so correctly for the one case that happened
to need CPU fallback.
