# WebSocket transport

The WebSocket adapter upgrades a claimed HTTP/1.1 connection in place. HTTP
continues to own its carrier, receive resources, application claim, and accepted
sends. Installing the receiver before sending `101` makes the first WebSocket
frame part of the same ordered receive stream. Carrier handlers never change.
HTTP reclamation releases the protocol only after the application and I/O have
both retired, including when the peer disconnects during an accepted CPU job.

`handshake.h` validates the RFC 6455 request and formats the challenge response.
A browser Origin requires an explicit exact allowlist match; clients without an
Origin use the same handshake without that browser policy. The caller supplies
its optional subprotocol. Compression and other extensions are not negotiated.
The protocol's fixed SHA-1 challenge requires no cryptographic dependency and
does not authenticate the peer.

`connection.h` owns bounded message assembly and publication. Client frames are
masked, lengths are canonical, and text and close reasons are valid UTF-8.
Control frames can arrive between fragments of a data message. The application
receives complete immutable messages in completion order, including ping, pong,
and close; it remains the sole output producer. Taking a message transfers its
receipt without copying or returning allocation credit. A CPU job can borrow
that receipt while the owner takes later controls. The application releases all
receipts after their borrowers retire, before relinquishing its HTTP claim.

The default limits are 512 KiB per message, 1 MiB of allocated payload across
queued, taken, and incomplete messages, and 64 records. Allocations grow on demand;
fragment growth is geometric, and allocated slack consumes the same byte budget.
An allocator growing a fragmented message may temporarily retain its old buffer,
bounded by one additional message capacity. Empty messages consume record credit.
Overflow fails only that connection rather than dropping input or allowing an
unbounded queue. These are transport budgets,
independent of native model input/output credit and session audio retention.

Network callbacks only frame, unmask, and publish bytes. They do not parse JSON,
invoke models, or wait for application work. The existing HTTP wake joins input,
send credit, and failure with the application owner. Outgoing complete frames
are generated directly into admitted carrier storage, copying payload once and
retaining no caller buffer after the send call. `finish` drains accepted sends;
`abort` cancels network delivery. Neither waits for the network.
