# Use Conversation

Start `loom-serve` with a prepared chat deployment. For example, with a
CPU-capable deployment at `models/chat/model.json`:

```sh
loom-serve --device=task --pool_backing=fixed \
  --listen=127.0.0.1:8080 --model=models/chat/model.json
```

Open `http://127.0.0.1:8080/dashboard/chat.html`, or choose **Conversation**
from **Telemetry** on the same server. Select a **Model**, enter a message,
and press **Send** or **Ctrl+Enter** (**⌘+Enter** on macOS). **Esc** stops
the current response. The browser loads the workspace from the executable.
Behind a relay, open the same path under the relay's mount prefix.
[Serving and relay setup](observe-model-server.md#watch-the-live-dashboard)

## Models, prompts, and tools

The model picker lists prepared deployments by name and groups them by what
the composer can do. Choose a chat model for conversation, an image model
to generate a picture, or a speech model to speak the input. **Refresh
models** picks up changes to the server. Attach or drop UTF-8 text files
into chat. A chat model advertising image input also accepts supported
image attachments.

Open **Inspector** on a narrow screen. **Prompts** holds system instructions
and a user template with one `{{input}}` placeholder. The composer counts
the next request with history, attachments, and offered tools. The two
prompt fields show their individual text counts. An ellipsis marks a pending
count. Leaving **Output limit** blank uses the deployment default and the
remaining context. Use **Sampling** for decoding controls, **Presets** for
named configurations, and **Set project defaults** for new chats.

Changes to the available tools take effect on the next input. Once the model
calls a tool, its arguments and results appear in the transcript. **Auto-run
and return results** lets enabled built-ins complete the turn. Switch it off
to inspect arguments, run a call, and choose when to continue.

| Tool | What happens when it runs |
| --- | --- |
| HTML preview | Displays a self-contained interactive document in a sandboxed frame. Inline styles, scripts, and event handlers work there. Drag the resize handle to adjust the frame. **Download HTML** saves the document. |
| Code mode (JavaScript) | Runs an async function body in a fresh offline worker. The return value or console output goes back to the model. **Stop** terminates the run. |
| Image generation | Sends the model's image prompt to the selected image deployment and displays the returned image. This option appears when an image deployment is available. |
| Function schemas | Offers the supplied function definitions to the model. Enter the results for those calls in the transcript. |

Under **Tools → MCP servers…**, enter a Streamable HTTP server URL and choose
**Discover**. Inspect the returned descriptions and input schemas, enable the
tools you want, and choose **Apply to chat**. Automatic execution requires an
explicit grant for each remote tool as well as the chat's auto-run setting.
Large tool results fold in the transcript and expand on demand.

MCP requests go directly from the browser to the chosen server. A
cross-origin endpoint must permit the workspace's origin and request headers
through CORS. For a stateful endpoint it must also expose `Mcp-Session-Id`
so the browser can continue the session. Enter bearer tokens or API keys
in **Credential · this tab**. Credentials live in tab memory. The saved
catalog contains server addresses and tool schemas, so use the credential
field for secrets.

## Keep, branch, or move a chat

**Project** selects a browser-local workspace. Use that picker to create or
delete a project. **Chat** selects, renames, pins, or deletes a conversation.
**Ctrl+K** (**⌘+K** on macOS) searches chat titles and message text in the
current project. Projects, sessions, attachments, and preview sizes live in
IndexedDB for this browser origin and server mount. Clearing site data
removes them.

Use **Fork here** on an eligible live reply to reserve its retained KV state.
The branch displays the server-confirmed checkpoint position. **Fork from
history** starts at the selected message using the saved transcript. The
first reply in the branch reports how many prompt tokens the server actually
reused, when the model supplies usage. **Edit in fork**, **Retry**, and
**Compare…** preserve the original conversation while you change the input,
model, or sampling settings.

**Export chat** or `/export` downloads a conversation JSON file, including
attachments, tool output, and preview sizes. On another server, choose
**Import chat**, pick the file, select an available model, and continue.
The import gets a fresh session identity. KV checkpoints belong to the
original server. Remote-tool grants and credentials stay with the browser
that granted them.

**Incognito** keeps its chats and tool configuration in tab memory.
Reloading or closing the page discards them. Inference still sends the
conversation to the selected deployment, and running a remote tool sends
its arguments to the configured MCP endpoint.

## Speak and inspect a turn

**Dictate** records a message through a prepared recognition model. Choose
**Finish dictation** to insert the transcript at the cursor. **Voice** also
accepts recorded audio files. To hear replies, select a speech model and
enter a voice ID accepted by that deployment.

With chat, recognition, and speech models available, choose **Start
turn-based voice chat**. A pause ends your utterance. The workspace
transcribes it, submits the chat turn, plays the reply, then listens again.
Microphone capture requires permission and a secure browser context such as
HTTPS or localhost.

Each streamed reply shows activity, including prefill progress when the
server reports it and tool arguments while they arrive. Select a bar under
**Tokens by turn** to inspect an earlier turn's prompt, cache, output, and
latency figures. Open **Telemetry** in a second tab for server-wide activity.
An open background tab retains up to 512 samples per chart and draws them
when you return. [Read the telemetry graphs](observe-model-server.md#watch-the-live-dashboard)
